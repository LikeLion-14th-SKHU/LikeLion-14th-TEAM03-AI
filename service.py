import os
import json
import re
import logging
import anthropic
import lookup  # lookup.py 연결
from datetime import datetime
from dotenv import load_dotenv
from schemas import RecommendRequest, ConcernRequest, JourneyRequest

load_dotenv()

log = logging.getLogger("ai-server")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    log.warning("⚠️ 경고: .env 파일에서 ANTHROPIC_API_KEY를 찾을 수 없습니다.")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)

MODEL_NAME = "claude-haiku-4-5-20251001"


def call_claude(prompt: str, temperature: float, max_tokens: int) -> str:
    """
    Claude를 호출하고 응답 텍스트를 반환한다.

    assistant 턴을 "{"로 미리 채워(prefill) 모델이 코드블록(```json)이나
    "알겠습니다" 같은 서두 없이 곧바로 JSON 본문부터 이어 쓰게 만든다.
    Claude API는 prefill한 문자열을 응답에 다시 포함시키지 않으므로,
    파싱 전에 "{"를 앞에 붙여 복원해야 한다.
    """
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"},
        ],
    )
    return "{" + response.content[0].text



def parse_json_safely(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)

    match = re.search(r'\{.*\}', stripped, re.DOTALL)
    if match:
        clean_text = match.group(0)
    else:
        open_idx = stripped.find('{')
        if open_idx == -1:
            log.error("AI 응답에서 JSON 시작 지점('{')조차 찾을 수 없음. 원문:\n%s", text)
            raise ValueError("AI 응답에서 JSON 포맷을 찾을 수 없습니다.")
        clean_text = stripped[open_idx:]
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        log.error("AI 응답 JSON 파싱 실패 (%s). 원문:\n%s", e, text)
        recovered = _try_recover_truncated_json(clean_text)
        if recovered is not None:
            log.warning("잘린 JSON에서 %d개 성분 카드를 복구해 사용합니다.",
                        len(recovered.get("cosmetic", {}).get("recommended", [])))
            return recovered
        raise


def _try_recover_truncated_json(clean_text: str):
    idx = clean_text.rfind('"recommended"')
    if idx == -1:
        return None

    arr_start = clean_text.find('[', idx)
    if arr_start == -1:
        return None

    depth = 0
    last_complete_end = None
    i = arr_start
    in_string = False
    escape = False
    obj_start = None

    while i < len(clean_text):
        ch = clean_text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and obj_start is not None:
                    last_complete_end = i
        i += 1

    if last_complete_end is None:
        return None

    partial_array = clean_text[arr_start:last_complete_end + 1] + "]"

    try:
        recovered_list = json.loads(partial_array)
    except json.JSONDecodeError:
        return None

    if not recovered_list:
        return None

    prefix = clean_text[:idx]
    try:
        skeleton = prefix.rstrip().rstrip(",")
        if not skeleton.endswith("}"):
            skeleton += "}}"
        else:
            skeleton += "}"
        base = json.loads(skeleton)
    except json.JSONDecodeError:
        base = {"type_description": "", "cosmetic": {"summary": ""}}

    base.setdefault("cosmetic", {})
    base["cosmetic"]["recommended"] = recovered_list
    base["cosmetic"].setdefault("excluded", [])
    base["cosmetic"].setdefault("closing", "")
    base.setdefault("card", {"type": "INITIAL"})
    return base


def get_recommendation_solution(req: RecommendRequest) -> dict:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    base_type = req.skin.base_type
    flags     = req.skin.flags.model_dump()
    safety    = req.skin.safety.model_dump()
    dday      = req.dday.remaining_days

    GOAL_MAP = {"피지량": "sebum", "모공": "pore", "댕김": "dry",
                "붉은기": "redness", "여드름": "acne", "흔적": "mark"}
    ts = req.skin.trouble_scores
    goal = GOAL_MAP.get(max(ts, key=ts.get)) if ts else None

    active_flags = lookup.get_active_attributes(flags, safety)
    ingredients = lookup.get_type_ingredients(base_type)
    ingredients, excluded_flag = lookup.filter_by_flags(
        ingredients, active_flags, safety.get("retinol_history", "never")
    )
    ingredients, excluded_dday = lookup.filter_by_dday(ingredients, dday, goal)

    if not ingredients:
        ingredients = lookup.get_safe_baseline(base_type)

    excluded_by_system = excluded_flag + excluded_dday
    ingredient_names = [i["name"] for i in ingredients]

    is_sensitive = "SENSITIVE" in active_flags

    products_by_ingredient = lookup.find_products_by_ingredient(
        ingredient_names=ingredient_names,
        active_flags=active_flags,
        sensitive=is_sensitive,
    )
    ingredient_product_map = {
        name: [p["product_id"] for p in prods]
        for name, prods in products_by_ingredient.items()
    }

    ing_info = [
        {"name": i["name"], "effect_days": i["effect_days"],
         "status": i["dday_status"], "evidence": i.get("evidence", "")}
        for i in ingredients
    ]

    type_data = lookup.get_type_data(base_type)
    cleansing_guide  = type_data.get("cleansing_guide", [])
    management_guide = type_data.get("management_guide", [])
    routine          = lookup.get_routine_guide()

    inflammatory_notice = ""
    if safety.get("inflammatory"):
        inflammatory_notice = (
            "[중요 — 반드시 반영할 것]\n"
            "    사용자가 붉고 아프며 크게 잡히는 여드름이 있다고 답했습니다.\n"
            "    - summary 첫 문장에서 홈케어의 한계를 언급하고 피부과 상담을 권할 것\n"
            "    - closing에도 다시 한 번 전문가 상담을 안내할 것\n"
            "    - 다만 중증도를 판정하지 말 것. '중등도', '심한 여드름' 같은 표현 금지"
        )

    prompt = f"""
    당신은 고객의 피부 고민에 깊이 공감하고 실질적인 도움을 주는 다정하고 전문적인 뷰티 어드바이저입니다.
    (주의: 응답은 반드시 다른 텍스트 없이 JSON 포맷으로만 출력하세요)

    [파이썬이 확정한 정답지 — 반드시 이 데이터만 사용]
    - 추천 성분: {ingredient_names}
    - 성분별 기간 정보: {ing_info}
    - 성분별 매칭 제품 ID: {ingredient_product_map}
    - 시스템이 제외한 성분과 사유: {excluded_by_system}

    [작성 절대 규칙: 디테일 및 톤앤매너]
    1. 어조와 호칭: 기계적인 말투를 버리고 "고객님"이라는 호칭을 사용하며, 다정하고 공감하는 '~해요', '~할 수 있어요' 등의 부드러운 말투를 사용하세요.
    2. 영어 타입명 번역: DRY, dehydrated, sensitive 등 영어 타입명을 그대로 노출하지 마세요. 반드시 "건성", "속당김", "민감성", "지성" 등 자연스러운 한국어로 풀어서 다정하게 작성하세요.
       (예시: "고객님은 건성 피부이시면서, 속당김과 붉은기가 고민인 민감성 피부 타입을 가지고 계시네요. 많이 불편하셨겠어요!")
    3. timeline 변주: 기계적으로 "N일 내 N일 후 효과를 기대할 수 있어..."라는 문구를 절대 반복하지 마세요. 성분마다 뉘앙스와 문장 구조를 다르게 변주하여 자연스럽게 작성하세요.
    4. cautions 구체화: "특별한 주의사항은 없습니다" 또는 "개인차가 있을 수 있습니다" 같은 성의 없는 답변은 절대 금지합니다. 사용 시간대, 제형 조합 등 구체적이고 실용적인 팁을 1문장으로 적어주세요.
    5. 성분 및 제품 제어: 추천 성분은 위 목록에 있는 것만 사용하며, products 배열은 해당 성분에 매칭된 ID만 정확히 넣으세요.
    6. "치료", "완치", 시술 언급은 절대 금지하며 "관리", "도움을 줄 수 있음"으로 표현하세요.

    {inflammatory_notice}

    [사용자 데이터]
    - 피부 타입: {base_type} / 중첩: {flags}
    - 축 점수: {req.skin.axis_scores}
    - 고민 점수: {ts}
    - D-Day: {dday}일 남음 (이벤트: {req.dday.event_type})
    - 주요 고민: {req.concern.get('raw')}
    - 히스토리: {[c.model_dump() for c in req.history_cards]}

    [출력 JSON 포맷]
    {{
      "type_description": "고객님의 피부 타입에 대한 다정한 한국어 요약과 공감",
      "cosmetic": {{
        "summary": "종합 코멘트 (고민 점수와 목표 이벤트를 자연스럽게 언급하며 공감, 2~3문장)",
        "recommended": [
          {{
            "ingredient": "성분명",
            "effect": "기대 효과 (20자 이내 명사형)",
            "timeline": "남은 D-Day와 효과 발현 시기를 연관 지은 자연스러운 문장 (반복 금지)",
            "cautions": ["형식적이지 않은 실용적인 주의사항 1문장"],
            "products": ["해당 성분에 매칭된 제품 ID만 배열로"]
          }}
        ],
        "excluded": [
          {{ "ingredient": "성분명", "reason": "제외 사유 1문장 (다정한 톤으로)" }}
        ],
        "closing": "목표 일정을 응원하는 따뜻한 마무리 멘트. 자외선 차단 안내 포함"
      }},
      "card": {{
        "type": "INITIAL",
        "created_at": "{now_str}",
        "dday_at_time": {dday},
        "concern_summary": "고민 1줄 요약",
        "prescribed_ingredients": {ingredient_names},
        "excluded_ingredients": {[x["ingredient"] for x in excluded_by_system]}
      }}
    }}
    """

    text = call_claude(prompt, temperature=0.3, max_tokens=4000)
    data = parse_json_safely(text)

    allowed = set(ingredient_names)
    cos = data.setdefault("cosmetic", {})
    cos["recommended"] = [
        r for r in cos.get("recommended", [])
        if r.get("ingredient") in allowed
    ]

    sys_excluded = {x["ingredient"] for x in excluded_by_system}
    cos["excluded"] = [
        e for e in cos.get("excluded", [])
        if e.get("ingredient") in sys_excluded
    ]

    for item in cos["recommended"]:
        own_candidates = products_by_ingredient.get(item.get("ingredient"), [])
        item["products"] = lookup.validate_product_ids(item.get("products", []), own_candidates)

    # ==========================
    # 💡 수정된 부분: recommended 배열에 있는 모든 제품을 빠짐없이 담음
    # (단, 서로 다른 성분에서 동일한 제품이 추천될 수 있으므로 product_id 기준으로 중복 제거)
    # ==========================
    detail_by_id = {}
    for item in cos["recommended"]:
        for pid in item.get("products", []):
            p = lookup.get_product(pid)
            if p and p["product_id"] not in detail_by_id:
                detail_by_id[p["product_id"]] = p

    data["products_detail"] = [
        {"product_id": p["product_id"], "name": p["name"], "category": p["category"]}
        for p in detail_by_id.values()
    ]
    # ==========================

    data["cleansing"] = {"guide": cleansing_guide, "management": management_guide}
    data["skincare_order"] = routine

    card = data.setdefault("card", {})
    card["type"] = "INITIAL"
    card["dday_at_time"] = dday
    card["prescribed_ingredients"] = ingredient_names
    card["excluded_ingredients"] = [x["ingredient"] for x in excluded_by_system]
    card.setdefault("created_at", now_str)
    card["concern_summary"] = req.concern.get("raw", "")

    data["needs_medical_consult"] = bool(safety.get("inflammatory"))

    return data


def get_concern_update(req: ConcernRequest) -> dict:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    base_type = req.skin.base_type
    flags     = req.skin.flags.model_dump()
    safety    = req.skin.safety.model_dump()
    dday      = req.dday.remaining_days

    GOAL_MAP = {"피지량": "sebum", "모공": "pore", "댕김": "dry",
                "붉은기": "redness", "여드름": "acne", "흔적": "mark"}
    ts = req.skin.trouble_scores
    goal = GOAL_MAP.get(max(ts, key=ts.get)) if ts else None

    active_flags = lookup.get_active_attributes(flags, safety)

    all_ingredients = [
        i for i in lookup._load()["ingredients"]
        if i.get("grade") == "cosmetic" and i.get("recommend_product")
    ]
    ingredients, excluded_flag = lookup.filter_by_flags(
        all_ingredients, active_flags, safety.get("retinol_history", "never")
    )
    ingredients, excluded_dday = lookup.filter_by_dday(ingredients, dday, goal)
    if not ingredients:
        ingredients = lookup.get_safe_baseline(base_type)

    allowed_names = [i["name"] for i in ingredients]
    excluded_by_system = excluded_flag + excluded_dday

    already = []
    for c in req.history_cards:
        already.extend(c.prescribed_ingredients)
    already = list(dict.fromkeys(already))

    newly_available = [n for n in allowed_names if n not in already]
    is_sensitive = "SENSITIVE" in active_flags

    # flags를 한국어 라벨로 미리 변환해 프롬프트에 넣는다.
    # 파이썬 dict({'sensitive': True, ...})를 그대로 넣으면 AI가 그 표기를
    # 그대로 옮겨 적을 수 있다. 1차에서 겪었던 영어 노출 문제의 재발을 막는다.
    FLAG_LABEL = {"dehydrated": "수분 부족", "sensitive": "민감성",
                  "acne": "여드름", "mark_prone": "흔적이 잘 남는 편"}
    active_flag_labels = [FLAG_LABEL[k] for k, v in flags.items() if v and k in FLAG_LABEL]
    TYPE_LABEL = {"OILY": "지성", "DRY": "건성", "COMBO": "복합성", "NORMAL": "정상"}
    base_type_label = TYPE_LABEL.get(base_type, base_type)

    prompt = f"""
    당신은 고객의 피부 고민에 깊이 공감하고, 다정하고 친절하게 해결책을 제안하는 '전문 스킨케어 어드바이저'입니다.
    사용자가 관리 도중 새로운 고민이나 변화를 등록했습니다.
    (주의: 응답은 반드시 다른 텍스트 없이 JSON 포맷으로만 출력하세요)

    [파이썬이 확정한 정답지 — 반드시 이 데이터만 사용]
    - 지금 사용 가능한 성분: {allowed_names}
    - 이미 처방했던 성분: {already}
    - 새로 추가 가능한 성분: {newly_available}
    - 시스템이 제외한 성분과 사유: {excluded_by_system}

    [페르소나 및 말투 (Tone & Manner)]
    1. 딱딱하고 차가운 전문가 말투를 피하고, 친근하고 따뜻한 어조(예: "~하셨군요", "~해 드릴게요", "많이 신경 쓰이셨겠어요!")를 사용하세요.
    2. 고객이 변화를 이야기했을 때 그 감정에 먼저 공감한 뒤 조치를 설명하세요. 바로 지시부터 하지 마세요.
    3. 걱정을 키우지 않되 가볍게 넘기지도 마세요. "충분히 조절할 수 있는 부분이에요"처럼 안심시키는 톤을 유지하세요.

    [절대 규칙 및 지식 제한 (Hallucination 제어)]
    1. 영문 타입 번역 강제: 피부 타입과 속성을 언급할 때 절대 영어(OILY, DRY, sensitive, dehydrated 등)나 파이썬 딕셔너리 형태를 그대로 쓰지 마세요. 반드시 한국어로 자연스럽게 풀어서 작성하세요.
       - (X) "고객님은 OILY 타입이시고 sensitive 상태시네요."
       - (O) "지성이시면서 민감성 고민도 함께 있으시네요."
    2. 원인을 섣불리 단정하지 마세요. "제품이 안 맞아서입니다" 대신 "초기 적응 반응일 수 있어요. 다만 ~가능성도 있으니" 형태로 가능성을 제시하세요.
    3. 위 [파이썬이 확정한 정답지] 목록에 없는 성분을 임의로 언급하거나 추가하지 마세요.
    4. 최초 고민을 잊지 말고, 이번 새 고민과 함께 고려해서 조언하세요.
    5. "치료", "완치" 표현을 쓰지 마세요. 시술 언급 및 중증도 판정을 금지합니다.
    6. [중요] 사용자가 '탄력', '미백' 등 새로운 피부 고민을 언급하거나 특정 성분의 추가를 원할 경우, 반드시 action을 "ADD"로 설정하고 adjusted_ingredients.add에 해당 성분을 포함시키세요.

    [항목별 세부 작성 가이드]
    - response: 기계적으로 지시만 나열하지 마세요. 먼저 공감하고, 이유를 설명하고, 구체적인 행동을 안내하는 흐름으로 2~3문장 작성하세요.
      (예: "많이 신경 쓰이셨겠어요. 세라마이드를 새로 시작하신 지 얼마 안 돼서 나타나는 초기 반응일 수 있어요. 며칠만 더 지켜봐 주시고, 계속되면 사용 빈도를 줄여볼게요.")
    - cautions: "개인차가 있습니다" 같은 성의 없는 답변 금지. 구체적인 관찰 기준이나 행동 팁을 담으세요.
      (예: "3일 정도 사용 빈도를 절반으로 줄여보시고, 그래도 그대로면 알려주세요.")

    [action 선택 기준]
    - MAINTAIN : 현재 루틴을 유지해도 될 때 (새로운 고민이나 성분 추가 요청이 없을 때)
    - REDUCE   : 사용 빈도를 줄이면 될 때
    - PAUSE    : 특정 성분을 일시 중단해야 할 때
    - ADD      : 사용자가 새로운 피부 고민을 말하거나 새 성분을 추가하는 것이 도움될 때
    - RECHECK  : 피부 상태가 크게 달라져 재검사가 필요할 때

    [사용자 데이터]
    - 새로운 고민: {req.new_concern}
    - 피부 타입: {base_type_label}{(' / 함께 있는 고민: ' + ', '.join(active_flag_labels)) if active_flag_labels else ''}
    - 처방약 사용 여부: {'예' if safety.get('on_medication') else '아니오'}
    - D-Day: {dday}일 남음
    - 이전 기록: {[c.model_dump() for c in req.history_cards]}

    [출력 JSON 포맷]
    {{
      "card": {{
        "type": "UPDATE",
        "created_at": "{now_str}",
        "dday_at_time": {dday},
        "response": "공감 → 이유 설명 → 행동 안내 흐름의 다정한 2~3문장",
        "action": "MAINTAIN | REDUCE | PAUSE | ADD | RECHECK 중 하나",
        "cautions": ["구체적인 관찰 기준 또는 행동 팁 1~2개"],
        "adjusted_ingredients": {{
          "pause": ["중단할 성분"],
          "keep": ["유지할 성분"],
          "add": ["새로 추가할 성분"]
        }}
      }}
    }}
    """

    text = call_claude(prompt, temperature=0.3, max_tokens=3500)
    data = parse_json_safely(text)

    card = data.setdefault("card", {})
    card.setdefault("created_at", now_str)
    card.setdefault("dday_at_time", dday)
    card["concern_summary"] = req.new_concern

    adj = card.get("adjusted_ingredients", {})
    if not isinstance(adj, dict):
        adj = {}
    adj["add"] = [n for n in adj.get("add", []) if n in newly_available]
    adj["keep"] = [n for n in adj.get("keep", []) if n in allowed_names]
    adj["pause"] = [n for n in adj.get("pause", []) if n in already or n in allowed_names]

    now_unsafe = [n for n in already if n not in allowed_names]
    adj["pause"] = list(dict.fromkeys(adj["pause"] + now_unsafe))

    paused = set(adj["pause"])
    adj["keep"] = [n for n in adj.get("keep", []) if n not in paused]
    adj["add"] = [n for n in adj.get("add", []) if n not in paused]
    card["adjusted_ingredients"] = adj

    if card.get("action") not in ("MAINTAIN", "REDUCE", "PAUSE", "ADD", "RECHECK"):
        card["action"] = "MAINTAIN"

    card["prescribed_ingredients"] = list(dict.fromkeys(
        (adj.get("keep") or []) + (adj.get("add") or [])
    ))
    card["excluded_ingredients"] = list(adj.get("pause") or [])

    final_target_ingredients = card["prescribed_ingredients"]
    if final_target_ingredients:
        products_by_ingredient = lookup.find_products_by_ingredient(
            ingredient_names=final_target_ingredients,
            active_flags=active_flags,
            sensitive=is_sensitive,
        )

        # 성분별로 골고루 배정한다.
        # (앞쪽 성분들만으로 8개 한도를 다 채워버려서 뒤쪽 성분의 제품이
        #  통째로 누락되는 문제가 있었다. 라운드로빈으로 순회해 각 성분이
        #  최소 1개씩은 대표 제품을 갖도록 한다.)
        PRODUCT_CAP = 8
        seen_ids = set()
        picked = []
        max_round = max(
            (len(products_by_ingredient.get(n, [])) for n in final_target_ingredients),
            default=0,
        )
        for round_idx in range(max_round):
            for name in final_target_ingredients:
                prods = products_by_ingredient.get(name, [])
                if round_idx >= len(prods):
                    continue
                p = prods[round_idx]
                if p["product_id"] in seen_ids:
                    continue
                picked.append(p)
                seen_ids.add(p["product_id"])
                if len(picked) >= PRODUCT_CAP:
                    break
            if len(picked) >= PRODUCT_CAP:
                break

        card["products"] = [p["product_id"] for p in picked]
        card["products_detail"] = [
            {"product_id": p["product_id"], "name": p["name"], "category": p["category"]}
            for p in picked
        ]
    else:
        card["products"] = []
        card["products_detail"] = []

    return data


def get_journey_summary(req: JourneyRequest) -> dict:
    sc = req.to_score_change()
    delta = sc.before - sc.after
    rate = (req.todo_stats.completed_days / req.todo_stats.total_days * 100
            if req.todo_stats.total_days else 0)

    all_prescribed = {ing for c in req.history_cards for ing in c.prescribed_ingredients}
    all_excluded = {ing for c in req.history_cards for ing in c.excluded_ingredients}
    strictly_excluded = list(all_excluded - all_prescribed)

    prompt = f"""
    당신은 사용자의 소중한 피부 관리 여정을 따뜻하게 마무리해 주는 뷰티 멘토입니다.
    (주의: 응답은 반드시 다른 텍스트 없이 JSON 포맷으로만 출력하세요)

    [여정 데이터]
    - 목표 이벤트: {req.event_type}
    - 실천: 총 {req.todo_stats.total_days}일 중 {req.todo_stats.completed_days}일 ({rate:.0f}%)
    - 가장 큰 고민이었던 항목: {sc.item}
    - 점수 변화: {sc.before}점 → {sc.after}점 ({'개선 ' + str(delta) + '점' if delta > 0 else '변화 없음' if delta == 0 else '악화 ' + str(-delta) + '점'})
    - 기간이 부족해 이번에 권하지 못한 성분: {strictly_excluded}
    - 상담 기록: {[c.model_dump() for c in req.history_cards]}

    [작성 절대 규칙]
    1. 어조: "고객님"께 말하듯 따뜻하고 격려하는 어조를 사용하세요.
    2. 점수 개선 시 구체적인 숫자로 칭찬하고, 점수 변화가 적더라도 꾸준한 실천 자체를 크게 칭찬해 주세요.
    3. next_step에는 '기간이 부족해 권하지 못한 성분'({strictly_excluded}) 중에서만 골라 자연스럽게 언급하며 앞으로의 관리 방향을 제시하세요. 그 목록에 없는 성분을 새로 지어내거나 언급하지 마세요.
    4. "치료", "완치", 시술 언급은 절대 금지합니다.
    5. 각 항목은 1~2문장으로 간결하고 읽기 편하게 작성하세요.

    [출력 JSON 포맷]
    {{
      "journey": {{
        "summary": "여정 전체에 대한 다정한 1~2문장 요약과 칭찬",
        "highlights": [
          "실천 달성률에 대한 따뜻한 코멘트",
          "점수 변화에 대한 구체적 코멘트"
        ],
        "next_step": "목표 일정 이후 이어갈 관리 방향 (못 쓴 성분 안내 포함)",
        "closing": "기분 좋은 마무리 인사. 자외선 차단 안내 포함"
      }}
    }}
    """

    text = call_claude(prompt, temperature=0.4, max_tokens=2000)
    data = parse_json_safely(text)

    journey_data = data.get("journey")
    if not isinstance(journey_data, dict):
        journey_data = {}
        data["journey"] = journey_data

    past_text = str(journey_data.get("summary", "")) + " " + " ".join(journey_data.get("highlights", []))

    for ex_ing in strictly_excluded:
        if ex_ing in past_text:
            log.warning("[안전 경고] 차단 성분(%s)이 과거 여정 요약에 언급되었습니다. 마스킹 진행.", ex_ing)
            if "summary" in journey_data:
                journey_data["summary"] = journey_data["summary"].replace(ex_ing, "맞춤 성분")
            if "highlights" in journey_data:
                journey_data["highlights"] = [
                    h.replace(ex_ing, "맞춤 성분") for h in journey_data["highlights"]
                ]

    # next_step은 미래 제안이므로 strictly_excluded 언급은 정상이다.
    # 다만 프롬프트 지시를 어기고 목록에 없는 성분(특히 이소트레티노인 같은
    # 전문의약품)을 지어내면, summary/highlights 검사만으로는 걸러지지 않는다.
    # 전체 성분 마스터에서 이름을 매칭해 "한 번도 언급된 적 없는" 성분만 가려낸다.
    #
    # 주의: strictly_excluded에만 없으면 무조건 마스킹하면, 실제로 계속
    # 처방해 온 성분(all_prescribed)까지 "지어낸 성분"으로 오인해 지워버린다.
    # (예: 제외 이력이 하나도 없는 사용자는 모든 성분명이 마스킹 대상이 되어
    #  "관리 성분, 관리 성분, 관리 성분..."처럼 문장이 깨지는 문제가 있었다)
    # 실제 처방 이력 + 이번에 못 쓴 성분까지는 정상적으로 언급될 수 있는
    # 성분이므로 허용 목록에 포함시킨다.
    allowed_mentions = set(strictly_excluded) | all_prescribed
    next_step_text = journey_data.get("next_step", "")
    if isinstance(next_step_text, str) and next_step_text:
        # 긴 이름부터 매칭한다. "트레티노인"이 "이소트레티노인"의 부분 문자열이라
        # 짧은 이름을 먼저 지우면 "이소"만 남는 등 어색한 결과가 생긴다.
        all_ingredient_names = sorted(
            {i["name"] for i in lookup._load()["ingredients"]}, key=len, reverse=True
        )
        for name in all_ingredient_names:
            if name in next_step_text and name not in allowed_mentions:
                log.warning(
                    "[안전 경고] 처방/제외 이력에 없는 성분(%s)이 next_step에 언급되었습니다. 마스킹 진행.", name
                )
                next_step_text = next_step_text.replace(name, "관리 성분")
        journey_data["next_step"] = next_step_text

    data["score_change"] = {
        "item": sc.item, "before": sc.before, "after": sc.after, "delta": delta,
    }
    data["completion_rate"] = round(rate)
    return data