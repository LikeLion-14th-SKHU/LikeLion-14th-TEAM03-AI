"""
fallback.py — AI 호출 실패 시 대체 응답

API 지연, rate limit, JSON 파싱 실패 등은 실제로 발생한다.
그때 500을 던지면 화면이 비므로, 정상과 동일한 형태로 응답한다.

여기서 반환하는 내용은 LLM이 생성하지 않는다.
마스터 데이터에서 꺼낸 값과 미리 작성한 문구만 사용한다.
"""

from datetime import datetime

import lookup

TYPE_LABEL = {"OILY": "지성", "DRY": "건성", "COMBO": "복합성", "NORMAL": "정상"}
FLAG_LABEL = {
    "dehydrated": "수분 부족",
    "sensitive": "민감성",
    "acne": "여드름이 잘 생기는",
    "mark_prone": "자국이 남기 쉬운",
}


def _type_sentence(base_type, flags):
    labels = [FLAG_LABEL[k] for k, v in flags.items() if v and k in FLAG_LABEL]
    base = TYPE_LABEL.get(base_type, base_type)
    if labels:
        return f"당신은 {base} 및 {', '.join(labels)} 경향으로 나타났습니다."
    return f"당신은 {base} 경향으로 나타났습니다."


def recommend_fallback(req, reason=""):
    """
    [1차 대체] 안전 기본 성분으로만 구성한다.

    필터링 결과가 아니라 safe_baseline을 쓰므로
    어떤 사용자에게도 자극이 되지 않는 조합이다.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_type = req.skin.base_type
    flags = req.skin.flags.model_dump()
    safety = req.skin.safety.model_dump()
    dday = req.dday.remaining_days

    try:
        type_data = lookup.get_type_data(base_type)
        cleansing = type_data.get("cleansing_guide", [])
        management = type_data.get("management_guide", [])
        routine = lookup.get_routine_guide()

        active = lookup.get_active_attributes(flags, safety)
        base_ing = lookup.get_safe_baseline(base_type)
        base_ing, _ = lookup.filter_by_flags(
            base_ing, active, safety.get("retinol_history", "never")
        )
        names = [i["name"] for i in base_ing]
        sensitive = "SENSITIVE" in active

        # 성분별로 실제 그 성분을 포함한 제품만 배정한다.
        # (예전에는 by_step에서 뽑은 4개짜리 세트를 모든 성분에 동일하게 붙여서,
        #  성분과 무관한 제품이 같이 나가거나 모든 성분 카드가 똑같아지는 문제가 있었다)
        products_by_ingredient = lookup.find_products_by_ingredient(
            names, active, sensitive=sensitive
        )
        per_ingredient_pids = {
            n: [p["product_id"] for p in prods]
            for n, prods in products_by_ingredient.items()
        }

        by_step = lookup.find_products_by_step(names, active, sensitive=sensitive)
        pids = [
            by_step[s][0]["product_id"]
            for s in ("toner", "serum", "ampoule", "cream")
            if by_step.get(s)
        ]
        detail = [
            {"product_id": p["product_id"], "name": p["name"], "category": p["category"]}
            for s in by_step.values()
            for p in s[:1]
        ]
    except Exception:
        # 데이터 로드까지 실패한 최악의 경우
        cleansing, management, routine = [], [], {}
        names, pids, detail = [], [], []
        per_ingredient_pids = {}

    return {
        "_fallback": True,
        "_reason": reason,
        "type_description": _type_sentence(base_type, flags),
        "cosmetic": {
            "summary": (
                f"{dday}일 남으셨네요. 지금은 새로운 성분을 늘리기보다 "
                "장벽을 지키는 기본 관리에 집중하시는 것이 좋습니다."
            ),
            "recommended": [
                {
                    "ingredient": n,
                    "effect": "장벽 유지와 수분 공급",
                    "timeline": "바르는 즉시 도움을 줄 수 있습니다",
                    "cautions": ["새 제품은 한 번에 하나씩만 추가해 주세요"],
                    "products": per_ingredient_pids.get(n, []),
                }
                for n in names
            ],
            "excluded": [],
            "closing": "낮에는 자외선 차단제를 꼭 사용해 주세요.",
        },
        "cleansing": {"guide": cleansing, "management": management},
        "skincare_order": routine,
        "products_detail": detail,
        "needs_medical_consult": bool(safety.get("inflammatory")),
        "card": {
            "type": "INITIAL",
            "created_at": now,
            "dday_at_time": dday,
            "concern_summary": req.concern.get("raw", ""),
            "prescribed_ingredients": names,
            "excluded_ingredients": [],
        },
    }


def concern_fallback(req, reason=""):
    """[2차 대체] 판단하지 않고 관찰을 권한다."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "_fallback": True,
        "_reason": reason,
        "card": {
            "type": "UPDATE",
            "created_at": now,
            "dday_at_time": req.dday.remaining_days,
            "concern_summary": req.new_concern,
            "response": (
                "말씀해 주신 변화를 확인했어요. 지금은 사용 중인 제품의 빈도를 "
                "절반으로 줄이고 보습과 진정에 집중해 보세요. "
                "일주일 뒤에도 나아지지 않으면 해당 제품을 중단하시는 것이 좋습니다."
            ),
            "action": "REDUCE",
            "cautions": [
                "새 제품을 추가하지 마세요",
                "통증이나 진물이 생기면 피부과 진료를 받아보세요",
            ],
            "adjusted_ingredients": {"pause": [], "keep": [], "add": []},
            "prescribed_ingredients": [],
            "excluded_ingredients": [],
            "products": [],
            "products_detail": [],
        },
    }


def journey_fallback(req, reason=""):
    """[3차 대체] 계산된 수치만으로 구성한다."""
    sc = req.to_score_change()
    delta = sc.before - sc.after
    total = req.todo_stats.total_days or 1
    rate = round(req.todo_stats.completed_days / total * 100)

    if delta > 0:
        highlight = f"{sc.item}이 {sc.before}점에서 {sc.after}점으로 {delta}점 나아졌어요."
    elif delta == 0:
        highlight = f"{sc.item}은 {sc.before}점을 유지했어요. 더 나빠지지 않은 것도 의미가 있습니다."
    else:
        highlight = f"{sc.item}은 이번에 변화가 크지 않았어요. 컨디션 영향도 있으니 너무 아쉬워하지 마세요."

    return {
        "_fallback": True,
        "_reason": reason,
        "journey": {
            "summary": f"{req.event_type}까지 {total}일 동안 꾸준히 관리하셨어요.",
            "highlights": [
                f"루틴을 {req.todo_stats.completed_days}일 지키셨어요 ({rate}%).",
                highlight,
            ],
            "next_step": (
                "일정이 끝났으니 이제 시간 여유를 두고 관리하실 수 있어요. "
                "기간이 짧아 미뤄뒀던 성분도 지금부터 천천히 시작해 보세요."
            ),
            "closing": "자외선 차단은 계속 지켜주세요.",
        },
        "score_change": {
            "item": sc.item, "before": sc.before, "after": sc.after, "delta": delta,
        },
        "completion_rate": rate,
    }