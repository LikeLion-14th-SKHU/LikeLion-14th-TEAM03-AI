"""
core/lookup.py — 성분·제품 조회 공통 모듈

1차 추천(skin_recommender)과 2차 재추천(concern_handler)이 함께 사용한다.
이 파일은 조회만 담당하며, D-Day 판정이나 LLM 호출은 하지 않는다.
"""

import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent          # flat 구조: JSON이 같은 폴더에 있음


# ─────────────────────────────────────────────
# 데이터 로드 (서버 시작 시 1회, 이후 캐시)
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load():
    ing = json.loads((DATA / "ingredients2.json").read_text(encoding="utf-8"))["ingredients_master"]
    prod = json.loads((DATA / "products2.json").read_text(encoding="utf-8"))["products_master"]
    types = json.loads((DATA / "skin_types.json").read_text(encoding="utf-8"))["skin_types_master"]

    # 성분 조회 인덱스: 표준명 + 별칭 → 성분 객체
    index = {}
    for i in ing:
        index[i["name"]] = i
        for a in i.get("aliases", []):
            index[a] = i

    return {
        "ingredients": ing,
        "products": prod,
        "types": types,
        "index": index,
    }


# ─────────────────────────────────────────────
# 성분 조회
# ─────────────────────────────────────────────

def lookup_ingredient(name):
    """
    성분 이름으로 조회. 별칭(aliases)도 함께 매칭한다.

    예)
        lookup_ingredient("히알루론산")             → 히알루론산 객체
        lookup_ingredient("소듐하이알루로네이트")    → 히알루론산 객체 (별칭)
        lookup_ingredient("없는성분")               → None
    """
    return _load()["index"].get(name)


def get_type_ingredients(base_type):
    """피부 타입의 추천 성분 목록을 성분 객체로 반환."""
    t = next((x for x in _load()["types"]["base_types"] if x["type_id"] == base_type), None)
    if not t:
        raise ValueError(f"알 수 없는 피부 타입: {base_type}")

    result = []
    for name in t["recommended_ingredients"]:
        ing = lookup_ingredient(name)
        if ing is None:
            # 데이터 오류. 조용히 넘기지 말고 로그로 남길 것
            print(f"[경고] skin_types의 '{name}'이 ingredients.json에 없습니다")
            continue
        result.append(ing)
    return result


def get_safe_baseline(base_type):
    """
    필터링 결과가 0개일 때 사용할 안전 기본 성분.
    제품 추천 대상(recommend_product=True)만 반환한다.
    """
    t = next((x for x in _load()["types"]["base_types"] if x["type_id"] == base_type), None)
    out = []
    for name in t.get("safe_baseline", []):
        ing = lookup_ingredient(name)
        if ing and ing.get("recommend_product"):
            # filter_by_dday를 거치지 않으므로 여기서 상태 필드를 채운다.
            # 채우지 않으면 dday=0 등 전량 제외 상황에서 KeyError가 발생한다.
            item = dict(ing)
            item.setdefault("dday_status", "가능")
            item.setdefault("effect_days", ing.get("effect_weeks", 0) * 7)
            out.append(item)
    return out


def get_active_attributes(flags, safety):
    """
    활성화된 중첩 속성을 priority 순으로 반환.
    flags 키와 safety.on_medication을 대문자 플래그로 변환한다.

    반환 예) ["ON_MEDICATION", "SENSITIVE", "ACNE"]
    """
    active = [k.upper() for k, v in flags.items() if v]
    if safety.get("on_medication"):
        active.append("ON_MEDICATION")

    attrs = _load()["types"]["overlapping_attributes"]
    ordered = sorted(
        [a for a in attrs if a["attr_id"] in active],
        key=lambda a: a["priority"],
    )
    return [a["attr_id"] for a in ordered]


def get_type_data(base_type):
    """피부 타입의 전체 데이터(설명·세안법·관리법 등)를 반환."""
    t = next((x for x in _load()["types"]["base_types"] if x["type_id"] == base_type), None)
    if not t:
        raise ValueError(f"알 수 없는 피부 타입: {base_type}")
    return t


@lru_cache(maxsize=1)
def get_routine_guide():
    """기초 화장품 바르는 순서. AI가 생성하지 않고 그대로 반환한다."""
    d = json.loads((DATA / "routine_guide.json").read_text(encoding="utf-8"))
    return d.get("skincare_routine_guide", d)


# ─────────────────────────────────────────────
# 제품 조회
# ─────────────────────────────────────────────

def find_products(ingredient_names, active_flags, category=None, sensitive=False, limit=None):
    """
    성분 이름으로 제품을 찾는다. (표준명 및 별칭 자동 확장 매칭)
    """
    # 1. 입력받은 성분명의 표준명 + 모든 별칭(aliases)을 하나의 검색 풀로 확장
    expanded_names = set()
    for name in ingredient_names:
        expanded_names.add(name)
        ing = lookup_ingredient(name)
        if ing:
            expanded_names.add(ing.get("name", ""))
            expanded_names.update(ing.get("aliases", []))
    
    expanded_names.discard("")  # 빈 문자열 제거
    out = []

    for p in _load()["products"]:
        if category and p["category"] != category:
            continue
        # 확장된 성분명 집합과 제품의 key_ingredients 교집합 확인
        if not expanded_names & set(p["key_ingredients"]):
            continue
        if any(f in p["banned_flags"] for f in active_flags):
            continue
        if sensitive and not p.get("verified", False):
            continue
        out.append(p)

    return out[:limit] if limit else out


def find_products_by_step(ingredient_names, active_flags, sensitive=False, per_step=2):
    """
    단계별로 제품을 나눠 반환한다. 결과 화면 구성에 사용.

    반환 예)
        {
          "toner":   [제품1, 제품2],
          "serum":   [제품3, 제품4],
          "ampoule": [...],
          "cream":   [...]
        }
    """
    return {
        cat: find_products(ingredient_names, active_flags,
                           category=cat, sensitive=sensitive, limit=per_step)
        for cat in ("toner", "serum", "ampoule", "cream")
    }


def get_product(product_id):
    """product_id로 제품 1개 조회. AI 응답 검증 및 상세 정보 결합에 사용."""
    return next((p for p in _load()["products"] if p["product_id"] == product_id), None)


def validate_product_ids(product_ids, candidates):
    """
    AI가 반환한 product_id 중 후보 목록에 실제로 있는 것만 남긴다.

    ※ 이 검증을 반드시 거칠 것.
      response_schema는 형식만 강제하며, 존재하지 않는 ID를 막지 못한다.
    """
    valid = {p["product_id"] for p in candidates}
    return [pid for pid in product_ids if pid in valid]


# ─────────────────────────────────────────────
# D-Day 필터
# ─────────────────────────────────────────────

def filter_by_dday(ingredients, dday, goal=None):
    """
    남은 일수로 성분을 거른다.

    반환: (keep, excluded)
      keep     = [성분객체 + dday_status("가능"|"부분")]
      excluded = [{"ingredient": 이름, "reason": 사유}]

    판정
      dday < min_dday_days      → 제외 (초기 자극 구간이 당일에 걸림)
      dday < effect_weeks * 7   → 부분 (효과가 일부만)
      그 외                      → 가능
    """
    keep, excluded = [], []
    for i in ingredients:
        if dday < i["min_dday_days"]:
            excluded.append({
                "ingredient": i["name"],
                "reason": f"적응 기간을 포함해 최소 {i['min_dday_days']}일이 필요한데 남은 기간이 {dday}일이라 이번엔 제외",
            })
            continue
        eff = i.get("effect_weeks_by_concern", {}).get(goal, i["effect_weeks"])
        item = dict(i)
        item["dday_status"] = "가능" if dday >= eff * 7 else "부분"
        item["effect_days"] = eff * 7
        keep.append(item)
    return keep, excluded


def filter_by_flags(ingredients, active_flags, retinol_history="never"):
    """
    플래그로 성분을 거른다. banned_flags가 유일한 판정 기준.

    retinol_history가 "irritated"(과거 자극으로 중단)이면
    레티놀 계열을 추가로 차단한다. 같은 자극이 반복될 수 있기 때문이다.
    """
    RETINOID = {"레티놀", "레틴알", "레티날", "레티노이드",
                "레티닐팔미테이트", "하이드록시피나콜론레티노에이트"}

    keep, excluded = [], []
    for i in ingredients:
        hit = [f for f in active_flags if f in i["banned_flags"]]
        if hit:
            label = {"ON_MEDICATION": "처방약 사용 중", "SENSITIVE": "민감성",
                     "DEHYDRATED": "수분 부족", "ACNE": "여드름"}.get(hit[0], hit[0])
            excluded.append({"ingredient": i["name"], "reason": f"{label} 상태에서는 자극이 될 수 있어 제외"})
            continue

        if retinol_history == "irritated" and i["name"] in RETINOID:
            excluded.append({
                "ingredient": i["name"],
                "reason": "이전에 자극으로 중단하신 이력이 있어 이번에는 제외",
            })
            continue

        keep.append(i)
    return keep, excluded


def find_products_by_ingredient(ingredient_names, active_flags, sensitive=False, per_ingredient=4):
    """
    성분별로 그 성분을 실제 포함한 제품만 조회한다.

    반환 예)
        {
          "나이아신아마이드": [제품1, 제품2],
          "히알루론산":      [제품3],
          "PHA":            []          ← 해당 제품이 없으면 빈 배열
        }

    ※ find_products_by_step은 전체 성분에서 카테고리별 1개씩 뽑으므로,
      그 결과를 모든 성분 카드에 붙이면 성분-제품 오매칭이 발생한다.
      성분 단위 배정에는 반드시 이 함수를 사용할 것.
    """
    out = {}
    for name in ingredient_names:
        found = find_products([name], active_flags, sensitive=sensitive)

        # 카테고리별로 고르게 뽑는다.
        # 단순히 앞에서 N개를 자르면 토너만 나오는 등 한쪽으로 쏠린다.
        picked, seen = [], set()
        for cat in ("toner", "serum", "ampoule", "cream"):
            for p in found:
                if p["category"] == cat and p["product_id"] not in seen:
                    picked.append(p)
                    seen.add(p["product_id"])
                    break
            if len(picked) >= per_ingredient:
                break

        # 카테고리가 부족하면 남은 제품으로 채운다
        for p in found:
            if len(picked) >= per_ingredient:
                break
            if p["product_id"] not in seen:
                picked.append(p)
                seen.add(p["product_id"])

        out[name] = picked
    return out


# ─────────────────────────────────────────────
# 자기 점검 (python lookup.py 로 실행)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    d = _load()
    print(f"성분 {len(d['ingredients'])}개 / 제품 {len(d['products'])}개 "
          f"/ 조회 인덱스 {len(d['index'])}개\n")

    # 별칭 매칭 확인
    for n in ["히알루론산", "소듐하이알루로네이트", "세라마이드엔피", "살리실릭애씨드", "레틴알"]:
        r = lookup_ingredient(n)
        print(f"  {n:22} → {r['name'] if r else '조회 실패'}")

    # products.json의 모든 성분명이 조회되는지
    print()
    miss = {k for p in d["products"] for k in p["key_ingredients"]
            if lookup_ingredient(k) is None}
    print(f"  제품 파일에서 조회 실패하는 성분: {miss or '없음'}")

    # skin_types.json의 모든 성분명이 조회되는지
    miss2 = set()
    for t in d["types"]["base_types"]:
        for k in t["recommended_ingredients"] + t.get("safe_baseline", []):
            if lookup_ingredient(k) is None:
                miss2.add(k)
    print(f"  타입 파일에서 조회 실패하는 성분: {miss2 or '없음'}")

    # 조합 시뮬레이션
    print("\n조합별 제품 수")
    for label, names, flags, sens in [
        ("지성 (제약 없음)",   ["나이아신아마이드", "살리실산", "히알루론산"], [], False),
        ("지성 + 민감성",      ["나이아신아마이드", "히알루론산"], ["SENSITIVE"], True),
        ("처방약 + 민감성",    ["히알루론산", "세라마이드", "판테놀"], ["ON_MEDICATION", "SENSITIVE"], True),
    ]:
        r = find_products(names, flags, sensitive=sens)
        step = find_products_by_step(names, flags, sensitive=sens)
        print(f"  {label:20} 총 {len(r):3}개  "
              f"단계별 { {k: len(v) for k, v in step.items()} }")