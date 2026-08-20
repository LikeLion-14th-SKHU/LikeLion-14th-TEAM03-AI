"""
schemas.py — 백엔드와 주고받는 데이터 정의

Dict[str, Any]로 두면 필드가 빠져도 런타임에야 알게 된다.
필수 필드는 명시해서 요청 시점에 걸러낸다.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal


# ─────────────────────────────────────────────
# 공통 구조
# ─────────────────────────────────────────────

class Flags(BaseModel):
    """중첩 속성. 설문 결과로 백엔드가 판정한다."""
    dehydrated: bool = False
    sensitive: bool = False
    acne: bool = False
    mark_prone: bool = False


class Safety(BaseModel):
    """안전 관련 플래그. on_medication이 가장 중요하다."""
    on_medication: bool = False
    retinol_history: Literal["current", "past", "irritated", "never"] = "never"
    inflammatory: bool = False


class Skin(BaseModel):
    base_type: Literal["OILY", "DRY", "COMBO", "NORMAL"]
    flags: Flags
    safety: Safety
    axis_scores: Dict[str, float] = Field(default_factory=dict)
    trouble_scores: Dict[str, int] = Field(default_factory=dict)


class DDay(BaseModel):
    remaining_days: int = Field(ge=0, description="남은 일수. 음수 불가")
    event_type: str = "일정"


class HistoryCard(BaseModel):
    """이전 상담 기록. 최초 카드 1개 + 최근 3개를 전달받는다."""
    date: Optional[str] = None
    dday_at_time: Optional[int] = None
    summary: Optional[str] = None
    concern_summary: Optional[str] = None
    prescribed_ingredients: List[str] = Field(default_factory=list)
    excluded_ingredients: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# 요청
# ─────────────────────────────────────────────

class RecommendRequest(BaseModel):
    """[1차] 온보딩 완료 후 최초 추천"""
    skin: Skin
    dday: DDay
    concern: Dict[str, str]
    history_cards: List[HistoryCard] = Field(default_factory=list)


class ConcernRequest(BaseModel):
    """
    [2차] 관리 중 새 고민 입력

    위험 신호(통증·진물·열감 등)는 백엔드에서 하드코딩으로 차단하며,
    해당 사용자의 요청은 이 엔드포인트로 오지 않는다.
    """
    new_concern: str
    history_cards: List[HistoryCard] = Field(default_factory=list)
    dday: DDay
    skin: Skin


class ScoreChange(BaseModel):
    """[3차] 개선도. 서버 내부에서 before_scores/after_scores로부터 계산한다."""
    item: str
    before: int
    after: int


class TodoStats(BaseModel):
    total_days: int = 0
    completed_days: int = 0
    rate: Optional[float] = None      # 백엔드가 보내면 받고, 없으면 서버에서 계산


class JourneyRequest(BaseModel):
    """
    [3차] D-Day 종료 후 여정 요약

    before_scores : 온보딩 시점의 trouble_scores 전체
    after_scores  : 슬라이더로 측정한 항목 1개 {"피지량": 4} 혹은 전체 항목

    비교 항목은 after_scores에 담긴 키를 기준으로 한다.
    (before의 최댓값 항목과 다를 수 있으므로 after 쪽을 신뢰한다)
    """
    history_cards: List[HistoryCard] = Field(default_factory=list)
    todo_stats: TodoStats
    before_scores: Dict[str, int] = Field(default_factory=dict)
    after_scores: Dict[str, int] = Field(default_factory=dict)
    event_type: str = "일정"

    def to_score_change(self) -> ScoreChange:
        """가장 개선폭(before - after)이 크거나, 가장 심각했던(before) 항목 1개를 뽑는다."""
        if not self.after_scores:
            return ScoreChange(item="피부 상태", before=0, after=0)
            
        best_item = None
        max_delta = -999
        max_before = -1

        # 1. after_scores에 들어온 모든 항목을 순회하며 가장 변화가 큰 항목을 찾음
        for item, after_val in self.after_scores.items():
            before_val = self.before_scores.get(item, 0)
            delta = before_val - after_val

            # 2. 개선폭(delta)이 가장 크거나, 개선폭이 같다면 원래 더 심각했던(before) 항목을 우선 선택
            if delta > max_delta or (delta == max_delta and before_val > max_before):
                max_delta = delta
                max_before = before_val
                best_item = item

        # 3. 만약 찾는 데 실패했다면 기존처럼 첫 번째 값을 폴백(Fallback)으로 사용
        if not best_item:
            best_item = next(iter(self.after_scores))

        return ScoreChange(
            item=best_item,
            before=self.before_scores.get(best_item, 0),
            after=self.after_scores[best_item],
        )