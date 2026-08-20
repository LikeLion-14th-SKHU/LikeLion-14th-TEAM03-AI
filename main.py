"""
main.py — AI 서버 엔드포인트

백엔드는 상황에 따라 세 엔드포인트 중 하나를 호출한다.
  온보딩 완료      → /ai/recommend
  고민 재입력      → /ai/concern
  D-Day 종료      → /ai/journey

AI 호출이 실패해도 빈 화면이 뜨지 않도록 Fallback으로 응답한다.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import fallback
import service
from schemas import ConcernRequest, JourneyRequest, RecommendRequest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai-server")

app = FastAPI(title="Mutsa Hackathon AI Server")

# 프론트에서 직접 호출할 경우를 대비 (배포 시 도메인 제한 권장)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """서버 상태 확인. 백엔드가 연동 전 점검용으로 사용."""
    return {"status": "ok"}


@app.post("/ai/recommend")
async def ai_recommend(req: RecommendRequest):
    """[1차] 온보딩 완료 후 초기 진단 및 추천"""
    try:
        return service.get_recommendation_solution(req)
    except Exception as e:
        # 에러 내용을 클라이언트에 그대로 노출하지 않는다.
        # 500을 던지면 화면이 비므로, 정상 형태의 Fallback으로 응답한다.
        log.exception("recommend 실패")
        return fallback.recommend_fallback(req, reason=type(e).__name__)


@app.post("/ai/concern")
async def ai_concern(req: ConcernRequest):
    """
    [2차] 관리 중 새 고민 입력

    위험 신호(통증·진물·열감 등)는 백엔드에서 하드코딩으로 차단한다.
    해당 사용자의 요청은 이 엔드포인트에 도달하지 않는다.
    """
    try:
        return service.get_concern_update(req)
    except Exception as e:
        log.exception("concern 실패")
        return fallback.concern_fallback(req, reason=type(e).__name__)


@app.post("/ai/journey")
async def ai_journey(req: JourneyRequest):
    """[3차] D-Day 종료 후 여정 요약"""
    try:
        return service.get_journey_summary(req)
    except Exception as e:
        log.exception("journey 실패")
        return fallback.journey_fallback(req, reason=type(e).__name__)