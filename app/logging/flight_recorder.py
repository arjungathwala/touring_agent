from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import logging
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

_STAGE_ORDER = [
    "ASR",
    "PLAN",
    "POLICY",
    "NER",
    "BOOK_TOUR",
    "SMS",
    "WS",
]


@dataclass
class StageEvent:
    stage: str
    message: str
    elapsed_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class FlightRecorder:
    def __init__(self) -> None:
        self.events: List[StageEvent] = []
        self.start_time = time.perf_counter()

    @contextmanager
    def stage(self, stage: str, **metadata: Any):
        if stage not in _STAGE_ORDER:
            logger.warning("flight_recorder.unknown_stage stage=%s", stage)
        stage_start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - stage_start) * 1000
            total_ms = (time.perf_counter() - self.start_time) * 1000
            event = StageEvent(
                stage=stage,
                message=f"{stage} completed",
                elapsed_ms=elapsed_ms,
                metadata={"total_ms": round(total_ms, 2), **_redact(metadata)},
            )
            self.events.append(event)
            redacted = _redact(metadata)
            logger.info(
                "flight_recorder.stage stage=%s elapsed_ms=%.2f total_ms=%.2f metadata=%s",
                stage,
                round(elapsed_ms, 2),
                round(total_ms, 2),
                redacted,
            )

    def log(self, stage: str, message: str, **metadata: Any) -> None:
        if stage not in _STAGE_ORDER:
            logger.warning("flight_recorder.unknown_stage stage=%s", stage)
        total_ms = (time.perf_counter() - self.start_time) * 1000
        event = StageEvent(
            stage=stage,
            message=message,
            elapsed_ms=0,
            metadata={"total_ms": round(total_ms, 2), **_redact(metadata)},
        )
        self.events.append(event)
        redacted = _redact(metadata)
        logger.info(
            "flight_recorder.log stage=%s message=%s total_ms=%.2f metadata=%s",
            stage,
            message,
            round(total_ms, 2),
            redacted,
        )


def _redact(payload: Dict[str, Any]) -> Dict[str, Any]:
    redacted = {}
    for key, value in payload.items():
        if key in {"phone", "email", "name"} and value:
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


class FlightRecorderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.flight_recorder = FlightRecorder()
        response = await call_next(request)
        return response


def register_log_middleware(app: FastAPI) -> None:
    app.add_middleware(FlightRecorderMiddleware)
