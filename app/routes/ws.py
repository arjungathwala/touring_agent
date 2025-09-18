from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.logging.flight_recorder import FlightRecorder
from app.models.realtime import TwilioMediaPayload
from app.services.realtime_loop import RealtimeLoop

router = APIRouter()


@router.websocket("/twilio/media-stream")
async def twilio_media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    recorder = FlightRecorder()
    realtime_loop = RealtimeLoop(recorder)
    try:
        while True:
            raw = await websocket.receive_text()
            payload = TwilioMediaPayload.model_validate(json.loads(raw))
            await realtime_loop.handle_event(payload, websocket)
    except WebSocketDisconnect:
        pass
