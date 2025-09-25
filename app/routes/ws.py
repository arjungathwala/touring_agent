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
    
    # Log memory service availability
    if realtime_loop.memory_service.is_available():
        recorder.log("WS", "connection_accepted_with_memory", remote_addr=str(websocket.client))
    else:
        recorder.log("WS", "connection_accepted_no_memory", remote_addr=str(websocket.client))
    
    recorder.log("WS", "connection_accepted", remote_addr=str(websocket.client))
    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")
            # Removed excessive message logging
            
            if message_type in {"websocket.disconnect", "websocket.close"}:
                recorder.log("WS", "disconnect", code=message.get("code"))
                break
            elif message_type == "websocket.pong":
                # Handle pong frames for keepalive
                recorder.log("WS", "pong_received")
                continue
            elif "text" in message and message["text"] is not None:
                raw = message["text"]
                payload = TwilioMediaPayload.model_validate(json.loads(raw))
                # Only log important events, not every message
                await realtime_loop.handle_event(payload, websocket)
            elif "bytes" in message and message["bytes"] is not None:
                # Twilio may send binary frames in some scenarios; log and ignore safely
                recorder.log("WS", "binary_ignored", bytes=len(message["bytes"]))
            else:
                recorder.log("WS", "unknown_message", keys=list(message.keys()))
    except WebSocketDisconnect:
        recorder.log("WS", "websocket_disconnect")
    except Exception as e:
        recorder.log("WS", "error", error=str(e))
        raise
