from __future__ import annotations

import os
from urllib.parse import urlunparse

from fastapi import APIRouter, Request
from fastapi.responses import Response

router = APIRouter()


def _media_stream_url(request: Request) -> str:
    override = os.getenv("TWILIO_MEDIA_STREAM_BASE")
    if override:
        base = override.rstrip("/")
        return f"{base}/twilio/media-stream"
    scheme = "wss"
    host = request.url.hostname or "localhost"
    port = request.url.port
    netloc = f"{host}:{port}" if port else host
    return urlunparse((scheme, netloc, "/twilio/media-stream", "", "", ""))


@router.post("/voice")
async def twilio_voice(request: Request) -> Response:
    stream_url = _media_stream_url(request)
    twiml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Response>
    <Say voice=\"Polly.Joanna\">Connecting you to your leasing specialist.</Say>
    <Connect>
        <Stream url=\"{stream_url}\"/>
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@router.post("/test-voice")
async def test_voice_endpoint(request: Request) -> Response:
    """Test endpoint that returns the TwiML with localhost WebSocket for debugging"""
    twiml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Response>
    <Say voice=\"Polly.Joanna\">This is a test connection to your leasing specialist.</Say>
    <Connect>
        <Stream url=\"ws://localhost:8000/twilio/media-stream\"/>
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")
