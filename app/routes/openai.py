from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/realtime-session")
async def create_realtime_session(body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    model = (body or {}).get("model") or os.getenv("OPENAI_REALTIME_MODEL") or "gpt-4o-realtime-preview"
    voice = (body or {}).get("voice") or os.getenv("OPENAI_REALTIME_VOICE") or "verse"
    modalities = (body or {}).get("modalities") or ["audio", "text"]

    payload = {
        "model": model,
        "voice": voice,
        "modalities": modalities,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
        resp = await client.post("https://api.openai.com/v1/realtime/sessions", json=payload, headers=headers)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc

    return resp.json()


