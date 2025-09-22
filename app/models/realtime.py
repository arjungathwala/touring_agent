from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TwilioMediaPayload(BaseModel):
    event: str
    media: Optional[Dict[str, Any]] = None
    streamSid: Optional[str] = Field(default=None, alias="streamSid")
    start: Optional[Dict[str, Any]] = None
    mark: Optional[Dict[str, Any]] = None
    stop: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)


class RealtimeTranscript(BaseModel):
    transcript: str
    confidence: float
    timestamp: datetime


class AgentResponse(BaseModel):
    text: str
    barge_in: bool = False
    tts_audio_b64: Optional[str] = None
    tool_calls: List["ToolCallRequest"] = Field(default_factory=list)


class ToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any]
    call_id: str


class ToolCallResult(BaseModel):
    call_id: str
    name: str
    output: Dict[str, Any]


AgentResponse.model_rebuild()
