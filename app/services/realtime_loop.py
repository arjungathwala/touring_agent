from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
from typing import Optional

import logging
from fastapi import WebSocket

from app.logging.flight_recorder import FlightRecorder
from app.models.realtime import AgentResponse, RealtimeTranscript, TwilioMediaPayload
from app.services.agent_planner import AgentPlanner
from app.services.tool_dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)


class DeepgramClient:
    def __init__(self, recorder: FlightRecorder) -> None:
        self.recorder = recorder
        self.audio_buffer = bytearray()

    async def transcribe_chunk(self, audio_chunk: bytes) -> Optional[RealtimeTranscript]:
        # Placeholder for real Deepgram streaming client
        self.audio_buffer.extend(audio_chunk)
        return None

    async def push_text(self, text: str) -> RealtimeTranscript:
        transcript = RealtimeTranscript(
            transcript=text,
            confidence=0.99,
            timestamp=datetime.now(timezone.utc),
        )
        self.recorder.log("ASR", "transcript", transcript=text)
        return transcript


class OpenAIRealtimeClient:
    def __init__(self, planner: AgentPlanner, recorder: FlightRecorder) -> None:
        self.planner = planner
        self.recorder = recorder

    async def infer(self, transcript: RealtimeTranscript) -> AgentResponse:
        self.recorder.log("PLAN", "planner_infer", transcript=transcript.transcript)
        text_response = await self.planner.process_transcript(transcript.transcript)
        response_text = text_response.get("text", "")
        return AgentResponse(text=response_text, barge_in=False)


class ElevenLabsClient:
    def __init__(self, recorder: FlightRecorder) -> None:
        self.recorder = recorder

    async def synthesize(self, text: str) -> str:
        # Placeholder for ElevenLabs API call. Encodes the text as base64 for tests.
        await asyncio.sleep(0)
        self.recorder.log("PLAN", "tts_prepared")
        return base64.b64encode(text.encode()).decode()


class RealtimeLoop:
    def __init__(self, recorder: FlightRecorder) -> None:
        self.recorder = recorder
        self.dispatcher = ToolDispatcher(recorder)
        self.planner = AgentPlanner(self.dispatcher, recorder)
        self.deepgram = DeepgramClient(recorder)
        self.openai = OpenAIRealtimeClient(self.planner, recorder)
        self.elevenlabs = ElevenLabsClient(recorder)
        self.call_sid: Optional[str] = None

    async def handle_event(self, payload: TwilioMediaPayload, websocket: WebSocket) -> None:
        if payload.event == "start":
            self.call_sid = payload.start.get("streamSid") if payload.start else payload.streamSid
            logger.info("realtime.call_started", call_sid=self.call_sid)
            return
        if payload.event == "media" and payload.media:
            audio_b64 = payload.media.get("payload")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                await self.deepgram.transcribe_chunk(audio_bytes)
            return
        if payload.event == "mark" and payload.mark:
            mark_name = payload.mark.get("name")
            if mark_name == "transcript":
                text = payload.mark.get("payload", {}).get("text", "")
                await self.process_text(text, websocket)
            elif mark_name == "gather":
                gather_text = payload.mark.get("payload", {}).get("text", "")
                await self.process_text(gather_text, websocket)
            return
        if payload.event == "stop":
            logger.info("realtime.call_stopped", call_sid=self.call_sid)
            return

    async def process_text(self, text: str, websocket: WebSocket) -> None:
        if not text:
            return
        transcript = await self.deepgram.push_text(text)
        agent_response = await self.openai.infer(transcript)
        tts_audio = await self.elevenlabs.synthesize(agent_response.text)
        payload = {
            "event": "agent_response",
            "text": agent_response.text,
            "barge_in": agent_response.barge_in,
            "audio": tts_audio,
        }
        await websocket.send_json(payload)
