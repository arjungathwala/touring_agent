from __future__ import annotations

import audioop
import asyncio
import base64
import io
import json
import math
import os
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import logging
import httpx
import websockets
from fastapi import WebSocket

from app.logging.flight_recorder import FlightRecorder
from app.models.realtime import AgentResponse, RealtimeTranscript, ToolCallRequest, ToolCallResult, TwilioMediaPayload
from app.services.agent_planner import AgentPlanner
from app.services.tool_dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)


class DeepgramClient:
    def __init__(self, recorder: FlightRecorder) -> None:
        self.recorder = recorder
        self.audio_buffer = bytearray()
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        self._http: Optional[httpx.AsyncClient] = None
        if self.api_key:
            self._http = httpx.AsyncClient(
                base_url="https://api.deepgram.com",
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/octet-stream",
                },
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
        self._min_flush_bytes = 3200
        openai_key = os.getenv("OPENAI_API_KEY")
        self._openai_http: Optional[httpx.AsyncClient] = None
        if openai_key:
            self._openai_http = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={"Authorization": f"Bearer {openai_key}"},
                timeout=httpx.Timeout(20.0, connect=5.0),
            )
        self._openai_model = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
        self._openai_min_flush_bytes = 16000
        self._no_asr_warned = False

    async def transcribe_chunk(self, audio_chunk: bytes) -> Optional[RealtimeTranscript]:
        self.audio_buffer.extend(audio_chunk)
        if self._http:
            if len(self.audio_buffer) < self._min_flush_bytes:
                return None
            return await self._flush_deepgram()
        if self._openai_http:
            if len(self.audio_buffer) < self._openai_min_flush_bytes:
                return None
            return await self._flush_openai()
        if not self._no_asr_warned:
            self.recorder.log("ASR", "no_transcriber_available")
            self._no_asr_warned = True
        self.audio_buffer.clear()
        return None

    async def push_text(self, text: str) -> RealtimeTranscript:
        transcript = RealtimeTranscript(
            transcript=text,
            confidence=0.99,
            timestamp=datetime.now(timezone.utc),
        )
        self.recorder.log("ASR", "transcript", transcript=text)
        return transcript

    async def finalize(self) -> Optional[RealtimeTranscript]:
        if not self.audio_buffer:
            return None
        if self._http:
            return await self._flush_deepgram()
        if self._openai_http:
            return await self._flush_openai()
        self.audio_buffer.clear()
        return None

    async def _flush_deepgram(self) -> Optional[RealtimeTranscript]:
        if not self._http:
            return None
        payload = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        try:
            with self.recorder.stage("ASR", bytes=len(payload)):
                response = await self._http.post(
                    "/v1/listen",
                    params={
                        "model": "nova-2",
                        "smart_format": "true",
                        "punctuate": "true",
                        "encoding": "mulaw",
                        "sample_rate": 8000,
                        "channels": 1,
                    },
                    content=payload,
                )
            response.raise_for_status()
            data = response.json()
            transcript_text = (
                data.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            )
            if transcript_text:
                transcript = RealtimeTranscript(
                    transcript=transcript_text,
                    confidence=
                    data.get("results", {})
                    .get("channels", [{}])[0]
                    .get("alternatives", [{}])[0]
                    .get("confidence", 0.9),
                    timestamp=datetime.now(timezone.utc),
                )
                self.recorder.log("ASR", "deepgram_transcript", transcript=transcript.transcript)
                return transcript
        except httpx.HTTPStatusError as exc:
            body = exc.response.text if exc.response is not None else ""
            logger.warning(
                "deepgram.http_error status=%s body=%s",
                exc.response.status_code if exc.response is not None else "?",
                body[:500],
                exc_info=True,
            )
            await self._disable_remote("http_status_error")
        except httpx.HTTPError as exc:
            logger.warning("deepgram.transport_error %s", exc, exc_info=True)
            await self._disable_remote("transport_error")
        return None

    async def _flush_openai(self) -> Optional[RealtimeTranscript]:
        if not self._openai_http:
            return None
        payload = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        pcm16 = _mulaw_to_pcm16(payload)
        if not pcm16:
            return None
        wav_bytes = _pcm16_to_wav(pcm16, sample_rate=8000, channels=1)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": self._openai_model,
            "response_format": "json",
            "temperature": "0",
        }
        try:
            with self.recorder.stage("ASR", provider="openai", bytes=len(payload)):
                response = await self._openai_http.post(
                    "/audio/transcriptions",
                    data=data,
                    files=files,
                )
            response.raise_for_status()
            json_data = response.json()
            transcript_text = json_data.get("text", "").strip()
            if transcript_text:
                transcript = RealtimeTranscript(
                    transcript=transcript_text,
                    confidence=0.85,
                    timestamp=datetime.now(timezone.utc),
                )
                self.recorder.log("ASR", "openai_transcript", transcript=transcript.transcript)
                return transcript
        except httpx.HTTPStatusError as exc:
            body = exc.response.text if exc.response is not None else ""
            logger.warning(
                "openai.asr_http_error status=%s body=%s",
                exc.response.status_code if exc.response is not None else "?",
                body[:500],
                exc_info=True,
            )
        except httpx.HTTPError as exc:
            logger.warning("openai.asr_transport_error %s", exc, exc_info=True)
        return None

    async def _disable_remote(self, reason: str) -> None:
        if not self._http:
            return
        await self._http.aclose()
        self._http = None
        self.recorder.log("ASR", "deepgram_remote_disabled", reason=reason)


class OpenAIRealtimeClient:
    def __init__(self, planner: AgentPlanner, dispatcher: ToolDispatcher, recorder: FlightRecorder) -> None:
        self.planner = planner
        self.dispatcher = dispatcher
        self.recorder = recorder
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
        self.voice = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
        self.sample_rate = int(os.getenv("OPENAI_REALTIME_SAMPLE_RATE", "16000"))
        self.system_prompt = os.getenv(
            "OPENAI_REALTIME_SYSTEM_PROMPT",
            "You are a helpful leasing assistant for luxury apartment tours. "
            "Answer succinctly and keep a welcoming tone.",
        )

    async def infer(self, transcript: RealtimeTranscript) -> AgentResponse:
        if not self.api_key:
            self.recorder.log("PLAN", "openai_missing_api_key")
            text_response = await self.planner.process_transcript(transcript.transcript)
            return AgentResponse(text=text_response.get("text", ""), barge_in=False)

        try:
            return await self._infer_via_realtime(transcript.transcript)
        except Exception as exc:  # noqa: BLE001
            logger.exception("openai.realtime_failure", error=str(exc))
            self.recorder.log("PLAN", "openai_realtime_error", error=str(exc))
            text_response = await self.planner.process_transcript(transcript.transcript)
            return AgentResponse(text=text_response.get("text", ""), barge_in=False)

    async def synthesize_text(self, text: str) -> bytes:
        if not self.api_key:
            return b""
        uri = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]
        try:
            async with websockets.connect(uri, additional_headers=headers) as ws:
                await self._initialize_session(ws)
                request = {
                    "type": "response.create",
                    "response": {
                        "conversation": "none",
                        "modalities": ["audio", "text"],
                        "instructions": f"Please say exactly: {text}",
                    },
                }
                await ws.send(json.dumps(request))
                captured = await self._collect_response(ws)
        except Exception as exc:  # noqa: BLE001
            logger.exception("openai.realtime_tts_failure", error=str(exc))
            self.recorder.log("PLAN", "openai_realtime_tts_error", error=str(exc))
            return b""

        if not captured.audio_pcm:
            return b""
        return _pcm_to_mulaw(captured.audio_pcm, captured.sample_rate or self.sample_rate)

    async def _infer_via_realtime(self, user_text: str) -> AgentResponse:
        uri = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]
        async with websockets.connect(uri, additional_headers=headers) as ws:
            await self._initialize_session(ws)
            await self._send_user_message(ws, user_text)
            await self._request_response(ws)
            captured = await self._collect_response(ws)

        audio_b64 = None
        if captured.audio_pcm:
            audio_mulaw = _pcm_to_mulaw(captured.audio_pcm, captured.sample_rate or self.sample_rate)
            audio_b64 = base64.b64encode(audio_mulaw).decode()

        return AgentResponse(
            text=captured.text.strip(),
            barge_in=False,
            tts_audio_b64=audio_b64,
            tool_calls=captured.tool_calls,
        )

    async def _initialize_session(self, ws: websockets.WebSocketClientProtocol) -> None:
        session_update = {
            "type": "session.update",
            "session": {
                "model": self.model,
                "instructions": self.system_prompt,
                "voice": self.voice,
                "output_audio_format": "pcm16",
            },
        }
        await ws.send(json.dumps(session_update))

    async def _send_user_message(self, ws: websockets.WebSocketClientProtocol, text: str) -> None:
        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
        await ws.send(json.dumps(message))

    async def _request_response(self, ws: websockets.WebSocketClientProtocol) -> None:
        request = {
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"],
                "instructions": "Respond to the latest guest utterance.",
            },
        }
        await ws.send(json.dumps(request))

    async def _collect_response(self, ws: websockets.WebSocketClientProtocol) -> "RealtimeCapture":
        text_chunks: List[str] = []
        audio_chunks: List[bytes] = []
        tool_calls: List[ToolCallRequest] = []
        sample_rate: Optional[int] = None

        while True:
            raw = await ws.recv()
            payload = json.loads(raw)
            event_type = payload.get("type")

            payload_for_log = payload
            if event_type == "response.output_audio.delta" and payload.get("delta"):
                payload_for_log = {**payload, "delta": f"<b64 {len(payload['delta'])} chars>"}
            elif event_type == "response.output_audio.delta" and payload.get("audio"):
                payload_for_log = {**payload, "audio": f"<b64 {len(payload['audio'])} chars>"}
            elif event_type == "response.output_audio.done" and payload.get("audio"):
                payload_for_log = {**payload, "audio": f"<b64 {len(payload['audio'])} chars>"}
            logger.debug("openai.realtime_event type=%s payload=%s", event_type, payload_for_log)

            if event_type == "response.output_text.delta":
                text_chunks.append(payload.get("delta", ""))
            elif event_type == "response.output_text.done":
                continue
            elif event_type == "response.output_audio.delta":
                audio_b64 = payload.get("audio") or payload.get("delta")
                if audio_b64:
                    audio_chunks.append(base64.b64decode(audio_b64))
                if payload.get("sample_rate"):
                    sample_rate = int(payload["sample_rate"])
            elif event_type == "response.output_tool_call.delta":
                tool_name = payload.get("name")
                arguments = payload.get("arguments", {})
                call_id = payload.get("id", "tool-call")
                tool_calls.append(ToolCallRequest(name=tool_name or "", arguments=arguments, call_id=call_id))
            elif event_type == "response.completed":
                break
            elif event_type == "error":
                logger.error("openai.realtime_error_event payload=%s", payload)
                raise RuntimeError(payload.get("error", {}).get("message", "openai realtime error"))

        return RealtimeCapture(
            text="".join(text_chunks),
            audio_pcm=b"".join(audio_chunks),
            sample_rate=sample_rate,
            tool_calls=tool_calls,
        )


@dataclass
class RealtimeCapture:
    text: str
    audio_pcm: bytes
    sample_rate: Optional[int]
    tool_calls: List[ToolCallRequest]


class RealtimeLoop:
    def __init__(self, recorder: FlightRecorder) -> None:
        self.recorder = recorder
        self.dispatcher = ToolDispatcher(recorder)
        self.planner = AgentPlanner(self.dispatcher, recorder)
        self.deepgram = DeepgramClient(recorder)
        self.openai = OpenAIRealtimeClient(self.planner, self.dispatcher, recorder)
        self.call_sid: Optional[str] = None
        self.stream_sid: Optional[str] = None

    async def handle_event(self, payload: TwilioMediaPayload, websocket: WebSocket) -> None:
        if payload.event == "start":
            start_payload = payload.start or {}
            self.call_sid = start_payload.get("callSid") or start_payload.get("streamSid")
            self.stream_sid = start_payload.get("streamSid") or payload.streamSid
            logger.info("realtime.call_started", call_sid=self.call_sid, stream_sid=self.stream_sid)
            # Proactive greeting so caller hears something immediately
            await self._handle_text(
                "Hi! Which building are you interested in touring from our portfolio?",
                websocket,
                infer=False,
            )
            return
        if payload.event == "media" and payload.media:
            audio_b64 = payload.media.get("payload")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                transcript = await self.deepgram.transcribe_chunk(audio_bytes)
                if transcript:
                    await self._handle_transcript(transcript, websocket)
            return
        if payload.event == "mark" and payload.mark:
            mark_name = payload.mark.get("name")
            if mark_name == "transcript":
                text = payload.mark.get("payload", {}).get("text", "")
                await self._handle_text(text, websocket)
            elif mark_name == "gather":
                gather_text = payload.mark.get("payload", {}).get("text", "")
                await self._handle_text(gather_text, websocket)
            return
        if payload.event == "stop":
            logger.info("realtime.call_stopped", call_sid=self.call_sid)
            final_transcript = await self.deepgram.finalize()
            if final_transcript:
                await self._handle_transcript(final_transcript, websocket)
            return

    async def _handle_text(self, text: str, websocket: WebSocket, *, infer: bool = True) -> None:
        if not text:
            return
        if not infer:
            self.recorder.log("PLAN", "direct_text", text=text)
            audio_bytes = await self._synthesize_with_timeout(text, timeout_sec=2.0)
            if not audio_bytes:
                audio_bytes = _generate_mulaw_tone(duration_s=0.5, freq_hz=440)
            await self._send_twilio_agent_events(text, audio_bytes, websocket)
            return
        transcript = await self.deepgram.push_text(text)
        await self._handle_transcript(transcript, websocket)

    async def _handle_transcript(self, transcript: RealtimeTranscript, websocket: WebSocket) -> None:
        if not transcript.transcript:
            return
        with self.recorder.stage("PLAN", transcript=transcript.transcript):
            agent_response = await self.openai.infer(transcript)

        if agent_response.tool_calls:
            tool_results = await self._execute_tool_calls(agent_response.tool_calls)
            if tool_results:
                summary = self._render_tool_results(tool_results)
                if agent_response.text:
                    agent_response.text = f"{agent_response.text} {summary}".strip()
                else:
                    agent_response.text = summary

        # Synthesize audio suitable for Twilio (8kHz mu-law) and stream back
        if agent_response.tts_audio_b64:
            try:
                audio_bytes = base64.b64decode(agent_response.tts_audio_b64)
            except Exception:  # noqa: BLE001
                audio_bytes = b""
        else:
            audio_bytes = await self._synthesize_with_timeout(agent_response.text or "", timeout_sec=4.0)
        if not audio_bytes:
            audio_bytes = _generate_mulaw_tone(duration_s=0.5, freq_hz=440)
        await self._send_twilio_agent_events(agent_response.text, audio_bytes, websocket)

    async def _execute_tool_calls(self, tool_calls: List[ToolCallRequest]) -> List[ToolCallResult]:
        results: List[ToolCallResult] = []
        stage_map = {
            "check_availability": "PLAN",
            "route_to_sister_property": "PLAN",
            "compute_net_effective_rent": "NER",
            "check_policy": "POLICY",
            "book_tour": "BOOK_TOUR",
            "send_sms": "SMS",
        }
        for call in tool_calls:
            try:
                stage_name = stage_map.get(call.name, "PLAN")
                with self.recorder.stage(stage_name, tool=call.name):
                    output = self.dispatcher.dispatch(call.name, call.arguments)
            except Exception as exc:  # noqa: BLE001
                logger.exception("tool.error name=%s err=%s", call.name, exc)
                output = {"error": str(exc)}
            if not isinstance(output, dict):
                output = {"result": output}
            result = ToolCallResult(call_id=call.call_id, name=call.name, output=output)
            results.append(result)
        return results

    async def _send_twilio_agent_events(self, text: str, audio_bytes: bytes, websocket: WebSocket) -> None:
        if self.stream_sid is None:
            logger.warning("realtime.no_stream_sid")
        mark_payload = {
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {
                "name": "agent_response",
                "payload": {"text": text},
            },
        }
        await websocket.send_json(mark_payload)
        # Twilio expects 8kHz mu-law 20ms frames (160 bytes) base64-encoded per message
        if audio_bytes:
            frame_size_bytes = 160
            frames_sent = 0
            for i in range(0, len(audio_bytes), frame_size_bytes):
                frame = audio_bytes[i : i + frame_size_bytes]
                if not frame:
                    continue
                payload_b64 = base64.b64encode(frame).decode()
                await websocket.send_json(
                    {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": payload_b64},
                    }
                )
                # Pace frames at ~20ms per chunk to approximate real-time playback
                await asyncio.sleep(0.02)
                frames_sent += 1
            logger.info("realtime.audio_frames_sent", frames=frames_sent, bytes=len(audio_bytes))
        else:
            logger.warning("realtime.audio_empty")
        await websocket.send_json(
            {
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {"name": "agent_response_complete"},
            }
        )

    def _render_tool_results(self, tool_results: List[ToolCallResult]) -> str:
        summaries: List[str] = []
        for result in tool_results:
            if result.name == "book_tour" and "booking_id" in result.output:
                summaries.append(
                    "Booking confirmed. Confirmation ID "
                    f"{result.output['booking_id']} for {result.output.get('property_id')} on "
                    f"{result.output.get('tour_time')}"
                )
            elif result.name == "check_policy":
                policies = result.output.get("policies") or result.output
                summaries.append(f"Policy details: {policies}")
            else:
                summaries.append(f"{result.name} result: {result.output}")
        return " ".join(summaries).strip()


    async def _synthesize_with_timeout(self, text: str, timeout_sec: float) -> bytes:
        if not text:
            return b""
        try:
            return await asyncio.wait_for(self.openai.synthesize_text(text), timeout=timeout_sec)
        except asyncio.TimeoutError:
            logger.warning("tts.timeout timeout=%s", timeout_sec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("tts.error %s", exc)
        return b""


def _mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    if not mulaw_bytes:
        return b""
    samples = []
    for value in mulaw_bytes:
        u_value = ~value & 0xFF
        sign = -1 if (u_value & 0x80) else 1
        exponent = (u_value >> 4) & 0x07
        mantissa = u_value & 0x0F
        sample = ((mantissa | 0x10) << (exponent + 3)) - 0x84
        samples.append(sign * sample)
    return struct.pack("<" + "h" * len(samples), *samples)


def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int) -> bytes:
    buffer = io.BytesIO()
    byte_rate = sample_rate * channels * 2
    block_align = channels * 2
    buffer.write(b"RIFF")
    buffer.write(struct.pack("<I", 36 + len(pcm_bytes)))
    buffer.write(b"WAVE")
    buffer.write(b"fmt ")
    buffer.write(struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, 16))
    buffer.write(b"data")
    buffer.write(struct.pack("<I", len(pcm_bytes)))
    buffer.write(pcm_bytes)
    return buffer.getvalue()


def _generate_mulaw_tone(duration_s: float, freq_hz: float = 440.0, sample_rate: int = 8000) -> bytes:
    num_samples = int(duration_s * sample_rate)
    pcm16 = []
    for n in range(num_samples):
        t = n / sample_rate
        sample = int(32767 * math.sin(2 * math.pi * freq_hz * t))
        pcm16.append(sample)

    def linear_to_mulaw(sample: int) -> int:
        # From ITU-T G.711 μ-law
        MU = 255
        sign = 0x80 if sample < 0 else 0x00
        sample = abs(sample)
        sample = min(sample, 32635)
        exponent = 7
        mask = 0x4000
        while exponent > 0 and not (sample & mask):
            mask >>= 1
            exponent -= 1
        mantissa = (sample >> (exponent + 3)) & 0x0F
        mulaw = ~(sign | (exponent << 4) | mantissa) & 0xFF
        return mulaw

    ulaw_bytes = bytes(linear_to_mulaw(s) for s in pcm16)
    return ulaw_bytes


def _pcm_to_mulaw(pcm_bytes: bytes, sample_rate: int) -> bytes:
    if not pcm_bytes:
        return b""
    if sample_rate != 8000:
        pcm_bytes, _ = audioop.ratecv(pcm_bytes, 2, 1, sample_rate, 8000, None)
    return audioop.lin2ulaw(pcm_bytes, 2)
