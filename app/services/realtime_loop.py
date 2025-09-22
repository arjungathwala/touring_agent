from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

import logging
import httpx
from deepgram import DeepgramClient as SDKDeepgramClient, FileSource, PrerecordedOptions
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
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
        self._client: Optional[SDKDeepgramClient] = None
        self._options: Optional[PrerecordedOptions] = None
        if self.api_key:
            self._client = SDKDeepgramClient(self.api_key)
            model = os.getenv("DEEPGRAM_MODEL", "nova-3")  # Use latest model from docs
            self._options = PrerecordedOptions(
                model=model,
                smart_format=True,
                punctuate=True,
                encoding="mulaw",
                sample_rate=8000,
                channels=1,
            )
        self._min_flush_bytes = 800  # Even faster response for better UX
        self._no_asr_warned = False

    async def transcribe_chunk(self, audio_chunk: bytes) -> Optional[RealtimeTranscript]:
        self.audio_buffer.extend(audio_chunk)
        self.recorder.log("ASR", "chunk_received", chunk_size=len(audio_chunk), buffer_total=len(self.audio_buffer))
# Debug: print(f"🎤 Audio chunk: {len(audio_chunk)} bytes, buffer total: {len(self.audio_buffer)} bytes")
        
        if self._client and self._options:
            if len(self.audio_buffer) < self._min_flush_bytes:
                self.recorder.log("ASR", "buffer_too_small", current=len(self.audio_buffer), needed=self._min_flush_bytes)
                return None
            # Analyze audio data before sending to Deepgram
            audio_data = bytes(self.audio_buffer)
            unique_values = len(set(audio_data))
            non_silence_bytes = sum(1 for b in audio_data if b not in [0x00, 0xFF])
            
            # Calculate some basic audio statistics
            avg_value = sum(audio_data) / len(audio_data) if audio_data else 0
            min_value = min(audio_data) if audio_data else 0
            max_value = max(audio_data) if audio_data else 0
            
            print(f"🔄 Transcribing {len(self.audio_buffer)} bytes: {unique_values} unique values, {non_silence_bytes} speech bytes")
            
            # Audio sample debugging (disabled)
            # if needed for debugging, uncomment the file saving logic
            
            self.recorder.log("ASR", "deepgram_flush", buffered=len(self.audio_buffer))
            result = await self._flush_deepgram()
            if result:
                print(f"✅ Transcript: '{result.transcript}' (confidence: {result.confidence:.2f})")
            else:
                print("❌ No speech detected")
            return result
        
        if not self._no_asr_warned:
            print("❌ No Deepgram client available!")
            self.recorder.log("ASR", "no_deepgram_client_available")
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
        if self._client and self._options:
            return await self._flush_deepgram()
        self.audio_buffer.clear()
        return None

    async def _flush_deepgram(self) -> Optional[RealtimeTranscript]:
        if not self._client or not self._options:
            return None
        payload = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        file_source: FileSource = {"buffer": payload}

        try:
            def _sync_transcribe() -> dict[str, Any]:
                response = self._client.listen.rest.v("1").transcribe_file(
                    file_source,
                    self._options,
                )
                return response.to_dict() if hasattr(response, "to_dict") else response

            with self.recorder.stage("ASR", bytes=len(payload)):
                data = await asyncio.to_thread(_sync_transcribe)
                self.recorder.log("ASR", "deepgram_raw_response", response=str(data)[:500])
                # Debug: print(f"🔍 Deepgram response: {data}")
                
                # Check confidence and alternatives  
                channels = data.get("results", {}).get("channels", [])
                if channels:
                    alternatives = channels[0].get("alternatives", [])
                    # Debug: print(f"📊 Alternatives found: {len(alternatives)}")
                    # for i, alt in enumerate(alternatives):
                    #     print(f"  Alt {i}: '{alt.get('transcript', '')}' (confidence: {alt.get('confidence', 0)})")

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
            self.recorder.log("ASR", "deepgram_empty_transcript", payload=data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("deepgram.sdk_error %s", exc, exc_info=True)
            self.recorder.log("ASR", "deepgram_sdk_error", error=str(exc))
        return None

class OpenAIResponsesClient:
    def __init__(self, planner: AgentPlanner, dispatcher: ToolDispatcher, recorder: FlightRecorder) -> None:
        self.planner = planner
        self.dispatcher = dispatcher
        self.recorder = recorder
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_RESPONSES_MODEL", os.getenv("OPENAI_MODEL", "gpt-5"))
        self.system_prompt = os.getenv(
            "OPENAI_SYSTEM_PROMPT",
            "You are a helpful leasing assistant for luxury apartment tours. "
            "Answer succinctly and keep a welcoming tone.",
        )
        self._http: Optional[httpx.AsyncClient] = None
        if self.api_key:
            self._http = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(20.0, connect=5.0),
            )

    async def infer(self, transcript: RealtimeTranscript) -> AgentResponse:
        if not self._http:
            self.recorder.log("PLAN", "openai_missing_api_key")
            text_response = await self.planner.process_transcript(transcript.transcript)
            return AgentResponse(text=text_response.get("text", ""), barge_in=False)

        payload = {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": transcript.transcript}],
                }
            ],
            "tools": self.dispatcher.get_tool_schemas(),
        }

        try:
            with self.recorder.stage("PLAN", provider="openai"):
                response = await self._http.post("/responses", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text if exc.response is not None else ""
            logger.warning(
                "openai.responses_http_error status=%s body=%s",
                exc.response.status_code if exc.response is not None else "?",
                body[:500],
                exc_info=True,
            )
            self.recorder.log("PLAN", "openai_responses_error", status=exc.response.status_code if exc.response else "?", body=body[:200])
            text_response = await self.planner.process_transcript(transcript.transcript)
            return AgentResponse(text=text_response.get("text", ""), barge_in=False)
        except httpx.HTTPError as exc:
            logger.warning("openai.responses_transport_error %s", exc, exc_info=True)
            self.recorder.log("PLAN", "openai_responses_transport_error", error=str(exc))
            text_response = await self.planner.process_transcript(transcript.transcript)
            return AgentResponse(text=text_response.get("text", ""), barge_in=False)

        data = response.json()
        text_chunks: List[str] = []
        tool_calls: List[ToolCallRequest] = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text_chunks.append(content.get("text", ""))
                    elif content.get("type") == "tool_call":
                        arguments = content.get("arguments") or {}
                        tool_calls.append(
                            ToolCallRequest(
                                name=content.get("name", ""),
                                arguments=arguments,
                                call_id=content.get("call_id", content.get("id", "tool_call")),
                            )
                        )

        text = " ".join(text_chunks).strip()
        return AgentResponse(text=text, barge_in=False, tool_calls=tool_calls)


class ElevenLabsClient:
    def __init__(self, recorder: FlightRecorder) -> None:
        self.recorder = recorder
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self._client: Optional[ElevenLabs] = None
        if self.api_key:
            self._client = ElevenLabs(api_key=self.api_key)

    async def synthesize(self, text: str) -> bytes:
        if not text:
            return b""
        if not self._client:
            self.recorder.log("PLAN", "tts_stubbed")
            return _generate_mulaw_tone(duration_s=0.5, freq_hz=440)
        
        try:
            with self.recorder.stage("PLAN", provider="elevenlabs"):
                # Use the official SDK's generate method with streaming
                audio_generator = self._client.generate(
                    text=text,
                    voice=self.voice_id,
                    model="eleven_turbo_v2_5",  # Fast model for real-time use
                    voice_settings={
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                    output_format="ulaw_8000",  # Twilio-compatible format
                    stream=True
                )
                
                # Collect all audio chunks
                audio_chunks = []
                def _sync_collect():
                    for chunk in audio_generator:
                        if chunk:
                            audio_chunks.append(chunk)
                
                # Run the synchronous generator in a thread
                await asyncio.to_thread(_sync_collect)
                audio_bytes = b"".join(audio_chunks)
                
            self.recorder.log("PLAN", "tts_generated", provider="elevenlabs", bytes=len(audio_bytes))
            return audio_bytes
            
        except Exception as exc:
            logger.warning("elevenlabs.tts_error %s", exc, exc_info=True)
            self.recorder.log("PLAN", "tts_error", provider="elevenlabs", error=str(exc))
            return b""

    async def close(self) -> None:
        # The official SDK client doesn't require explicit closing
        self._client = None


class RealtimeLoop:
    def __init__(self, recorder: FlightRecorder) -> None:
        self.recorder = recorder
        self.dispatcher = ToolDispatcher(recorder)
        self.planner = AgentPlanner(self.dispatcher, recorder)
        self.deepgram = DeepgramClient(recorder)
        self.openai = OpenAIResponsesClient(self.planner, self.dispatcher, recorder)
        self.elevenlabs = ElevenLabsClient(recorder)
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
                self.recorder.log("ASR", "media_received", audio_bytes=len(audio_bytes), b64_length=len(audio_b64))
                transcript = await self.deepgram.transcribe_chunk(audio_bytes)
                if transcript:
                    self.recorder.log("ASR", "transcript_ready", text=transcript.transcript[:100])
                    await self._handle_transcript(transcript, websocket)
                else:
                    self.recorder.log("ASR", "no_transcript_yet", buffer_size=len(self.deepgram.audio_buffer))
            else:
                self.recorder.log("ASR", "media_no_payload", media_keys=list(payload.media.keys()) if payload.media else [])
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
            audio_bytes = await self.elevenlabs.synthesize(text)
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
            audio_bytes = await self.elevenlabs.synthesize(agent_response.text or "")
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
                # Pace frames at ~10ms per chunk for faster playback and prevent timeouts
                await asyncio.sleep(0.01)
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
