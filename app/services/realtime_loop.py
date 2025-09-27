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
from typing import Any, AsyncIterator, List, Optional

import logging
import re
import threading
from deepgram import DeepgramClient as SDKDeepgramClient, LiveOptions, LiveTranscriptionEvents
from elevenlabs.client import ElevenLabs
from elevenlabs import stream, VoiceSettings
from fastapi import WebSocket
from collections import deque

from openai import AsyncOpenAI

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
        self._connection = None
        self._latest_transcript = None
        self._transcript_queue = []
        self._transcript_buffer = ""  # Buffer for accumulating partial transcripts
        self._last_transcript_time = None
        self._keepalive_task: Optional[asyncio.Task] = None
        if self.api_key:
            self._client = SDKDeepgramClient(self.api_key)
        self._no_asr_warned = False

    def _has_voice_activity(self, audio_data: bytes) -> bool:
        """Simple Voice Activity Detection for μ-law audio"""
        if len(audio_data) < 160:
            return False
            
        # μ-law silence is typically 0xFF (255) or 0x7F (127)
        silence_values = {0x00, 0x7F, 0xFF}
        
        # Calculate voice activity metrics
        unique_values = len(set(audio_data))
        non_silence_count = sum(1 for b in audio_data if b not in silence_values)
        
        # Calculate energy/amplitude variation
        if len(audio_data) > 1:
            # Check for amplitude variation (sign of speech)
            amplitude_changes = 0
            for i in range(1, len(audio_data)):
                if abs(audio_data[i] - audio_data[i-1]) > 10:  # Significant change
                    amplitude_changes += 1
        else:
            amplitude_changes = 0
            
        # More sophisticated VAD thresholds
        variation_threshold = len(audio_data) * 0.05  # At least 5% non-silence (lowered)
        energy_threshold = 3  # At least 3 unique values (lowered)
        change_threshold = len(audio_data) * 0.02  # At least 2% amplitude changes
        
        has_variation = non_silence_count > variation_threshold
        has_energy = unique_values > energy_threshold  
        has_changes = amplitude_changes > change_threshold
        
        # More lenient: any two of the three conditions
        voice_indicators = sum([has_variation, has_energy, has_changes])

        speaking = voice_indicators >= 2
        if speaking:
            # Immediately interrupt any ongoing TTS playback
            if self._current_tts_task and not self._current_tts_task.done():
                self._current_tts_task.cancel()
            self._barge_in_detected = True
        return speaking

    def _is_complete_sentence(self, text: str) -> bool:
        """Determine if the buffered text represents a complete sentence/thought"""
        if not text.strip():
            return False
            
        text = text.strip().lower()
        
        # Ignore meaningless fragments and incomplete phrases
        meaningless_fragments = [
            'okay', 'ok', 'yeah', 'yes', 'no', 'um', 'uh', 'hmm', 'well', 'so', 
            'and', 'but', 'or', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
            'questions', 'question', 'can you', 'i want', 'i need', 'i would'
        ]
        
        # If it's just a meaningless fragment, don't process
        if text in meaningless_fragments:
            return False
        
        # If it's too short (less than 3 words) and not a clear command, ignore
        words = text.split()
        if len(words) < 3:
            # Allow some short but meaningful phrases
            meaningful_short = [
                'book it', 'yes please', 'no thanks', 'thank you', 'sounds good',
                '21 west end', 'hudson 360', 'riverview lofts', '1 bedroom', '2 bedroom',
                'studio apartment', 'tomorrow morning', 'next week'
            ]
            if not any(phrase in text for phrase in meaningful_short):
                return False
        
        # Check for sentence endings
        if text.endswith(('.', '!', '?')):
            return True
            
        # Check for complete phrases/questions that indicate intent
        complete_patterns = [
            'can you book me',
            'i want to tour',
            'i\'m interested in',
            'interested in touring',
            'what time',
            'when can i',
            'how much',
            'do you have',
            'i have pets',
            'my name is',
            'my email is',
            'my phone is',
            'book me for',
            'schedule me',
            'next tuesday',
            'next wednesday',
            'next thursday',
            'next friday',
            'next monday',
            'this weekend',
            'tomorrow at'
        ]
        
        for pattern in complete_patterns:
            if pattern in text:
                return True
                
        # Check for time-based completeness (day + time mentions)
        has_day = any(day in text for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'tomorrow', 'today', 'weekend', 'weekday'])
        has_time = any(time_word in text for time_word in ['am', 'pm', 'morning', 'afternoon', 'evening', 'noon', 'midnight', '11am', '2pm', 'at 11'])
        
        if has_day and has_time:
            return True
            
        # Check for property mentions (including fuzzy matches)
        property_patterns = [
            '21 west end', '21 west', 'twenty one west', '21 best end', '21 vest end',
            'hudson 360', 'hudson', 'riverview lofts', 'riverview', 'river view'
        ]
        if any(prop in text for prop in property_patterns):
            return True
            
        # If buffer is getting long (>6 words), consider it complete to avoid hanging
        if len(text.split()) > 6:
            return True
            
        return False

    def _flush_transcript_buffer_if_stale(self) -> None:
        """Flush transcript buffer if it's been sitting for too long without completion"""
        if (self._transcript_buffer.strip() and 
            self._last_transcript_time and 
            (datetime.now(timezone.utc) - self._last_transcript_time).total_seconds() > 2.0):
            
            # Force flush the stale buffer
            complete_transcript = RealtimeTranscript(
                transcript=self._transcript_buffer.strip(),
                confidence=0.8,  # Lower confidence for forced flush
                timestamp=datetime.now(timezone.utc),
            )
            self._transcript_queue.append(complete_transcript)
            print(f"🎤 STT (timeout): '{self._transcript_buffer.strip()}'")
            self._transcript_buffer = ""

    async def start_streaming(self) -> None:
        """Initialize Deepgram streaming connection following documentation exactly"""
        if not self._client:
            return
            
        try:
            # Create streaming connection exactly as per docs
            self._connection = self._client.listen.websocket.v("1")
            
            # Event handlers using method call syntax (decorator syntax has issues)
            def handle_transcript(self_dg, result, **kwargs):
                # Handle the exact response format from docs: {"type":"Results","channel":{"alternatives":[...]}}
                if hasattr(result, 'channel') and result.channel and result.channel.alternatives:
                    transcript_text = result.channel.alternatives[0].transcript
                    confidence = result.channel.alternatives[0].confidence
                    is_final = getattr(result, 'is_final', False)
                    speech_final = getattr(result, 'speech_final', False)
                    
                    if transcript_text.strip():
                        current_time = datetime.now(timezone.utc)
                        
                        if is_final or speech_final:
                            # This is a final transcript - replace buffer with clean text
                            self._transcript_buffer = transcript_text.strip()
                            self._last_transcript_time = current_time
                            
                            # Check if we have a complete sentence/thought
                            if self._is_complete_sentence(self._transcript_buffer):
                                # Process the complete buffered transcript
                                complete_transcript = RealtimeTranscript(
                                    transcript=self._transcript_buffer.strip(),
                                    confidence=confidence,
                                    timestamp=current_time,
                                )
                                self._transcript_queue.append(complete_transcript)
                                print(f"🎤 STT: '{self._transcript_buffer.strip()}'")
                                self._transcript_buffer = ""  # Reset buffer
                        else:
                            # This is an interim result - just update the buffer without processing
                            self._transcript_buffer = transcript_text.strip()
                            self._last_transcript_time = current_time
            
            def handle_error(self_dg, error, **kwargs):
                error_str = str(error)
                if "1011" in error_str:
                    print(f"⚠️ Deepgram timeout (will reconnect)")
                else:
                    print(f"❌ Deepgram error: {error}")
                # Mark connection as failed for auto-restart
                self._is_connected = False
                self._connection = None
            
            # Register handlers using method call syntax
            self._connection.on(LiveTranscriptionEvents.Transcript, handle_transcript)
            self._connection.on(LiveTranscriptionEvents.Error, handle_error)
            
            # Correct parameters based on documentation (strings vs integers)
            options = LiveOptions(
                model="nova-3",
                language="en",
                encoding="mulaw",
                sample_rate="8000",  # String as per docs
                channels="1",        # String as per docs  
                interim_results=True,
                smart_format=True,
                punctuate=True,
                endpointing="10",    # String as per docs (10ms default)
                utterance_end_ms="1000",  # String as per docs
            )
            
            self._connection.start(options)
            print("🔗 Deepgram streaming ready")
            self._is_connected = True
            
            # Start keepalive task to prevent 1011 timeouts
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            
        except Exception as e:
            print(f"❌ Streaming connection failed: {e}")
            self._is_connected = False
            self._connection = None

    async def _keepalive_loop(self) -> None:
        """Send KeepAlive messages to prevent timeout as per Deepgram docs"""
        try:
            while self._is_connected and self._connection:
                await asyncio.sleep(8)  # Send keepalive every 8 seconds (before 10s timeout)
                if self._connection and self._is_connected:
                    try:
                        # Send KeepAlive message as per documentation
                        keepalive_msg = {"type": "KeepAlive"}
                        self._connection.send(json.dumps(keepalive_msg))
                    except Exception as e:
                        print(f"⚠️ Keepalive failed: {e}")
                        break
        except asyncio.CancelledError:
            pass  # Task was cancelled, exit gracefully

    async def transcribe_chunk(self, audio_chunk: bytes) -> Optional[RealtimeTranscript]:
        """Robust streaming transcription with auto-reconnection"""
        if not self._client:
            if not self._no_asr_warned:
                self.recorder.log("ASR", "no_deepgram_client_available")
                self._no_asr_warned = True
            return None
            
        # Ensure streaming connection is active
        if not self._connection or not self._is_connected:
            await self.start_streaming()
            
        # Send audio chunk to streaming connection
        if self._connection and self._is_connected:
            try:
                self._connection.send(audio_chunk)
                self._last_audio_time = datetime.now()
                
                # Check for buffered transcript timeout (flush if no activity for 2 seconds)
                self._flush_transcript_buffer_if_stale()
                
                # Check for any new transcripts
                if self._transcript_queue:
                    transcript = self._transcript_queue.pop(0)
                    return transcript
                    
            except Exception as e:
                print(f"❌ Error sending audio: {e}")
                self.recorder.log("ASR", "streaming_send_error", error=str(e))
                # Mark for reconnection
                self._is_connected = False
                self._connection = None
                
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
        """Finalize streaming connection and get final transcript"""
        if self._connection:
            try:
                # Send any remaining audio
                if self.audio_buffer:
                    self._connection.send(bytes(self.audio_buffer))
                    self.audio_buffer.clear()
                
                # Close connection to get final results
                self._connection.finish()
                self._connection = None
                self._is_connected = False
                
                # Return any remaining transcript
                if self._transcript_queue:
                    return self._transcript_queue.pop(0)
                    
            except Exception as e:
                print(f"❌ Deepgram finalize error: {e}")
        
        self.audio_buffer.clear()
        return None

    async def close(self) -> None:
        """Close the streaming connection and cleanup"""
        # Cancel keepalive task
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            
        # Close connection
        if self._connection:
            try:
                self._connection.finish()
                self._connection = None
                self._is_connected = False
            except Exception as e:
                print(f"⚠️ Deepgram close error: {e}")

class GroqResponsesClient:
    def __init__(self, planner: AgentPlanner, dispatcher: ToolDispatcher, recorder: FlightRecorder) -> None:
        self.planner = planner
        self.dispatcher = dispatcher
        self.recorder = recorder
        self.api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = (
            os.getenv("GROQ_RESPONSES_MODEL")
            or os.getenv("GROQ_MODEL")
            or os.getenv("OPENAI_RESPONSES_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "openai/gpt-oss-20b"
        )
        self.system_prompt = os.getenv(
            "GROQ_SYSTEM_PROMPT",
            "You are a helpful leasing specialist whose primary goal is to book apartment tours by gathering required information. "
            "Ask specific, bite-sized questions to collect: Property, Bedroom size, Tour time, Contact info. "
            "Be conversational and helpful, but always ask the next question needed to complete the booking. "
            "Keep asking questions until you have all the information needed to schedule their tour. And then use the tour booking tools."
            "Focus on being informative and systematically gathering the information needed to schedule their tour.",
        )
        if not self.system_prompt:
            self.system_prompt = os.getenv(
                "OPENAI_SYSTEM_PROMPT",
                "You are a helpful leasing specialist whose primary goal is to book apartment tours by gathering required information. "
                "Ask specific, bite-sized questions to collect: Property, Bedroom size, Tour time, Contact info. "
                "Be conversational and helpful, but always ask the next question needed to complete the booking. "
                "Keep asking questions until you have all the information needed to schedule their tour. And then use the tour booking tools."
                "Focus on being informative and systematically gathering the information needed to schedule their tour.",
            )
        self._client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
            )

    async def infer(self, transcript: RealtimeTranscript) -> AgentResponse:
        if not self._client:
            self.recorder.log("PLAN", "groq_missing_api_key")
            text_response = await self.planner.process_transcript(transcript.transcript)
            # Enable barge-in for longer responses
            response_text = text_response.get("text", "")
            enable_barge_in = len(response_text) > 100  # Enable for responses longer than 100 chars
            return AgentResponse(text=response_text, barge_in=enable_barge_in)

        # Inject current planner state and booking requirements for better tool selection
        state_summary = self._summarize_state()
        booking_context = self.planner._get_booking_requirements_context()
        missing_info = self.planner._missing_booking_info()
        history = getattr(self.planner.state, "transcript_history", [])
        history_text = " | ".join(f"{turn['role']}: {turn['text']}" for turn in history[-12:]) if history else "(none)"

        booking_rules = (
            "CRITICAL: Until tour is booked, EVERY response must end with a question to gather the next missing piece of info. "
            "Required to book: 1) Property name, 2) Bedroom size, 3) Tour date/time, 4) Contact info. "
            "NEVER use sales phrases like 'units move fast', 'book you right now', 'these slots move fast'. "
            "Instead ask specific questions like: 'What size apartment—studio, 1BR, or 2BR?' or 'What day works for your tour?' "
            f"Currently missing: {', '.join(missing_info) if missing_info else 'NOTHING - READY TO BOOK'}. "
            "Ask for the FIRST missing item only. Be helpful and conversational, not pushy."
        )

        orchestration_rules = (
            "You must write the entire response (no canned prompts). "
            "Keep replies under 2 short sentences. "
            "Do not try to book a tour until property, bedrooms, move-in date, budget, tour time, and contact are collected. "
            "If no availability or property can't be found, recommend a sister property using the latest tool summary. "
            "Whenever the caller supplies new information (property, bedrooms, move-in, budget, contact, tour time), call the matching planner_set_* tool with the structured value before replying. "
            "Use planner_check_availability before offering to book and planner_recommend_sister when the primary property has no matches. "
            "When referencing tool results, summarize in natural language."
        )

        brevity_rules = (
            "Be concise. Ask exactly one focused question at a time. "
            "Prefer short sentences. Avoid multi-part lists unless strictly necessary. "
            "Acknowledge new info briefly before asking the next question."
        )
        
        payload = {
            "model": self.model,
            "instructions": (
                f"{self.system_prompt}\n\n"
                f"{booking_rules}\n\n"
                f"Operational Instructions: {orchestration_rules}\n\n"
                f"Style Rules: {brevity_rules}\n\n"
                f"Booking Status: {booking_context}\n\n"
                f"Conversation Memory: {history_text}\n\n"
                f"Conversation State: {state_summary}"
            ),
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": transcript.transcript}],
                }
            ],
            "tools": self._collect_llm_tools(),
        }

        try:
            with self.recorder.stage("PLAN", provider="groq"):
                response = await self._client.responses.create(**payload)
        except Exception as exc:
            logger.warning("groq.responses_error %s", exc, exc_info=True)
            self.recorder.log("PLAN", "groq_responses_error", error=str(exc))
            text_response = await self.planner.process_transcript(transcript.transcript)
            return AgentResponse(text=text_response.get("text", ""), barge_in=False)

        data = response.dict()
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
        # Enable barge-in for longer responses or when tools are involved
        enable_barge_in = len(text) > 100 or len(tool_calls) > 0

        tool_results: List[ToolCallResult] = []
        for call in tool_calls:
            try:
                output = self.planner.execute_llm_tool(call.name, call.arguments)
                tool_results.append(
                    ToolCallResult(call_id=call.call_id, name=call.name, output=output)
                )
                text_chunks.append(output.get("text", ""))
            except Exception as exc:  # noqa: BLE001
                logger.exception("planner.tool_execution_error %s", call.name)
                tool_results.append(
                    ToolCallResult(call_id=call.call_id, name=call.name, output={"error": str(exc)})
                )

        text = " ".join(chunk for chunk in text_chunks if chunk).strip()

        # Ensure the planner captures the final agent response for memory
        if text:
            self.planner.state.transcript_history.append({"role": "agent", "text": text})

        # Enforce follow-up question when required
        missing = self.planner._missing_booking_info()
        if missing and not text.endswith("?"):
            next_question = self.planner._get_next_question(missing[0])
            if next_question and next_question not in text:
                text = (text + " " + next_question).strip()

        enable_barge_in = len(text) > 100 or bool(tool_results)
        return AgentResponse(text=text, barge_in=enable_barge_in, tool_calls=tool_calls, tools=tool_results)

    def _summarize_state(self) -> str:
        s = self.planner.state
        parts = []
        if s.property_name:
            parts.append(f"property={s.property_name}")
        if s.desired_bedrooms is not None:
            parts.append(f"bedrooms={s.desired_bedrooms}")
        if s.has_pets is not None:
            parts.append(f"pets={'yes' if s.has_pets else 'no'}")
        if s.tour_type:
            parts.append(f"tour_type={s.tour_type}")
        if s.move_in_date:
            parts.append(f"move_in={s.move_in_date}")
        if s.budget_min or s.budget_max:
            parts.append(f"budget=${s.budget_min or ''}-{s.budget_max or ''}")
        if s.requested_time:
            parts.append(f"time={s.requested_time}")
        if s.prospect_name:
            parts.append("have_name")
        if s.prospect_email or s.prospect_phone:
            parts.append("have_contact")
        if s.last_tool_summary:
            parts.append(f"tool={s.last_tool_summary}")
        if s.last_error:
            parts.append(f"error={s.last_error}")
        if s.sister_recommendation:
            name = s.sister_recommendation.get("name")
            parts.append(f"sister_option={name}")
        parts.append(f"last_action={s.last_action}")
        return ", ".join(parts) or "(none)"

    def _collect_llm_tools(self) -> List[Dict[str, Any]]:
        planner_tools = getattr(self.planner, "get_llm_tool_schemas", None)
        return planner_tools() if callable(planner_tools) else []


class ElevenLabsClient:
    def __init__(self, recorder: FlightRecorder) -> None:
        self.recorder = recorder
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self._client: Optional[ElevenLabs] = None
        self._stream_lock = threading.Lock()
        self._stream_buffer: deque[bytes] = deque()
        self._stream_event = threading.Event()
        if self.api_key:
            self._client = ElevenLabs(api_key=self.api_key)

    async def synthesize(self, text: str) -> bytes:
        """Retained for compatibility: generate full audio buffer."""
        if not text:
            return b""
        if not self._client:
            self.recorder.log("PLAN", "tts_stubbed")
            return _generate_mulaw_tone(duration_s=0.5, freq_hz=440)

        def _collect_audio() -> bytes:
            chunks: List[bytes] = []
            for part in self._client.text_to_speech.convert(
                self.voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                output_format="ulaw_8000",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                ),
            ):
                if isinstance(part, (bytes, bytearray)):
                    chunks.append(bytes(part))
            return b"".join(chunks)

        try:
            with self.recorder.stage("PLAN", provider="elevenlabs"):
                audio_bytes = await asyncio.to_thread(_collect_audio)
            self.recorder.log("PLAN", "tts_generated", provider="elevenlabs", bytes=len(audio_bytes))
            return audio_bytes
        except Exception as exc:
            logger.warning("elevenlabs.tts_error %s", exc, exc_info=True)
            self.recorder.log("PLAN", "tts_error", provider="elevenlabs", error=str(exc))
            return b""

    def stream_tts(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio chunks for immediate playback."""
        if not text or not self._client:
            async def _empty():
                if not text:
                    yield b""
                else:
                    yield _generate_mulaw_tone(duration_s=0.5, freq_hz=440)
            return _empty()

        async def _stream_generator() -> AsyncIterator[bytes]:
            queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
            class _Sentinel:
                pass
            sentinel = _Sentinel()
            loop = asyncio.get_running_loop()

            def _producer() -> None:
                try:
                    with self.recorder.stage("PLAN", provider="elevenlabs", operation="stream"):
                        for part in self._client.text_to_speech.convert_as_stream(
                            self.voice_id,
                            text=text,
                            model_id="eleven_turbo_v2_5",
                            output_format="ulaw_8000",
                            voice_settings=VoiceSettings(
                                stability=0.5,
                                similarity_boost=0.75,
                                style=0.0,
                                use_speaker_boost=True,
                                speed=1.0,
                            ),
                        ):
                            if part:
                                asyncio.run_coroutine_threadsafe(queue.put(bytes(part)), loop)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("elevenlabs.stream_error %s", exc, exc_info=True)
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)

            threading.Thread(target=_producer, daemon=True).start()

            while True:
                chunk = await queue.get()
                if isinstance(chunk, _Sentinel):
                    break
                if chunk:
                    yield chunk
            if queue.empty():
                return

        return _stream_generator()


class RealtimeLoop:
    def __init__(self, recorder: FlightRecorder) -> None:
        self.recorder = recorder
        self.dispatcher = ToolDispatcher(recorder)
        self.planner = AgentPlanner(self.dispatcher, recorder)
        self.deepgram = DeepgramClient(recorder)
        self.groq = GroqResponsesClient(self.planner, self.dispatcher, recorder)
        self.elevenlabs = ElevenLabsClient(recorder)
        self.call_sid: Optional[str] = None
        self.stream_sid: Optional[str] = None
        self._is_agent_speaking = False
        self._barge_in_detected = False
        self._current_response_task: Optional[asyncio.Task] = None
        self._processing_transcript = False  # Lock to prevent concurrent transcript processing

    async def handle_event(self, payload: TwilioMediaPayload, websocket: WebSocket) -> None:
        if payload.event == "start":
            start_payload = payload.start or {}
            self.call_sid = start_payload.get("callSid") or start_payload.get("streamSid")
            self.stream_sid = start_payload.get("streamSid") or payload.streamSid
            logger.info("realtime.call_started call_sid=%s stream_sid=%s", self.call_sid, self.stream_sid)
            
            # Session started - planner state will be maintained for this session
            
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
                # Removed excessive media logging
                
                # Always process audio for transcription first
                transcript = await self.deepgram.transcribe_chunk(audio_bytes)
                
                # Check for barge-in during agent responses
                if self._is_agent_speaking:
                    if await self._detect_barge_in(audio_bytes):
                        await self._handle_barge_in(websocket)
                        # Still process the transcript that caused the barge-in
                        if transcript and transcript.transcript.strip():
                            await self._handle_transcript(transcript, websocket)
                        return
                
                # Normal transcript processing when agent is not speaking
                if transcript and transcript.transcript.strip():
                    await self._handle_transcript(transcript, websocket)
            else:
                pass  # No audio payload - normal
            return
        if payload.event == "mark" and payload.mark:
            mark_name = payload.mark.get("name")
            if mark_name == "transcript":
                text = payload.mark.get("payload", {}).get("text", "")
                await self._handle_text(text, websocket)
            elif mark_name == "gather":
                gather_text = payload.mark.get("payload", {}).get("text", "")
                await self._handle_text(gather_text, websocket)
            elif mark_name == "force_transcription":
                # Manual trigger for transcription when VAD fails
                print("🔧 Manual transcription triggered")
                if len(self.deepgram.audio_buffer) > 0:
                    final_transcript = await self.deepgram.finalize()
                    if final_transcript:
                        await self._handle_transcript(final_transcript, websocket)
            return
        if payload.event == "stop":
            logger.info("realtime.call_stopped call_sid=%s", self.call_sid)
            # Close streaming connection and get final transcript
            final_transcript = await self.deepgram.finalize()
            if final_transcript:
                await self._handle_transcript(final_transcript, websocket)
            # Cleanup streaming connection
            await self.deepgram.close()
            return

    async def _detect_barge_in(self, audio_bytes: bytes) -> bool:
        """Detect if human is speaking during agent response (barge-in)"""
        # More sensitive barge-in detection
        unique_values = len(set(audio_bytes))
        non_silence_count = sum(1 for b in audio_bytes if b not in [0x00, 0x7F, 0xFF])
        
        # Lower threshold for barge-in detection (more sensitive)
        variation_threshold = len(audio_bytes) * 0.02  # Just 2% variation needed
        energy_threshold = 2  # Just 2 unique values needed
        
        has_variation = non_silence_count > variation_threshold
        has_energy = unique_values > energy_threshold
        
        if has_variation or has_energy:  # Either condition triggers barge-in
            print("🚨 BARGE-IN detected!")
            return True
        return False
    
    async def _handle_barge_in(self, websocket: WebSocket) -> None:
        """Handle barge-in interruption"""
        # Cancel current response task if running
        if self._current_response_task and not self._current_response_task.done():
            self._current_response_task.cancel()
        
        # Mark that we're no longer speaking
        self._is_agent_speaking = False
        self._barge_in_detected = True
        
        # Send immediate acknowledgment
        await websocket.send_json({
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {
                "name": "barge_in_ack",
                "payload": {"text": "Sorry, go ahead..."}
            }
        })
        
        # Reset Deepgram buffer to start fresh listening  
        if hasattr(self.deepgram, 'audio_buffer'):
            self.deepgram.audio_buffer.clear()

    async def _handle_text(self, text: str, websocket: WebSocket, *, infer: bool = True) -> None:
        if not text:
            return
        if not infer:
            self.recorder.log("PLAN", "direct_text", text=text)
            audio_bytes = await self.elevenlabs.synthesize(text)
            if not audio_bytes:
                audio_bytes = _generate_mulaw_tone(duration_s=0.5, freq_hz=440)
            await self._send_twilio_agent_events(text, audio_bytes, None, websocket)
            return
        transcript = await self.deepgram.push_text(text)
        await self._handle_transcript(transcript, websocket)

    async def _handle_transcript(self, transcript: RealtimeTranscript, websocket: WebSocket) -> None:
        if not transcript.transcript:
            # Skip empty transcripts silently
            return
            
        # Prevent concurrent transcript processing
        if self._processing_transcript:
            print(f"⏳ Skipping transcript (already processing): '{transcript.transcript}'")
            return
            
        self._processing_transcript = True
        try:
            # Reset barge-in state when processing new transcript
            self._barge_in_detected = False
            self._is_agent_speaking = False
                
            print(f"🧠 THINKING: '{transcript.transcript}'")
            
            # Send engagement prompt for longer processing
            if len(transcript.transcript) > 20:  # Longer queries might take time
                await websocket.send_json({
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {
                        "name": "processing_prompt",
                        "payload": {"text": "Let me check that for you right now..."}
                    }
                })
            
            with self.recorder.stage("PLAN", transcript=transcript.transcript):
                # CRITICAL: Update planner state BEFORE generating response
                # This ensures OpenAI has the latest state when making decisions
                print(f"🔄 UPDATING STATE with: '{transcript.transcript}'")
                planner_response = await self.planner.process_transcript(transcript.transcript)
                print(f"🏢 STATE AFTER UPDATE - Property: {self.planner.state.property_name}, Bedrooms: {self.planner.state.desired_bedrooms}, Phone: {self.planner.state.prospect_phone}, Email: {self.planner.state.prospect_email}, Name: {self.planner.state.prospect_name}")
                
                # Now get Groq response with updated state
                agent_response = await self.groq.infer(transcript)
                
                # If OpenAI didn't generate a good response, use planner's response
                if not agent_response.text or not agent_response.text.strip():
                    agent_response.text = planner_response.get("text", "")
                
            # Ensure we have a meaningful response based on current state
            if not agent_response.text or not agent_response.text.strip():
                # Use planner state to generate context-aware fallback
                missing = self.planner._missing_booking_info()
                if missing:
                    agent_response.text = self.planner._get_next_question(missing[0])
                else:
                    agent_response.text = "Perfect! Ready to book your tour?"
            
            # Ensure every response contains a question until tour is booked
            missing = self.planner._missing_booking_info()
            if missing and agent_response.text and not agent_response.text.endswith("?"):
                next_question = self.planner._get_next_question(missing[0])
                if next_question not in agent_response.text:
                    agent_response.text = agent_response.text.rstrip(".") + f". {next_question}"
            
            # Enforce concise runtime style: trim to ~2 sentences or 180 chars
            def _condense(text: str) -> str:
                if not text:
                    return text
                # Split on sentence enders
                parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
                short = " ".join(parts[:2])
                if len(short) > 180:
                    short = short[:177].rstrip() + "..."
                return short

            agent_response.text = _condense(agent_response.text)
            print(f"💬 RESPONSE: '{agent_response.text}'")

            if agent_response.tool_calls:
                print(f"🔧 TOOLS: {', '.join(call.name for call in agent_response.tool_calls)}")
            
            # Send engagement prompt during tool execution
            tool_names = [call.name for call in agent_response.tool_calls]
            if "check_availability" in tool_names:
                await websocket.send_json({
                    "event": "mark", 
                    "streamSid": self.stream_sid,
                    "mark": {
                        "name": "tool_engagement",
                        "payload": {"text": "Checking live availability for you right now..."}
                    }
                })
            elif "book_tour" in tool_names:
                await websocket.send_json({
                    "event": "mark",
                    "streamSid": self.stream_sid, 
                    "mark": {
                        "name": "booking_engagement",
                        "payload": {"text": "Booking your tour now - stay on the line..."}
                    }
                })
            
            # Execute tool calls (run independent tools concurrently for latency reduction)
            tool_results = await self._execute_tool_calls(agent_response.tool_calls)
            if tool_results:
                summary = self._render_tool_results(tool_results)
                if agent_response.text:
                    agent_response.text = f"{agent_response.text} {summary}".strip()
                else:
                    agent_response.text = summary
                # Removed verbose tool results logging

            # Synthesize audio suitable for Twilio (8kHz mu-law) and stream back
            print(f"🗣️ TTS: Generating audio ({'interruptible' if agent_response.barge_in else 'standard'})")

            if agent_response.tts_audio_b64:
                try:
                    audio_bytes = base64.b64decode(agent_response.tts_audio_b64)
                    audio_stream = None
                except Exception:  # noqa: BLE001
                    audio_bytes = b""
                    audio_stream = None
            else:
                audio_bytes = b""
                audio_stream = self.elevenlabs.stream_tts(agent_response.text or "")

            if not audio_bytes and audio_stream is None:
                audio_bytes = _generate_mulaw_tone(duration_s=0.5, freq_hz=440)

            self._current_response_task = asyncio.create_task(
                self._send_twilio_agent_events(agent_response.text, audio_bytes, audio_stream, websocket)
            )
            
            try:
                await self._current_response_task
                print(f"✅ SENT")
            except asyncio.CancelledError:
                print(f"🚨 INTERRUPTED")
            finally:
                self._current_response_task = None
        finally:
            # Always release the transcript processing lock
            self._processing_transcript = False

    async def _execute_tool_calls(self, tool_calls: List[ToolCallRequest]) -> List[ToolCallResult]:
        results: List[ToolCallResult] = []
        if not tool_calls:
            return results

        async def execute_call(call: ToolCallRequest) -> ToolCallResult:
            stage_map = {
                "check_availability": "PLAN",
                "route_to_sister_property": "PLAN",
                "compute_net_effective_rent": "NER",
                "check_policy": "POLICY",
                "book_tour": "BOOK_TOUR",
                "send_sms": "SMS",
            }
            try:
                stage_name = stage_map.get(call.name, "PLAN")
                with self.recorder.stage(stage_name, tool=call.name):
                    if call.name == "book_tour":
                        missing = self.planner._missing_booking_info()
                        if missing:
                            output = {"missing_booking_info": missing}
                        else:
                            output = self.dispatcher.dispatch(call.name, call.arguments)
                    else:
                        output = self.dispatcher.dispatch(call.name, call.arguments)
            except Exception as exc:  # noqa: BLE001
                logger.exception("tool.error name=%s err=%s", call.name, exc)
                output = {"error": str(exc)}
            if not isinstance(output, dict):
                output = {"result": output}
            return ToolCallResult(call_id=call.call_id, name=call.name, output=output)

        # Run tool calls concurrently (they use IO-bound operations)
        tasks = [execute_call(call) for call in tool_calls]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _send_twilio_agent_events(self, text: str, audio_bytes: bytes, audio_stream: Optional[AsyncIterator[bytes]], websocket: WebSocket) -> None:
        if self.stream_sid is None:
            print("⚠️ No stream ID available")
            return
            
        # Mark that agent is starting to speak
        self._is_agent_speaking = True
        self._barge_in_detected = False
        
        mark_payload = {
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {
                "name": "agent_response",
                "payload": {"text": text},
            },
        }
        await websocket.send_json(mark_payload)
        
        if audio_stream is not None:
            await self._send_streaming_audio(audio_stream, websocket)
        else:
            if audio_bytes:
                await self._send_interruptible_audio(audio_bytes, websocket)
            else:
                print("⚠️ No audio generated")
        
        if not self._barge_in_detected:
            await websocket.send_json(
                {
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {"name": "agent_response_complete"},
                }
            )
        
        self._is_agent_speaking = False

    async def _send_streaming_audio(self, audio_stream: AsyncIterator[bytes], websocket: WebSocket) -> None:
        frame_size_bytes = 160
        frame_interval = 0.01  # 10 ms
        async for chunk in audio_stream:
            if self._barge_in_detected:
                print("🛑 Streaming interrupted by barge-in")
                break
            start = 0
            while start < len(chunk):
                if self._barge_in_detected:
                    print("🛑 Streaming interrupted mid-chunk")
                    break
                frame = chunk[start : start + frame_size_bytes]
                start += frame_size_bytes
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
                await asyncio.sleep(frame_interval)
        if self._barge_in_detected:
            print("🛑 Streaming audio interrupted")

    async def _send_interruptible_audio(self, audio_bytes: bytes, websocket: WebSocket) -> None:
        """Send audio frames with barge-in detection"""
        frame_size_bytes = 160
        frames_sent = 0
        total_frames = len(audio_bytes) // frame_size_bytes
        
        # Send audio frames with barge-in detection (minimal logging)
        for i in range(0, len(audio_bytes), frame_size_bytes):
            # Check for barge-in before each frame
            if self._barge_in_detected:
                print(f"🛑 INTERRUPTED at {frames_sent}/{total_frames}")
                break
                
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
            frames_sent += 1
            
            # Shorter sleep for more responsive barge-in detection
            await asyncio.sleep(0.005)  # 5ms instead of 10ms for faster interruption
        
        # Minimal completion logging
        if self._barge_in_detected:
            print(f"🛑 Audio interrupted after {frames_sent} frames")
            # Notify stop to Twilio so TTS playback ceases immediately
            await websocket.send_json(
                {
                    "event": "stop",
                    "streamSid": self.stream_sid,
                }
            )

    def _render_tool_results(self, tool_results: List[ToolCallResult]) -> str:
        summaries: List[str] = []
        for result in tool_results:
            if isinstance(result.output, dict) and result.output.get("missing_booking_info"):
                missing = result.output.get("missing_booking_info")
                # Ask for missing info in a natural way
                prompt = self.planner._prompt_for_missing(missing)
                summaries.append(prompt)
                continue
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
