from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
import json
from typing import Any, Dict, List, Optional

import logging
from dateutil import parser as dateparser
from dateutil import tz

from app.logging.flight_recorder import FlightRecorder
from openai import AsyncOpenAI

from app.models.slot_updates import (
    BedroomUpdate,
    BudgetUpdate,
    ContactUpdate,
    EmptyPayload,
    PlannerStateExtraction,
    MoveInUpdate,
    PropertyUpdate,
    TourTimeUpdate,
)
from app.models.tooling import AvailabilityQuery, NetEffectiveRentRequest, SisterPropertyRouteRequest, TourRequest
from app.services import portfolio
from app.services.tool_dispatcher import ToolDispatcher
from word2number import w2n
import os
logger = logging.getLogger(__name__)


@dataclass
class PlannerState:
    property_id: Optional[str] = None
    property_name: Optional[str] = None
    property_guess: Optional[str] = None
    property_guess_source: Optional[str] = None
    property_guess_confidence: Optional[float] = None
    desired_bedrooms: Optional[int] = None
    availability_shared: bool = False
    sister_route_offered: bool = False
    booked_confirmation: Optional[Dict[str, str]] = None
    prospect_name: Optional[str] = None
    prospect_email: Optional[str] = None
    prospect_phone: Optional[str] = None
    availability_cache: List[Dict[str, Any]] = field(default_factory=list)
    has_available_units: bool = False
    sister_recommendation: Optional[Dict[str, Any]] = None
    selected_unit: Optional[Dict[str, Any]] = None
    last_recommendation_reason: Optional[str] = None
    last_action: Optional[str] = None
    last_tool_summary: Optional[str] = None
    last_error: Optional[str] = None
    last_units_viewed: List[str] = field(default_factory=list)
    transcript_history: List[Dict[str, str]] = field(default_factory=list)
    # Additional conversation slots
    has_pets: Optional[bool] = None
    tour_type: Optional[str] = None  # "in_person" | "virtual"
    move_in_date: Optional[date] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    target_rent: Optional[int] = None
    requested_time: Optional[datetime] = None
    
    # Call retention and booking objective tracking
    conversation_turns: int = 0
    booking_attempts: int = 0
    last_engagement_time: Optional[datetime] = None
    has_shown_urgency: bool = False
    is_booking_focused: bool = False
    retention_prompts_used: int = 0


class AgentPlanner:
    def __init__(self, dispatcher: ToolDispatcher, recorder: FlightRecorder) -> None:
        self.dispatcher = dispatcher
        self.recorder = recorder
        self.state = PlannerState()
        self._state_client: Optional[AsyncOpenAI] = None
        self._state_model = os.getenv("GROQ_STATE_MODEL", "openai/gpt-oss-20b")
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            self._state_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        self._response_cache = {
            "get_property": "Hi! Which building are you interested in—21 West End, Hudson 360, or Riverview Lofts?",
            "get_bedrooms_default": "Great! What size apartment—studio, 1BR, or 2BR?",
            "get_bedrooms_21we": "Perfect! 21 West End is excellent. What size apartment—studio, 1BR, or 2BR?",
            "get_bedrooms_hudson-360": "Great choice! Hudson 360 is modern. What size apartment—studio, 1BR, or 2BR?",
            "get_bedrooms_riverview-lofts": "Excellent! Riverview Lofts has great views. What size apartment—studio, 1BR, or 2BR?",
            "get_move_in": "Great! When would you like to move in? A specific date really helps me check availability.",
            "get_budget": "Thanks! What's your ideal monthly budget so I can make sure I recommend the right options?",
            "get_time": "Excellent! What day and time work for your tour? I have openings today, tomorrow, and this weekend.",
            "get_contact": "Perfect! I just need your phone number or email to send the tour confirmation."
        }

    async def process_transcript(self, transcript: str) -> Dict[str, str]:
        logger.info("planner.transcript %s", transcript)
        text = transcript.lower()
        
        # Track conversation progress
        self.state.conversation_turns += 1
        self.state.last_engagement_time = datetime.now()
        
        # Memory service removed - using session-only state management
        
        logger.info(
            "BEFORE capture - Property: %s, Bedrooms: %s, Move-in: %s, Budget: %s, Time: %s",
            self.state.property_name,
            self.state.desired_bedrooms,
            self.state.move_in_date,
            self.state.target_rent,
            self.state.requested_time,
        )

        await self._update_state_from_llm(transcript)

        logger.info(
            "AFTER capture - Property: %s, Bedrooms: %s, Move-in: %s, Budget: %s, Time: %s",
            self.state.property_name,
            self.state.desired_bedrooms,
            self.state.move_in_date,
            self.state.target_rent,
            self.state.requested_time,
        )

        # Record user transcript in session memory
        self.state.transcript_history.append({"role": "user", "text": transcript.strip()})
        
        # Create session summary and determine next action
        session_summary = await self._create_session_summary(transcript)
        next_action = self._determine_next_action()
        booking_context = self._get_booking_requirements_context()
        
        logger.info("Session Summary: %s", session_summary)
        logger.info("Next Action: %s", next_action)
        
        # Execute the determined action
        response = await self._execute_action(next_action, text)
        
        # Add booking requirements context to help with tool selection
        response["booking_context"] = booking_context
        response["missing_info"] = self._missing_booking_info()
        response["next_action"] = next_action

        if response.get("text"):
            self.state.transcript_history.append({"role": "agent", "text": response["text"].strip()})
        
        # Session state maintained without persistent memory
        logger.info("SESSION STATE - Property: %s, Bedrooms: %s, Phone: %s", 
                   self.state.property_name, self.state.desired_bedrooms, self.state.prospect_phone)
        
        # Log current state and context for debugging
        logger.info("Booking Context: %s", booking_context)
        self._log_state()
        
        return response

    async def _update_state_from_llm(self, transcript: str) -> None:
        if not self._state_client:
            return

        history_text = "\n".join(
            f"{turn['role']}: {turn['text']}" for turn in self.state.transcript_history[-12:]
        )

        state_snapshot = {
            "property_id": self.state.property_id,
            "property_name": self.state.property_name,
            "desired_bedrooms": self.state.desired_bedrooms,
            "move_in_date": self.state.move_in_date.isoformat() if self.state.move_in_date else None,
            "budget_min": self.state.budget_min,
            "budget_max": self.state.budget_max,
            "target_rent": self.state.target_rent,
            "requested_time": self.state.requested_time.isoformat() if self.state.requested_time else None,
            "prospect_name": self.state.prospect_name,
            "prospect_email": self.state.prospect_email,
            "prospect_phone": self.state.prospect_phone,
        }

        context_text = (
            "You update booking state for an apartment leasing agent.\n"
            "Conversation history (newest last):\n"
            f"{history_text or '(none)'}\n\n"
            "Current captured state (JSON):\n"
            f"{json.dumps(state_snapshot)}\n\n"
            "Output ONLY a JSON object with the fields that changed in the latest caller utterance. "
            "Leave fields out entirely if they did not change."
        )
        messages = [
            {"role": "system", "content": "Extract leasing booking fields from the latest caller message."},
            {"role": "assistant", "content": context_text},
            {"role": "user", "content": transcript},
        ]

        schema = {
            "name": "planner_state_delta",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "property_name": {"type": ["string", "null"]},
                    "desired_bedrooms": {"type": ["integer", "null"]},
                    "move_in_date": {"type": ["string", "null"], "description": "ISO or natural language date"},
                    "budget_min": {"type": ["number", "null"]},
                    "budget_max": {"type": ["number", "null"]},
                    "target_rent": {"type": ["number", "null"]},
                    "prospect_name": {"type": ["string", "null"]},
                    "prospect_email": {"type": ["string", "null"]},
                    "prospect_phone": {"type": ["string", "null"]},
                    "requested_time": {"type": ["string", "null"], "description": "ISO datetime or natural language"},
                    "has_pets": {"type": ["boolean", "null"]},
                    "tour_type": {"type": ["string", "null"], "enum": ["in_person", "virtual", None]},
                    "reasoning": {"type": ["string", "null"]},
                },
                "required": [],
            },
        }

        try:
            response = await self._state_client.chat.completions.create(
                model=self._state_model,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": schema},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("state.llm_error %s", exc)
            return

        if not response.choices:
            logger.info("state.extractor_no_output")
            return

        parsed_data: Dict[str, Any] = {}
        message = response.choices[0].message
        content_items = getattr(message, "content", None)
        if isinstance(content_items, list):
            for item in content_items:
                if item.get("type") == "output_json":
                    parsed_data = item.get("json", {}) or {}
                    break
                if item.get("type") == "text":
                    try:
                        parsed_data = json.loads(item.get("text", ""))
                        break
                    except json.JSONDecodeError:
                        continue
        else:
            # Fallback for string responses
            try:
                parsed_data = json.loads(getattr(message, "content", "") or "{}")
            except json.JSONDecodeError:
                parsed_data = {}

        if not parsed_data:
            logger.info("state.extractor_no_parse")
            return

        reasoning = parsed_data.pop("reasoning", None)
        if reasoning:
            logger.info("state.extractor_reasoning %s", reasoning)

        updates = {k: v for k, v in parsed_data.items() if v not in (None, "")}
        if not updates:
            logger.info("state.extractor_no_updates")
            return

        for field, value in updates.items():
            if value in (None, ""):
                continue
            if field == "property_name":
                self._identify_property(str(value).lower())
            elif field == "desired_bedrooms":
                self.state.desired_bedrooms = int(value)
                self.state.last_action = "captured_bedrooms"
            elif field == "move_in_date":
                try:
                    self.state.move_in_date = datetime.fromisoformat(value).date() if isinstance(value, str) else value
                    self.state.last_action = "captured_move_in"
                except Exception:  # noqa: BLE001
                    logger.warning("state.extractor_invalid_move_in %s", value)
            elif field in {"budget_min", "budget_max", "target_rent"}:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    logger.warning("state.extractor_invalid_budget %s", value)
                    continue
                if field == "target_rent":
                    self.state.target_rent = numeric
                    self.state.budget_min = min(self.state.budget_min or numeric, numeric)
                    self.state.budget_max = max(self.state.budget_max or numeric, numeric)
                else:
                    setattr(self.state, field, numeric)
                    if self.state.target_rent is None:
                        self.state.target_rent = numeric
                self.state.last_action = "captured_budget"
            elif field == "prospect_name":
                self.state.prospect_name = str(value)
                self.state.last_action = "captured_name"
            elif field == "prospect_email":
                self.state.prospect_email = str(value)
                self.state.last_action = "captured_contact"
            elif field == "prospect_phone":
                digits = "".join(filter(str.isdigit, str(value)))
                if len(digits) >= 10:
                    self.state.prospect_phone = f"+1{digits[-10:]}"
                    self.state.last_action = "captured_contact"
            elif field == "requested_time":
                try:
                    parsed_time = value
                    if isinstance(parsed_time, str):
                        parsed_time = datetime.fromisoformat(parsed_time)
                    if parsed_time.tzinfo is None:
                        parsed_time = parsed_time.replace(tzinfo=tz.tzlocal())
                    self.state.requested_time = parsed_time
                    self.state.last_action = "captured_time"
                except Exception:  # noqa: BLE001
                    logger.warning("state.extractor_invalid_time %s", value)
            elif field == "has_pets":
                self.state.has_pets = bool(value)
            elif field == "tour_type":
                if value in {"in_person", "virtual"}:
                    self.state.tour_type = value
            elif field == "reasoning":
                self.state.last_tool_summary = str(value)
            else:
                logger.debug("state.extractor_unhandled_field %s=%s", field, value)

    def _identify_property(self, text: str) -> None:
        # Exact phrase matches
        exact_mapping = {
            "21 west end": ("21we", "21 West End"),
            "21we": ("21we", "21 West End"),
            "21 west": ("21we", "21 West End"),
            "twenty one west end": ("21we", "21 West End"),
            "twenty one west": ("21we", "21 West End"),
            "hudson": ("hudson-360", "Hudson 360"),
            "hudson 360": ("hudson-360", "Hudson 360"),
            "riverview": ("riverview-lofts", "Riverview Lofts"),
            "riverview lofts": ("riverview-lofts", "Riverview Lofts"),
        }
        
        # Pre-normalize known hyphen variations
        normalized_text = text.replace("riverview loft", "riverview lofts")

        for phrase, (prop_id, name) in exact_mapping.items():
            if phrase in normalized_text:
                self.state.property_id = prop_id
                self.state.property_name = name
                self.state.property_guess = None
                self.state.property_guess_source = None
                self.state.property_guess_confidence = None
                self.state.sister_recommendation = None
                self.state.last_action = "property_captured"
                logger.info("planner.property_identified %s from exact match '%s'", prop_id, phrase)
                return
        
        # Fuzzy fallback with difflib
        candidates = {
            "21 West End": ["21 west end", "twenty one west end", "21 west"],
            "Hudson 360": ["hudson", "hudson three sixty", "hudson 360"],
            "Riverview Lofts": ["riverview", "river view lofts", "river view", "riverview lofts"],
        }

        best_match = None
        best_score = 0.0
        for name, patterns in candidates.items():
            for pattern in patterns:
                score = SequenceMatcher(None, pattern, normalized_text).ratio()
                if score > best_score:
                    best_score = score
                    best_match = name

        confidence_threshold = 0.70
        if best_match and best_score >= confidence_threshold:
            property_map = {
                "21 West End": "21we",
                "Hudson 360": "hudson-360",
                "Riverview Lofts": "riverview-lofts",
            }
            prop_id = property_map[best_match]
            self.state.property_id = prop_id
            self.state.property_name = best_match
            self.state.property_guess = None
            self.state.property_guess_source = None
            self.state.property_guess_confidence = None
            self.state.sister_recommendation = None
            self.state.last_action = "property_captured"
            logger.info(
                "planner.property_identified %s via fuzzy match score=%.2f text='%s'",
                prop_id,
                best_score,
                text,
            )
            return
        
        if best_match:
            property_map = {
                "21 West End": "21we",
                "Hudson 360": "hudson-360",
                "Riverview Lofts": "riverview-lofts",
            }
            self.state.property_guess = property_map[best_match]
            self.state.property_guess_source = best_match
            self.state.property_guess_confidence = best_score
            logger.info(
                "planner.property_guess %s score=%.2f text='%s'",
                self.state.property_guess,
                best_score,
                text,
            )
        else:
            self.state.property_guess = None
            self.state.property_guess_source = None
            self.state.property_guess_confidence = None
        
        bedrooms_mapping = {
            0: ["studio", "0 bedroom", "zero bedroom"],
            1: ["1 bedroom", "one bedroom", "1br", "1 br", "1 b r", "one br", "single bedroom"],
            2: ["2 bedroom", "two bedroom", "2br", "2 br", "2 b r", "two br", "double bedroom"],
            3: ["3 bedroom", "three bedroom", "3br", "3 br", "3 b r", "three br", "triple bedroom"],
        }
        for count, keywords in bedrooms_mapping.items():
            if any(keyword in text for keyword in keywords):
                self.state.desired_bedrooms = count
                break

    def _respond_with_availability(self) -> bool:
        assert self.state.property_id
        args = AvailabilityQuery(
            property_id=self.state.property_id,
            bedrooms=self.state.desired_bedrooms,
            move_in_date=self.state.move_in_date,
            limit=5,
        )
        try:
            with self.recorder.stage("PLAN", property_id=self.state.property_id):
                availability = self.dispatcher.dispatch("check_availability", args.model_dump())
        except Exception as exc:  # noqa: BLE001
            logger.exception("availability.error %s", exc)
            self.state.last_error = f"availability_error: {exc}"
            self.state.last_action = "availability_error"
            return False

        filtered: List[Dict[str, Any]] = []
        for unit in availability:
            rent = unit.get("rent")
            if self.state.budget_min and rent and rent < self.state.budget_min:
                continue
            if self.state.budget_max and rent and rent > self.state.budget_max:
                continue
            if self.state.target_rent and rent and rent > self.state.target_rent:
                continue
            filtered.append(unit)

        if not filtered:
            self.recorder.log("PLAN", "no_availability", property_id=self.state.property_id)
            self.state.availability_shared = False
            self.state.has_available_units = False
            self.state.availability_cache = []
            self.state.last_units_viewed = []

            self.state.last_recommendation_reason = "NO_MATCH_PRIMARY"
            recommended = self._route_to_sister()
            if recommended and self.state.sister_recommendation:
                sister = self.state.sister_recommendation
                unit_label = sister.get("unit_id") or "a unit"
                sister_bed = sister.get("bedrooms")
                sister_bed_text = f"{sister_bed}BR " if sister_bed is not None else ""
                sister_rent = sister.get("rent")
                sister_rent_text = f" for ${sister_rent:,.0f}" if sister_rent else ""
                criteria_parts = [
                    f"a {self.state.desired_bedrooms or ''}BR".strip(),
                ]
                if self.state.move_in_date:
                    criteria_parts.append(f"available around {self.state.move_in_date.isoformat()}")
                if self.state.target_rent:
                    criteria_parts.append(f"near ${self.state.target_rent:,.0f}")
                criteria_text = " with ".join([p for p in criteria_parts if p])
                reason_text = (
                    f"I couldn't find {criteria_text or 'an available unit'} at {self.state.property_name or 'that property'}. "
                    f"A good alternative is {sister.get('name', 'a sister property')} {unit_label} ({sister_bed_text.strip() or ''}{sister_rent_text})"
                ).strip()
                self.state.last_tool_summary = reason_text
            else:
                self.state.last_tool_summary = (
                    f"I couldn't find any matching availability at {self.state.property_name or 'that property'}, "
                    "and there were no sister properties with similar options right now."
                )
            return recommended

        # We have matching units; surface the options immediately
        self.state.availability_shared = True
        self.state.availability_cache = filtered
        self.state.last_units_viewed = [unit.get("unit_id", "?") for unit in filtered]
        self.state.has_available_units = True
        self.state.selected_unit = filtered[0]
        self.state.last_recommendation_reason = "PRIMARY_AVAILABILITY"

        def _format_unit(unit: Dict[str, Any]) -> str:
            rent = unit.get("rent")
            ner = unit.get("net_effective_rent")
            move_in = unit.get("available_on")
            bedroom = unit.get("bedrooms")
            parts = [unit.get("unit_id", "unit")]
            if bedroom is not None:
                parts.append(f"{bedroom}BR")
            if rent is not None:
                parts.append(f"${rent:,.0f}")
            if ner is not None and ner != rent:
                parts.append(f"(${ner:,.0f} net)")
            if move_in:
                parts.append(f"avail {move_in}")
            return " ".join(parts)

        top_units = " | ".join(_format_unit(unit) for unit in filtered[:2])
        self.state.last_tool_summary = (
            f"AVAILABILITY {self.state.property_name or self.state.property_id}: {top_units}"
        )
        self.state.last_action = "shared_availability"
        return True

    def _route_to_sister(self) -> bool:
        target_property_id = self.state.property_id or self.state.property_guess
        if not target_property_id:
            self.state.last_action = "recommend_sister_unavailable"
            return False

        request = SisterPropertyRouteRequest(
            origin_property_id=target_property_id,
            bedrooms=self.state.desired_bedrooms,
        )

        try:
            with self.recorder.stage("PLAN", origin=target_property_id):
                options: List[Dict[str, Any]] = self.dispatcher.dispatch(
                    "route_to_sister_property", request.model_dump()
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("sister.error %s", exc)
            self.state.last_error = f"sister_error: {exc}"
            self.state.last_action = "recommend_sister_error"
            return False

        if not options:
            # Fallback to portfolio inventory data
            sisters = portfolio.iter_sister_properties(target_property_id)
            best_option: Optional[Dict[str, Any]] = None
            for sister in sisters:
                units = portfolio.list_units(sister.id)
                if not units:
                    continue
                sorted_units = sorted(
                    units,
                    key=lambda u: (
                        abs((u.bedrooms or 0) - (self.state.desired_bedrooms or u.bedrooms or 0)),
                        u.available_on,
                        u.rent,
                    ),
                )
                unit = sorted_units[0]
                best_option = {
                    "property_id": sister.id,
                    "name": sister.name,
                    "unit_id": unit.unit_id,
                    "bedrooms": unit.bedrooms,
                    "rent": unit.rent,
                    "available_on": unit.available_on.isoformat(),
                    "distance_miles": None,
                }
                break
            if best_option:
                options = [best_option]

        if not options:
            self.state.last_action = "recommend_sister_unavailable"
            self.state.last_tool_summary = "NO_SISTER_OPTION"
            self.state.availability_shared = False
            self.state.availability_cache = []
            self.state.last_units_viewed = []
            self.state.has_available_units = False
            self.state.sister_recommendation = None
            return False

        self.state.sister_route_offered = True
        top = options[0]
        self.state.sister_recommendation = top
        self.state.last_action = "recommend_sister"
        self.state.availability_shared = False
        self.state.availability_cache = []
        self.state.last_units_viewed = []
        self.state.has_available_units = False
        self.state.selected_unit = top
        self.state.last_recommendation_reason = "SISTER_AVAILABILITY"
        summary_parts = [top.get("name", "a sister property")]
        if top.get("unit_id"):
            summary_parts.append(f"unit {top['unit_id']}")
        bedrooms = top.get("bedrooms")
        if bedrooms is not None:
            summary_parts.append(f"{bedrooms}BR")
        rent = top.get("rent")
        if rent is not None:
            summary_parts.append(f"${rent:,.0f}")
        move_in = top.get("available_on")
        if move_in:
            summary_parts.append(f"available {move_in}")
        distance = top.get("distance_miles")
        if distance is not None:
            summary_parts.append(f"{distance}mi away")
        self.state.last_tool_summary = " ".join(summary_parts)
        return True

    def _book_tour(self, booking_time: datetime) -> Dict[str, str]:
        assert self.state.property_id and self.state.property_name
        tour_time = booking_time
        if tour_time.tzinfo is None:
            tour_time = tour_time.replace(tzinfo=tz.tzlocal())
        request = TourRequest(
            property_id=self.state.property_id,
            tour_time=tour_time,
            prospect_name=self.state.prospect_name or "Prospect",
            prospect_email=self.state.prospect_email or "prospect@example.com",
            prospect_phone=self.state.prospect_phone or "+15555555555",
        )
        with self.recorder.stage("BOOK_TOUR", property_id=self.state.property_id):
            confirmation = self.dispatcher.dispatch("book_tour", request.model_dump())
        sms_body = (
            f"You're confirmed for {self.state.property_name} on {tour_time.strftime('%A %b %d at %I:%M %p')}. "
            f"Confirmation ID {confirmation['booking_id']}."
        )
        with self.recorder.stage("SMS", phone=request.prospect_phone):
            self.dispatcher.dispatch(
                "send_sms",
                {"to_phone": request.prospect_phone, "body": sms_body},
            )
        self.state.booked_confirmation = confirmation
        return confirmation

    def _respond_with_policy(self, text: str) -> Optional[str]:
        if "policy" not in text and "pet" not in text and "income" not in text:
            return None
        if not self.state.property_id:
            return None
        policy_type = None
        if "pet" in text:
            policy_type = "pet"
            self.state.has_pets = True
        elif "income" in text or "guarantor" in text:
            policy_type = "income"
        request_args = {"property_id": self.state.property_id}
        if policy_type:
            request_args["policy_type"] = policy_type
        with self.recorder.stage("POLICY", property_id=self.state.property_id, policy_type=policy_type or "all"):
            policies = self.dispatcher.dispatch("check_policy", request_args)
        if not policies:
            return "I don't have those details handy, but I can follow up by text if you'd like."
        fragments = []
        for policy in policies[:2]:
            description = policy.get("description")
            label = policy.get("policy_type", "policy")
            fragments.append(f"{label.title()}: {description}")
        ack = "Noted on pets. " if self.state.has_pets else ""
        follow_up = (
            " What size—studio, 1BR, or 2BR?" if self.state.desired_bedrooms is None else " What day and time works for your tour?"
        )
        return ack + "Policy: " + " | ".join(fragments[:1]) + "." + follow_up

    def _respond_with_net_effective(self, text: str) -> Optional[str]:
        if "net effective" not in text and "concession" not in text:
            return None
        if not self.state.property_id:
            return None
        if not self.state.availability_cache:
            self._respond_with_availability()
        if not self.state.availability_cache:
            return None
        unit = self.state.availability_cache[0]
        term = unit.get("term_months") or 12
        request = NetEffectiveRentRequest(
            property_id=self.state.property_id,
            unit_id=unit["unit_id"],
            term_months=term,
        )
        with self.recorder.stage("NER", unit_id=unit["unit_id"], term=term):
            ner = self.dispatcher.dispatch("compute_net_effective_rent", request.model_dump())
        if not ner:
            return "I'm double checking concessions, one sec."
        return (
            f"That pencils out to ${ner['net_effective_rent']:,.0f} net effective on a {ner['term_months']} month term "
            "after concessions. Want to pick a tour time?"
        )

    def _maybe_capture_contact(self, transcript: str) -> None:
        if not self.state.prospect_email:
            email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", transcript)
            if email_match:
                self.state.prospect_email = email_match.group(0)
                logger.info("Captured email: %s", self.state.prospect_email)
        
        if not self.state.prospect_phone:
            # Enhanced phone number patterns to handle various formats
            phone_patterns = [
                r"\((\d{3})\)\s*(\d{3})[-\s]?(\d{4})",  # (805) 319-0650
                r"(\+?1[\s-]?)?(\d{3})[\s-]?(\d{3})[\s-]?(\d{4})",  # 805-319-0650 or +1-805-319-0650
                r"(\d{3})\.(\d{3})\.(\d{4})",  # 805.319.0650
                r"(\d{10})",  # 8053190650
            ]
            
            logger.info("Trying to capture phone from: '%s'", transcript)
            for i, pattern in enumerate(phone_patterns):
                phone_match = re.search(pattern, transcript)
                logger.info("Pattern %d (%s): %s", i + 1, pattern, "MATCH" if phone_match else "NO MATCH")
                if not phone_match:
                    continue
                digits = "".join(filter(str.isdigit, phone_match.group(0)))
                if len(digits) >= 10:
                    self.state.prospect_phone = f"+1{digits[-10:]}"
                    logger.info(
                        "✅ CAPTURED PHONE: %s from text '%s' using pattern %d",
                        self.state.prospect_phone,
                        transcript,
                        i + 1,
                    )
                    self.state.last_action = "captured_contact_phone"
                    break
            
            if not self.state.prospect_phone:
                logger.warning("❌ FAILED to capture phone from: '%s'", transcript)
        
        if not self.state.prospect_name:
            # Enhanced name capture patterns
            name_patterns = [
                r"my name is ([a-zA-Z\s]+)",
                r"i'm ([a-zA-Z\s]+)",
                r"this is ([a-zA-Z\s]+)",
                r"call me ([a-zA-Z\s]+)"
            ]
            
            for pattern in name_patterns:
                name_match = re.search(pattern, transcript.lower())
                if not name_match:
                    continue
                name = name_match.group(1).strip().title()
                if name and len(name.split()) <= 3 and not any(
                    word in name.lower() for word in ["interested", "looking", "calling", "here"]
                ):
                    self.state.prospect_name = name
                    logger.info("Captured name: %s", self.state.prospect_name)
                    self.state.last_action = "captured_name"
                    break

    def _detect_booking_intent(self, text: str) -> bool:
        """Detect if user wants to book a tour"""
        booking_keywords = [
            "book", "schedule", "reserve", "appointment", "tour", "visit", 
            "see the place", "come by", "check it out", "when can i", "available"
        ]
        return any(keyword in text for keyword in booking_keywords)
    
    def _detect_hangup_intent(self, text: str) -> bool:
        """Detect if user might hang up"""
        hangup_indicators = [
            "thanks", "thank you", "goodbye", "bye", "gotta go", "talk later",
            "call back", "think about it", "let me think", "not interested",
            "too expensive", "out of budget", "not right now"
        ]
        return any(indicator in text for indicator in hangup_indicators)
    
    def _apply_retention_strategy(self) -> Dict[str, str]:
        """Apply call retention when hangup intent detected"""
        self.state.retention_prompts_used += 1
        
        if self.state.retention_prompts_used == 1:
            if not self.state.availability_shared:
                return {"text": "Wait! Before you go, let me quickly check what we have available. It'll just take 5 seconds - you might be surprised by our current deals."}
            else:
                return {"text": "Hold on! I can get you locked in for a tour right now - it only takes 30 seconds. What day works best for you?"}
        
        elif self.state.retention_prompts_used == 2:
            weekend_day = "Saturday" if datetime.now().weekday() < 5 else "Sunday"
            return {"text": f"I totally understand wanting to think it over. How about I just hold a {weekend_day} morning slot for you? No commitment, and I'll text you the details in case you change your mind."}
        
        else:
            # Final retention attempt
            return {"text": "No problem at all! I'll send you a quick text with the property details and my direct line. Have a great day!"}
    
    def _get_property_prompt(self) -> str:
        """Get property identification prompt with urgency"""
        if self.state.conversation_turns > 2:
            return "I want to help you find the perfect place! Which building caught your eye - 21 West End, Hudson 360, or Riverview Lofts? I can check live availability right now."

            return "Hi! Which building are you interested in touring from our portfolio?"
    
    def _handle_booking_request(self, text: str) -> Dict[str, str]:
        """Handle direct booking requests with urgency"""
        self.state.booking_attempts += 1
        self.state.is_booking_focused = True
        
        # Check if we have all required info first
        missing = self._missing_booking_info()
        if missing:
            return {"text": self._prompt_for_missing(missing[:1])}
        
        # Try to parse specific time
        if "saturday" in text or "tomorrow" in text or any(day in text for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]):
            # Groq should call planner_set_tour_time; this fallback handles rare misses
            parsed_time = dateparser.parse(text, fuzzy=True)
            if parsed_time and parsed_time.tzinfo is None:
                parsed_time = parsed_time.replace(tzinfo=tz.tzlocal())
            if parsed_time:
                confirmation = self._book_tour(parsed_time)
            return {
                "text": f"Locked for {confirmation['tour_time']}. I just sent your confirmation."
            }
        
        # Ask for specific time if not provided
        if self.state.requested_time is None:
            return {"text": "Absolutely! What day and time work for you?"}
        
        # If we have time but missing other info
        return {"text": self._prompt_for_missing(missing[:1]) if missing else "Let me book that for you now."}
    
    def _get_booking_focused_response(self) -> str:
        """Generate booking-focused responses based on conversation state"""
        missing = self._missing_booking_info()
        if missing:
            return self._prompt_for_missing(missing[:1])  # Ask for just the first missing item
        
        # If we have property but no availability shown yet
        if self.state.property_id and not self.state.availability_shared:
            return self._respond_with_availability()
            
        # If we have availability but missing specific booking details
        if self.state.availability_shared:
            if self.state.desired_bedrooms is None:
                return "What size—studio, 1BR, or 2BR?"
            elif self.state.requested_time is None:
                return "What day and time works for you?"
            elif not (self.state.prospect_phone or self.state.prospect_email):
                return "What's your phone or email for the confirmation?"
        
        # Default booking prompts
        if self.state.conversation_turns > 4 and not self.state.is_booking_focused:
            self.state.is_booking_focused = True
            weekend_day = "Saturday" if datetime.now().weekday() < 5 else "Sunday"
            return f"Can I book you this {weekend_day}? 11am, 2pm, or 4pm?"
        elif self.state.booking_attempts > 0:
            return "What day and time should I lock in?"
        elif self.state.availability_shared:
            return "Should I grab you a weekend tour slot?"
        else:
            return "Which works better—weekday or weekend?"

    # --- New helpers: slot capture, state logging, gating ---
    def ingest_transcript(self, transcript: str) -> None:
        """Update state from freeform transcript without producing a response or calling tools."""
        self.state.conversation_turns += 1
        self.state.last_engagement_time = datetime.now()
        self._log_state()
        self.state.last_action = "ingest"

    def _fallback_slot_capture(self, transcript: str) -> None:
        """Fallback slot capture when the LLM extractor skips updates."""
        return


    def _missing_booking_info(self) -> List[str]:
        missing: List[str] = []
        if not self.state.property_id:
            missing.append("property")
        if self.state.desired_bedrooms is None:
            missing.append("bedrooms")
        if self.state.move_in_date is None:
            missing.append("move_in")
        if self.state.target_rent is None:
            missing.append("budget")
        if self.state.requested_time is None:
            missing.append("time")
        if not (self.state.prospect_phone or self.state.prospect_email):
            missing.append("contact")
        return missing
    
    def _get_booking_requirements_context(self) -> str:
        """Generate context about what info is needed to complete booking."""
        required_info = {
            "property": self.state.property_name or "NOT SET",
            "bedrooms": f"{self.state.desired_bedrooms}BR" if self.state.desired_bedrooms is not None else "NOT SET",
            "availability": "SHOWN" if self.state.availability_shared else "NOT SHOWN",
            "move_in": self.state.move_in_date.isoformat() if self.state.move_in_date else "NOT SET",
            "budget": f"${self.state.target_rent:,}" if self.state.target_rent else "NOT SET",
            "time": self.state.requested_time.strftime("%A %I:%M %p") if self.state.requested_time else "NOT SET",
            "contact": (self.state.prospect_phone or self.state.prospect_email or "NOT SET")
        }
        
        missing = self._missing_booking_info()

        if self.state.property_guess and not self.state.property_id:
            guess_conf = f"~{int((self.state.property_guess_confidence or 0.0) * 100)}%"
            guess_text = f"{self.state.property_guess_source} ({guess_conf})"
        else:
            guess_text = "NONE"
        
        context_parts = [
            "BOOKING REQUIREMENTS STATUS:",
            f"Property: {required_info['property']}",
            f"Property Guess: {guess_text}",
            f"Bedrooms: {required_info['bedrooms']}",
            f"Availability: {required_info['availability']}",
            f"Move-In: {required_info['move_in']}",
            f"Budget: {required_info['budget']}",
            f"Tour Time: {required_info['time']}",
            f"Contact Info: {required_info['contact']}",
            f"MISSING: {', '.join(missing).upper() if missing else 'NONE - READY TO BOOK'}",
            f"NEXT QUESTION MUST ASK FOR: {missing[0].upper() if missing else 'CONFIRMATION TO BOOK'}"
        ]
        
        return " | ".join(context_parts)
    
    def _get_next_question(self, missing_info: str) -> str:
        """Get the appropriate question for the missing information."""
        questions = {
            "property": "Which building interests you—21 West End, Hudson 360, or Riverview Lofts?",
            "bedrooms": "What size apartment—studio, 1BR, or 2BR?",
            "move_in": "When would you like to move in? A specific date helps me check availability.",
            "budget": "What's your ideal monthly budget so I can target the best fits?",
            "time": "What day and time should I book your tour?",
            "contact": "What's the best phone or email for your confirmation?",
        }
        if missing_info in questions:
            return questions[missing_info]
        fallback = self._prompt_for_missing([missing_info])
        return fallback if fallback else "Could you share a bit more so I can keep moving?"

    def _prompt_for_missing(self, missing: List[str]) -> str:
        prompts: List[str] = []
        if "property" in missing:
            prompts.append("Which building are you interested in—21 West End, Hudson 360, or Riverview Lofts?")
        if "bedrooms" in missing:
            prompts.append("What size are you looking for—studio, 1BR, or 2BR?")
        if "move_in" in missing:
            prompts.append("When would you like to move in? Knowing the date lets me check availability.")
        if "budget" in missing:
            prompts.append("What's your ideal monthly budget so I can target the best fits?")
        if "time" in missing:
            prompts.append("What day and time should I book your tour? For example, 'tomorrow at 11am'.")
        if "contact" in missing:
            prompts.append("What's the best phone or email for your confirmation?")
        # Add empathetic acknowledgement if we know pets
        if self.state.has_pets and prompts:
            prompts[0] = "Noted on pets. " + prompts[0]
        return " ".join(prompts)

    async def _create_session_summary(self, current_transcript: str) -> str:
        """Create a condensed summary of the current session state and progress."""
        summary_parts = []
        
        # What we know about the user
        if self.state.property_id:
            summary_parts.append(f"Property: {self.state.property_name}")
        if self.state.desired_bedrooms is not None:
            bedroom_text = "studio" if self.state.desired_bedrooms == 0 else f"{self.state.desired_bedrooms}BR"
            summary_parts.append(f"Size: {bedroom_text}")
        if self.state.has_pets:
            summary_parts.append("Has pets")
        if self.state.prospect_name:
            summary_parts.append(f"Name: {self.state.prospect_name}")
        if self.state.prospect_phone or self.state.prospect_email:
            summary_parts.append("Has contact info")
        if self.state.requested_time:
            summary_parts.append(f"Time: {self.state.requested_time.strftime('%A %I:%M %p')}")
        if self.state.availability_shared:
            summary_parts.append("Saw availability")
        
        # Current transcript context
        summary_parts.append(f"User said: '{current_transcript}'")
        
        return " | ".join(summary_parts) if summary_parts else "New conversation"
    
    def _determine_next_action(self) -> str:
        """Determine the next action needed to progress toward booking."""
        # Check what's missing for booking
        missing = self._missing_booking_info()
        
        # Debug current state
        logger.info("Determining next action - Property: %s, Bedrooms: %s, Missing: %s", 
                   self.state.property_name, self.state.desired_bedrooms, missing)
        
        # Priority order for gathering information
        if not self.state.property_id:
            logger.info("Next action: get_property (no property set)")
            return "get_property"
        elif self.state.desired_bedrooms is None:
            logger.info("Next action: get_bedrooms (property: %s)", self.state.property_name)
            return "get_bedrooms"
        elif not self.state.availability_shared:
            logger.info("Next action: show_availability")
            return "show_availability"
        elif "time" in missing:
            logger.info("Next action: get_time")
            return "get_time"
        elif "contact" in missing:
            logger.info("Next action: get_contact")
            return "get_contact"
        elif not missing:
            logger.info("Next action: book_tour (all info collected)")
            return "book_tour"
        else:
            logger.info("Next action: clarify_requirements")
            return "clarify_requirements"
    
    async def _execute_action(self, action: str, text: str) -> Dict[str, str]:
        """Execute the determined action and return appropriate response."""
        response: Dict[str, str] = {"text": ""}
        
        if action == "get_property":
            # Double-check if property was just mentioned but not captured
            self._identify_property(text)
            if self.state.property_id:
                # Property was just identified, move to next step
                logger.info("Property captured during get_property action: %s", self.state.property_name)
                return await self._execute_action("get_bedrooms", text)
            else:
                response["text"] = "Hi! Which building are you interested in—21 West End, Hudson 360, or Riverview Lofts?"
        
        elif action == "get_bedrooms":
            # Check if bedroom info was just provided
            if self.state.desired_bedrooms is not None:
                # Bedrooms already captured, move to next step
                return await self._execute_action("show_availability", text)
            else:
                cache_key = f"get_bedrooms_{self.state.property_id}" if self.state.property_id else "get_bedrooms_default"
                response["text"] = self._response_cache.get(cache_key, self._response_cache["get_bedrooms_default"])
        
        if not self.state.availability_shared and self.state.desired_bedrooms is not None:
            if self.state.move_in_date is None:
                response["text"] = self._response_cache["get_move_in"]
                return response
            if self.state.target_rent is None:
                response["text"] = self._response_cache["get_budget"]
                return response
            if self._respond_with_availability():
                summary = self.state.last_tool_summary or "Let me know if one of these works."
                response["text"] = summary
                return response
            reason_bits = [
                "I'm not seeing anything at",
                self.state.property_name or "that property",
                "meeting",
                f"{self.state.desired_bedrooms or ''}BR",
            ]
            if self.state.move_in_date:
                reason_bits.extend(["for a", self.state.move_in_date.isoformat(), "move-in"])
            if self.state.target_rent:
                reason_bits.extend(["around", f"${self.state.target_rent:,.0f}"])
            sister = self.state.last_tool_summary or "A nearby sister property may fit better."
            response["text"] = " ".join(reason_bits).strip() + f". {sister}"
            self.state.last_recommendation_reason = "NO_MATCH_PRIMARY"
            return response
        
        elif action == "show_availability":
            if self.state.move_in_date is None:
                response["text"] = self._response_cache["get_move_in"]
                return response
            if self.state.target_rent is None:
                response["text"] = self._response_cache["get_budget"]
                return response
            if self._respond_with_availability():
                response["text"] = self.state.last_tool_summary or "Let me know if one of these works."
                return response
            reason_bits = [
                "I'm not seeing anything at",
                self.state.property_name or "that property",
                "meeting",
                f"{self.state.desired_bedrooms or ''}BR",
            ]
            if self.state.move_in_date:
                reason_bits.extend(["for a", self.state.move_in_date.isoformat(), "move-in"])
            if self.state.target_rent:
                reason_bits.extend(["around", f"${self.state.target_rent:,.0f}"])
            sister = self.state.last_tool_summary or "A nearby sister property may fit better."
            response["text"] = " ".join(reason_bits).strip() + f". {sister}"
            self.state.last_recommendation_reason = "NO_MATCH_PRIMARY"
        
        elif action == "get_time":
            if not (self.state.availability_shared or self.state.sister_recommendation):
                response["text"] = "Let me walk you through some options first, then we can lock a time."
            else:
                response["text"] = self._response_cache["get_time"]
        
        elif action == "get_contact":
            if not (self.state.availability_shared or self.state.sister_recommendation):
                response["text"] = "Once you pick the apartment you like, I'll grab your phone or email for confirmation."
            elif self.state.prospect_phone or self.state.prospect_email:
                return await self._execute_action("book_tour", text)
            elif any(phrase in text for phrase in ["already gave", "already provided", "i gave you", "you have my"]):
                logger.warning("User claims contact already provided, but we don't have it. Missing: phone=%s, email=%s", 
                               self.state.prospect_phone, self.state.prospect_email)
                response["text"] = "I couldn't find it—could you share your phone or email one more time?"
            else:
                response["text"] = self._response_cache["get_contact"]
        
        elif action == "book_tour":
            if not self.state.availability_shared or not self.state.has_available_units:
                reason = self.state.last_tool_summary or "I couldn't find a direct match there."
                return {
                    "text": f"{reason} Would you like to go with one of the sister-property options I mentioned?"
                }

            if not self.state.requested_time:
                fallback_time = dateparser.parse(text, fuzzy=True)
                if fallback_time and fallback_time.tzinfo is None:
                    fallback_time = fallback_time.replace(tzinfo=tz.tzlocal())
                self.state.requested_time = fallback_time
            
            if self.state.requested_time and (self.state.prospect_phone or self.state.prospect_email):
                unit_to_book = self.state.selected_unit or (self.state.availability_cache[0] if self.state.availability_cache else None)
                try:
                    confirmation = self._book_tour(self.state.requested_time)
                    if unit_to_book:
                        confirmation["unit_id"] = unit_to_book.get("unit_id")
                        self.state.selected_unit = unit_to_book
                    contact = self.state.prospect_phone or self.state.prospect_email
                    unit_desc = unit_to_book.get("unit_id") if unit_to_book else self.state.property_name
                    response["text"] = f"Perfect! You're booked for {confirmation['tour_time']} at {unit_desc}. Confirmation sent to {contact}!"
                except Exception as e:
                    logger.error("Booking failed: %s", e)
                    response["text"] = "I'm having trouble with the booking system. Let me try again - what's your preferred day and time?"
            elif not self.state.requested_time:
                response["text"] = "Almost ready to book! What day and time work best?"
            else:
                response["text"] = "Perfect! I just need your phone number or email to send the confirmation."
        
        elif action == "clarify_requirements":
            missing = self._missing_booking_info()
            response["text"] = self._prompt_for_missing(missing[:1])
        
        # Handle policy and pricing questions while maintaining goal focus
        if "policy" in text or "pet" in text:
            policy_response = self._respond_with_policy(text)
            if policy_response:
                next_missing = self._missing_booking_info()
                if next_missing and action != "book_tour":
                    response["text"] = policy_response + f" Now, {self._prompt_for_missing(next_missing[:1]).lower()}"
                else:
                    response["text"] = policy_response + " Ready to book your tour?"
        
        elif "rent" in text or "price" in text or "cost" in text:
            ner_response = self._respond_with_net_effective(text)
            if ner_response:
                next_missing = self._missing_booking_info()
                if next_missing and action != "book_tour":
                    response["text"] = ner_response + f" {self._prompt_for_missing(next_missing[:1])}"
                else:
                    response["text"] = ner_response + " Should we book your tour?"
        
        return response

    def _log_state(self) -> None:
        self.recorder.log(
            "PLAN",
            "state",
            property_id=self.state.property_id,
            property_name=self.state.property_name,
            bedrooms=self.state.desired_bedrooms,
            has_pets=self.state.has_pets,
            tour_type=self.state.tour_type,
            move_in=str(self.state.move_in_date) if self.state.move_in_date else None,
            budget_min=self.state.budget_min,
            budget_max=self.state.budget_max,
            requested_time=self.state.requested_time.isoformat() if self.state.requested_time else None,
            name=self.state.prospect_name,
            email=self.state.prospect_email,
            phone=self.state.prospect_phone,
        )

    # --- LLM Tooling Interface ---

    def get_llm_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "planner_set_property",
                "description": "Record the property the caller wants to tour.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "property_name": {"type": "string", "description": "Name of the property the caller mentioned."}
                    },
                    "required": ["property_name"],
                },
            },
            {
                "type": "function",
                "name": "planner_set_bedrooms",
                "description": "Store the desired bedroom count.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bedrooms": {"type": "integer", "minimum": 0, "maximum": 5},
                    },
                    "required": ["bedrooms"],
                },
            },
            {
                "type": "function",
                "name": "planner_set_move_in_date",
                "description": "Record the caller's target move-in date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "move_in_date": {"type": "string", "description": "ISO date or natural language date."}
                    },
                    "required": ["move_in_date"],
                },
            },
            {
                "type": "function",
                "name": "planner_set_budget",
                "description": "Store the caller's monthly budget range.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min_amount": {"type": "number"},
                        "max_amount": {"type": "number"},
                    },
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "planner_set_contact",
                "description": "Store the caller's contact information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "planner_set_tour_time",
                "description": "Store the requested tour datetime.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tour_time": {"type": "string", "description": "Preferred tour datetime in natural language or ISO."}
                    },
                    "required": ["tour_time"],
                },
            },
            {
                "type": "function",
                "name": "planner_check_availability",
                "description": "Check availability for the stored property, bedrooms, move-in date, and budget.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "planner_recommend_sister",
                "description": "Recommend a sister property when primary property has no matches.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "planner_end_session",
                "description": "Politely end the call when no viable tour can be booked.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def execute_llm_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        handlers = {
            "planner_set_property": self._llm_set_property,
            "planner_set_bedrooms": self._llm_set_bedrooms,
            "planner_set_move_in_date": self._llm_set_move_in_date,
            "planner_set_budget": self._llm_set_budget,
            "planner_set_contact": self._llm_set_contact,
            "planner_set_tour_time": self._llm_set_tour_time,
            "planner_check_availability": self._llm_check_availability,
            "planner_recommend_sister": self._llm_recommend_sister,
            "planner_end_session": self._llm_end_session,
        }
        handler = handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown planner tool: {name}")
        try:
            return handler(arguments or {})
        except ValidationError as exc:
            return {"error": exc.errors()}

    def _llm_set_property(self, args: Dict[str, Any]) -> Dict[str, Any]:
        payload = PropertyUpdate(**args)
        before = self.state.property_name
        self._identify_property(payload.property_name.lower())
        if self.state.property_name and self.state.property_name != before:
            return {"text": f"Noted {self.state.property_name}."}
        return {"text": "I'm not sure which building that is; could you name the property again?"}

    def _llm_set_bedrooms(self, args: Dict[str, Any]) -> Dict[str, Any]:
        payload = BedroomUpdate(**args)
        self.state.desired_bedrooms = int(payload.desired_bedrooms)
        self.state.last_action = "captured_bedrooms"
        return {"text": f"Got it—{payload.desired_bedrooms} bedroom."}

    def _llm_set_move_in_date(self, args: Dict[str, Any]) -> Dict[str, Any]:
        payload = MoveInUpdate(**args)
        self.state.move_in_date = payload.move_in_date
        self.state.last_action = "captured_move_in"
        return {"text": f"Move-in logged for {payload.move_in_date.isoformat()}."}

    def _llm_set_budget(self, args: Dict[str, Any]) -> Dict[str, Any]:
        payload = BudgetUpdate(**args)
        min_amount = payload.min_amount
        max_amount = payload.max_amount
        if min_amount is None and max_amount is None:
            return {"text": "Just checking—what budget works for you?"}
        if min_amount is not None and max_amount is None:
            max_amount = min_amount
        if max_amount is not None and min_amount is None:
            min_amount = max_amount
        if min_amount is not None and max_amount is not None:
            if min_amount > max_amount:
                min_amount, max_amount = max_amount, min_amount
            self.state.budget_min = float(min_amount)
            self.state.budget_max = float(max_amount)
            self.state.target_rent = float(max_amount)
            self.state.last_action = "captured_budget"
            if min_amount == max_amount:
                return {"text": f"Budget noted at ${max_amount:,.0f}."}
            return {"text": f"Budget range set for ${min_amount:,.0f}-${max_amount:,.0f}."}
        return {"text": "I couldn't tell the budget—could you restate it?"}

    def _llm_set_contact(self, args: Dict[str, Any]) -> Dict[str, Any]:
        payload = ContactUpdate(**args)
        phone = payload.phone
        email = payload.email
        if not phone and not email:
            return {"text": "Let me know the best phone or email when you can."}
        if phone:
            digits = "".join(filter(str.isdigit, phone))
            if len(digits) >= 10:
                self.state.prospect_phone = f"+1{digits[-10:]}"
        if email:
            self.state.prospect_email = email
        self.state.last_action = "captured_contact"
        contact = self.state.prospect_phone or self.state.prospect_email
        return {"text": f"Thanks! I'll use {contact} for confirmation." if contact else "Thanks, I'll hold onto that."}

    def _llm_set_tour_time(self, args: Dict[str, Any]) -> Dict[str, Any]:
        payload = TourTimeUpdate(**args)
        parsed = payload.tour_time
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz.tzlocal())
        self.state.requested_time = parsed
        self.state.last_action = "captured_time"
        return {"text": f"Tour time saved for {parsed.strftime('%A %b %d at %I:%M %p')}."}

    def _llm_check_availability(self, _: Dict[str, Any]) -> Dict[str, Any]:
        success = self._respond_with_availability()
        summary = self.state.last_tool_summary or "No availability yet."
        return {"text": summary if success else summary}

    def _llm_recommend_sister(self, _: Dict[str, Any]) -> Dict[str, Any]:
        success = self._route_to_sister()
        summary = self.state.last_tool_summary or "No sister options available."
        return {"text": summary}

    def _llm_end_session(self, _: Dict[str, Any]) -> Dict[str, Any]:
        self.state.last_action = "ended_session"
        return {"text": "Understood—I'll pause here. Feel free to reach out if you'd like to explore other options."}

    # Memory service methods removed - using session-only state management


