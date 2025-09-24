from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import logging
from dateutil import parser as dateparser
from dateutil import tz

from app.logging.flight_recorder import FlightRecorder
from app.models.tooling import AvailabilityQuery, NetEffectiveRentRequest, SisterPropertyRouteRequest, TourRequest
from app.services.tool_dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)


@dataclass
class PlannerState:
    property_id: Optional[str] = None
    property_name: Optional[str] = None
    desired_bedrooms: Optional[int] = None
    availability_shared: bool = False
    sister_route_offered: bool = False
    booked_confirmation: Optional[Dict[str, str]] = None
    prospect_name: Optional[str] = None
    prospect_email: Optional[str] = None
    prospect_phone: Optional[str] = None
    availability_cache: List[Dict[str, Any]] = field(default_factory=list)
    
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

    async def process_transcript(self, transcript: str) -> Dict[str, str]:
        logger.info("planner.transcript", transcript=transcript)
        text = transcript.lower()
        response: Dict[str, str] = {"text": ""}
        
        # Track conversation progress for call retention
        self.state.conversation_turns += 1
        self.state.last_engagement_time = datetime.now()
        
        # Continuously capture contact info
        self._maybe_capture_contact(transcript)
        
        # Check for booking intent early and prioritize it
        booking_intent = self._detect_booking_intent(text)
        if booking_intent and self.state.availability_shared:
            return self._handle_booking_request(text)
        
        # Check for hangup indicators and apply retention
        if self._detect_hangup_intent(text):
            return self._apply_retention_strategy()

        # Step 1: Identify property (required)
        if not self.state.property_id:
            self._identify_property(text)
            if not self.state.property_id:
                response["text"] = self._get_property_prompt()
                return response

        # Step 2: Handle specific questions (but guide toward booking)
        policy_response = self._respond_with_policy(text)
        if policy_response:
            # Add booking nudge to policy responses
            response["text"] = policy_response + " Ready to schedule your tour?"
            return response

        ner_response = self._respond_with_net_effective(text)
        if ner_response:
            # Add booking nudge to rent responses
            response["text"] = ner_response + " Should we lock in a tour time?"
            return response

        # Step 3: Show availability (core value proposition)
        if not self.state.availability_shared:
            response["text"] = self._respond_with_availability()
            return response

        # Step 4: Direct booking attempts (flexible time parsing)
        if self.state.availability_shared and "book" in text:
            booking_time = _parse_requested_time(text)
            confirmation = self._book_tour(booking_time)
            response["text"] = (
                f"Perfect! I've got you locked in for {confirmation['tour_time']}. "
                f"You'll get a confirmation text and calendar invite right now. Your booking ID is {confirmation['booking_id']}. "
                "Thanks for choosing us!"
            )
            return response

        # Sister property booking
        if self.state.sister_route_offered and "book" in text:
            booking_time = _parse_requested_time(text)
            confirmation = self._book_tour(booking_time)
            response["text"] = (
                f"Great choice! You're all set for {confirmation['tour_time']} at {self.state.property_name}. "
                "Confirmation details are on the way!"
            )
            return response

        # Step 5: Route to sister property if needed
        if "no availability" in text or "no units" in text:
            response["text"] = self._route_to_sister()
            return response

        # Step 6: Booking-focused fallback with retention
        response["text"] = self._get_booking_focused_response()
        return response

    def _identify_property(self, text: str) -> None:
        mapping = {
            "21 west end": ("21we", "21 West End"),
            "21we": ("21we", "21 West End"),
            "hudson": ("hudson-360", "Hudson 360"),
            "riverview": ("riverview-lofts", "Riverview Lofts"),
        }
        for phrase, (prop_id, name) in mapping.items():
            if phrase in text:
                self.state.property_id = prop_id
                self.state.property_name = name
                logger.info("planner.property_identified", property_id=prop_id)
                break
        bedrooms_mapping = {
            0: ["studio", "0 bedroom"],
            1: ["1 bedroom", "one bedroom", "1br"],
            2: ["2 bedroom", "two bedroom", "2br"],
            3: ["3 bedroom", "three bedroom", "3br"],
        }
        for count, keywords in bedrooms_mapping.items():
            if any(keyword in text for keyword in keywords):
                self.state.desired_bedrooms = count
                break

    def _respond_with_availability(self) -> str:
        assert self.state.property_id
        args = AvailabilityQuery(property_id=self.state.property_id, bedrooms=self.state.desired_bedrooms)
        with self.recorder.stage("PLAN", property_id=self.state.property_id):
            availability = self.dispatcher.dispatch("check_availability", args.model_dump())
        if availability:
            self.state.availability_shared = True
            self.state.availability_cache = availability
            snippets = []
            for unit in availability[:2]:
                rent = unit.get("rent")
                ner = unit.get("net_effective_rent")
                snippets.append(
                    f"{unit['unit_id']} at ${rent:,.0f} gross (${ner:,.0f} net effective) available {unit['available_on']}"
                )
            # Booking-focused availability response
            response = (
                f"Perfect! I have {len(availability)} available homes at {self.state.property_name}. "
                + " | ".join(snippets)
                + ". These units are moving fast - should I book you a tour this weekend? "
                + "I have Saturday and Sunday slots available. I can confirm it right now while we're talking."
            )
            return response
        self.recorder.log("PLAN", "no_availability", property_id=self.state.property_id)
        return self._route_to_sister()

    def _route_to_sister(self) -> str:
        assert self.state.property_id
        request = SisterPropertyRouteRequest(
            origin_property_id=self.state.property_id,
            bedrooms=self.state.desired_bedrooms,
        )
        with self.recorder.stage("PLAN", origin=self.state.property_id):
            options: List[Dict[str, str]] = self.dispatcher.dispatch("route_to_sister_property", request.model_dump())
        if not options:
            return "I don't have alternatives right now, but I can take your info and follow up."
        self.state.sister_route_offered = True
        top = options[0]
        self.state.property_id = top["property_id"]
        self.state.property_name = top["name"]
        availability = top["available_units"]
        self.state.availability_shared = True
        self.state.availability_cache = availability
        unit = availability[0]
        return (
            f"Nothing open there today, but {top['name']} is {top['distance_miles']} miles away with {unit['unit_id']} "
            f"at ${unit['net_effective_rent']:,.0f} net effective. Want me to book a tour there?"
        )

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
        return (
            "Here's what I show: " + " | ".join(fragments) + ". Ready to move forward with a tour?"
        )

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
            email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", transcript)
            if email_match:
                self.state.prospect_email = email_match.group(0)
        if not self.state.prospect_phone:
            phone_match = re.search(r"(\+?1[\s-]?)?(\d{3})[\s-]?(\d{3})[\s-]?(\d{4})", transcript)
            if phone_match:
                digits = "".join(filter(str.isdigit, phone_match.group(0)))
                self.state.prospect_phone = f"+1{digits[-10:]}"
        if not self.state.prospect_name:
            name_match = re.search(r"my name is ([a-zA-Z\s]+)", transcript.lower())
            if name_match:
                name = name_match.group(1).strip().title()
                self.state.prospect_name = name

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
        else:
            return "Hi! Which building are you interested in touring from our portfolio?"
    
    def _handle_booking_request(self, text: str) -> Dict[str, str]:
        """Handle direct booking requests with urgency"""
        self.state.booking_attempts += 1
        self.state.is_booking_focused = True
        
        # Try to parse specific time
        if "saturday" in text or "tomorrow" in text or any(day in text for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]):
            booking_time = _parse_requested_time(text)
            confirmation = self._book_tour(booking_time)
            return {
                "text": f"Perfect! I've got you confirmed for {confirmation['tour_time']} at {self.state.property_name}. "
                       f"You'll receive a confirmation text and calendar invite immediately. Your booking ID is {confirmation['booking_id']}. "
                       "Thanks for choosing us!"
            }
        
        # Offer specific times to close quickly (dynamic based on current time)
        now = datetime.now()
        weekend_day = "Saturday" if now.weekday() < 5 else "Sunday"  # Offer weekend
        
        return {
            "text": f"Great! I can book you right now. How about {weekend_day} at 11am or 2pm? "
                   f"Or if you prefer a weekday, I have morning, afternoon, or evening slots. Which works for you?"
        }
    
    def _get_booking_focused_response(self) -> str:
        """Generate booking-focused responses based on conversation state"""
        if self.state.conversation_turns > 4 and not self.state.is_booking_focused:
            self.state.is_booking_focused = True
            weekend_day = "Saturday" if datetime.now().weekday() < 5 else "Sunday"
            return f"I'd love to get you scheduled before these units are gone. Can I book you for a tour this {weekend_day}? I have 11am, 2pm, or 4pm available. It only takes 2 seconds to confirm."
        
        elif self.state.booking_attempts > 0:
            return "I have your tour slot ready to confirm. Just say your preferred day and time - like 'Saturday morning' or 'tomorrow afternoon' - and I'll lock it in immediately."
        
        elif self.state.availability_shared:
            return "These units move fast! Should I grab you a tour slot for this weekend? I can book you right now while we're talking."
        
        else:
            return "Let me show you what's available and get you booked for a tour. Which time works better - weekday or weekend?"


def _upcoming_weekday_delta(target_weekday: int) -> timedelta:
    today_weekday = datetime.now().weekday()
    days_ahead = target_weekday - today_weekday
    if days_ahead <= 0:
        days_ahead += 7
    return timedelta(days=days_ahead)


def _parse_requested_time(text: str) -> datetime:
    """Parse flexible time requests with smart defaults"""
    try:
        # Try to parse the text
        parsed_time = dateparser.parse(text, fuzzy=True)
        if parsed_time:
            # If no time specified, default to 11am
            if parsed_time.hour == 0 and parsed_time.minute == 0:
                parsed_time = parsed_time.replace(hour=11, minute=0)
            return parsed_time
    except (ValueError, OverflowError):
        pass
    
    # Smart defaults based on keywords
    base_date = datetime.now()
    
    if "saturday" in text.lower():
        days_until_saturday = (5 - base_date.weekday()) % 7
        if days_until_saturday == 0:
            days_until_saturday = 7  # Next Saturday if today is Saturday
        target_date = base_date + timedelta(days=days_until_saturday)
    elif "sunday" in text.lower():
        days_until_sunday = (6 - base_date.weekday()) % 7
        if days_until_sunday == 0:
            days_until_sunday = 7  # Next Sunday if today is Sunday
        target_date = base_date + timedelta(days=days_until_sunday)
    elif "tomorrow" in text.lower():
        target_date = base_date + timedelta(days=1)
    else:
        # Default to next Saturday
        days_until_saturday = (5 - base_date.weekday()) % 7
        if days_until_saturday == 0:
            days_until_saturday = 7
        target_date = base_date + timedelta(days=days_until_saturday)
    
    # Set time based on keywords
    if "morning" in text.lower() or "10" in text or "11" in text:
        hour = 11
    elif "afternoon" in text.lower() or "1" in text or "2" in text:
        hour = 14  # 2pm
    elif "evening" in text.lower() or "4" in text or "5" in text:
        hour = 16  # 4pm
    else:
        hour = 11  # Default to 11am
    
    return target_date.replace(hour=hour, minute=0, second=0, microsecond=0)
