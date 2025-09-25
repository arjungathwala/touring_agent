from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
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
    # Additional conversation slots
    has_pets: Optional[bool] = None
    tour_type: Optional[str] = None  # "in_person" | "virtual"
    move_in_date: Optional[date] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
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

    async def process_transcript(self, transcript: str) -> Dict[str, str]:
        logger.info("planner.transcript %s", transcript)
        text = transcript.lower()
        
        # Track conversation progress
        self.state.conversation_turns += 1
        self.state.last_engagement_time = datetime.now()
        
        # Memory service removed - using session-only state management
        
        # ALWAYS capture information from current transcript
        logger.info("BEFORE capture - Property: %s, Bedrooms: %s", self.state.property_name, self.state.desired_bedrooms)
        self._maybe_capture_contact(transcript)
        self._identify_property(text)
        self._capture_slots(transcript)
        logger.info("AFTER capture - Property: %s, Bedrooms: %s", self.state.property_name, self.state.desired_bedrooms)
        
        # Create session summary and determine next action
        session_summary = await self._create_session_summary(transcript)
        next_action = self._determine_next_action()
        
        logger.info("Session Summary: %s", session_summary)
        logger.info("Next Action: %s", next_action)
        
        # Execute the determined action
        response = await self._execute_action(next_action, text)
        
        # Add booking requirements context to help with tool selection
        booking_context = self._get_booking_requirements_context()
        
        # Ensure response contains a question if tour not yet booked
        missing = self._missing_booking_info()
        if missing and not response["text"].endswith("?"):
            # Add a question to gather missing info
            next_question = self._get_next_question(missing[0])
            if next_question and not next_question in response["text"]:
                response["text"] = response["text"].rstrip(".") + f". {next_question}"
        
        # Add context for tool selection (this will be used by the OpenAI planner)
        response["booking_context"] = booking_context
        response["missing_info"] = missing
        response["next_action"] = next_action
        
        # Session state maintained without persistent memory
        logger.info("SESSION STATE - Property: %s, Bedrooms: %s, Phone: %s", 
                   self.state.property_name, self.state.desired_bedrooms, self.state.prospect_phone)
        
        # Log current state and context for debugging
        logger.info("Booking Context: %s", booking_context)
        self._log_state()
        
        return response

    def _identify_property(self, text: str) -> None:
        # Exact matches first
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
        
        # Try exact matches first
        for phrase, (prop_id, name) in exact_mapping.items():
            if phrase in text:
                self.state.property_id = prop_id
                self.state.property_name = name
                logger.info("planner.property_identified %s from exact match '%s'", prop_id, text)
                return
        
        # Fuzzy matching for common STT errors
        fuzzy_patterns = [
            # 21 West End variations (common STT errors)
            (["21 best end", "21 vest end", "21 rest end", "21 west and", "twenty one best end", 
              "twenty one vest end", "21 blessed", "21 blessed end", "cleveland west end", 
              "21 worst end", "21 first west end", "20 first west end", "21 verse end"], ("21we", "21 West End")),
            
            # Hudson 360 variations  
            (["hudson three sixty", "hudson 3 60", "hudson three six zero", "hudson tree sixty"], 
             ("hudson-360", "Hudson 360")),
            
            # Riverview Lofts variations
            (["river view lofts", "river view", "riverview loft", "river lofts"], 
             ("riverview-lofts", "Riverview Lofts"))
        ]
        
        for patterns, (prop_id, name) in fuzzy_patterns:
            for pattern in patterns:
                if pattern in text:
                    self.state.property_id = prop_id
                    self.state.property_name = name
                    logger.info("planner.property_identified %s from fuzzy match '%s' -> '%s'", prop_id, pattern, name)
                    return
        
        # If no match found, try partial word matching for very garbled text
        if "21" in text and any(word in text for word in ["west", "best", "vest", "rest", "blessed"]):
            self.state.property_id = "21we"
            self.state.property_name = "21 West End"
            logger.info("planner.property_identified %s from partial match in '%s'", "21we", text)
        elif "hudson" in text:
            self.state.property_id = "hudson-360"
            self.state.property_name = "Hudson 360"
            logger.info("planner.property_identified %s from partial match in '%s'", "hudson-360", text)
        elif any(word in text for word in ["river", "riverview"]):
            self.state.property_id = "riverview-lofts"
            self.state.property_name = "Riverview Lofts"
            logger.info("planner.property_identified %s from partial match in '%s'", "riverview-lofts", text)
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
            # Concise availability + single next question
            response = (
                f"I have {len(availability)} at {self.state.property_name}. "
                + " | ".join(snippets[:1])
                + ". What size—studio, 1BR, or 2BR?"
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
                logger.info("Pattern %d (%s): %s", i+1, pattern, "MATCH" if phone_match else "NO MATCH")
                if phone_match:
                    digits = "".join(filter(str.isdigit, phone_match.group(0)))
                    if len(digits) >= 10:
                        self.state.prospect_phone = f"+1{digits[-10:]}"
                        logger.info("✅ CAPTURED PHONE: %s from text '%s' using pattern %d", self.state.prospect_phone, transcript, i+1)
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
                if name_match:
                    name = name_match.group(1).strip().title()
                    # Filter out common non-names
                    if name and len(name.split()) <= 3 and not any(word in name.lower() for word in ["interested", "looking", "calling", "here"]):
                        self.state.prospect_name = name
                        logger.info("Captured name: %s", self.state.prospect_name)
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
        else:
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
            booking_time = _parse_requested_time(text)
            confirmation = self._book_tour(booking_time)
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
        self._maybe_capture_contact(transcript)
        self._identify_property(transcript.lower())
        self._capture_slots(transcript)
        self._log_state()

    def _capture_slots(self, transcript: str) -> None:
        text = transcript.lower()
        # Pets
        if any(word in text for word in ["pet", "pets", "dog", "cat"]):
            self.state.has_pets = True
        # Tour type
        if "virtual" in text:
            self.state.tour_type = "virtual"
        if "in person" in text or "in-person" in text or "inperson" in text:
            self.state.tour_type = "in_person"
        # Budget (simple extraction of dollar amounts/range)
        dollars = re.findall(r"\$\s*([0-9]{3,5})", transcript)
        nums = [int(n) for n in dollars]
        if len(nums) == 1:
            self.state.budget_min = nums[0]
        elif len(nums) >= 2:
            lo, hi = sorted(nums[:2])
            self.state.budget_min, self.state.budget_max = lo, hi
        # Move-in date
        if any(kw in text for kw in ["move in", "move-in", "movein", "start", "begin", "lease begins"]):
            try:
                dt = dateparser.parse(transcript, fuzzy=True)
                if dt:
                    self.state.move_in_date = dt.date()
            except Exception:  # noqa: BLE001
                pass
        # Desired bedrooms (including common STT variations)
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
        # Requested tour time - look for day keywords
        if any(day in text for day in ["saturday", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "weekend", "weekday", "tomorrow"]):
            try:
                parsed = dateparser.parse(transcript, fuzzy=True)
                if parsed:
                    # Set default time to 11am if no time specified
                    if parsed.hour == 0 and parsed.minute == 0:
                        parsed = parsed.replace(hour=11, minute=0)
                    self.state.requested_time = parsed
            except Exception:  # noqa: BLE001
                # Fallback for simple day parsing
                if "saturday" in text:
                    base = datetime.now().replace(hour=11, minute=0, second=0, microsecond=0)
                    days_ahead = (5 - base.weekday()) % 7  # Saturday is weekday 5
                    if days_ahead == 0:
                        days_ahead = 7
                    self.state.requested_time = base + timedelta(days=days_ahead)
                elif "sunday" in text:
                    base = datetime.now().replace(hour=11, minute=0, second=0, microsecond=0)
                    days_ahead = (6 - base.weekday()) % 7  # Sunday is weekday 6
                    if days_ahead == 0:
                        days_ahead = 7
                    self.state.requested_time = base + timedelta(days=days_ahead)

    def _missing_booking_info(self) -> List[str]:
        missing: List[str] = []
        if not self.state.property_id:
            missing.append("property")
        if self.state.desired_bedrooms is None:
            missing.append("bedrooms")
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
            "time": self.state.requested_time.strftime("%A %I:%M %p") if self.state.requested_time else "NOT SET",
            "contact": (self.state.prospect_phone or self.state.prospect_email or "NOT SET")
        }
        
        missing = self._missing_booking_info()
        
        context_parts = [
            "BOOKING REQUIREMENTS STATUS:",
            f"Property: {required_info['property']}",
            f"Bedrooms: {required_info['bedrooms']}",
            f"Availability: {required_info['availability']}",
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
            "time": "What day and time work for your tour?",
            "contact": "What's your phone number or email for the confirmation?"
        }
        return questions.get(missing_info, "What else do you need to know?")

    def _prompt_for_missing(self, missing: List[str]) -> str:
        prompts: List[str] = []
        if "property" in missing:
            prompts.append("Which building are you interested in—21 West End, Hudson 360, or Riverview Lofts?")
        if "bedrooms" in missing:
            prompts.append("What size are you looking for—studio, 1BR, or 2BR?")
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
                response["text"] = f"Perfect! {self.state.property_name} is excellent. What size apartment—studio, 1BR, or 2BR?"
        
        elif action == "show_availability":
            availability_response = self._respond_with_availability()
            response["text"] = availability_response
        
        elif action == "get_time":
            # Try to parse any time information from the current text
            if any(word in text for word in ["today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "morning", "afternoon", "evening"]):
                parsed_time = _parse_requested_time(text)
                if parsed_time:
                    self.state.requested_time = parsed_time
                    return await self._execute_action("get_contact", text)
            
            if any(word in text for word in ["week", "next week", "upcoming"]):
                response["text"] = "Great! What specific day works best—Monday through Friday, or would you prefer a weekend tour?"
            else:
                response["text"] = "Excellent! What day and time work for your tour? I have openings today, tomorrow, and this weekend."
        
        elif action == "get_contact":
            # Check if contact info was just provided
            if self.state.prospect_phone or self.state.prospect_email:
                # Contact info captured, proceed to book
                return await self._execute_action("book_tour", text)
            elif any(phrase in text for phrase in ["already gave", "already provided", "i gave you", "you have my"]):
                # User claims they already provided contact info - check if we missed it
                logger.warning("User claims contact already provided, but we don't have it. Missing: phone=%s, email=%s", 
                              not self.state.prospect_phone, not self.state.prospect_email)
                response["text"] = "I apologize, I don't see your contact info. Could you please provide your phone number or email again?"
            else:
                response["text"] = "Perfect! I just need your phone number or email to send the tour confirmation."
        
        elif action == "book_tour":
            # We have everything needed - book the tour
            if not self.state.requested_time:
                # Try to parse time from current text
                self.state.requested_time = _parse_requested_time(text)
            
            if self.state.requested_time and (self.state.prospect_phone or self.state.prospect_email):
                # We have time and contact info - book it!
                try:
                    confirmation = self._book_tour(self.state.requested_time)
                    contact = self.state.prospect_phone or self.state.prospect_email
                    response["text"] = f"Perfect! You're booked for {confirmation['tour_time']} at {self.state.property_name}. Confirmation sent to {contact}!"
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

    # Memory service methods removed - using session-only state management


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
