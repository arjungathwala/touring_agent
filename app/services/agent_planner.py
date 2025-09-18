from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import logging
from dateutil import parser as dateparser
from dateutil import tz

from app.logging.flight_recorder import FlightRecorder
from app.models.tooling import AvailabilityQuery, SisterPropertyRouteRequest, TourRequest
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


class AgentPlanner:
    def __init__(self, dispatcher: ToolDispatcher, recorder: FlightRecorder) -> None:
        self.dispatcher = dispatcher
        self.recorder = recorder
        self.state = PlannerState()

    async def process_transcript(self, transcript: str) -> Dict[str, str]:
        logger.info("planner.transcript", transcript=transcript)
        text = transcript.lower()
        response: Dict[str, str] = {"text": ""}

        if not self.state.property_id:
            self._identify_property(text)
            if not self.state.property_id:
                response["text"] = "Hi! Which building are you interested in touring from our portfolio?"
                return response

        if not self.state.availability_shared:
            response["text"] = self._respond_with_availability()
            return response

        if self.state.availability_shared and "saturday" in text and "11" in text and "book" in text:
            confirmation = self._book_tour(datetime.now().replace(hour=11, minute=0, second=0, microsecond=0) + _upcoming_weekday_delta(5))
            response["text"] = (
                "Done! I booked you for Saturday at 11:00. "
                f"I'll text you the details and send a calendar invite. Confirmation ID {confirmation['booking_id']}."
            )
            return response

        if self.state.availability_shared and "book" in text and ("saturday" in text or "tomorrow" in text):
            booking_time = _parse_requested_time(text)
            confirmation = self._book_tour(booking_time)
            response["text"] = (
                f"You're locked in for {confirmation['tour_time']} at {self.state.property_name}. "
                "Watch for a confirmation text and calendar invite."
            )
            return response

        if self.state.sister_route_offered and "book" in text:
            booking_time = _parse_requested_time(text)
            confirmation = self._book_tour(booking_time)
            response["text"] = (
                f"Booked at {self.state.property_name} for {confirmation['tour_time']}. "
                "Sending confirmation now."
            )
            return response

        if "no availability" in text or "no units" in text:
            response["text"] = self._route_to_sister()
            return response

        response["text"] = "Let me know if you want to see another option or pick a time to tour."
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
        availability = self.dispatcher.dispatch("check_availability", args.model_dump())
        if availability:
            self.state.availability_shared = True
            snippets = []
            for unit in availability[:2]:
                rent = unit.get("rent")
                ner = unit.get("net_effective_rent")
                snippets.append(
                    f"{unit['unit_id']} at ${rent:,.0f} gross (${ner:,.0f} net effective) available {unit['available_on']}"
                )
            response = (
                f"I have {len(availability)} homes at {self.state.property_name}. "
                + " | ".join(snippets)
                + ". Ready to lock in a tour?"
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
        options: List[Dict[str, str]] = self.dispatcher.dispatch("route_to_sister_property", request.model_dump())
        if not options:
            return "I don't have alternatives right now, but I can take your info and follow up."
        self.state.sister_route_offered = True
        top = options[0]
        self.state.property_id = top["property_id"]
        self.state.property_name = top["name"]
        availability = top["available_units"]
        self.state.availability_shared = True
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
        confirmation = self.dispatcher.dispatch("book_tour", request.model_dump())
        sms_body = (
            f"You're confirmed for {self.state.property_name} on {tour_time.strftime('%A %b %d at %I:%M %p')}. "
            f"Confirmation ID {confirmation['booking_id']}."
        )
        self.dispatcher.dispatch(
            "send_sms",
            {"to_phone": request.prospect_phone, "body": sms_body},
        )
        self.state.booked_confirmation = confirmation
        return confirmation


def _upcoming_weekday_delta(target_weekday: int) -> timedelta:
    today_weekday = datetime.now().weekday()
    days_ahead = target_weekday - today_weekday
    if days_ahead <= 0:
        days_ahead += 7
    return timedelta(days=days_ahead)


def _parse_requested_time(text: str) -> datetime:
    try:
        return dateparser.parse(text, fuzzy=True)
    except (ValueError, OverflowError):
        return datetime.now() + timedelta(days=1)
