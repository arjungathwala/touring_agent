from __future__ import annotations

import hashlib
import os
from datetime import date
from typing import Dict, List, Optional

import logging
from math import asin, cos, radians, sin, sqrt
from dateutil import parser as dateparser

from app.logging.flight_recorder import FlightRecorder
from app.models.tooling import (
    AvailabilityQuery,
    NetEffectiveRentRequest,
    NetEffectiveRentResponse,
    PolicyQuery,
    PolicySummary,
    SisterPropertyOption,
    SisterPropertyRouteRequest,
    TourConfirmation,
    TourRequest,
    UnitSummary,
)
from app.services import calendar
from app.services.portfolio import Property, Unit, get_property, iter_sister_properties, list_policies, list_units
from app.services.sms import SmsClient

logger = logging.getLogger(__name__)

_BOOKED_TOURS: Dict[str, TourConfirmation] = {}


def check_availability(query: AvailabilityQuery, recorder: Optional[FlightRecorder] = None) -> List[UnitSummary]:
    units = list_units(query.property_id)
    move_in_date = query.move_in_date
    if move_in_date and isinstance(move_in_date, str):
        move_in_date = dateparser.parse(move_in_date).date()

    def is_match(unit: Unit) -> bool:
        if query.bedrooms is not None and unit.bedrooms != query.bedrooms:
            return False
        if move_in_date and unit.available_on > move_in_date:
            return False
        return True

    matches = [unit for unit in units if is_match(unit)]
    results: List[UnitSummary] = []
    for unit in matches[: query.limit]:
        ner = compute_net_effective_rent(
            NetEffectiveRentRequest(
                property_id=unit.property_id,
                unit_id=unit.unit_id,
                term_months=unit.concessions.get("term_months", 12) or 12,
            ),
            recorder,
        )
        summary = UnitSummary(
            property_id=unit.property_id,
            unit_id=unit.unit_id,
            bedrooms=unit.bedrooms,
            bathrooms=unit.bathrooms,
            rent=unit.rent,
            available_on=unit.available_on,
            net_effective_rent=ner.net_effective_rent if ner else None,
            term_months=ner.term_months if ner else None,
            concessions_weeks=unit.concessions.get("weeks_free"),
            sq_ft=unit.sq_ft,
        )
        results.append(summary)
    if recorder:
        recorder.log("PLAN", "availability_results", count=len(results), property_id=query.property_id)
    return results


def check_policy(query: PolicyQuery, recorder: Optional[FlightRecorder] = None) -> List[PolicySummary]:
    policies = list_policies(query.property_id)
    summaries = []
    for policy in policies:
        if query.policy_type and policy.policy_type != query.policy_type:
            continue
        summaries.append(
            PolicySummary(
                policy_type=policy.policy_type,
                description=policy.description,
                notes=policy.notes,
            )
        )
    if recorder:
        recorder.log("POLICY", "policy_lookup", count=len(summaries), property_id=query.property_id)
    return summaries


def compute_net_effective_rent(request: NetEffectiveRentRequest, recorder: Optional[FlightRecorder] = None) -> Optional[NetEffectiveRentResponse]:
    units = list_units(request.property_id)
    unit = next((u for u in units if u.unit_id == request.unit_id), None)
    if not unit:
        logger.warning("net_effective.not_found", unit_id=request.unit_id, property_id=request.property_id)
        return None
    term_months = request.term_months
    weeks_free = unit.concessions.get("weeks_free", 0)
    gross_rent_total = unit.rent * term_months
    weekly_rent = unit.rent * 12 / 52
    total_concession = weekly_rent * weeks_free
    net_effective = (gross_rent_total - total_concession) / term_months
    response = NetEffectiveRentResponse(
        property_id=request.property_id,
        unit_id=request.unit_id,
        term_months=term_months,
        gross_rent=unit.rent,
        total_concession=round(total_concession, 2),
        net_effective_rent=round(net_effective, 2),
    )
    if recorder:
        recorder.log("NER", "net_effective_computed", unit_id=request.unit_id, ner=response.net_effective_rent)
    return response


def _booking_key(request: TourRequest) -> str:
    raw = f"{request.property_id}:{request.tour_time.isoformat()}:{request.prospect_email.lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def book_tour(request: TourRequest, recorder: Optional[FlightRecorder] = None) -> TourConfirmation:
    key = _booking_key(request)
    if key in _BOOKED_TOURS:
        if recorder:
            recorder.log("BOOK_TOUR", "tour_idempotent_hit", booking_id=_BOOKED_TOURS[key].booking_id)
        return _BOOKED_TOURS[key]

    prop = get_property(request.property_id)
    if not prop:
        raise ValueError(f"Unknown property_id={request.property_id}")

    summary = f"Tour for {prop.name}"
    details = f"Prospect: {request.prospect_name}\nPhone: {request.prospect_phone}\nEmail: {request.prospect_email}\nNotes: {request.agent_notes or 'None'}"

    artifacts = calendar.write_calendar_artifacts(
        summary=summary,
        start=request.tour_time,
        duration_minutes=45,
        details=details,
        location=prop.address,
        timezone=prop.timezone,
    )

    booking_id = key[:12]
    confirmation = TourConfirmation(
        booking_id=booking_id,
        property_id=request.property_id,
        tour_time=request.tour_time,
        calendar_url=artifacts["google_link"],
        ics_path=artifacts["ics_path"],
    )
    _BOOKED_TOURS[key] = confirmation
    if recorder:
        recorder.log(
            "BOOK_TOUR",
            "tour_booked",
            booking_id=booking_id,
            property_id=request.property_id,
            tour_time=request.tour_time.isoformat(),
        )
    return confirmation


def send_sms(to_phone: str, body: str, recorder: Optional[FlightRecorder] = None) -> dict[str, str]:
    client = SmsClient()
    result = {
        "status": "skipped",
        "to": to_phone,
    }

    if not to_phone:
        logger.warning("sms.missing_phone")
        return result

    if os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"):
        # When credentials are provided, send the actual SMS
        client_result = {}
        try:
            # This call is asynchronous, but for the MVP we execute synchronously via loop.run_until_complete
            import asyncio

            client_result = asyncio.get_event_loop().run_until_complete(client.send(to_phone, body))
        except RuntimeError:
            # When there is no running loop we create a temporary one
            import asyncio

            result_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(result_loop)
            client_result = result_loop.run_until_complete(client.send(to_phone, body))
            result_loop.close()
            asyncio.set_event_loop(None)
        result.update(client_result)
    else:
        logger.info("sms.stub", to=_redact(to_phone), body_preview=body[:120])
        result.update({"status": "stubbed"})

    if recorder:
        recorder.log("SMS", "sms_sent", to=_redact(to_phone), status=result["status"])
    return result


def _redact(phone: str) -> str:
    if len(phone) <= 4:
        return "***"
    return f"***{phone[-4:]}"


def route_to_sister_property(request: SisterPropertyRouteRequest, recorder: Optional[FlightRecorder] = None) -> List[SisterPropertyOption]:
    origin = get_property(request.origin_property_id)
    if not origin:
        raise ValueError(f"Unknown property {request.origin_property_id}")

    origin_point = (origin.lat, origin.lng)
    options: List[SisterPropertyOption] = []
    for sister in iter_sister_properties(request.origin_property_id):
        sister_point = (sister.lat, sister.lng)
        distance = _haversine_miles(origin_point, sister_point)
        availability = check_availability(
            AvailabilityQuery(
                property_id=sister.id,
                bedrooms=request.bedrooms,
                move_in_date=request.move_in_date,
                limit=2,
            ),
            recorder,
        )
        if not availability:
            continue
        options.append(
            SisterPropertyOption(
                property_id=sister.id,
                name=sister.name,
                address=sister.address,
                distance_miles=round(distance, 2),
                available_units=availability,
            )
        )

    options.sort(key=lambda opt: opt.distance_miles)
    selected = options[: request.limit]
    if recorder:
        recorder.log(
            "PLAN",
            "sister_routing",
            origin=request.origin_property_id,
            count=len(selected),
        )
    return selected


def _haversine_miles(origin: tuple[float, float], destination: tuple[float, float]) -> float:
    lat1, lon1 = origin
    lat2, lon2 = destination
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    earth_radius_miles = 3958.8
    return c * earth_radius_miles
