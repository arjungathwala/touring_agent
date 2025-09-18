from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class AvailabilityQuery(BaseModel):
    property_id: str
    bedrooms: Optional[int] = None
    move_in_date: Optional[date] = None
    limit: int = 3


class UnitSummary(BaseModel):
    property_id: str
    unit_id: str
    bedrooms: int
    bathrooms: int
    rent: float
    available_on: date
    net_effective_rent: Optional[float] = None
    term_months: Optional[int] = None
    concessions_weeks: Optional[int] = None
    sq_ft: Optional[int] = None


class PolicyQuery(BaseModel):
    property_id: str
    policy_type: Optional[str] = None


class PolicySummary(BaseModel):
    policy_type: str
    description: str
    notes: Optional[str] = None


class NetEffectiveRentRequest(BaseModel):
    property_id: str
    unit_id: str
    term_months: int


class NetEffectiveRentResponse(BaseModel):
    property_id: str
    unit_id: str
    term_months: int
    gross_rent: float
    total_concession: float
    net_effective_rent: float


class TourRequest(BaseModel):
    property_id: str
    tour_time: datetime
    prospect_name: str
    prospect_email: str
    prospect_phone: str
    agent_notes: Optional[str] = None


class TourConfirmation(BaseModel):
    booking_id: str
    property_id: str
    tour_time: datetime
    calendar_url: Optional[str]
    ics_path: Optional[str]


class SmsRequest(BaseModel):
    to_phone: str
    body: str


class SisterPropertyRouteRequest(BaseModel):
    origin_property_id: str
    bedrooms: Optional[int] = None
    move_in_date: Optional[date] = None
    limit: int = 2


class SisterPropertyOption(BaseModel):
    property_id: str
    name: str
    address: str
    distance_miles: float
    available_units: List[UnitSummary]
