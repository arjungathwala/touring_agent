from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, ConfigDict


class PropertyUpdate(BaseModel):
    property_name: str = Field(..., description="The property name the caller mentioned.")


class BedroomUpdate(BaseModel):
    desired_bedrooms: int = Field(..., ge=0, le=5, description="Desired number of bedrooms.")


class MoveInUpdate(BaseModel):
    move_in_date: date = Field(..., description="Desired move-in date.")


class BudgetUpdate(BaseModel):
    min_amount: Optional[float] = Field(default=None, ge=0)
    max_amount: Optional[float] = Field(default=None, ge=0)
    currency: Optional[str] = Field(default="USD")


class ContactUpdate(BaseModel):
    phone: Optional[str] = Field(default=None, description="E.164 or raw phone string.")
    email: Optional[EmailStr] = Field(default=None, description="Prospect email address.")


class TourTimeUpdate(BaseModel):
    tour_time: datetime = Field(..., description="Desired tour date and time.")


class EmptyPayload(BaseModel):
    model_config = {"extra": "ignore"}


class PlannerStateExtraction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    property_name: Optional[str] = None
    desired_bedrooms: Optional[int] = None
    move_in_date: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    target_rent: Optional[float] = None
    prospect_name: Optional[str] = None
    prospect_email: Optional[str] = None
    prospect_phone: Optional[str] = None
    requested_time: Optional[str] = None
    has_pets: Optional[bool] = None
    tour_type: Optional[str] = None
    reasoning: Optional[str] = None

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        schema["required"] = []
        schema["additionalProperties"] = False
        return schema

