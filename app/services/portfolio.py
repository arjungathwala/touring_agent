from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

from dateutil import parser as dateparser

from app.utils.fixture_loader import load_inventory, load_policies, load_portfolio


@dataclass
class Property:
    id: str
    name: str
    address: str
    lat: float
    lng: float
    timezone: str
    sisters: List[str]


@dataclass
class Unit:
    property_id: str
    unit_id: str
    bedrooms: int
    bathrooms: int
    rent: float
    available_on: date
    concessions: Dict[str, float]
    sq_ft: int


@dataclass
class Policy:
    property_id: str
    policy_type: str
    description: str
    notes: Optional[str]


@lru_cache(maxsize=1)
def get_portfolio() -> Dict[str, Property]:
    properties = {}
    for raw in load_portfolio():
        properties[raw["id"]] = Property(
            id=raw["id"],
            name=raw["name"],
            address=raw["address"],
            lat=raw["lat"],
            lng=raw["lng"],
            timezone=raw["timezone"],
            sisters=raw.get("sisters", []),
        )
    return properties


@lru_cache(maxsize=1)
def get_inventory() -> Dict[str, List[Unit]]:
    inventory: Dict[str, List[Unit]] = {}
    for raw in load_inventory():
        unit = Unit(
            property_id=raw["property_id"],
            unit_id=raw["unit_id"],
            bedrooms=raw["bedrooms"],
            bathrooms=raw["bathrooms"],
            rent=float(raw["rent"]),
            available_on=dateparser.parse(raw["available_on"]).date(),
            concessions=raw.get("concessions", {}),
            sq_ft=raw.get("sq_ft", 0),
        )
        inventory.setdefault(unit.property_id, []).append(unit)
    return inventory


@lru_cache(maxsize=1)
def get_policies() -> Dict[str, List[Policy]]:
    policies: Dict[str, List[Policy]] = {}
    for raw in load_policies():
        policy = Policy(
            property_id=raw["property_id"],
            policy_type=raw["policy_type"],
            description=raw["description"],
            notes=raw.get("notes"),
        )
        policies.setdefault(policy.property_id, []).append(policy)
    return policies


def list_units(property_id: str) -> List[Unit]:
    return get_inventory().get(property_id, [])


def list_policies(property_id: str) -> List[Policy]:
    return get_policies().get(property_id, [])


def get_property(property_id: str) -> Optional[Property]:
    return get_portfolio().get(property_id)


def iter_sister_properties(property_id: str) -> Iterable[Property]:
    prop = get_property(property_id)
    if not prop:
        return []
    return [get_portfolio()[sister_id] for sister_id in prop.sisters if sister_id in get_portfolio()]
