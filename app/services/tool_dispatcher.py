from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Optional

import logging

from app.logging.flight_recorder import FlightRecorder
from app.models.tooling import (
    AvailabilityQuery,
    NetEffectiveRentRequest,
    PolicyQuery,
    SisterPropertyRouteRequest,
    TourRequest,
)
from app.services import tools

logger = logging.getLogger(__name__)


class ToolDispatcher:
    def __init__(self, recorder: Optional[FlightRecorder] = None) -> None:
        self.recorder = recorder
        self.registry: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            "check_availability": self._wrap(self._check_availability),
            "check_policy": self._wrap(self._check_policy),
            "compute_net_effective_rent": self._wrap(self._compute_net_effective_rent),
            "book_tour": self._wrap(self._book_tour),
            "send_sms": self._wrap(self._send_sms),
            "route_to_sister_property": self._wrap(self._route_to_sister_property),
        }

    def _wrap(self, func: Callable[[Dict[str, Any]], Any]) -> Callable[[Dict[str, Any]], Any]:
        def wrapped(args: Dict[str, Any]) -> Any:
            logger.info("tool.call", tool=func.__name__)
            return func(args)

        return wrapped

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if tool_name not in self.registry:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self.registry[tool_name](arguments)

    def _check_availability(self, args: Dict[str, Any]):
        query = AvailabilityQuery(**args)
        return [item.model_dump() for item in tools.check_availability(query, self.recorder)]

    def _check_policy(self, args: Dict[str, Any]):
        query = PolicyQuery(**args)
        return [item.model_dump() for item in tools.check_policy(query, self.recorder)]

    def _compute_net_effective_rent(self, args: Dict[str, Any]):
        request = NetEffectiveRentRequest(**args)
        response = tools.compute_net_effective_rent(request, self.recorder)
        return response.model_dump() if response else None

    def _book_tour(self, args: Dict[str, Any]):
        if isinstance(args.get("tour_time"), str):
            args["tour_time"] = datetime.fromisoformat(args["tour_time"])
        request = TourRequest(**args)
        confirmation = tools.book_tour(request, self.recorder)
        return confirmation.model_dump()

    def _send_sms(self, args: Dict[str, Any]):
        to_phone = args.get("to_phone") or args.get("phone")
        body = args.get("body") or args.get("message")
        result = tools.send_sms(to_phone=to_phone, body=body, recorder=self.recorder)
        return result

    def _route_to_sister_property(self, args: Dict[str, Any]):
        if "move_in_date" in args and isinstance(args["move_in_date"], str):
            args["move_in_date"] = datetime.fromisoformat(args["move_in_date"]).date()
        request = SisterPropertyRouteRequest(**args)
        options = tools.route_to_sister_property(request, self.recorder)
        return [option.model_dump() for option in options]
