from __future__ import annotations

import os
from typing import Optional

import logging

logger = logging.getLogger(__name__)


class SmsClient:
    def __init__(self, account_sid: Optional[str] = None, auth_token: Optional[str] = None, from_phone: Optional[str] = None) -> None:
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.from_phone = from_phone or os.getenv("TWILIO_FROM_PHONE")

    async def send(self, to_phone: str, body: str) -> dict[str, str]:
        # In the MVP environment we log instead of hitting Twilio directly
        logger.info("sms.send", to_phone=_redact(to_phone), body_preview=body[:120])
        # Integrate with Twilio API here when credentials are present
        return {"status": "queued", "to": to_phone}


def _redact(phone: str) -> str:
    if len(phone) <= 4:
        return "***"
    return f"***{phone[-4:]}"
