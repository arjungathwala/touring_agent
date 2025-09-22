from __future__ import annotations

import os
from typing import Optional

import logging
import httpx

logger = logging.getLogger(__name__)


class SmsClient:
    def __init__(self, account_sid: Optional[str] = None, auth_token: Optional[str] = None, from_phone: Optional[str] = None) -> None:
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.from_phone = from_phone or os.getenv("TWILIO_FROM_PHONE")

    async def send(self, to_phone: str, body: str) -> dict[str, str]:
        if not self.account_sid or not self.auth_token or not self.from_phone:
            logger.info("sms.stub", to_phone=_redact(to_phone), body_preview=body[:120])
            return {"status": "stubbed", "to": to_phone}

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        payload = {
            "To": to_phone,
            "From": self.from_phone,
            "Body": body,
        }
        auth = (self.account_sid, self.auth_token)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0), auth=auth) as client:
                response = await client.post(url, data=payload)
            response.raise_for_status()
            data = response.json()
            logger.info("sms.sent", to=_redact(to_phone), sid=data.get("sid"))
            return {
                "status": data.get("status", "queued"),
                "to": data.get("to", to_phone),
                "sid": data.get("sid"),
            }
        except httpx.HTTPError as exc:
            logger.exception("sms.error to=%s err=%s", _redact(to_phone), exc)
            return {"status": "error", "to": to_phone, "error": str(exc)}


def _redact(phone: str) -> str:
    if len(phone) <= 4:
        return "***"
    return f"***{phone[-4:]}"
