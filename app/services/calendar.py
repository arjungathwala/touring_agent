from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path
from typing import Optional

from dateutil import tz

ICS_DIR = Path(__file__).resolve().parent.parent.parent / "tmp" / "ics"


def ensure_ics_dir() -> None:
    ICS_DIR.mkdir(parents=True, exist_ok=True)


def create_google_event_link(summary: str, start: datetime, duration_minutes: int, details: str, location: str) -> str:
    end = start + timedelta(minutes=duration_minutes)
    start_str = start.strftime("%Y%m%dT%H%M%S")
    end_str = end.strftime("%Y%m%dT%H%M%S")
    params = {
        "action": "TEMPLATE",
        "text": summary,
        "dates": f"{start_str}/{end_str}",
        "details": details,
        "location": location,
    }
    query = "&".join(f"{key}={value.replace(' ', '+')}" for key, value in params.items())
    return f"https://calendar.google.com/calendar/render?{query}"


def create_ics_file(summary: str, start: datetime, duration_minutes: int, description: str, location: str, timezone: str) -> Path:
    ensure_ics_dir()
    event_uid = f"{uuid.uuid4()}@touring-agent"
    end = start + timedelta(minutes=duration_minutes)
    tzinfo = tz.gettz(timezone)
    start_local = start.astimezone(tzinfo) if start.tzinfo else start.replace(tzinfo=tzinfo)
    end_local = end.astimezone(tzinfo) if end.tzinfo else end.replace(tzinfo=tzinfo)
    dtstamp = datetime.now(dt_timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Touring Agent//EN
BEGIN:VEVENT
UID:{event_uid}
DTSTAMP:{dtstamp}
DTSTART;TZID={timezone}:{start_local.strftime('%Y%m%dT%H%M%S')}
DTEND;TZID={timezone}:{end_local.strftime('%Y%m%dT%H%M%S')}
SUMMARY:{summary}
DESCRIPTION:{description}
LOCATION:{location}
END:VEVENT
END:VCALENDAR
"""
    file_path = ICS_DIR / f"{event_uid}.ics"
    file_path.write_text(ics_content)
    return file_path


def create_m365_deeplink(summary: str, start: datetime, duration_minutes: int, details: str, location: str) -> str:
    end = start + timedelta(minutes=duration_minutes)
    start_str = start.strftime("%Y-%m-%dT%H:%M:%S")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%S")
    return (
        "https://outlook.live.com/calendar/0/deeplink/compose?"
        f"subject={summary.replace(' ', '+')}&body={details.replace(' ', '+')}"
        f"&startdt={start_str}&enddt={end_str}&location={location.replace(' ', '+')}"
    )


def write_calendar_artifacts(summary: str, start: datetime, duration_minutes: int, details: str, location: str, timezone: str) -> dict[str, Optional[str]]:
    google_link = create_google_event_link(summary, start, duration_minutes, details, location)
    m365_link = create_m365_deeplink(summary, start, duration_minutes, details, location)
    ics_path = create_ics_file(summary, start, duration_minutes, details, location, timezone)
    return {
        "google_link": google_link,
        "m365_link": m365_link,
        "ics_path": str(ics_path),
    }
