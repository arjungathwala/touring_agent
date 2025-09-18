import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services import tools


def setup_function():
    tools._BOOKED_TOURS.clear()


def _start_call(ws, stream_sid="CA123"):
    ws.send_text(
        json.dumps({
            "event": "start",
            "streamSid": stream_sid,
            "start": {"streamSid": stream_sid},
        })
    )


def test_happy_path_match_and_book(tmp_path):
    client = TestClient(app)
    with client.websocket_connect("/twilio/media-stream") as ws:
        _start_call(ws, stream_sid="CA123")
        ws.send_text(
            json.dumps(
                {
                    "event": "mark",
                    "streamSid": "CA123",
                    "mark": {
                        "name": "transcript",
                        "payload": {"text": "Hi, I'd like to tour 21 West End"},
                    },
                }
            )
        )
        response = ws.receive_json()
        assert "21 West End" in response["text"]

        ws.send_text(
            json.dumps(
                {
                    "event": "mark",
                    "streamSid": "CA123",
                    "mark": {
                        "name": "transcript",
                        "payload": {"text": "Please book me for Saturday at 11"},
                    },
                }
            )
        )
        response = ws.receive_json()
        text = response["text"].lower()
        assert "book" in text

    assert tools._BOOKED_TOURS
    confirmation = next(iter(tools._BOOKED_TOURS.values()))
    assert confirmation.property_id == "21we"
    assert Path(confirmation.ics_path).exists()


def test_happy_path_route_and_book():
    client = TestClient(app)
    with client.websocket_connect("/twilio/media-stream") as ws:
        _start_call(ws, stream_sid="CA124")
        ws.send_text(
            json.dumps(
                {
                    "event": "mark",
                    "streamSid": "CA124",
                    "mark": {
                        "name": "transcript",
                        "payload": {"text": "I'm hunting for a studio at 21 West End"},
                    },
                }
            )
        )
        response = ws.receive_json()
        assert "Hudson 360" in response["text"]

        ws.send_text(
            json.dumps(
                {
                    "event": "mark",
                    "streamSid": "CA124",
                    "mark": {
                        "name": "transcript",
                        "payload": {"text": "Yes book that for Saturday at 11"},
                    },
                }
            )
        )
        response = ws.receive_json()
        text = response["text"].lower()
        assert "book" in text

    confirmations = list(tools._BOOKED_TOURS.values())
    assert confirmations
    # The latest booking should be for the sister property
    assert any(c.property_id == "hudson-360" for c in confirmations)
