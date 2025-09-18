from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

_FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"


def load_json(filename: str) -> List[Dict[str, Any]]:
    file_path = _FIXTURE_DIR / filename
    with file_path.open() as f:
        return json.load(f)


def load_portfolio() -> List[Dict[str, Any]]:
    return load_json("portfolio.json")


def load_inventory() -> List[Dict[str, Any]]:
    return load_json("inventory.json")


def load_policies() -> List[Dict[str, Any]]:
    return load_json("policies.json")
