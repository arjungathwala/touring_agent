from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    # Load from project root .env if present
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
except Exception:
    # dotenv is optional; proceed if not available
    pass

