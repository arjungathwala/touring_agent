**Project description (1–2 lines)**
Voice-first, portfolio-aware leasing agent that answers 24/7, checks live availability/policies, computes net-effective rent, and **books tours on-call**—routing to sister properties when the target building can’t fit.

**MVP build list (8 items)**

1. **Twilio Media Streams → FastAPI WS** (answer <1s; fallback `<Gather>`).
2. **Realtime loop:** Deepgram STT + OpenAI Realtime (function calling, barge-in, TTS <900ms).
3. **Tool layer (5 funcs):** `check_availability`, `check_policy`, `compute_net_effective_rent`, `book_tour` (idempotent), `send_sms`.
4. **Portfolio routing:** `route_to_sister_property` (haversine + availability; return top 1–2).
5. **Fixtures (JSON):** `portfolio.json`, `inventory.json`, `policies.json`.
6. **Writes:** Google/M365 event (+ `.ics` fallback) and Twilio SMS confirmation.
7. **Flight-recorder logs:** `[ASR] → [PLAN] → [POLICY] → [NER] → [BOOK_TOUR] → [SMS]` with timings; PII redacted.
8. **Happy paths:**
   A) Match at 21WE → offer 2 units → **book Sat 11:00** → SMS/ICS.
   B) No match → **route to sister property** → **book** → SMS/ICS.
