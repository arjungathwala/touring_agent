from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import health, twilio, ws
from app.logging.flight_recorder import register_log_middleware


def create_app() -> FastAPI:
    app = FastAPI(title="Touring Agent", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_log_middleware(app)

    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(twilio.router, prefix="/twilio", tags=["twilio"])
    app.include_router(ws.router, tags=["realtime"])

    return app


app = create_app()
