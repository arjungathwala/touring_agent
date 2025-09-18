from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
