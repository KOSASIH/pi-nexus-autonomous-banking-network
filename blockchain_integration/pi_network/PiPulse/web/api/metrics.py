from fastapi import APIRouter
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

metrics_router = APIRouter()

@metrics_router.get("/metrics")
async def get_metrics():
    registry = CollectorRegistry()
    metrics_page = generate_latest(registry)
    return JSONResponse(content=metrics_page, media_type=CONTENT_TYPE_LATEST)
