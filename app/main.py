"""FastAPI entry — `uvicorn app.main:app --reload --port 8000`"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.analysis import router as analysis_router
from .routes.scan import router as scan_router

app = FastAPI(title="台股技術分析 API", version="0.1.0")

# Allowed origins: localhost defaults for dev, plus anything in ALLOWED_ORIGINS
# (comma-separated) for production deployment.
_default_origins = [
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:3030", "http://127.0.0.1:3030",
]
_extra = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
allowed_origins = _default_origins + _extra

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router)
app.include_router(scan_router)


@app.get("/health")
def health():
    return {"ok": True}
