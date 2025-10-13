# app.py — Vercel Gateway for GutBot
# Serves static frontend + proxies chat requests to Railway ML backend.
# Ultra-lightweight, zero heavy ML dependencies.

import os
import uuid
import logging
import pathlib
import httpx
import time
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

# ================================================================
# 🌐 Environment
# ================================================================

ML_BACKEND_URL = os.getenv("ML_BACKEND_URL", "").rstrip("/")
ML_BACKEND_API_KEY = os.getenv("ML_BACKEND_API_KEY", "").strip()
ALLOWED_HOSTS = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "*").split(",") if h.strip()]
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
REQUEST_TIMEOUT = float(os.getenv("ML_REQUEST_TIMEOUT", "20.0"))  # seconds

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("GutBot-Gateway")

if not ML_BACKEND_URL:
    logger.warning("⚠️  ML_BACKEND_URL not set — /chat will return 503")

# ================================================================
# 📡 Models
# ================================================================

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None

    @validator("message")
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()

# ================================================================
# 🌍 FastAPI App
# ================================================================

app = FastAPI(
    title="GutBot Vercel Gateway",
    version="2.0.0",
    description="Serves GutBot frontend and proxies chat requests to the ML backend on Railway.",
)

# Middlewares
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS or ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ================================================================
# 🛡 Security & Telemetry
# ================================================================

def get_request_id(req: Request) -> str:
    return req.headers.get("X-Request-ID", str(uuid.uuid4()))

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=()"
    return response

@app.middleware("http")
async def add_telemetry(request: Request, call_next):
    req_id = get_request_id(request)
    t0 = time.time()
    try:
        resp = await call_next(request)
    finally:
        elapsed = time.time() - t0
        try:
            resp.headers["X-Request-ID"] = req_id
            resp.headers["X-Process-Time"] = f"{elapsed:.4f}"
        except Exception:
            pass
    return resp

# ================================================================
# 🧠 Proxy: /chat → Railway /process
# ================================================================

@app.post("/chat")
async def chat_proxy(payload: ChatMessage):
    """
    Lightweight proxy to ML backend.
    Forwards payload to {ML_BACKEND_URL}/process and returns the response.
    """
    if not ML_BACKEND_URL:
        raise HTTPException(status_code=503, detail="ML backend not configured")

    target_url = f"{ML_BACKEND_URL}/process"
    headers = {"Content-Type": "application/json"}
    if ML_BACKEND_API_KEY:
        headers["X-API-Key"] = ML_BACKEND_API_KEY

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            resp = await client.post(target_url, json=payload.dict(), headers=headers)
        except httpx.RequestError as e:
            logger.error("ML backend unreachable: %s", e)
            raise HTTPException(status_code=503, detail="ML backend unreachable or timed out")

    if resp.status_code != 200:
        try:
            err_json = resp.json()
        except Exception:
            err_json = {"error": resp.text}
        logger.warning("ML backend error %s: %s", resp.status_code, err_json)
        raise HTTPException(status_code=resp.status_code, detail=err_json)

    return resp.json()

# ================================================================
# 🩺 Health Checks
# ================================================================

@app.get("/health")
async def health():
    """Basic health for Vercel function itself."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/vercel-health")
async def vercel_health():
    """Used by Vercel build system to verify the lambda deploy."""
    return {"status": "ok", "message": "GutBot Vercel gateway running 🚀"}

# ================================================================
# 🌐 Static Frontend (public/)
# ================================================================

FRONTEND_DIR = pathlib.Path(__file__).resolve().parent / "public"

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="public")
    logger.info(f"Serving static frontend from: {FRONTEND_DIR}")
else:
    logger.warning(f"No public folder found at {FRONTEND_DIR}")

@app.get("/chat-page")
async def serve_chat_page():
    file = FRONTEND_DIR / "chat.html"
    if file.exists():
        return FileResponse(file)
    raise HTTPException(status_code=404, detail="chat.html not found")

@app.get("/")
async def serve_index():
    file = FRONTEND_DIR / "index.html"
    if file.exists():
        return FileResponse(file)
    raise HTTPException(status_code=404, detail="index.html not found")

# ================================================================
# ❌ Error Handlers
# ================================================================

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    req_id = get_request_id(request)
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder({"error": str(exc.detail), "request_id": req_id}),
    )

@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    req_id = get_request_id(request)
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder({
            "error": "Internal server error",
            "request_id": req_id,
            "details": str(exc) if os.getenv("ENVIRONMENT") != "production" else None
        }),
    )

# ================================================================
# ✅ Vercel Entrypoint
# ================================================================

app = app
