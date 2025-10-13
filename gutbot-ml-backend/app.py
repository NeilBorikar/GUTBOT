# app.py — Railway ML Backend for GutBot
# Loads heavy ML stack once (spaCy / transformers / torch / sklearn),
# exposes /process for the Vercel gateway, and stays fast & stable.

import os
import time
import uuid
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from starlette.concurrency import run_in_threadpool

# Optional dependencies (loaded if available)
sentry_available = False
try:
    import sentry_sdk
    sentry_available = True
except Exception:
    sentry_sdk = None

prom_available = False
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    prom_available = True
except Exception:
    Instrumentator = None

redis_available = False
try:
    import redis.asyncio as aioredis
    redis_available = True
except Exception:
    aioredis = None

# ---- Import your local ML engine
# Must be in the same repo as this app (or installed as a package).
# HealthChatbot must load models in __init__ and expose .process_message(session_id, text)
from public_health_chatbot import HealthChatbot, Config

# =====================================================================================
# Environment & Logging
# =====================================================================================

ENV = os.getenv("ENVIRONMENT", "production")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ALLOWED_HOSTS = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "*").split(",") if h.strip()]
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
API_KEYS = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]  # optional
SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()
REDIS_URL = os.getenv("REDIS_URL", "").strip()  # optional for future rate limit/caching
MODEL_WARMUP_TEXT = os.getenv("MODEL_WARMUP_TEXT", "hello")  # used to warm up pipeline
SERVICE_NAME = os.getenv("SERVICE_NAME", "gutbot-ml")
BUILD_SHA = os.getenv("BUILD_SHA", "")
START_TS = time.time()

# Torch perf hints (safe no-ops if torch not present)
try:
    import torch
    torch.set_num_threads(max(1, int(os.getenv("TORCH_THREADS", "1"))))
    torch.set_num_interop_threads(max(1, int(os.getenv("TORCH_INTEROP_THREADS", "1"))))
except Exception:
    pass

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("GutBot-ML")

if sentry_available and SENTRY_DSN:
    sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=float(os.getenv("SENTRY_TRACES", "0.05")))
    logger.info("Sentry initialized.")

# =====================================================================================
# FastAPI app & security
# =====================================================================================

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_request_id(req: Request) -> str:
    return req.headers.get("X-Request-ID", str(uuid.uuid4()))

async def require_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Optional API key enforcement (set API_KEYS to enable)."""
    if not API_KEYS:  # not configured -> open (use at your own risk)
        return None
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return api_key

# =====================================================================================
# Models
# =====================================================================================

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None

    @validator("message")
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()

class EntityModel(BaseModel):
    text: str
    type: str
    confidence: float

class IntentModel(BaseModel):
    type: str
    confidence: float

class ChatResponse(BaseModel):
    response: str
    session_id: str
    entities: List[EntityModel]
    intent: IntentModel
    timestamp: str
    processing_time_ms: float
    disclaimer: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str
    service: str
    uptime: float
    version: str
    build: str
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    request_id: str
    details: Optional[str] = None

# =====================================================================================
# Lifespan — load heavy models once
# =====================================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional Redis (not currently required, but we keep the hook for caching/rate limits)
    app.state.redis = None
    if redis_available and REDIS_URL:
        try:
            app.state.redis = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
            logger.info("Connected to Redis.")
        except Exception as e:
            logger.warning("Redis connection failed: %s", e)

    # Load your ML engine once
    logger.info("Loading HealthChatbot models… (this may take a while on first boot)")
    t0 = time.time()
    app.state.chatbot = HealthChatbot()
    load_ms = (time.time() - t0) * 1000
    logger.info("HealthChatbot loaded in %.1f ms", load_ms)

    # Warm-up pass (avoids first-request cold latency)
    try:
        _ = await run_in_threadpool(app.state.chatbot.process_message, "warmup-session", MODEL_WARMUP_TEXT)
        logger.info("Warm-up completed.")
    except Exception as e:
        logger.warning("Warm-up failed (continuing): %s", e)

    # Prometheus /metrics (optional)
    if prom_available:
        try:
            Instrumentator().instrument(app).expose(app)
            logger.info("Prometheus metrics exposed at /metrics")
        except Exception as e:
            logger.warning("Prometheus init failed: %s", e)

    yield

    # Graceful shutdown
    if app.state.redis:
        try:
            await app.state.redis.close()
        except Exception:
            pass

app = FastAPI(
    title="GutBot ML Backend",
    version="2.0.0",
    description="Heavy ML service (spaCy/transformers/torch) for GutBot — deployed on Railway.",
    lifespan=lifespan,
)

# Middlewares
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS if ALLOWED_HOSTS else ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Security headers
@app.middleware("http")
async def secure_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=()"
    return response

# Process-time + request-id
@app.middleware("http")
async def add_telemetry(request: Request, call_next):
    req_id = get_request_id(request)
    t0 = time.time()
    try:
        resp = await call_next(request)
    except Exception as exc:
        if sentry_available and SENTRY_DSN:
            sentry_sdk.capture_exception(exc)
        resp = JSONResponse(status_code=500, content={"detail": "Internal server error"})
        raise
    finally:
        elapsed = time.time() - t0
        try:
            resp.headers["X-Process-Time"] = f"{elapsed:.4f}"
            resp.headers["X-Request-ID"] = req_id
        except Exception:
            pass
    return resp

# =====================================================================================
# Helpers
# =====================================================================================

def _normalize_result(raw: Any, session_id: str, elapsed_ms: float) -> Dict[str, Any]:
    """Normalize HealthChatbot outputs into ChatResponse schema."""
    timestamp = datetime.utcnow().isoformat()
    disclaimer_text = getattr(Config, "RESPONSE_TEMPLATES", {}).get("disclaimer", "")

    if isinstance(raw, str):
        return {
            "response": raw,
            "session_id": session_id,
            "entities": [],
            "intent": {"type": "unknown", "confidence": 0.0},
            "timestamp": timestamp,
            "processing_time_ms": round(elapsed_ms, 2),
            "disclaimer": disclaimer_text or None,
        }

    if isinstance(raw, dict):
        out = dict(raw)
        out.setdefault("session_id", session_id)
        out.setdefault("timestamp", timestamp)
        out.setdefault("processing_time_ms", round(elapsed_ms, 2))
        out.setdefault("disclaimer", disclaimer_text or None)

        # entities normalization
        ents = out.get("entities", [])
        norm_ents = []
        for e in ents:
            if isinstance(e, dict):
                norm_ents.append({
                    "text": e.get("text", ""),
                    "type": e.get("type", e.get("label", "unknown")),
                    "confidence": float(e.get("confidence", 0.0)),
                })
            else:
                norm_ents.append({
                    "text": getattr(e, "text", ""),
                    "type": getattr(e, "type", getattr(e, "label", "unknown")),
                    "confidence": float(getattr(e, "confidence", 0.0)),
                })
        out["entities"] = norm_ents

        # intent normalization
        intent = out.get("intent", {})
        if isinstance(intent, dict):
            out["intent"] = {
                "type": intent.get("type", "unknown"),
                "confidence": float(intent.get("confidence", 0.0)),
            }
        else:
            out["intent"] = {
                "type": getattr(intent, "type", "unknown"),
                "confidence": float(getattr(intent, "confidence", 0.0)),
            }
        return out

    raise ValueError("Unsupported chatbot result type")

# =====================================================================================
# Routes
# =====================================================================================

@app.get("/health", response_model=HealthCheckResponse, tags=["meta"])
async def health():
    """Lightweight health check that does NOT hit the model."""
    return HealthCheckResponse(
        status="healthy",
        service=SERVICE_NAME,
        uptime=time.time() - START_TS,
        version=app.version,
        build=BUILD_SHA,
        timestamp=datetime.utcnow().isoformat(),
    )

@app.get("/ready", tags=["meta"])
async def ready():
    """Readiness probe — verifies model object exists."""
    if getattr(app.state, "chatbot", None) is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}

@app.post("/process", response_model=ChatResponse, tags=["inference"])
async def process_chat(payload: ChatMessage, background: BackgroundTasks, _api_key: Optional[str] = Depends(require_api_key)):
    """
    Heavy endpoint. Accepts:
      { "message": str, "session_id": str? }
    Returns a structured ChatResponse.
    """
    if getattr(app.state, "chatbot", None) is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    session_id = payload.session_id or str(uuid.uuid4())
    t0 = time.time()

    try:
        raw = await run_in_threadpool(app.state.chatbot.process_message, session_id, payload.message)
        elapsed = (time.time() - t0) * 1000.0
        normalized = _normalize_result(raw, session_id, elapsed)

        # Async log (you can wire this to Redis, DB, S3, etc.)
        def _log():
            short = payload.message[:250].replace("\n", " ")
            logger.info("session=%s | %.1fms | text=%s", session_id, elapsed, short)
        background.add_task(_log)

        return ChatResponse(**normalized)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Processing error: %s", e)
        if sentry_available and SENTRY_DSN:
            sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail="Error processing message")

# ---- Error handlers

@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    req_id = get_request_id(request)
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(ErrorResponse(error=str(exc.detail), request_id=req_id)),
    )

@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    req_id = get_request_id(request)
    logger.error("Unhandled error: %s", exc, exc_info=True)
    if sentry_available and SENTRY_DSN:
        sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder(ErrorResponse(
            error="Internal server error",
            request_id=req_id,
            details=str(exc) if ENV != "production" else None
        )),
    )

# =====================================================================================
# Local dev entrypoint (Railway will use the CMD below)
# =====================================================================================
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    uvicorn.run("app:app", host=host, port=port, workers=workers, log_level=LOG_LEVEL.lower())
