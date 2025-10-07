# app.py - production-ready FastAPI entrypoint for Public Health Chatbot
import os
import time
import uuid
import logging
import pathlib
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from functools import lru_cache

from fastapi import FastAPI, Request, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator

# Concurrency helper for running blocking calls in threadpool
from starlette.concurrency import run_in_threadpool
from datetime import datetime
import redis

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.Redis.from_url(redis_url)




# Try optional integrations
fastapi_limiter_available = False
try:
    from fastapi_limiter import FastAPILimiter
    from fastapi_limiter.depends import RateLimiter
    fastapi_limiter_available = True
except Exception:
    FastAPILimiter = None
    RateLimiter = None

instrumentator_available = False
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    instrumentator_available = True
except Exception:
    Instrumentator = None

sentry_available = False
try:
    import sentry_sdk
    sentry_available = True
except Exception:
    sentry_sdk = None

# Import chatbot engine
from public_health_chatbot import HealthChatbot, Config

# -------------------- Logging --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.getenv("APP_LOG", "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HealthChatbotAPI")

# -------------------- Security & Rate limit defaults --------------------
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

# -------------------- Pydantic models --------------------
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")

    @validator('message')
    def message_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
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
    version: str
    timestamp: str
    uptime: float

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    request_id: str

# -------------------- Lifespan / startup / shutdown --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      - initialize optional services (Redis + FastAPILimiter)
      - build chatbot instance and attach to app.state.chatbot
      - instrument Prometheus if available
    Shutdown:
      - gracefully close optional services
    """
    # Sentry init (if configured)
    if sentry_available and os.getenv("SENTRY_DSN"):
        sentry_sdk.init(dsn=os.getenv("SENTRY_DSN"), traces_sample_rate=0.05)

    logger.info("Starting Public Health Chatbot API...")

    # Rate limiter initialization (Redis)
    app.state.rate_limiter_available = False
    if fastapi_limiter_available:
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            import redis.asyncio as redis  # local import
            redis_conn = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            await FastAPILimiter.init(redis_conn)
            app.state.rate_limiter_available = True
            logger.info("FastAPILimiter initialized with Redis at %s", redis_url)
        except Exception as e:
            logger.warning("FastAPILimiter initialization failed: %s. Continuing without limiter.", e)
            app.state.rate_limiter_available = False

    # Initialize chatbot instance once and attach to app state
    try:
        app.state.chatbot = HealthChatbot()
        logger.info("HealthChatbot instance created and attached to app.state.chatbot")
    except Exception as e:
        logger.exception("Failed to initialize HealthChatbot: %s", e)
        raise

    # Prometheus instrumentator (optional)
    if instrumentator_available:
        try:
            Instrumentator().instrument(app).expose(app)
            logger.info("Prometheus Instrumentator initialized")
        except Exception as e:
            logger.warning("Instrumentator initialization failed: %s", e)

    # Mark start time
    app.state.start_time = time.time()

    try:
        yield
    finally:
        logger.info("Shutting down Public Health Chatbot API...")
        # Close FastAPILimiter if it was inited
        if fastapi_limiter_available and app.state.rate_limiter_available:
            try:
                await FastAPILimiter.close()
            except Exception as e:
                logger.warning("Error closing FastAPILimiter: %s", e)

# -------------------- Application --------------------
app = FastAPI(
    title="Public Health Chatbot API",
    description="AI-powered public health information and disease awareness chatbot",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(","))
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=()"
    return response


# ---------- Frontend / static files ----------
# ---------- Frontend / Static Files (Vercel-Optimized) ----------
FRONTEND_DIR = pathlib.Path(__file__).resolve().parent / "public"

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="public")
    logger.info("Mounted /public as static frontend directory")
else:
    logger.warning("Public frontend folder not found at %s. Static file serving will fail until it is created.", FRONTEND_DIR)

# Optional pretty URL for chat
@app.get("/chat-page")
async def serve_chat_page():
    chat_file = FRONTEND_DIR / "chat.html"
    if chat_file.exists():
        return FileResponse(chat_file)
    raise HTTPException(status_code=404, detail="chat.html not found")




# -------------------- Utilities & dependencies --------------------
def get_request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))

async def get_api_key(api_key: str = Depends(api_key_header)):
    """
    Validate API key if API_KEYS are configured.
    If API_KEYS env var is empty, skip validation (developer mode).
    """
    configured = os.getenv("API_KEYS", "").strip()
    if not configured:
        # No keys configured → skip authentication (dev)
        return None

    valid_keys = [k.strip() for k in configured.split(",") if k.strip()]
    if not api_key or api_key not in valid_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return api_key

def get_chatbot(request: Request) -> HealthChatbot:
    """
    Return the chatbot instance created during lifespan startup.
    If not present for any reason, create one and attach it (safety net).
    """
    chatbot = getattr(request.app.state, "chatbot", None)
    if chatbot is None:
        logger.warning("Chatbot not found in app.state – creating a new instance on demand")
        chatbot = HealthChatbot()
        request.app.state.chatbot = chatbot
    return chatbot

# -------------------- Middlewares & Exception handlers --------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    request_id = get_request_id(request)
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as exc:
        # Ensure headers exist even on exceptions
        response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
        raise
    finally:
        process_time = time.time() - start_time
        # Add headers if response object supports it
        try:
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
        except Exception:
            pass
    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = get_request_id(request)
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(ErrorResponse(error=exc.detail, request_id=request_id))
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id = get_request_id(request)
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    if sentry_available and os.getenv("SENTRY_DSN"):
        sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(ErrorResponse(
            error="Internal server error",
            details=str(exc) if os.getenv("ENVIRONMENT") == "development" else None,
            request_id=request_id
        ))
    )

# -------------------- Background logger --------------------
async def log_conversation(session_id: str, message: str, response: Dict[str, Any], processing_time_ms: float):
    # Consider writing to DB, ELK, or other storage in production
    logger.info("Conversation logged - session=%s processing_ms=%.2f message=%s", session_id, processing_time_ms, (message[:200] + "...") if len(message) > 200 else message)

# -------------------- Helper to normalize chatbot result --------------------
def _normalize_chatbot_result(raw_result: Any, session_id: str, processing_time_ms: float) -> Dict[str, Any]:
    """
    Normalize the possible outputs from HealthChatbot.process_message into a dict
    that matches ChatResponse fields.
    Accepts:
      - str (simple response)
      - dict (detailed response as implemented in some variants)
    """
    timestamp = datetime.utcnow().isoformat()
    disclaimer_text = getattr(Config, "RESPONSE_TEMPLATES", {}).get("disclaimer", "")

    if isinstance(raw_result, str):
        return {
            "response": raw_result,
            "session_id": session_id,
            "entities": [],
            "intent": {"type": "unknown", "confidence": 0.0},
            "timestamp": timestamp,
            "processing_time_ms": round(processing_time_ms, 2),
            "disclaimer": disclaimer_text
        }
    elif isinstance(raw_result, dict):
        normalized = raw_result.copy()
        normalized.setdefault("session_id", session_id)
        normalized.setdefault("processing_time_ms", round(processing_time_ms, 2))
        normalized.setdefault("timestamp", timestamp)
        # ensure entities format
        entities = normalized.get("entities", [])
        normalized_entities = []
        for e in entities:
            # accept either dict-like from chatbot or our EntityModel-compatible items
            if isinstance(e, dict):
                normalized_entities.append({
                    "text": e.get("text", ""),
                    "type": e.get("type", e.get("label", "unknown")),
                    "confidence": float(e.get("confidence", 0.0))
                })
            else:
                # fallback; convert objects with attributes
                normalized_entities.append({
                    "text": getattr(e, "text", ""),
                    "type": getattr(e, "type", getattr(e, "label", "")),
                    "confidence": float(getattr(e, "confidence", 0.0))
                })
        normalized["entities"] = normalized_entities
        intent = normalized.get("intent", {})
        if isinstance(intent, dict):
            normalized["intent"] = {"type": intent.get("type", "unknown"), "confidence": float(intent.get("confidence", 0.0))}
        else:
            # If intent is object, try to pull attributes
            normalized["intent"] = {"type": getattr(intent, "type", "unknown"), "confidence": float(getattr(intent, "confidence", 0.0))}
        normalized.setdefault("disclaimer", disclaimer_text)
        return normalized
    else:
        raise ValueError("Unsupported chatbot result type")

# -------------------- Rate-limited endpoint factory --------------------
# We'll define two flavors of the chat endpoint depending on whether FastAPILimiter is initialized.
# If FastAPILimiter is available and inited, we use RateLimiter decorator; otherwise we define a non-rate-limited endpoint.
def _register_endpoints():
    """
    Registers endpoints on the app dynamically depending on availability of rate-limiter.
    """

    async def _process_chat(chat_message: ChatMessage, background_tasks: BackgroundTasks, api_key: str, chatbot: HealthChatbot, request: Request):
        start_time = time.time()
        session_id = chat_message.session_id or str(uuid.uuid4())

        try:
            # Run potentially blocking ML/IO work in threadpool
            raw_result = await run_in_threadpool(chatbot.process_message, session_id, chat_message.message)

            processing_time_ms = (time.time() - start_time) * 1000.0
            normalized = _normalize_chatbot_result(raw_result, session_id, processing_time_ms)

            # background logging
            background_tasks.add_task(log_conversation, session_id, chat_message.message, normalized, processing_time_ms)

            # return as Pydantic model
            return ChatResponse(**normalized)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error in chat processing: %s", e)
            if sentry_available and os.getenv("SENTRY_DSN"):
                sentry_sdk.capture_exception(e)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing message")

    # Conditional decorator application based on rate-limiter availability
    if fastapi_limiter_available:
        # If the FastAPILimiter is available we can wrap the endpoint with RateLimiter on registration.
        decorator = app.post(
            "/chat",
            response_model=ChatResponse,
            summary="Process user message",
            description="Process a user message and return an appropriate response with intent and entity information",
        )
        # Register with a wrapper that applies RateLimiter as dependency inside function (safer than using decorator in some envs)
        async def chat_endpoint_with_optional_limit(chat_message: ChatMessage, background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key), chatbot: HealthChatbot = Depends(get_chatbot), request: Request = None):
            # If the rate limiter was inited in lifespan, apply RateLimiter logic manually:
            if getattr(request.app.state, "rate_limiter_available", False):
                # call RateLimiter dependency to enforce limit
                await RateLimiter(times=RATE_LIMIT_PER_MINUTE, minutes=1)(request=request)
            return await _process_chat(chat_message, background_tasks, api_key, chatbot, request)
        decorator(chat_endpoint_with_optional_limit)
    else:
        # No fastapi limiter; register simple endpoint
        app.post(
            "/chat",
            response_model=ChatResponse,
            summary="Process user message",
            description="Process a user message and return an appropriate response with intent and entity information"
        )(_process_chat)

# Register endpoints during import
_register_endpoints()

# -------------------- Health & session endpoints --------------------
@app.get("/health", response_model=HealthCheckResponse, summary="Health check")
async def health_check(request: Request, chatbot: HealthChatbot = Depends(get_chatbot)):
    """
    Lightweight health check. Avoids heavy ML operations.
    """
    try:
        uptime = time.time() - getattr(request.app.state, "start_time", time.time())
        return HealthCheckResponse(status="healthy", version=app.version, timestamp=datetime.utcnow().isoformat(), uptime=uptime)
    except Exception as e:
        logger.exception("Health check failed: %s", e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service unhealthy")

@app.get("/sessions/{session_id}", summary="Get session information")
async def get_session(session_id: str, api_key: str = Depends(get_api_key), chatbot: HealthChatbot = Depends(get_chatbot)):
    # public_health_chatbot.Session uses .history and .last_active
    if session_id in chatbot.sessions:
        session = chatbot.sessions[session_id]
        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": getattr(session, "last_active", session.created_at).isoformat(),
            "message_count": len(getattr(session, "history", []))
        }
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

@app.delete("/sessions/{session_id}", summary="Delete session")
async def delete_session(session_id: str, api_key: str = Depends(get_api_key), chatbot: HealthChatbot = Depends(get_chatbot)):
    if session_id in chatbot.sessions:
        del chatbot.sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

# -------------------- Startup time --------------------
app.state.start_time = time.time()

# -------------------- Vercel Compatibility --------------------
from fastapi import APIRouter

vercel_router = APIRouter()

@vercel_router.get("/vercel-health")
async def vercel_health():
    """
    Vercel health check endpoint — used during cold starts and build verification.
    """
    return {"status": "ok", "message": "GutBot running on Vercel 🚀"}

app.include_router(vercel_router)

# ✅ Expose FastAPI app for Vercel
app = app


