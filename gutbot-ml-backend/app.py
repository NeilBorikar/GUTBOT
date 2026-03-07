"""
app.py — GutBot Local Backend

Runs GutBot ML Chatbot + Frontend UI on localhost.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import uuid
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

# ---- Import your ML Chatbot Core ----
from public_health_chatbot import HealthChatbotService as HealthChatbot


# ===============================================================
# CONFIGURATION
# ===============================================================
HOST = "127.0.0.1"
PORT = 8000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("GutBot-Local")

# ===============================================================
# APP INITIALIZATION
# ===============================================================

app = FastAPI(
    title="GutBot Local Backend",
    description="Local ML Chatbot backend + static frontend.",
    version="2.0-local",
)

# Allow local frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local use only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static frontend files
# Mount the frontend folder for static assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "gutbot-frontend")
ASSETS_DIR = os.path.join(FRONTEND_DIR, "assets")

print("🧭 CWD:", os.getcwd())
print("📂 BASE_DIR:", BASE_DIR)
print("📂 PROJECT_ROOT:", PROJECT_ROOT)
print("📂 FRONTEND_DIR:", FRONTEND_DIR)
print("📂 ASSETS_DIR:", ASSETS_DIR)
print("✅ Exists:", os.path.exists(ASSETS_DIR), "Files:", os.listdir(ASSETS_DIR) if os.path.exists(ASSETS_DIR) else "Missing")

templates = Jinja2Templates(directory=FRONTEND_DIR)

# Serve everything from /assets → actual assets folder
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# Also serve /frontend (optional, for testing)
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")



# ===============================================================
# LIFECYCLE: Load Chatbot once on startup
# ===============================================================

@app.on_event("startup")
async def startup_event():
    """Load ML chatbot once."""
    app.state.start_time = time.time()
    logger.info("🧠 Initializing HealthChatbot... please wait.")
    t0 = time.time()
    try:
        app.state.chatbot = HealthChatbot()
        warmup_text = "hello"
        await run_in_threadpool(app.state.chatbot.process_message, "warmup", warmup_text)
        logger.info(f"✅ Chatbot loaded successfully in {(time.time()-t0):.2f}s")
    except Exception as e:
        logger.exception("❌ Failed to initialize chatbot: %s", e)
        raise

@app.get("/info")
async def info():
    return {
        "app": "GutBot Local Backend",
        "version": "2.0-local",
        "model_status": "loaded" if getattr(app.state, "chatbot", None) else "not loaded",
        "frontend": "/",
        "api": "/process"
    }


# ===============================================================
# ROUTES — FRONTEND PAGES
# ===============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the landing page (index.html)."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat-page", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Serve the chat interface page (chat.html)."""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/faq", response_class=HTMLResponse)
async def faq_page(request: Request):
    """Serve the FAQ page (faq.html)."""
    return templates.TemplateResponse("faq.html", {"request": request})

@app.get("/journal", response_class=HTMLResponse)
async def journal_page(request: Request):
    """Serve the Health Journal page."""
    return templates.TemplateResponse("journal.html", {"request": request})

@app.get("/bio-database", response_class=HTMLResponse)
async def bio_database(request: Request):
    """Serve the Bio-Database (personal health record) page."""
    return templates.TemplateResponse("bio_database.html", {"request": request})


@app.get("/static/your-logo.png")
async def get_logo():
    logo_path = os.path.join(ASSETS_DIR, "your-logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path)
    raise HTTPException(status_code=404, detail="Logo not found")

# ===============================================================
# API ENDPOINTS — Chat, Health, etc.
# ===============================================================

@app.get("/health")
async def health():
    """Health check."""
    uptime = time.time() - getattr(app.state, "start_time", time.time())
    return {"status": "ok", "uptime": f"{uptime:.2f}s"}

@app.post("/process")
async def process_chat(request: Request, background: BackgroundTasks):
    """
    Handle POST requests from frontend chat UI.
    Expects: {"message": "text", "session_id": "..."}
    """
    if not getattr(app.state, "chatbot", None):
        raise HTTPException(status_code=503, detail="Chatbot not initialized")

    data = await request.json()
    message = data.get("message", "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    t0 = time.time()
    try:
        raw = await run_in_threadpool(app.state.chatbot.process_message, session_id, message)
        elapsed = (time.time() - t0) * 1000

        # Log in background
        def _log():
            short = message[:100].replace("\n", " ")
            logger.info(f"[{session_id}] {short} | {elapsed:.1f}ms")
        background.add_task(_log)

        response_text = ""
        if isinstance(raw, dict):
            response_text = raw.get("response", "⚠️ No response generated.")
        else:
                response_text = str(raw)

        return JSONResponse({
                "response": response_text,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": round(elapsed, 2),
            })

        

    except Exception as e:
        logger.exception("Error during chat processing: %s", e)
        raise HTTPException(status_code=500, detail="Internal error: " + str(e))

# ===============================================================
# RUN LOCALLY
# ===============================================================
if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Starting GutBot Local at http://{HOST}:{PORT}")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
