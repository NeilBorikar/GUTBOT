"""
public_health_chatbot.py
AI-Driven Public Health Chatbot for Disease Awareness

✅ Lightweight wrapper around Healthbot.py (the real ML brain)
✅ Handles session management + CLI interface
✅ No heavy ML or API imports here

Run:
    python public_health_chatbot.py
"""

import logging
import uuid
import io
import sys
from datetime import datetime
from pathlib import Path

from Healthbot import HealthChatbot as CoreBot  # full pipeline brain

# ==================== LOGGING ====================
LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(exist_ok=True)
file_handler = logging.FileHandler(LOGS_DIR / "chatbot.log", encoding="utf-8")
stream_handler = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
logger = logging.getLogger("PublicHealthCLI")

# ==================== WRAPPER APP ====================
class ChatApp:
    """
    Thin wrapper around the core Healthbot for sessions + CLI.
    """
    def __init__(self):
        self.core = CoreBot()  # initializes models/KB internally
        logger.info("✅ ChatApp wrapper initialized.")

    def process_message(self, user_id: str, text: str):
        # Core returns a dict with 'response', 'entities', 'intent', etc.
        return self.core.process_message(user_id, text)

    def get_session(self, user_id: str):
        return self.core.get_session(user_id)

# ==================== CLI INTERFACE ====================
def run_cli():
    bot = ChatApp()
    user_id = str(uuid.uuid4())[:8]

    print("\n" + "=" * 50)
    print(" Public Health Chatbot - Command Line Mode")
    print("=" * 50)
    print("Type 'quit', 'exit', or 'bye' to end")
    print("Type 'clear' to clear conversation history")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in {"quit", "exit", "bye"}:
                print("👋 Goodbye! Stay healthy!")
                break

            if user_input.lower() == "clear":
                session = bot.get_session(user_id)
                # Clear only the history; timestamps/ids remain intact
                session.conversation_history.clear()
                session.last_activity = datetime.utcnow()
                print("🧹 Conversation history cleared")
                continue

            if not user_input:
                continue

            resp = bot.process_message(user_id, user_input)
            # Only print the human-readable part
            print(f"🤖 Bot: {resp['response']}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"CLI error: {e}", exc_info=True)

if __name__ == "__main__":
    run_cli()
