"""
fallback_kb.py

Ultra-resilient fallback knowledge base for Public Health Chatbot.
Acts as a last line of defense when the main KB is missing, corrupted,
or when queries are vague.
"""

import logging
from typing import List, Dict

logger = logging.getLogger("FallbackKnowledgeBase")


class FallbackKnowledgeBase:
    """
    Lightweight, rule-based knowledge base.
    """

    def __init__(self):
        self.responses: Dict[str, str] = {
            "greeting": "Hello! I'm here to share general health information. How can I help you?",
            "thanks": "You're welcome! Stay safe and take care of your health.",
            "unknown": "I'm not sure I fully understand your question. Could you please rephrase?",
            "emergency": "⚠️ This sounds like an emergency. Please call your local emergency number immediately."
        }
        logger.info("FallbackKnowledgeBase initialized with default responses.")

    def search(self, intent: str, query: str = "") -> List[Dict[str, str]]:
        """
        Return a response for a given intent.
        """
        response = self.responses.get(intent, self.responses["unknown"])
        return [{"intent": intent, "response": response, "score": 1.0}]


if __name__ == "__main__":
    fb = FallbackKnowledgeBase()
    print(fb.search("greeting"))
    print(fb.search("unknown"))
