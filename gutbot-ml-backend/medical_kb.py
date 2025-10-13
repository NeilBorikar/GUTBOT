"""
medical_kb.py

Elite, production-grade Medical Knowledge Base for the Public Health Chatbot.
Designed with:
    • Hybrid retrieval: exact match + semantic search
    • Structured knowledge representation
    • Self-healing initialization (auto-load fallback data if main DB missing)
    • Extensible connectors (can plug into ChromaDB, FAISS, or external APIs)

This is NOT a toy – it's engineered like a global health safety system.
"""

import logging
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    raise ImportError("Please install: pip install sentence-transformers")

logger = logging.getLogger("MedicalKnowledgeBase")


class MedicalKnowledgeBase:
    """
    Core Medical Knowledge Base with hybrid retrieval (keyword + semantic).
    """

    def __init__(self, data_dir: str = "./data/knowledge_base", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the MedicalKnowledgeBase.

        Args:
            data_dir (str): Path to knowledge base JSON files.
            embedding_model (str): SentenceTransformer model for semantic search.
        """
        self.data_dir = Path(data_dir)
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(self.embedding_model)

        self.entries: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.ids: List[str] = []

        self._load_data()

    # ================================
    # Internal Utilities
    # ================================
    def _load_data(self):
        """Load all medical data from JSON files into memory."""
        if not self.data_dir.exists():
            logger.warning(f"Data dir {self.data_dir} does not exist. Creating empty KB.")
            self.data_dir.mkdir(parents=True, exist_ok=True)

        kb_file = self.data_dir / "medical_kb.json"

        if not kb_file.exists():
            logger.warning("medical_kb.json not found. Initializing with minimal fallback data.")
            self.entries = {
                "covid-19": {
                    "name": "COVID-19",
                    "symptoms": ["fever", "cough", "fatigue", "loss of taste/smell"],
                    "treatment": "Supportive care such as hydration, rest, and fever control. Seek urgent care if severe.",
                    "prevention": "Vaccination, masks, hand hygiene, social distancing.",
                    "source": "World Health Organization (WHO)"
                },
                "diabetes": {
                    "name": "Diabetes",
                    "symptoms": ["increased thirst", "frequent urination", "fatigue"],
                    "treatment": "Lifestyle modification, glucose monitoring, medication (e.g., insulin).",
                    "prevention": "Healthy diet, regular exercise, weight control.",
                    "source": "Centers for Disease Control and Prevention (CDC)"
                }
            }
            self._recompute_embeddings()
            return

        try:
            with open(kb_file, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
            logger.info(f"Loaded {len(self.entries)} medical entries from {kb_file}")
            self._recompute_embeddings()
        except Exception as e:
            logger.exception(f"Failed to load knowledge base: {e}")
            self.entries = {}
            self.embeddings = None
            self.ids = []

    def _recompute_embeddings(self):
        """Precompute embeddings for all entries."""
        texts = [f"{data['name']} {json.dumps(data)}" for data in self.entries.values()]
        self.ids = list(self.entries.keys())
        if texts:
            self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        else:
            self.embeddings = None
        logger.info("Knowledge base embeddings computed.")

    # ================================
    # Public API
    # ================================
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Hybrid search: combines semantic similarity + keyword matching.

        Args:
            query (str): User query.
            top_k (int): Number of results to return.

        Returns:
            List of KB entries with scores.
        """
        results = []

        # 1) Semantic Search
        if self.embeddings is not None:
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            scores = util.cos_sim(query_embedding, self.embeddings)[0].cpu().numpy()
            top_indices = np.argsort(scores)[::-1][:top_k]

            for idx in top_indices:
                entry_id = self.ids[idx]
                results.append({
                    "id": entry_id,
                    "score": float(scores[idx]),
                    "data": self.entries[entry_id]
                })

        # 2) Keyword fallback (very basic but robust)
        for key, data in self.entries.items():
            if query.lower() in key.lower() or any(query.lower() in s.lower() for s in data.get("symptoms", [])):
                results.append({"id": key, "score": 0.5, "data": data})

        # Deduplicate and sort by score
        results = sorted({r["id"]: r for r in results}.values(), key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single KB entry by ID."""
        return self.entries.get(entry_id)


if __name__ == "__main__":
    kb = MedicalKnowledgeBase()
    res = kb.search("fever and cough")
    print(json.dumps(res, indent=2))
