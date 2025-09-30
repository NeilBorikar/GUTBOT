"""
public_health_chatbot.py

AI-Driven Public Health Chatbot for Disease Awareness
MAIN APPLICATION FILE - Complete implementation with ML integration
"""

# ==================== IMPORTS ====================
import os
import sys
import io
import re
import json
import torch
import torch.nn.functional as F
import time
import logging
import argparse
import asyncio
import uuid
from difflib import get_close_matches
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Deque, Union,Callable
from enum import Enum
from datetime import datetime
from collections import deque, defaultdict
from functools import lru_cache
from medical_kb import MedicalKnowledgeBase
from fallback_kb import FallbackKnowledgeBase
from Healthbot import RuleBasedIntentClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import unicodedata
import fnmatch
import heapq


# ==================== LOGGING SETUP ====================
LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(exist_ok=True)

file_handler = logging.FileHandler(LOGS_DIR / "chatbot.log", encoding="utf-8")
stream_handler = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
     handlers=[file_handler, stream_handler],
)
logger = logging.getLogger("HealthChatbot")

# ==================== CONDITIONAL IMPORTS ====================
# Initialize flags for optional dependencies
ml_available = False
api_available = False
custom_model_available = False

# Try to import ML libraries
try:
    import torch
    import torch.nn as nn
    ml_available = True
    logger.info(" PyTorch imported successfully")
except ImportError:
    logger.warning(" PyTorch not available. Install with: pip install torch torchvision")

try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    ml_available = True
    logger.info(" Transformers imported successfully")
except ImportError:
    logger.warning(" Transformers not available. Install with: pip install transformers")

try:
    from sentence_transformers import SentenceTransformer, util
    ml_available = True
    logger.info("SentenceTransformers imported successfully")
except ImportError:
    logger.warning(" SentenceTransformers not available. Install with: pip install sentence-transformers")

# Try to import API libraries
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    api_available = True
    logger.info("✅ FastAPI imported successfully")
except ImportError:
    logger.warning(" FastAPI not available. Install with: pip install fastapi uvicorn")

# Try to import the custom healthbot model
from Healthbot import CustomHealthBotModel
model_dir = Path(__file__).resolve().parents[0] / "models" / "intent_classifier"

chatbot = None
try:
    chatbot = CustomHealthBotModel.load(str(model_dir))
    if chatbot is None:
        raise RuntimeError("CustomHealthBotModel.load() returned None")
    logger.info("✅ CustomHealthBotModel loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load CustomHealthBotModel from {model_dir}: {e}")
    raise  # Stop execution instead of falling back silently

    # Create a mock class for type checking
class HealthBotCore:
        @staticmethod
        def load(model_path):
            return None
        def predict_intent(self, text):
            return "unknown", 0.5
        def extract_entities(self, text):
            return []
        def generate_response(self, intent, text):
            return "Fallback response"

# ==================== CONFIGURATION ====================
class Config:
    """Central configuration for the chatbot"""
    
    # Model settings
    INTENT_MODEL = "bert-base-uncased"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Knowledge base
    DATA_DIR = Path("./data")
    KB_DIR = DATA_DIR / "knowledge_base"
    
    # Conversation settings
    MAX_HISTORY = 10
    SESSION_TIMEOUT = 300  # 5 minutes in seconds
    
    # Safety settings
    MIN_CONFIDENCE_THRESHOLD = 0.6
    EMERGENCY_KEYWORDS = {
        "heart attack", "stroke", "suicide", "chest pain", 
        "difficulty breathing", "severe pain", "bleeding heavily",
        "choking", "unconscious", "not breathing"
    }
    
    # API settings
    HOST = "127.0.0.1"
    PORT = 8000
    VERSION = "v1"
    
    # Response templates
    RESPONSE_TEMPLATES = {
        "greeting": " Hello! I'm your health assistant. How can I help you today?",
        "thanks": "You're welcome! I'm glad I could help. Stay healthy! ",
        "emergency": " **EMERGENCY ALERT**: This sounds serious. Please call emergency services immediately or go to the nearest hospital. I'm a chatbot and cannot provide emergency care.",
        "fallback": "I'm not sure I understand. Could you please rephrase your question?",
        "disclaimer": "\n\n*Disclaimer: I am an AI assistant and not a medical professional. Always consult with a healthcare provider for medical advice.*"
    }

# Ensure directories exist
for d in [Config.DATA_DIR, Config.KB_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# ==================== DATA CLASSES ====================
class IntentType(str, Enum):
    GREETING = "greeting"
    DISEASE_INFO = "disease_info"
    SYMPTOM_QUERY = "symptom_query"
    PREVENTION = "prevention"
    TREATMENT = "treatment"
    MEDICATION = "medication"
    EMERGENCY = "emergency"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"
    THANKS = "thanks"
    UNKNOWN = "unknown"

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0

@dataclass
class Intent:
    type: IntentType
    confidence: float
    entities: List[Entity] = field(default_factory=list)

@dataclass
class ChatMessage:
    user_id: str
    text: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Session:
    user_id: str
    history: deque = field(default_factory=lambda: deque(maxlen=Config.MAX_HISTORY))
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    
    def is_expired(self) -> bool:
        return (datetime.utcnow() - self.last_active).total_seconds() > Config.SESSION_TIMEOUT
    
    def update_activity(self):
        self.last_active = datetime.utcnow()


class FallbackKnowledgeBase:
    """Fallback knowledge base for when the main knowledge base fails to load"""
    
    def __init__(self):
        self.fallback_data = {
            "diseases": {
                "covid": {
                    "name": "COVID-19",
                    "description": "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus.",
                    "symptoms": ["Fever", "Cough", "Shortness of breath", "Fatigue", "Loss of taste or smell"],
                    "prevention": ["Vaccination", "Mask-wearing", "Social distancing", "Hand hygiene"],
                    "treatments": ["Rest", "Hydration", "Over-the-counter fever reducers", "Medical care for severe cases"]
                },
                "influenza": {
                    "name": "Influenza (Flu)",
                    "description": "Influenza is a viral infection that attacks your respiratory system.",
                    "symptoms": ["Fever", "Muscle aches", "Chills and sweats", "Headache", "Dry cough"],
                    "prevention": ["Annual flu vaccine", "Hand washing", "Avoiding sick people"],
                    "treatments": ["Rest", "Fluids", "Antiviral medications if prescribed early"]
                }
            },
            "symptoms": {
                "fever": {
                    "name": "Fever",
                    "description": "A temporary increase in body temperature, often due to an illness.",
                    "common_causes": ["Infections", "Inflammatory conditions", "Medications"],
                    "when_to_seek_help": "Temperature above 103°F (39.4°C) or fever lasting more than 3 days"
                },
                "cough": {
                    "name": "Cough",
                    "description": "A reflex action to clear your airways of mucus and irritants.",
                    "common_causes": ["Common cold", "Flu", "COVID-19", "Allergies", "Asthma"],
                    "when_to_seek_help": "Cough lasting more than 3 weeks or accompanied by difficulty breathing"
                }
            }
        }
    
    def query(self, query: str) -> List[Dict]:
        """Simple query method for fallback knowledge base"""
        results = []
        query_lower = query.lower()
        
        # Search through all categories
        for category, items in self.fallback_data.items():
            for key, data in items.items():
                # Check if query matches any field in the data
                for field_value in data.values():
                    if isinstance(field_value, str) and query_lower in field_value.lower():
                        results.append({
                            "data": data,
                            "category": category,
                            "source": "fallback_knowledge_base",
                            "relevance_score": 0.7  # Medium confidence for fallback
                        })
                        break
                    elif isinstance(field_value, list):
                        for item in field_value:
                            if query_lower in item.lower():
                                results.append({
                                    "data": data,
                                    "category": category,
                                    "source": "fallback_knowledge_base",
                                    "relevance_score": 0.6  # Lower confidence for list matches
                                })
                                break
        
        return results
    
    def get_disease_info(self, disease_name: str) -> Optional[Dict]:
        """Get disease information from fallback data"""
        disease_key = disease_name.lower()
        if disease_key in self.fallback_data["diseases"]:
            return self.fallback_data["diseases"][disease_key]
        
        # Try fuzzy matching
        for key in self.fallback_data["diseases"].keys():
            if disease_name.lower() in key or key in disease_name.lower():
                return self.fallback_data["diseases"][key]
        
        return None
    
    def get_symptom_info(self, symptom_name: str) -> Optional[Dict]:
        """Get symptom information from fallback data"""
        symptom_key = symptom_name.lower()
        if symptom_key in self.fallback_data["symptoms"]:
            return self.fallback_data["symptoms"][symptom_key]
        
        # Try fuzzy matching
        for key in self.fallback_data["symptoms"].keys():
            if symptom_name.lower() in key or key in symptom_name.lower():
                return self.fallback_data["symptoms"][key]
        
        return None

class MedicalKnowledgeBase:
    """Enhanced medical knowledge base with advanced search capabilities"""
    
    def __init__(self, config):
        self.config = config
        self.kb = {
            "diseases": {}, 
            "symptoms": {}, 
            "prevention": {}, 
            "treatments": {},
            "medications": {}  # Added medications category
        }
        self._indices = {
            "symptom_to_diseases": defaultdict(set),
            "disease_to_prevention": defaultdict(list),
            "disease_to_treatments": defaultdict(list),
            "disease_to_medications": defaultdict(list),
            "keyword_index": defaultdict(set)  # Inverted index for full-text search
        }
        self._normalized_keys = {}  # Cache for normalized keys
        self._search_cache = lru_cache(maxsize=1000)  # Cache for search results
        
    def initialize(self, medical_data_path: Optional[Path] = None) -> bool:
        """
        Robust loader for medical_data JSON files with enhanced capabilities.
        Returns True if successful, False otherwise.
        """
        try:
            # Find medical_data directory with multiple fallbacks
            base_path = self._find_medical_data_path(medical_data_path)
            if not base_path or not base_path.exists():
                logger.warning(f"Medical data directory not found: {base_path}")
                return False
                
            self.medical_data_dir = base_path
            logger.info(f"Loading medical knowledge from: {self.medical_data_dir}")
            
            # Load all categories
            categories = {
                "diseases": "diseases",
                "symptoms": "symptoms", 
                "prevention": "prevention",
                "treatments": "treatments",
                "medications": "medications"  # New category
            }
            
            for category_key, folder_name in categories.items():
                self._load_category(base_path / folder_name, category_key)
            
            # Build indices
            self._build_indices()
            
            logger.info(
                f"Knowledge base loaded: {len(self.kb['diseases'])} diseases, "
                f"{len(self.kb['symptoms'])} symptoms, {len(self.kb['prevention'])} prevention items, "
                f"{len(self.kb['treatments'])} treatments, {len(self.kb['medications'])} medications."
            )
            return True
            
        except Exception as e:
            logger.exception(f"Failed to initialize knowledge base: {e}")
            return False
    
    def _find_medical_data_path(self, medical_data_path: Optional[Path]) -> Optional[Path]:
        """Find medical_data directory with multiple fallback strategies"""
        if medical_data_path:
            return Path(medical_data_path)
        
        # Try project root (assuming this file is in Backend/)
        project_root = Path(__file__).resolve().parents[1]
        candidate = project_root / "medical_data"
        if candidate.exists():
            return candidate
            
        # Try current working directory
        candidate = Path.cwd() / "medical_data"
        if candidate.exists():
            return candidate
            
        # Try config data directory
        candidate = Path(getattr(self.config, "DATA_DIR", Path("./medical_data")))
        if candidate.exists():
            return candidate
            
        return None
    
    def _load_category(self, dir_path: Path, category_key: str):
        """Load all JSON files from a category directory"""
        if not dir_path.exists():
            logger.warning(f"KB category folder not found: {dir_path}")
            return
            
        # Support both JSON and JSONL files
        patterns = ["*.json", "*.jsonl"]
        file_count = 0
        
        for pattern in patterns:
            for file_path in dir_path.glob(pattern):
                try:
                    if file_path.suffix == ".jsonl":
                        self._load_jsonl_file(file_path, category_key)
                    else:
                        self._load_json_file(file_path, category_key)
                    file_count += 1
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {file_count} files for category: {category_key}")
    
    def _load_json_file(self, file_path: Path, category_key: str):
        """Load a single JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Determine the primary key for this item
        primary_key = self._get_primary_key(data, category_key, file_path.stem)
        normalized_key = self._normalize_text(primary_key)
        
        # Store the data
        self.kb[category_key][normalized_key] = {
            "data": data,
            "source_file": str(file_path),
            "original_key": primary_key
        }
        
        # Cache the normalized key
        self._normalized_keys[primary_key] = normalized_key
    
    def _load_jsonl_file(self, file_path: Path, category_key: str):
        """Load a JSONL file (each line is a JSON object)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    primary_key = self._get_primary_key(data, category_key, f"{file_path.stem}_line{line_num}")
                    normalized_key = self._normalize_text(primary_key)
                    
                    self.kb[category_key][normalized_key] = {
                        "data": data,
                        "source_file": f"{file_path}:{line_num}",
                        "original_key": primary_key
                    }
                    
                    self._normalized_keys[primary_key] = normalized_key
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON at line {line_num} in {file_path}: {e}")
    
    def _get_primary_key(self, data: Dict, category_key: str, fallback: str) -> str:
        """Extract the primary key from a data object"""
        key_fields = {
            "diseases": ["name", "disease", "title"],
            "symptoms": ["name", "symptom", "title"],
            "prevention": ["name", "title", "prevention_method"],
            "treatments": ["name", "title", "treatment", "medication"],
            "medications": ["name", "title", "drug", "medication"]
        }
        
        for field in key_fields.get(category_key, ["name", "title"]):
            if field in data and data[field]:
                return str(data[field])
                
        return fallback
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent indexing"""
        if not isinstance(text, str):
            return ""
            
        # Unicode normalization
        text = unicodedata.normalize("NFKD", text)
        
        # Convert to lowercase and remove special characters
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        
        return text
    
    def _build_indices(self):
        """Build all search indices from the loaded knowledge base"""
        logger.info("Building search indices...")
        
        # Build symptom-to-disease index
        for disease_key, disease_data in self.kb["diseases"].items():
            symptoms = disease_data["data"].get("symptoms", [])
            for symptom in symptoms:
                normalized_symptom = self._normalize_text(symptom)
                self._indices["symptom_to_diseases"][normalized_symptom].add(disease_key)
        
        # Build disease-to-prevention index
        for prevention_key, prevention_data in self.kb["prevention"].items():
            diseases = prevention_data["data"].get("effective_against", [])
            for disease in diseases:
                normalized_disease = self._normalize_text(disease)
                self._indices["disease_to_prevention"][normalized_disease].append(prevention_data["data"])
        
        # Build disease-to-treatments index
        for treatment_key, treatment_data in self.kb["treatments"].items():
            diseases = treatment_data["data"].get("used_for", [])
            for disease in diseases:
                normalized_disease = self._normalize_text(disease)
                self._indices["disease_to_treatments"][normalized_disease].append(treatment_data["data"])
        
        # Build disease-to-medications index
        for medication_key, medication_data in self.kb["medications"].items():
            diseases = medication_data["data"].get("treats", [])
            for disease in diseases:
                normalized_disease = self._normalize_text(disease)
                self._indices["disease_to_medications"][normalized_disease].append(medication_data["data"])
        
        # Build full-text keyword index
        for category, items in self.kb.items():
            for key, data in items.items():
                # Index all text fields
                text_to_index = self._extract_indexable_text(data["data"])
                for word in text_to_index.split():
                    normalized_word = self._normalize_text(word)
                    if len(normalized_word) > 2:  # Skip very short words
                        self._indices["keyword_index"][normalized_word].add((category, key))
        
        logger.info("Search indices built successfully")
    
    def _extract_indexable_text(self, data: Any) -> str:
        """Extract all text content from a data structure for indexing"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            return " ".join(str(v) for v in data.values() if v and isinstance(v, (str, int, float)))
        elif isinstance(data, list):
            return " ".join(self._extract_indexable_text(item) for item in data)
        else:
            return str(data)
    
    def search(self, query: str, category: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Advanced search across the knowledge base
        Returns relevant items with relevance scores
        """
        normalized_query = self._normalize_text(query)
        query_terms = normalized_query.split()
        
        # Score results based on term frequency
        results = defaultdict(float)
        
        for term in query_terms:
            if term in self._indices["keyword_index"]:
                for category_key, item_key in self._indices["keyword_index"][term]:
                    if category and category != category_key:
                        continue
                    # Boost score for matches in title/name fields
                    item_data = self.kb[category_key][item_key]
                    if "name" in item_data["data"] and term in self._normalize_text(item_data["data"]["name"]):
                        results[(category_key, item_key)] += 2.0
                    else:
                        results[(category_key, item_key)] += 1.0
        
        # Get top results
        top_results = heapq.nlargest(limit, results.items(), key=lambda x: x[1])
        
        # Format results
        formatted_results = []
        for (category_key, item_key), score in top_results:
            item_data = self.kb[category_key][item_key].copy()
            item_data["relevance_score"] = score / len(query_terms)  # Normalize score
            item_data["category"] = category_key
            formatted_results.append(item_data)
        
        return formatted_results
    
    def get_disease_info(self, name: str) -> Optional[Dict]:
        """Get disease information with fuzzy matching"""
        return self._get_item("diseases", name)
    
    def get_symptom_info(self, name: str) -> Optional[Dict]:
        """Get symptom information with fuzzy matching"""
        return self._get_item("symptoms", name)
    
    def get_related_diseases(self, symptom: str) -> List[Dict]:
        """Get diseases related to a symptom"""
        normalized_symptom = self._normalize_text(symptom)
        disease_keys = self._indices["symptom_to_diseases"].get(normalized_symptom, set())
        
        # Try fuzzy matching if no exact match
        if not disease_keys:
            closest_match = self._find_closest_key(
                list(self._indices["symptom_to_diseases"].keys()), 
                normalized_symptom
            )
            if closest_match:
                disease_keys = self._indices["symptom_to_diseases"][closest_match]
        
        return [self.kb["diseases"][key]["data"] for key in disease_keys]
    
    def _get_item(self, category: str, name: str) -> Optional[Dict]:
        """Get an item from a category with fuzzy matching"""
        if not name:
            return None
            
        normalized_name = self._normalize_text(name)
        
        # Exact match
        if normalized_name in self.kb[category]:
            return self.kb[category][normalized_name]["data"]
        
        # Fuzzy match
        closest_key = self._find_closest_key(list(self.kb[category].keys()), normalized_name)
        if closest_key:
            return self.kb[category][closest_key]["data"]
        
        return None
    
    def _find_closest_key(self, keys: List[str], target: str, cutoff: float = 0.6) -> Optional[str]:
        """Find the closest matching key using fuzzy matching"""
        if not keys or not target:
            return None
            
        matches = get_close_matches(target, keys, n=1, cutoff=cutoff)
        return matches[0] if matches else None

# --- Integration with HealthChatbot class ---

# In the HealthChatbot class, replace the initialize_knowledge_base method:



# ==================== CORE CHATBOT CLASS ====================
import logging
from pathlib import Path
import torch
import spacy  # keep here instead of importing inside method
from Healthbot import CustomHealthBotModel as HealthBotModel
  # adjust import path if needed

logger = logging.getLogger("Healthbot")


# ==================== WRAPPER CHATBOT ====================

class HealthChatbot:
    """
    Thin wrapper around HealthBotModel to handle sessions, logging,
    and user-facing interfaces (CLI/API).
    """

    def __init__(self):
        # Initialize the real brain
        self.model = HealthBotModel()
        self.model.sessions = {}  # Session storage

        # Initialize knowledge base
        self.model.initialize_knowledge_base()

        logger.info("HealthChatbot initialized with HealthBotModel")

    def process_message(self, user_id: str, text: str) -> str:
        """
        Delegate message processing to HealthBotModel.
        """
        return self.model.process_message(user_id, text)

    def get_session(self, user_id: str):
        """
        Access session via HealthBotModel.
        """
        return self.model.get_session(user_id)


class HealthBotModel:
    def __init__(self):
        logger.info("Initializing HealthBotModel...")

    # Track ML model state
        self.ml_models_loaded = False
        self.custom_model = None   # <--- THIS was missing

        # Knowledge bases
        self.medical_kb = None
        self.fallback_kb = None

         # Rule-based intent classifier
        self.rule_based_classifier = RuleBasedIntentClassifier()

        # Load NLP / ML models
        self.initialize_models()
        self.initialize_knowledge_base()

        # Initialize user sessions
        self.sessions: Dict[str, Session] = {}

        logger.info("HealthBotModel initialized successfully!")
    
    def initialize_models(self):
        """Initialize ML/NLP models"""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy NLP model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
            self.nlp = None

        # Load your custom HealthBot ML model
        try:
            from Healthbot import CustomHealthBotModel as CustomMLModel
 # import your ML wrapper

            model_dir = Path(__file__).resolve().parents[0] / "models" / "intent_classifier"

            if (model_dir / "model.pt").exists():
                self.custom_model = CustomMLModel.load(str(model_dir))
                if self.custom_model:
                    self.ml_models_loaded = True
                    logger.info("Custom ML model (HealthBotModel) loaded successfully")
                else:
                    logger.warning("Custom ML model could not be loaded")
                    self.custom_model = None
                    self.ml_models_loaded = False
            else:
                logger.warning(f"No model file found at {model_dir}")
                self.custom_model = None
                self.ml_models_loaded = False

        except Exception as e:
            logger.exception(f"Failed to load custom ML model: {e}")
            self.custom_model = None
            self.ml_models_loaded = False



    def predict_intent(self, text: str) -> Intent:
        # Dummy logic if no real model is trained
        if not self.model:
            return Intent(IntentType.UNKNOWN, 0.5)
        # If real model exists, implement inference here
        return Intent(IntentType.UNKNOWN, 0.5)

    def extract_entities(self, text: str):
        # Dummy fallback if no real model
        return []

    @staticmethod
    def load(model_path: str):
        """
        Robust loader:
          - Accepts either a directory (expected files: model.pt, tokenizer.pt) or direct file path.
          - Returns a HealthBotModel instance with real objects, or None if loading failed.
        """
        try:
            p = Path(model_path)
            # If a direct file to a torch model is passed
            if p.is_file():
                model = torch.load(str(p), map_location="cpu")
                tokenizer = None
            else:
                model_file = p / "model.pt"
                tokenizer_file = p / "tokenizer.pt"
                if model_file.exists():
                    model = torch.load(str(model_file), map_location="cpu")
                else:
                    logger.warning("Model file not found at %s", model_file)
                    model = None

                if tokenizer_file.exists():
                    tokenizer = torch.load(str(tokenizer_file))
                else:
                    logger.warning("Tokenizer file not found at %s", tokenizer_file)
                    tokenizer = None

            if model is None and tokenizer is None:
                logger.warning("No model or tokenizer loaded from %s", model_path)
                return None

            return HealthBotModel(model, tokenizer)
        except Exception as e:
            logger.exception("Error loading custom model from %s: %s", model_path, e)
            return None


    def initialize_knowledge_base(self):
        """Initialize the enhanced medical knowledge base"""
        try:
            self.medical_kb = MedicalKnowledgeBase(Config)
            success = self.medical_kb.initialize()
        
            if not success:
                logger.warning("Knowledge base initialization failed, using fallback")
                self.fallback_kb = FallbackKnowledgeBase()
            else:
                logger.info("Enhanced medical knowledge base initialized successfully")
                self.fallback_kb = None
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            self.fallback_kb = FallbackKnowledgeBase()

    
    def initialize_faiss_index(self):
        """Load FAISS medical index if available"""
        try:
            index_dir = Path(__file__).resolve().parents[0] / "models" / "medical_index"
            index_path = index_dir / "medical.index"
            metadata_path = index_dir / "metadata.json"

            if index_path.exists() and metadata_path.exists():
                self.faiss_index = faiss.read_index(str(index_path))
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.faiss_metadata = json.load(f)
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("FAISS medical index loaded successfully")
            else:
                self.faiss_index = None
                self.faiss_metadata = []
                self.embedder = None
                logger.warning("No FAISS medical index found, fallback only")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.faiss_index, self.faiss_metadata, self.embedder = None, [], None

    def search_faiss(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search medical FAISS index"""
        if not self.faiss_index or not self.embedder:
            return []

        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.faiss_index.search(query_emb, top_k)

        results = []
        for idx in I[0]:
            if idx < len(self.faiss_metadata):
                results.append(self.faiss_metadata[idx])
        return results



    def retrieve_medical_knowledge(self, query: str, category: Optional[str] = None) -> List[Dict]:
        """Search the medical knowledge base with fallback support"""
        if self.medical_kb is not None:
            return self.medical_kb.search(query, category)
        elif self.fallback_kb is not None:
            return self.fallback_kb.query(query)
        else:
            logger.error("No knowledge base available")
            return []

    def get_disease_information(self, disease_name: str) -> Optional[Dict]:
        """Get detailed information about a disease with fallback support"""
        if self.medical_kb is not None:
            result = self.medical_kb.search(disease_name, category="diseases")
            if result:
                return result[0]
        elif self.fallback_kb is not None:
            return self.fallback_kb.get_disease_info(disease_name)
    
        logger.error("No knowledge base available")
        return None


    def get_symptom_information(self, symptom_name: str) -> Optional[Dict]:
        """Get detailed information about a symptom with fallback support"""
        if self.medical_kb is not None:
            # Search symptom knowledge base
            result = self.medical_kb.search(symptom_name, category="symptoms")
            if result:
                return result[0]  # return best match
        elif self.fallback_kb is not None:
            return self.fallback_kb.get_symptom_info(symptom_name)
    
        logger.error("No knowledge base available")
        return None
    
    def get_treatment_information(self, treatment_name: str) -> Optional[Dict]:
        """Get detailed information about a treatment with fallback support"""
        if self.medical_kb is not None:
            result = self.medical_kb.search(treatment_name, category="treatments")
            if result:
                return result[0]
        elif self.fallback_kb is not None:
            return self.fallback_kb.get_treatment_info(treatment_name)
    
        logger.error("No knowledge base available")
        return None


    def get_prevention_information(self, prevention_name: str) -> Optional[Dict]:
        """Get detailed information about a prevention method with fallback support"""
        if self.medical_kb is not None:
            result = self.medical_kb.search(prevention_name, category="prevention")
            if result:
                return result[0]
        elif self.fallback_kb is not None:
            return self.fallback_kb.get_prevention_info(prevention_name)
    
        logger.error("No knowledge base available")
        return None


    def get_medication_information(self, medication_name: str) -> Optional[Dict]:
        """Get detailed information about a medication with fallback support"""
        if self.medical_kb is not None:
            result = self.medical_kb.search(medication_name, category="medications")
            if result:
                return result[0]
        elif self.fallback_kb is not None:
            return self.fallback_kb.get_medication_info(medication_name)
    
        logger.error("No knowledge base available")
        return None



    def get_related_diseases(self, symptom: str) -> List[Dict]:
        """Get diseases related to a symptom with fallback support"""
        if self.medical_kb is not None:
            return self.medical_kb.get_related_diseases(symptom)
        else:
            # Fallback: return empty list as we don't have this mapping in fallback KB
            return []
    
    def get_session(self, user_id: str) -> Session:
        """Get or create a user session"""
        if user_id in self.sessions:
            session = self.sessions[user_id]
            if session.is_expired():
                logger.info(f"Session expired for user {user_id}, creating new session")
                self.sessions[user_id] = Session(user_id)
            else:
                session.update_activity()
        else:
            self.sessions[user_id] = Session(user_id)
            logger.info(f"Created new session for user {user_id}")
        
        return self.sessions[user_id]
    
    def detect_intent(self, text: str) -> Intent:
        """Detect intent with: emergency -> KB mention -> custom model -> rule-based fallback."""
        if not text or not text.strip():
            return Intent(IntentType.UNKNOWN, 0.0)

        text_lower = text.lower()

        # 1) Emergency keyword check (use word boundary)
        for keyword in Config.EMERGENCY_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                return Intent(IntentType.EMERGENCY, 1.0)

        # 2) KB mention checks (prefer explicit KB matches to avoid false positives)
        try:
            if self.medical_kb:
                # if the KB returns any result for the text in a category, assume that intent
             kb_checks = [
                    ("diseases", IntentType.DISEASE_INFO),
                    ("symptoms", IntentType.SYMPTOM_QUERY),
                    ("medications", IntentType.MEDICATION),
                    ("treatments", IntentType.TREATMENT),
                    ("prevention", IntentType.PREVENTION),
                ]
            for cat, mapped_intent in kb_checks:
                    res = self.medical_kb.search(text, category=cat, limit=1)
                    if res:
                    # high confidence when KB finds an explicit match
                        return Intent(mapped_intent, 0.95)
        except Exception as e:
            logger.debug(f"KB pre-check failed: {e}")

        # 3) If you have a custom model, ask it
        if self.custom_model:
            try:
                intent_type, confidence = self.custom_model.predict_intent(text)
                try:
                    mapped_type = IntentType(intent_type)
                except ValueError:
                    logger.warning(f"Unknown intent type from custom model: {intent_type}")
                    mapped_type = IntentType.UNKNOWN
                return Intent(type=mapped_type, confidence=confidence)
            except Exception as e:
                logger.error(f"Custom model prediction error: {e}")

        # 4) Fall back to rule-based detection
        return self.rule_based_intent_detection(text)


    def rule_based_intent_detection(self, text: str) -> Intent:
        """Rule-based detection using word-boundary matching to avoid substring false-positives."""
        text_lower = text.lower().strip()

        def word_match(words):
            # build pattern like r'\b(?:word1|word2|word3)\b'
            if not words:
                return False
            pattern = r'\b(?:' + '|'.join(re.escape(w) for w in words) + r')\b'
            return bool(re.search(pattern, text_lower))

        # Thanks
        if word_match(["thank", "thanks", "appreciate", "grateful"]):
            return Intent(IntentType.THANKS, 0.85)

        # Symptom detection (use words; avoid substring)
        if word_match(["symptom", "symptoms", "feel", "pain", "hurt", "ache",
                    "fever", "cough", "headache", "nausea", "sore throat", "fatigue"]):
            return Intent(IntentType.SYMPTOM_QUERY, 0.80)

        # Disease name detection via common disease keywords
        if word_match(["covid", "flu", "malaria", "diabetes", "asthma", "cancer", "tb"]):
            return Intent(IntentType.DISEASE_INFO, 0.80)

        # Prevention detection
        if word_match(["prevent", "avoid", "protection", "safe", "mask", "hygiene", "wash hands"]):
            return Intent(IntentType.PREVENTION, 0.75)

        # Treatment
        if word_match(["treat", "cure", "medicine", "medication", "therapy", "treatment"]):
            return Intent(IntentType.TREATMENT, 0.75)

        # Medication intent detection
        if word_match(["medication", "drug", "pill", "tablet", "dose", "prescription", "aspirin", "ibuprofen", "amoxicillin"]):
            return Intent(IntentType.MEDICATION, 0.75)

        # Greeting detection — run **after** other checks to avoid false positives
        if word_match(["hello", "hi", "hey", "greetings"]):
            return Intent(IntentType.GREETING, 0.90)

        # Fallback: use the regex-based classifier (RuleBasedIntentClassifier)
        return self.rule_based_classifier.predict(text)


    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using custom model or spaCy"""
        entities = []

        # If custom ML model exists
        if self.custom_model and hasattr(self.custom_model, "extract_entities"):
            try:
                raw_entities = self.custom_model.extract_entities(text)
                return [
                    Entity(
                        text=e.get("text", ""),
                        label=e.get("label", "UNKNOWN"),
                        start=e.get("start", -1),
                        end=e.get("end", -1),
                        confidence=e.get("confidence", 0.5)
                    )
                    for e in raw_entities
                ]
            except Exception as e:
                logger.error(f"Custom model entity extraction failed: {e}")

        # Fallback: spaCy
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append(
                    Entity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.8
                    )
                )

        return entities

    
    def generate_response(self, intent: Intent, user_input: str, user_id: str, entities: List[Entity] = None) -> str:
        """Generate appropriate response based on intent using knowledge base"""
        session = self.get_session(user_id)

        # Handle emergency case first
        if intent.type == IntentType.EMERGENCY:
            return Config.RESPONSE_TEMPLATES["emergency"]

        # Extract entities if not provided
        if entities is None:
            entities = self.extract_entities(user_input)

        # If custom model exists and confident, prefer it (keeps prior behavior)
        if (self.custom_model and intent.confidence >= Config.MIN_CONFIDENCE_THRESHOLD):
            try:
                response = self.custom_model.generate_response(intent.type.value, user_input)
                return response + Config.RESPONSE_TEMPLATES["disclaimer"]
            except Exception as e:
                logger.error(f"Custom model response generation error: {str(e)}")

        # Dispatch to the appropriate generator
        if intent.type == IntentType.GREETING:
            response = Config.RESPONSE_TEMPLATES["greeting"]

        elif intent.type == IntentType.THANKS:
            response = Config.RESPONSE_TEMPLATES["thanks"]

        elif intent.type == IntentType.SYMPTOM_QUERY:
            response = self._generate_symptom_response(entities, user_input)

        elif intent.type == IntentType.DISEASE_INFO:
            response = self._generate_disease_response(entities, user_input)

        elif intent.type == IntentType.PREVENTION:
            response = self._generate_prevention_response(entities, user_input)

        elif intent.type == IntentType.TREATMENT:
            response = self._generate_treatment_response(entities, user_input)

        elif intent.type == IntentType.MEDICATION:
            response = self._generate_medication_response(entities, user_input)

        else:
            response = Config.RESPONSE_TEMPLATES["fallback"]

        # Add disclaimer to all responses except emergency
        if intent.type != IntentType.EMERGENCY:
            response += Config.RESPONSE_TEMPLATES["disclaimer"]

        return response


# ---------- Helper utilities used by the generators ----------
    def _resolve_kb(self):
        """Return the active KB object (primary or fallback)."""
        return self.medical_kb if self.medical_kb is not None else self.fallback_kb


    def _unwrap_kb_item(self, item: Optional[Dict]) -> Optional[Dict]:
        """
        KB items might be stored as {'data': {...}, 'source_file': ...} or as raw dict.
        Normalize to the inner dict or return None.
        """
        if item is None:
            return None
        if isinstance(item, dict) and "data" in item and isinstance(item["data"], dict):
            return item["data"]
        return item if isinstance(item, dict) else None


    def _format_disease_summary(self, disease_data: Dict, limit_fields: int = 5) -> str:
        """Create a concise multi-section summary for a disease dict."""
        if not disease_data:
            return ""
        parts = []
        name = disease_data.get("name") or disease_data.get("title") or disease_data.get("disease") or "This disease"
        parts.append(f"=== {name} ===")
        if "description" in disease_data and disease_data["description"]:
            parts.append(disease_data["description"].strip())
        if "symptoms" in disease_data and disease_data["symptoms"]:
            parts.append("Symptoms: " + ", ".join(str(s) for s in disease_data["symptoms"][:limit_fields]))
        if "prevention" in disease_data and disease_data["prevention"]:
            parts.append("Prevention: " + ", ".join(str(p) for p in disease_data["prevention"][:limit_fields]))
        # treatments may be list or string
        if "treatments" in disease_data and disease_data["treatments"]:
            if isinstance(disease_data["treatments"], (list, tuple)):
                parts.append("Treatments: " + ", ".join(str(t) for t in disease_data["treatments"][:limit_fields]))
            else:
                parts.append("Treatments: " + str(disease_data["treatments"]))
        if "medications" in disease_data and disease_data["medications"]:
            parts.append("Medications: " + ", ".join(str(m) for m in disease_data["medications"][:limit_fields]))
        # optional source
        if "source" in disease_data:
            parts.append(f"Source: {disease_data.get('source')}")
        return "\n\n".join(parts) + "\n\n"


# ---------- Disease response ----------
    def _generate_disease_response(self, entities: List[Entity], user_input: str) -> str:
        """If disease(s) mentioned → return aggregated information from all KB categories."""
        kb = self._resolve_kb()

        # Try to find disease name from entities first
        disease_entities = [e for e in entities if self._map_entity_label(e.label) == "disease"]
        candidate_names = [e.text for e in disease_entities]

        # If none, attempt to search KB directly using the whole user_input
        if not candidate_names:
            # Try exact KB lookup
            try:
                if hasattr(kb, "get_disease_info"):
                    info = kb.get_disease_info(user_input)
                    if info:
                        candidate_names = [info.get("name", user_input)]
                    else:
                        # try full-text search
                        results = kb.search(user_input, category="diseases", limit=3)
                        candidate_names = [self._unwrap_kb_item(r).get("name") for r in results if self._unwrap_kb_item(r) and self._unwrap_kb_item(r).get("name")]
            except Exception:
                candidate_names = []

        if not candidate_names:
            return "I can provide information about diseases, but I couldn't identify a specific disease in your message. Which disease would you like to know about?"

        # Limit number of diseases returned (to keep answers readable)
        MAX_DISEASES = 3
        candidate_names = candidate_names[:MAX_DISEASES]

        response_chunks = []
        for name in candidate_names:
            # Use KB function for a consistent result
            disease_info = None
            try:
                if hasattr(kb, "get_disease_info"):
                    disease_info = kb.get_disease_info(name)
            except Exception:
                disease_info = None

            # If KB returned a 'search' style item, unwrap
            disease_info = self._unwrap_kb_item(disease_info)

            # If not found, try a broader search
            if not disease_info:
                try:
                    search_res = kb.search(name, category="diseases", limit=1)
                    disease_info = self._unwrap_kb_item(search_res[0]) if search_res else None
                except Exception:
                    disease_info = None

            # Build aggregated content
            if disease_info:
                response_chunks.append(self._format_disease_summary(disease_info))
            else:
                response_chunks.append(f"I don't have information about '{name}' in the knowledge base.")

        # Combine and return
        return "\n".join(response_chunks).strip()


    # ---------- Symptom response ----------
    def _generate_symptom_response(self, entities: List[Entity], user_input: str) -> str:
        """
        If a symptom is mentioned:
        1) show symptom description (if any),
        2) list likely related diseases (top N),
        3) for each disease return aggregated disease info (as in disease response).
        """
        kb = self._resolve_kb()

        symptom_entities = [e for e in entities if self._map_entity_label(e.label) == "symptom"]
        candidate_symptoms = [e.text for e in symptom_entities]

        # If none from entities, try searching KB for symptoms using user_input
        if not candidate_symptoms:
            try:
                if hasattr(kb, "get_symptom_info"):
                    s_info = kb.get_symptom_info(user_input)
                    if s_info:
                        candidate_symptoms = [s_info.get("name", user_input)]
                    else:
                        # Attempt fulltext search for symptom terms
                        search_res = kb.search(user_input, category="symptoms", limit=2)
                        candidate_symptoms = [self._unwrap_kb_item(r).get("name") for r in search_res if self._unwrap_kb_item(r) and self._unwrap_kb_item(r).get("name")]
            except Exception:
                candidate_symptoms = []

        if not candidate_symptoms:
            return "I can help with symptoms — could you tell me the symptom (e.g. 'fever', 'cough') so I can list possible causes and related disease information?"

        # We'll handle up to 2 symptoms per request to keep answers readable
        candidate_symptoms = candidate_symptoms[:2]
        out_parts = []

        for symptom in candidate_symptoms:
            # symptom-level info if available
            symptom_info = None
            try:
                if hasattr(kb, "get_symptom_info"):
                    symptom_info = kb.get_symptom_info(symptom)
            except Exception:
                symptom_info = None
            symptom_info = self._unwrap_kb_item(symptom_info)

            header = f"Information on symptom: {symptom}"
            out_parts.append(header)

            if symptom_info:
                if "description" in symptom_info:
                    out_parts.append(symptom_info["description"])
                if "common_causes" in symptom_info:
                    out_parts.append("Common causes: " + ", ".join(symptom_info.get("common_causes", [])[:5]))
                if "when_to_seek_help" in symptom_info:
                    out_parts.append("When to seek help: " + str(symptom_info["when_to_seek_help"]))
            else:
                out_parts.append("No standalone symptom metadata found in KB.")

            # Get related diseases (best-effort)
            related = []
            try:
                if hasattr(kb, "get_related_diseases"):
                    related = kb.get_related_diseases(symptom) or []
            except Exception:
                related = []

            # If KB returns wrapped dicts, unwrap
            related_unwrapped = []
            for r in related:
                item = self._unwrap_kb_item(r)
                if item and item.get("name"):
                    related_unwrapped.append(item)
                elif isinstance(r, dict) and r.get("name"):
                    related_unwrapped.append(r)

            if not related_unwrapped:
                out_parts.append("I couldn't find associated diseases for this symptom in the knowledge base.")
                continue

            # Limit to top 3 diseases
            MAX = 5
            out_parts.append("Possible related conditions (top results): " + ", ".join(d.get("name", "unknown") for d in related_unwrapped[:MAX]))
            out_parts.append("Details on those conditions:\n")

            # For each disease, append formatted disease summary
            for d in related_unwrapped[:MAX]:
                disease_name = d.get("name") or d.get("title") or d.get("disease")
                if not disease_name:
                    continue
                disease_full = None
                try:
                    disease_full = kb.get_disease_info(disease_name) if hasattr(kb, "get_disease_info") else None
                except Exception:
                    disease_full = None
                disease_full = self._unwrap_kb_item(disease_full) or d
                out_parts.append(self._format_disease_summary(disease_full))

        return "\n".join(out_parts).strip()


# ---------- Prevention response ----------
    def _generate_prevention_response(self, entities: List[Entity], user_input: str) -> str:
        """If a prevention method is mentioned, show which disease(s) it prevents and then disease summaries."""
        kb = self._resolve_kb()

        prevention_entities = [e for e in entities if self._map_entity_label(e.label) == "prevention"]
        candidate_preventions = [e.text for e in prevention_entities]

        if not candidate_preventions:
            # Try to search KB for prevention by user_input
            try:
                search_res = kb.search(user_input, category="prevention", limit=3)
                candidate_preventions = [self._unwrap_kb_item(r).get("name") for r in search_res if self._unwrap_kb_item(r) and self._unwrap_kb_item(r).get("name")]
            except Exception:
                candidate_preventions = []

        if not candidate_preventions:
            return "Which prevention method are you asking about? (e.g. 'vaccination', 'hand hygiene', 'mask wearing')"

        # Limit
        candidate_preventions = candidate_preventions[:2]
        out = []

        for p in candidate_preventions:
            # Get prevention entry
            prevention_entry = None
            try:
                # Many KB entries use 'get_prevention_information'
                if hasattr(kb, "get_prevention_information"):
                    prevention_entry = kb.get_prevention_information(p)
            except Exception:
                prevention_entry = None

            prevention_entry = self._unwrap_kb_item(prevention_entry)

            if not prevention_entry:
                # fallback: search
                try:
                    res = kb.search(p, category="prevention", limit=1)
                    prevention_entry = self._unwrap_kb_item(res[0]) if res else None
                except Exception:
                    prevention_entry = None

            out.append(f"Prevention: {p}")
            if prevention_entry:
                if "description" in prevention_entry:
                    out.append(prevention_entry["description"])
                # try to discover diseases this prevention acts against
                diseases_list = prevention_entry.get("effective_against") or prevention_entry.get("prevents") or prevention_entry.get("targets") or []
                if diseases_list:
                    out.append("This prevention is effective against: " + ", ".join(diseases_list[:10]))
                    # Add disease summaries for up to 3 of them
                    for disease_name in diseases_list[:3]:
                        dinfo = None
                        try:
                            dinfo = kb.get_disease_info(disease_name)
                        except Exception:
                            dinfo = None
                        dinfo = self._unwrap_kb_item(dinfo)
                        if dinfo:
                            out.append(self._format_disease_summary(dinfo))
                else:
                    # if no explicit mapping, try a reverse search: find diseases mentioning this prevention in their 'prevention' field
                    reverse_hits = []
                    try:
                        # Use a KB search for the prevention string across diseases
                        reverse_hits = kb.search(p, category="diseases", limit=5)
                    except Exception:
                        reverse_hits = []
                    if reverse_hits:
                        names = [self._unwrap_kb_item(r).get("name") for r in reverse_hits if self._unwrap_kb_item(r) and self._unwrap_kb_item(r).get("name")]
                        out.append("Likely prevented conditions (by KB mentions): " + ", ".join(names[:5]))
                        for r in reverse_hits[:3]:
                            out.append(self._format_disease_summary(self._unwrap_kb_item(r)))
                    else:
                        out.append("No explicit disease mappings found for this prevention in the knowledge base.")
            else:
                out.append("I couldn't find a knowledge-base entry for that prevention method.")

        return "\n".join(out).strip()


# ---------- Treatment response ----------
    def _generate_treatment_response(self, entities: List[Entity], user_input: str) -> str:
        """If treatment mentioned → say which diseases it's used for, then include disease info for those diseases."""
        kb = self._resolve_kb()

        treatment_entities = [e for e in entities if self._map_entity_label(e.label) == "treatment"]
        candidate_treatments = [e.text for e in treatment_entities]

        if not candidate_treatments:
            # fallback search
            try:
                res = kb.search(user_input, category="treatments", limit=3)
                candidate_treatments = [self._unwrap_kb_item(r).get("name") for r in res if self._unwrap_kb_item(r) and self._unwrap_kb_item(r).get("name")]
            except Exception:
                candidate_treatments = []

        if not candidate_treatments:
            return "Which treatment are you asking about? (e.g. 'antiviral therapy', 'radiation', 'surgery')"

        out = []
        for t in candidate_treatments[:2]:
            # try to get treatment entry
            treat_entry = None
            try:
                res = kb.search(t, category="treatments", limit=1)
                treat_entry = self._unwrap_kb_item(res[0]) if res else None
            except Exception:
                treat_entry = None

            out.append(f"Treatment: {t}")
            if treat_entry:
                if "description" in treat_entry:
                    out.append(treat_entry["description"])
                # common fields that indicate which diseases it's used for
                used_for = treat_entry.get("used_for") or treat_entry.get("indications") or treat_entry.get("treats") or treat_entry.get("applied_to") or []
                if used_for:
                    out.append("Used to treat: " + ", ".join(str(x) for x in used_for[:10]))
                    for disease_name in used_for[:3]:
                        dinfo = None
                        try:
                            dinfo = kb.get_disease_info(disease_name)
                        except Exception:
                            dinfo = None
                        dinfo = self._unwrap_kb_item(dinfo)
                        if dinfo:
                            out.append(self._format_disease_summary(dinfo))
                else:
                    # try reverse: search diseases mentioning this treatment
                    rev = []
                    try:
                        rev = kb.search(t, category="diseases", limit=5)
                    except Exception:
                        rev = []
                    if rev:
                        names = [self._unwrap_kb_item(r).get("name") for r in rev if self._unwrap_kb_item(r) and self._unwrap_kb_item(r).get("name")]
                        out.append("Diseases where this treatment is mentioned: " + ", ".join(names[:6]))
                        for r in rev[:3]:
                            out.append(self._format_disease_summary(self._unwrap_kb_item(r)))
                    else:
                        out.append("I couldn't find explicit mapping from this treatment to diseases in the KB.")
            else:
                out.append("No KB entry found for that treatment.")

        return "\n".join(out).strip()


    # ---------- Medication response ----------
    def _generate_medication_response(self, entities: List[Entity], user_input: str) -> str:
        """If medication mentioned → show which disease(s) it treats and provide disease summaries."""
        kb = self._resolve_kb()

        med_entities = [e for e in entities if self._map_entity_label(e.label) == "medication"]
        candidate_meds = [e.text for e in med_entities]

        if not candidate_meds:
            # quick-pattern fallback
            med_patterns = [r'\b(?:aspirin|ibuprofen|paracetamol|amoxicillin|amox|metformin|insulin)\b']
            for pat in med_patterns:
                m = re.findall(pat, user_input, re.IGNORECASE)
                if m:
                    candidate_meds.extend(m)
            # try KB search if still empty
            if not candidate_meds:
                try:
                    res = kb.search(user_input, category="medications", limit=3)
                    candidate_meds = [self._unwrap_kb_item(r).get("name") for r in res if self._unwrap_kb_item(r) and self._unwrap_kb_item(r).get("name")]
                except Exception:
                    candidate_meds = []

        if not candidate_meds:
            return "Which medication are you asking about? Please provide the medication name (e.g. 'aspirin')."

        out = []
        for med in candidate_meds[:2]:
            med_entry = None
            try:
                med_entry = kb.get_medication_information(med) if hasattr(kb, "get_medication_information") else None
            except Exception:
                med_entry = None

            med_entry = self._unwrap_kb_item(med_entry)
            out.append(f"Medication: {med}")
            if med_entry:
                if "description" in med_entry:
                    out.append(med_entry["description"])
                if "uses" in med_entry and med_entry["uses"]:
                    out.append("Used for: " + ", ".join(med_entry["uses"][:6]))
                # Many med entries use 'treats' or similar
                treats = med_entry.get("treats") or med_entry.get("indications") or med_entry.get("used_for") or med_entry.get("for") or []
                if treats:
                    out.append("Treats: " + ", ".join(str(x) for x in treats[:10]))
                    # Add disease summaries for top 3
                    for disease_name in treats[:3]:
                        dinfo = None
                        try:
                            dinfo = kb.get_disease_info(disease_name)
                        except Exception:
                            dinfo = None
                        dinfo = self._unwrap_kb_item(dinfo)
                        if dinfo:
                            out.append(self._format_disease_summary(dinfo))
                else:
                    # try reverse search into diseases
                    rev = []
                    try:
                        rev = kb.search(med, category="diseases", limit=5)
                    except Exception:
                        rev = []
                    if rev:
                        names = [self._unwrap_kb_item(r).get("name") for r in rev if self._unwrap_kb_item(r) and self._unwrap_kb_item(r).get("name")]
                        out.append("Appears in the context of: " + ", ".join(names[:6]))
                        for r in rev[:3]:
                            out.append(self._format_disease_summary(self._unwrap_kb_item(r)))
                    else:
                        out.append("No explicit disease mappings found for this medication in the KB.")
            else:
                out.append("Medication not found in the knowledge base.")

        return "\n".join(out).strip()

    
    def _map_entity_label(self, label: str) -> str:
        """Map various entity labels to standardized categories"""
        label_lower = label.lower()
    
        if any(term in label_lower for term in ["disease", "illness", "condition", "covid", "flu"]):
            return "disease"
        elif any(term in label_lower for term in ["symptom", "pain", "ache", "fever", "cough"]):
            return "symptom"
        elif any(term in label_lower for term in ["treatment", "medicine", "medication", "therapy"]):
            return "treatment"
        elif any(term in label_lower for term in ["prevention", "vaccine", "protection"]):
            return "prevention"
        elif any(term in label_lower for term in ["medication", "drug", "pill", "tablet"]):
            return "medication"

        else:
            return "unknown"
        

    def process_message(self, user_id: str, text: str) -> str:
        """
        Full pipeline: intent → entities → knowledge base → response
        with fallbacks, session handling, and disclaimer.
        """
        try:
            # --- 0. Validate input ---
            if not text or not text.strip():
                return "⚠️ Please type a health-related query so I can assist you."

            # --- 1. Get / refresh session ---
            session = self.get_session(user_id)

            # --- 2. Detect intent ---
            intent = self.detect_intent(text)

            # --- 3. Extract entities ---
            entities = self.extract_entities(text)
            intent.entities = entities

            # --- 4. Generate response ---
            response = self.generate_response(intent, text, user_id, entities)

            # --- 5. Store in conversation history ---
            session.history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "user_input": text,
                "intent": intent.type.value,
                "confidence": intent.confidence,
                "entities": [e.__dict__ for e in entities],
                "response": response
            })

            logger.info(
                f"Processed user={user_id} | text='{text}' "
                f"→ intent={intent.type.value} (conf={intent.confidence:.2f}) "
                f"| entities={[e.text for e in entities]}"
            )

            return response

        except Exception as e:
            logger.exception(f"Critical error while processing message for user {user_id}: {e}")
            return (
                "⚠️ Sorry, I ran into an unexpected error while processing your request. "
                "Please try rephrasing your question.\n\n"
                "*Disclaimer: I am an AI assistant and not a medical professional.*"
            )


# ==================== FASTAPI APPLICATION ====================
if api_available:
    app = FastAPI(title="Public Health Chatbot", version=Config.VERSION)
    
    # Initialize chatbot
    chatbot = HealthChatbot()
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {"message": "Public Health Chatbot API", "version": Config.VERSION}
    
    @app.post("/chat/{user_id}")
    async def chat_endpoint(user_id: str, request: Request):
        try:
            data = await request.json()
            text = data.get("message", "").strip()
            
            if not text:
                raise HTTPException(status_code=400, detail="Message is required")
            
            response = chatbot.process_message(user_id, text)
            return JSONResponse(content={"reply": response, "user_id": user_id})
            
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "model_loaded": chatbot.ml_models_loaded}
else:
    # Create a dummy app for type checking when API is not available
    app = None

# ==================== COMMAND LINE INTERFACE ====================
def run_cli():
    """Run the chatbot in command line mode"""
    bot = HealthChatbot()
    user_id = str(uuid.uuid4())[:8]  # Generate a simple user ID
    
    print("\n" + "="*50)
    print(" Public Health Chatbot - Command Line Mode")
    print("="*50)
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'clear' to clear the conversation history")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in {"quit", "exit", "bye"}:
                print(" Goodbye! Stay healthy!")
                break
                
            if user_input.lower() == "clear":
                bot.sessions[user_id] = Session(user_id)
                print("🗑️ Conversation history cleared")
                continue
                
            if not user_input:
                continue
                
            # Process the message
            response = bot.process_message(user_id, user_input)
            
            # Print the response
            print(f" Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n Goodbye! Stay healthy!")
            break
        except Exception as e:
            print(f" Error: {str(e)}")
            logger.error(f"CLI error: {str(e)}")

# ==================== MAIN EXECUTION ====================
def main():
    parser = argparse.ArgumentParser(description="Public Health Chatbot")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli", 
                       help="Run mode: CLI interface or API server")
    parser.add_argument("--host", default=Config.HOST, help="API host address")
    parser.add_argument("--port", type=int, default=Config.PORT, help="API port")
    
    args = parser.parse_args()
    
    if args.mode == "cli":
        run_cli()
    elif args.mode == "api":
        if api_available:
            import uvicorn
            logger.info(f" Starting API server on {args.host}:{args.port}")
            uvicorn.run(app, host=args.host, port=args.port)
        else:
            logger.error("API mode requested but FastAPI is not available")
            print("FastAPI is not available. Install with: pip install fastapi uvicorn")
    else:
        print("Invalid mode. Use 'cli' or 'api'")

if __name__ == "__main__":
    main()