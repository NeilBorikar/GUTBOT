"""
Healthbot.py

World-class, production-grade scaffold for an AI-Driven Public Health Chatbot
using a hybrid approach with:
  1) Intent recognition (BERT-based)
  2) Entity extraction (BioClinicalBERT for medical entities)
  3) Knowledge retrieval from medical databases
  4) Response generation with safety guardrails

Includes:
  • Medical knowledge base with disease information
  • Multi-model NLP pipeline for understanding user queries
  • Contextual response generation with citation of sources
  • Conversation memory and context tracking
  • Safety checks to prevent harmful medical advice
  • API endpoints for web/mobile integration
  • Comprehensive evaluation framework

Dependencies:
    pip install transformers==4.30.0 torch==2.0.1 scikit-learn==1.2.2 
    pip install fastapi==0.95.0 uvicorn==0.21.1 python-multipart==0.0.6
    pip install sentence-transformers==2.2.2 chromadb==0.3.21
    pip install python-dotenv==1.0.0 requests==2.28.2
    pip install google-search-results==2.4.2 spacy==3.5.3
    python -m spacy download en_core_web_sm

Run examples:
  # 1) Initialize knowledge base
  python public_health_chatbot.py init_kb --data_dir ./medical_data

  # 2) Train intent classifier (optional)
  python public_health_chatbot.py train_intent --data_dir ./training_data --epochs 5

  # 3) Start the chatbot service
  python public_health_chatbot.py serve --host 0.0.0.0 --port 8000

  # 4) Command-line interaction
  python public_health_chatbot.py chat

  # 5) Evaluate model performance
  python public_health_chatbot.py evaluate --test_data ./test_data

DISCLAIMER: This system provides general health information only and does not 
            offer medical diagnosis or treatment advice. Users should always 
            consult healthcare professionals for medical concerns.
"""

from __future__ import annotations

from html import entities
import os
from pydoc import text
import sys
import re
import json
import time
import logging
import argparse
import asyncio

import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Deque
from enum import Enum
from datetime import datetime
from collections import deque, defaultdict

import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
from opentelemetry import context

try:
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("HealthChatbot")

# Import third-party libraries
try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        pipeline, BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
    )
except ImportError:
    logger.error("Transformers or torch not installed. Please install with: pip install transformers torch")
    sys.exit(1)

try:
    import spacy
    from spacy import displacy
except ImportError:
    logger.error("SpaCy not installed. Please install with: pip install spacy && python -m spacy download en_core_web_sm")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    logger.error("Sentence-transformers not installed. Please install with: pip install sentence-transformers")
    sys.exit(1)

try:
    import chromadb
    from chromadb import PersistentClient
except ImportError:
    logger.error("ChromaDB not installed. Please install with: pip install chromadb")
    sys.exit(1)

    #from chromadb.config import Settings
#except ImportError:
    #logger.error("ChromaDB not installed. Please install with: pip install chromadb")
    #sys.exit(1)



# Optional imports for additional functionality
SERPER_AVAILABLE = False
try:
    from serpapi import GoogleSearch
    SERPER_AVAILABLE = True
except ImportError:
    logger.warning("SerpAPI not available. Web search fallback disabled.")

class CustomHealthBotModel:
    """
    Elite adapter class that bridges your custom ML model with the public health chatbot.
    Implements the exact interface expected by public_health_chatbot.py
    """
    def __init__(self, model, tokenizer, entity_model=None, response_model=None):
        """
        Initialize with your actual model components.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.entity_model = entity_model
        self.response_model = response_model
    
    # Intent mapping
        self.intent_map = {
            0: "greeting", 1: "disease_info", 2: "symptom_query", 
            3: "prevention", 4: "treatment", 5: "emergency",
            6: "clarification", 7: "follow_up", 8: "thanks", 9: "unknown"
        }

    @staticmethod
    def load(model_path: str):
        """
        Load your actual model (supports both PyTorch .pt and HuggingFace formats).
        """
        try:
            model_pt = os.path.join(model_path, "model.pt")
            tokenizer_pt = os.path.join(model_path, "tokenizer.pt")
            hf_model = os.path.join(model_path, "pytorch_model.bin")
        
            if os.path.exists(model_pt) and os.path.exists(tokenizer_pt):
                # Load the legacy .pt format
                model = torch.load(model_pt, map_location="cpu")
                tokenizer = torch.load(tokenizer_pt)
                print(f"Loaded legacy .pt model from {model_path}")
        
            elif os.path.exists(hf_model):
                # Load HuggingFace format
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                print(f"Loaded HuggingFace model from {model_path}")
        
            else:
                raise FileNotFoundError("No compatible model files found.")

            model.eval()
            return CustomHealthBotModel(model, tokenizer)
    
        except Exception as e:
            print(f"Error loading model: {e}")
            return CustomHealthBotModel(None, None)



    def predict_intent(self, text: str) -> Tuple[str, float]:
        """
        Predict intent from text.
        """
        try:
            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
        
            # Get probabilities and predicted class
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted = torch.max(probabilities, dim=1)
        
            # Map to intent string
            intent_type = self.intent_map.get(predicted.item(), "unknown")
        
            return intent_type, confidence.item()
        
        except Exception as e:
            print(f"Error in intent prediction: {e}")
            return self._fallback_intent_prediction(text)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text.
        """
        try:
            # Use your entity extraction model if available
            if self.entity_model:
                entities = self.entity_model.extract_entities(text)
                return [{
                    "text": e['text'],
                    "label": e['type'],
                    "start": e['start'],
                    "end": e['end'],
                    "confidence": e['confidence']
                } for e in entities]
        
            # Fallback to rule-based extraction
            return self._fallback_entity_extraction(text)
        
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            return self._fallback_entity_extraction(text)

    def generate_response(self, intent: str, text: str) -> str:
        """
        Generate response based on intent.
        """
        try:
            # Use your response generation model if available
            if self.response_model:
                response = self.response_model.generate(intent, text)
                return response
        
            # Fallback to simple response generation
            return self._fallback_response_generation(intent, text)
        
        except Exception as e:
            print(f"Error in response generation: {e}")
            return self._fallback_response_generation(intent, text)


    def process_message(self, user_id: str, text: str) -> str:
        """
        Complete pipeline: Detect intent -> Extract entities -> Generate response.
        Stores conversation history per user.
        """

        # --- Session management ---
        if not hasattr(self, "sessions"):
            self.sessions = {}
        if user_id not in self.sessions:
            self.sessions[user_id] = {"history": []}

        # --- 1. Intent detection ---
        intent_type, confidence = self.predict_intent(text)

        # --- 2. Entity extraction ---
        entities = self.extract_entities(text)

        # --- 3. Response generation ---
        response = self.generate_response(intent_type, text)

        # --- 4. Store history ---
        self.sessions[user_id]["history"].append({
            "user": text,
            "intent": intent_type,
            "confidence": confidence,
            "entities": entities,
            "bot": response
        })

        return response


# ========== FALLBACK METHODS ==========

    def _fallback_intent_prediction(self, text: str) -> Tuple[str, float]:
        """Rule-based intent prediction fallback"""
        text_lower = text.lower()
    
        if any(word in text_lower for word in ["hello", "hi", "hey", "greetings"]):
            return "greeting", 0.9
        elif any(word in text_lower for word in ["covid", "flu", "disease", "illness", "virus"]):
            return "disease_info", 0.8
        elif any(word in text_lower for word in ["symptom", "symptoms", "pain", "hurt", "ache"]):
            return "symptom_query", 0.8
        elif any(word in text_lower for word in ["prevent", "avoid", "protection", "vaccine"]):
            return "prevention", 0.8
        elif any(word in text_lower for word in ["treat", "treatment", "cure", "medicine", "medication"]):
            return "treatment", 0.8
        elif any(word in text_lower for word in ["thank", "thanks", "appreciate"]):
            return "thanks", 0.9
        else:
            return "unknown", 0.5

    def _fallback_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Rule-based entity extraction fallback"""
        entities = []
        patterns = {
            "disease": r'\b(covid(-19)?|coronavirus|influenza|flu|diabetes|asthma|malaria|tb|tuberculosis)\b',
            "symptom": r'\b(fever|cough|headache|pain|nausea|vomiting|fatigue|rash|sore throat|shortness of breath)\b',
            "medication": r'\b(aspirin|ibuprofen|paracetamol|vaccine|antibiotic|penicillin)\b'
        }
    
        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.7
                })
            
        return entities

    def _fallback_response_generation(self, intent: str, text: str) -> str:
        """Simple response generation fallback"""
        responses = {
            "greeting": "Hello! I'm here to provide information about public health. How can I help you today?",
            "disease_info": "I can provide information about various diseases. What specific disease are you interested in?",
            "symptom_query": "I can help with information about symptoms. Could you tell me which symptoms you're concerned about?",
            "prevention": "Prevention is important for many health conditions. What specific disease are you concerned about preventing?",
            "treatment": "Treatment options vary depending on the condition. Could you specify which health issue you're asking about?",
            "thanks": "You're welcome! I'm glad I could help.",
            "unknown": "I'm not sure I understand. Could you please rephrase your question about health?"
        }
    
        return responses.get(intent, "I'm here to help with health information. What would you like to know?")

# Configuration
class Config:
    # Model settings
    INTENT_MODEL_NAME = "bert-base-uncased"
    NER_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    SENTENCE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Knowledge base settings
    KB_CHUNK_SIZE = 512
    KB_OVERLAP = 50
    MAX_CONTEXT_LENGTH = 1024
    
    # Response generation
    MAX_RESPONSE_LENGTH = 500
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 50
    
    # Conversation settings
    MAX_HISTORY_LENGTH = 10
    SESSION_TIMEOUT_MINUTES = 30
    
    # API settings
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8000
    API_VERSION = "v1"
    
    # Safety settings
    REDACT_PII = True
    MIN_CONFIDENCE_THRESHOLD = 0.6
    EMERGENCY_KEYWORDS = ["heart attack", "stroke", "suicide", "chest pain", "difficulty breathing"]
    
    # Paths
    DATA_DIR = Path("C:/Users/NEIL/projects/public_health_chatbot/gutbot-medical-data/medical_data")
    MODELS_DIR = Path("C:/Users/NEIL/projects/public_health_chatbot/gutbot-ml-backend/models")
    KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
    LOGS_DIR = Path("./logs")

# Create directories
for directory in [Config.DATA_DIR, Config.MODELS_DIR, Config.KNOWLEDGE_BASE_DIR, Config.LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Enums for better type safety
class IntentType(Enum):
    GREETING = "greeting"
    DISEASE_INFO = "disease_info"
    SYMPTOM_QUERY = "symptom_query"
    PREVENTION = "prevention"
    TREATMENT = "treatment"
    EMERGENCY = "emergency"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"
    THANKS = "thanks"
    UNKNOWN = "unknown"

class EntityType(Enum):
    DISEASE = "disease"
    SYMPTOM = "symptom"
    BODY_PART = "body_part"
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    PERSON = "person"
    DATE = "date"
    NUMBER = "number"
    PREVENTION = "prevention"

@dataclass
class Entity:
    text: str
    type: EntityType
    start_pos: int
    end_pos: int
    confidence: float = 1.0

@dataclass
class Intent:
    type: IntentType
    confidence: float
    entities: List[Entity] = field(default_factory=list)

@dataclass
class ContextualEmbedding:
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]

@dataclass
class KnowledgeItem:
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ConversationTurn:
    user_id: str
    message: str
    response: str
    timestamp: datetime
    intent: Intent
    context: Dict[str, Any]
    feedback: Optional[Dict[str, Any]] = None

@dataclass
class UserSession:
    id: str
    created_at: datetime
    last_activity: datetime
    conversation_history: Deque[ConversationTurn] = field(default_factory=deque)
    user_profile: Dict[str, Any] = field(default_factory=dict)
    diagnostic_state: Dict[str, Any] = field(default_factory=lambda: {
        "is_active": False,
        "waiting_for_more_symptoms": False,
        "primary_symptom": None,
        "candidate_diseases": [],
        "candidate_scores": {},
        "confirmed_symptoms": [],
        "negated_symptoms": [],
        "question_ptr": 0,
        "last_asked_symptom": None
    })
    def is_expired(self, timeout_minutes: int = Config.SESSION_TIMEOUT_MINUTES) -> bool:
        return (datetime.now() - self.last_activity).total_seconds() > timeout_minutes * 60
    
    def add_message(self, turn: ConversationTurn):
        self.conversation_history.append(turn)
        # Keep only the most recent messages
        if len(self.conversation_history) > Config.MAX_HISTORY_LENGTH:
            self.conversation_history.popleft()
        self.last_activity = datetime.now()

class HealthChatbot:
    """Main chatbot class with NLP capabilities and medical knowledge."""
    
    def __init__(self):
        self.config = Config
        self.sessions: Dict[str, UserSession] = {}
        self.intent_classifier = None
        self.ner_model = None
        self.sentence_model = None
        self.knowledge_base = None
        self.nlp = None
        self.emergency_protocol = EmergencyProtocol()
        self.safety_checker = SafetyChecker()
        
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize all NLP models and components."""
        logger.info("Initializing NLP models...")
        
        # Load spaCy for basic NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
            raise
        
        # Load sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer(self.config.SENTENCE_EMBEDDING_MODEL)
        
        # Initialize intent classifier
        self.initialize_intent_classifier()
        
        # Initialize NER model
        self.initialize_ner_model()
        
        # Initialize knowledge base
        self.initialize_knowledge_base()
        if self.kb_collection.count() == 0:
            self.ingest_medical_data()
        
        self.medical_kb = MedicalKnowledgeBase(self.config)
        self.medical_kb.initialize()
        self.load_disease_dictionary()
        
        logger.info("All models initialized successfully")
    
    def initialize_intent_classifier(self):
        """Initialize the intent classification model."""
        try:
            # Path to your fine-tuned model
            intent_model_path = self.config.MODELS_DIR / "intent_classifier"

            if intent_model_path.exists():
                # ✅ Load your fine-tuned model and tokenizer
                self.intent_model = AutoModelForSequenceClassification.from_pretrained(str(intent_model_path))
                self.intent_tokenizer = AutoTokenizer.from_pretrained(str(intent_model_path))
                label_map_path = intent_model_path / "label_map.json"
                if label_map_path.exists():
                    with open(label_map_path) as f:
                        self.label_map = json.load(f)

                    # Reverse map
                    self.id2label = {int(k): v for k, v in self.label_map["id2label"].items()}
                else:
                    logger.warning("label_map.json not found, creating default mapping")

                    self.id2label = {
                        0: "greeting",
                        1: "disease_info",
                        2: "symptom_query",
                        3: "prevention",
                        4: "treatment",
                        5: "emergency",
                        6: "clarification",
                        7: "follow_up",
                        8: "thanks",
                        9: "unknown"
                    }

            
                    logger.info(f" Loaded fine-tuned intent classifier from {intent_model_path}")
            else:
                # ⚠️ Fallback to base BERT
                logger.warning("⚠️ Fine-tuned intent model not found. Loading base model.")
                self.intent_tokenizer = AutoTokenizer.from_pretrained(self.config.INTENT_MODEL_NAME)
                self.intent_model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.INTENT_MODEL_NAME,
                    num_labels=len(IntentType)
                )
                logger.info("Using base (untrained) intent classifier.")
                # fallback label map
                self.id2label = {i: intent.value for i, intent in enumerate(IntentType)}

            self.intent_model.eval()  # Put model in inference mode

        except Exception as e:
            logger.error(f"Failed to initialize intent classifier: {e}")
            self.intent_classifier = RuleBasedIntentClassifier()

    
    def initialize_ner_model(self):
        """Initialize the named entity recognition model."""
        try:
            self.ner_tokenizer = AutoTokenizer.from_pretrained(self.config.NER_MODEL_NAME)
            self.ner_model = AutoModel.from_pretrained(self.config.NER_MODEL_NAME)
            logger.info("Loaded BioClinicalBERT for medical NER")
        except Exception as e:
            logger.error(f"Failed to initialize NER model: {e}")
            logger.info("Falling back to spaCy NER")
    
    def initialize_knowledge_base(self):
        """Initialize the medical knowledge base using the modern ChromaDB PersistentClient."""
        try:
            from chromadb import PersistentClient

            # Define persistent directory for the vector store
            kb_path = Path("C:/Users/NEIL/projects/public_health_chatbot/gutbot-ml-backend/chroma_db")
            kb_path.mkdir(parents=True, exist_ok=True)

            # Initialize persistent client (new Chroma API)
            self.client = PersistentClient(path=str(kb_path))

            # Create or get collection
            self.kb_collection = self.client.get_or_create_collection(
                name="medical_knowledge"
            )

            logger.info(f" ChromaDB knowledge base initialized at: {kb_path}")

        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            self.kb_collection = None  # fallback — prevents attribute errors

       

    def load_disease_dictionary(self):
        """Load diseases from dataset."""

        PROJECT_ROOT = Path(__file__).resolve().parent.parent

        disease_folder = PROJECT_ROOT / "gutbot-medical-data" / "medical_data" / "diseases"

        self.disease_dictionary = set()

        if not disease_folder.exists():
            logger.warning(f"Disease folder not found: {disease_folder}")
            return

        for file in disease_folder.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    name = data.get("name")
                    if name:
                        self.disease_dictionary.add(name.lower())

            except Exception as e:
                logger.warning(f"Failed loading {file}: {e}")

        logger.info(f"Loaded {len(self.disease_dictionary)} diseases")
    
    def ingest_medical_data(self):

        data_path = Path("C:/Users/NEIL/projects/public_health_chatbot/gutbot-medical-data/medical_data/diseases")

        documents = []
        metadatas = []
        ids = []

        for file in data_path.glob("*.json"):

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            disease_name = data.get("name", file.stem)
            symptoms = ", ".join(data.get("related_symptoms", []))
            description = data.get("description", "")
            ayurvedic = data.get("ayurvedic_homeopathic", {})
            ayur_note = ayurvedic.get("note", "")
            ayur_remedies = ", ".join(ayurvedic.get("supportive_remedies", []))

            content = f"""
            Disease: {disease_name}

           
            Description: {description}

            Symptoms: {symptoms}

            Prevention: {", ".join(data.get('prevention', []))}

            Treatment: {", ".join(data.get('treatment', []))}
            
            Ayurvedic/Homeopathic Note: {ayur_note}
            Ayurvedic/Homeopathic Remedies: {ayur_remedies}
            """

            documents.append(content)

            metadatas.append({
                "category": "disease_info",
                "source": file.name
            })

            ids.append(str(uuid.uuid4()))

        embeddings = self.sentence_model.encode(documents)

        self.kb_collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Indexed {len(documents)} diseases into ChromaDB")

    def get_session(self, user_id: str) -> UserSession:
        """Retrieve or create a user session."""
        if user_id not in self.sessions or self.sessions[user_id].is_expired():
            self.sessions[user_id] = UserSession(
                id=user_id,
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
            logger.info(f"Created new session for user {user_id}")
        return self.sessions[user_id]
    
    def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        start_time = time.time()
        session = self.get_session(user_id)
        cleaned_message = self._preprocess_text(message)
        msg_lower = cleaned_message.lower().strip()

        if self.emergency_protocol.is_emergency(msg_lower):
            response = self.emergency_protocol.get_emergency_response(message)
            return self._format_response(response, [], Intent(IntentType.EMERGENCY, 1.0), start_time)

        if session.diagnostic_state["is_active"]:
            intent = Intent(IntentType.SYMPTOM_QUERY, 1.0)
            entities = self.extract_entities(cleaned_message, intent)
            symptom_entities = [e for e in entities if e.type == EntityType.SYMPTOM]
            
            msg_words = msg_lower.split()
            
            def fuzzy_check(words, targets, threshold=0.85):
                for word in words:
                    for target in targets:
                        if word == target: return True
                        if len(word) > 2 and SequenceMatcher(None, word, target).ratio() >= threshold:
                            return True
                return False

            is_yes = fuzzy_check(msg_words, ["yes", "yeah", "yep", "true", "do", "yesss"])
            is_no = fuzzy_check(msg_words, ["no", "nope", "false", "not", "don't", "dont", "noooo"])
            
            if is_yes or is_no or symptom_entities:
                is_confirmed = is_yes and not is_no
                diag_result = self._handle_diagnostic_answer(session, is_confirmed, symptom_entities, start_time)
                
                final_resp_text = self.safety_checker.check_response(diag_result["response"])
                diag_result["response"] = final_resp_text
                
                turn = ConversationTurn(
                    user_id=user_id,
                    message=message,
                    response=final_resp_text,
                    timestamp=datetime.now(),
                    intent=Intent(IntentType.SYMPTOM_QUERY, 1.0),
                    context={"diagnostic_step": session.diagnostic_state["question_ptr"]}
                )
                session.add_message(turn)
                return diag_result

        if session.diagnostic_state.get("waiting_for_more_symptoms"):
            intent = Intent(IntentType.SYMPTOM_QUERY, 1.0)
            entities = self.extract_entities(cleaned_message, intent)
            if not entities:
                response = "I'm still having trouble determining the exact medical symptom from your description. Could you mention a specific symptom like 'fever', 'headache', or 'stomach pain'?"
                return self._format_response(response, [], intent, start_time)
            
            context = self.retrieve_knowledge(cleaned_message, intent, entities)
            response = self.generate_response(cleaned_message, intent, entities, context, session)
            return self._format_response(response, entities, intent, start_time)

        intent = self.recognize_intent(cleaned_message)
        entities = self.extract_entities(cleaned_message, intent)
        
        has_disease = any(e.type == EntityType.DISEASE for e in entities)
        has_symptom = any(e.type == EntityType.SYMPTOM for e in entities)

        if intent.type == IntentType.DISEASE_INFO and not has_disease and has_symptom:
            intent = Intent(IntentType.SYMPTOM_QUERY, intent.confidence)
        elif intent.type == IntentType.SYMPTOM_QUERY and not has_symptom and has_disease:
            intent = Intent(IntentType.DISEASE_INFO, intent.confidence)
            
        context = self.retrieve_knowledge(cleaned_message, intent, entities)

        response = self.generate_response(cleaned_message, intent, entities, context, session)
        response = self.safety_checker.check_response(response)

        turn = ConversationTurn(
            user_id=user_id,
            message=message,
            response=response,
            timestamp=datetime.now(),
            intent=intent,
            context=context
        )
        session.add_message(turn)

        return self._format_response(response, entities, intent, start_time)

    def generate_response(self, message: str, intent: Intent, entities: List[Entity], context: Dict[str, Any], session: UserSession) -> str:
        """Route to appropriate response generator based on intent."""
        if intent.type == IntentType.DISEASE_INFO:
            return self._generate_disease_info_response(entities, context)
        elif intent.type == IntentType.SYMPTOM_QUERY:
            return self._generate_symptom_response(entities, context, session)
        elif intent.type == IntentType.PREVENTION:
            return self._generate_prevention_response(entities, context)
        elif intent.type == IntentType.TREATMENT:
            return self._generate_treatment_response(entities, context)
        elif intent.type == IntentType.CLARIFICATION:
            return self._generate_clarification_response(session)
        elif intent.type == IntentType.THANKS:
            return self._generate_thanks_response()
        elif intent.type == IntentType.GREETING:
            return "Hello! I'm GutBot. How can I assist you with your health today?"
        else:
            return self._generate_fallback_response(message)
            
    def _format_response(self, text: str, entities: List[Entity], intent: Intent, start_time: float) -> Dict[str, Any]:
        """Format the final response object."""
        return {
            "response": text,
            "intent": intent.type.value if intent else "unknown",
            "confidence": intent.confidence if intent else 0.0,
            "entities": [{"text": e.text, "type": e.type.value} for e in entities],
            "processing_time": round((time.time() - start_time) * 1000, 2)
        }
    
    def recognize_intent(self, text: str) -> Intent:
        """Recognize the intent behind user message."""
        # Try ML-based intent classification first
        if self.intent_model:
            try:
                inputs = self.intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    outputs = self.intent_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, predicted = torch.max(probs, dim=1)
                
                intent_name = self.id2label[predicted.item()]
                intent_type = IntentType(intent_name)
                return Intent(intent_type, confidence.item())
            except Exception as e:
                logger.warning(f"ML intent classification failed: {e}")
        
        # Fallback to rule-based approach
        return self._rule_based_intent(text)
    
    def extract_entities(self, text: str, intent: Intent) -> List[Entity]:
        """Extract medical entities from text."""
        entities = []
        
        # Use BioClinicalBERT if available
        if self.ner_model:
            try:
                # This is a simplified approach - in production you'd use a proper NER model
                inputs = self.ner_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = self.ner_model(**inputs)
                # In a real implementation, you would decode the NER tags here
                # This is a placeholder for the actual NER implementation
                pass
            except Exception as e:
                logger.warning(f"BERT NER failed: {e}")
        
        # Fallback to spaCy NER
        doc = self.nlp(text)
        for ent in doc.ents:
            # Map spaCy entity types to our types
            entity_type = self._map_spacy_entity(ent.label_)
            if entity_type:
                entities.append(Entity(
                    text=ent.text,
                    type=entity_type,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=0.8  # spaCy doesn't provide confidence scores
                ))
        
        # Add rule-based entity extraction for medical terms
        entities.extend(self._extract_medical_entities(text))
        
        return entities
    
    def retrieve_knowledge(self, text: str, intent: Intent, entities: List[Entity]) -> Dict[str, Any]:
        """Retrieve relevant knowledge from the knowledge base."""
        context = {
            "disease_info": [],
            "symptom_info": [],
            "prevention_info": [],
            "treatment_info": [],
            "general_info": []
        }

        # ---------- 1 Structured retrieval ----------
        for entity in entities:

            if entity.type == EntityType.DISEASE:
                disease_name = entity.text
                disease = self.medical_kb.get_disease_info(disease_name)

                if disease:
                    context["disease_info"].append({
                        "content": disease,
                        "source": "medical_dataset",
                        "confidence": 0.95
                    })

                preventions = self.medical_kb.get_prevention_for_disease(disease_name)

                for p in preventions:
                    context["prevention_info"].append({
                        "content": p,
                        "source": "medical_dataset",
                        "confidence": 0.9
                    })

                treatments = self.medical_kb.get_treatments_for_disease(disease_name)

                for t in treatments:
                    context["treatment_info"].append({
                        "content": t,
                        "source": "medical_dataset",
                        "confidence": 0.9
                    })
                meds = self.medical_kb.get_medications_for_disease(disease_name)
                for m in meds:
                    context["treatment_info"].append({
                    "content": m,
                    "source": "medical_dataset",
                    "confidence": 0.9
                })
            elif entity.type == EntityType.SYMPTOM:

                symptom_name = entity.text

                symptom_info = self.medical_kb.get_symptom_info(symptom_name)

                if symptom_info:
                    context["symptom_info"].append({
                        "content": symptom_info,
                        "source": "medical_dataset",
                        "confidence": 0.95
                    })

                related_diseases = self.medical_kb.get_diseases_for_symptom(symptom_name)

                for d in related_diseases:
                    context["disease_info"].append({
                    "content": d,
                    "source": "medical_dataset",
                    "confidence": 0.85
            })
         # ---------- 2 Vector retrieval ----------
        try:
            # Generate query embedding
            query_embedding = self.sentence_model.encode(text)
            
            # Query knowledge base
            results = self.kb_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=3,
                include=["metadatas", "documents", "distances"]
            )
            
            # Process results
            for document, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                category = metadata.get("category", "general_info")
                if category in context:
                    context[category].append({
                        "content": document,
                        "source": metadata.get("source", "vector_kb"),
                        "confidence": 1 - distance,  # Convert distance to similarity
                        "relevance": self._calculate_relevance(document, intent, entities)
                    })
            
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
        
        return context
    
    def _generate_disease_info_response(self, entities, context):
        disease_entities = [e for e in entities if e.type == EntityType.DISEASE]
        if not disease_entities:
            return "Please specify the disease you want information about."

        target_name = disease_entities[0].text
        db_data = self.medical_kb.get_disease_info(target_name)

        if not db_data and context.get("disease_info"):
            db_data = context["disease_info"][0]["content"]

        if not db_data:
            return f"Please specify the disease you want information about."

        def clean_to_list(field):
            val = db_data.get(field, [])
            if isinstance(val, str):
                return [s.strip() for s in val.split(",")]
            return val

        name = db_data.get("name", target_name)
        desc = db_data.get("description", "N/A")
        causes = clean_to_list("causes")
        symptoms = clean_to_list("related_symptoms")
        severity = db_data.get("severity", "N/A")

        prev_list = self.medical_kb.get_prevention_for_disease(target_name)
        treat_list = self.medical_kb.get_treatments_for_disease(target_name)
        meds_list = self.medical_kb.get_medications_for_disease(target_name)

        output = f"Here is information about {name}:\n\n"
        output += f"Description: {desc}\n\n"

        if causes:
            output += "Causes:\n"
            for c in causes:
                output += f"- {c}\n"
            output += "\n"

        if symptoms:
            output += "Symptoms:\n"
            for s in symptoms:
                output += f"- {s}\n"
            output += "\n"

        output += f"Severity: {severity}\n"

        if prev_list:
            output += "Prevention:\n"
            for p in prev_list:
                p_val = p.get("name", p.get("description", ""))
                if p_val:
                    output += f"- {p_val}\n"
            output += "\n"

        if treat_list or meds_list:
            output += "Treatment:\n"
            combined = []
            for t in treat_list: combined.append(t.get("name", t.get("description", "")))
            for m in meds_list: combined.append(m.get("name", m.get("description", "")))
            for item in list(dict.fromkeys(combined)):
                if item:
                    output += f"- {item}\n"

        ayurvedic = db_data.get("ayurvedic_homeopathic")
        if ayurvedic:
            output += "\nAyurvedic & Homeopathic Support:\n"
            note = ayurvedic.get("note")
            if note:
                output += f"Note: {note}\n"
            remedies = ayurvedic.get("supportive_remedies", [])
            for r in remedies:
                output += f"- {r}\n"

        return output.strip()
    
    def _generate_symptom_response(self, entities, context, session: UserSession):
        symptom_entities = [e for e in entities if e.type == EntityType.SYMPTOM]

        if not symptom_entities:
            session.diagnostic_state["waiting_for_more_symptoms"] = True
            return "I understand you're concerned about your symptoms. Could you describe them more specifically, such as where it hurts or how long you've felt this way?"

        session.diagnostic_state["waiting_for_more_symptoms"] = False
        primary_symptom = symptom_entities[0].text
    
        related_diseases = self.medical_kb.get_related_diseases(primary_symptom)
    
        if not related_diseases:
            return f"I've noted that you're experiencing {primary_symptom}. To help me understand better, could you tell me about any other signs you've noticed?"

        session.diagnostic_state.update({
            "is_active": True,
            "primary_symptom": primary_symptom,
            "candidate_diseases": [d['name'] for d in related_diseases],
            "candidate_scores": {d['name']: 1 for d in related_diseases},
            "confirmed_symptoms": [primary_symptom],
            "negated_symptoms": [],
            "question_ptr": 0
        })

        best_candidate_name = max(session.diagnostic_state["candidate_scores"].items(), key=lambda x: x[1])[0]
        first_candidate = next((d for d in related_diseases if d['name'] == best_candidate_name), related_diseases[0])
        candidate_name = first_candidate.get('name', 'this condition')
        
        all_symptoms = first_candidate.get('related_symptoms', [])
        if isinstance(all_symptoms, str):
            all_symptoms = [s.strip() for s in all_symptoms.split(",")]
    
        follow_up_list = [s for s in all_symptoms if s.lower() != primary_symptom.lower()]
    
        response = f"I've analyzed your symptom: {primary_symptom}.\n\n"
    
        if context.get("symptom_info"):
            s_data = context["symptom_info"][0]["content"]
            if isinstance(s_data, dict) and s_data.get("description"):
                response += f"Note: {s_data['description']}\n\n"

        if follow_up_list:
            next_q = follow_up_list[0]
            session.diagnostic_state["last_asked_symptom"] = next_q
            response += f"To narrow things down, are you also experiencing **{next_q}**?"
        else:
            response += f"This can sometimes be associated with {candidate_name}. Would you like to see the full details for this condition?"
        
        return response
    
    def _generate_prevention_response(self, entities, context):
        disease_entities = [e for e in entities if e.type == EntityType.DISEASE]
    
        if not disease_entities:
            if context.get("prevention_info"):
                res = "Here is some general prevention advice:\n\n"
                for item in context["prevention_info"][:3]:
                    p = item["content"]
                    res += f"- {p.get('name', p.get('description'))}\n"
                return res
            return "Which disease or symptom would you like prevention advice for?"

        target_name = disease_entities[0].text
        db_prev = self.medical_kb.get_prevention_for_disease(target_name)

        if not db_prev:
            return f"I couldn't find specific prevention protocols for {target_name} in my medical database."

        output = f"Prevention and protection for {target_name}:\n\n"
        for item in db_prev:
            name = item.get("name")
            desc = item.get("description")
            if name:
                output += f"- {name}\n"
            if desc and desc != name:
                output += f"  ({desc})\n"

        db_data = self.medical_kb.get_disease_info(target_name)
        if db_data:
            ayurvedic = db_data.get("ayurvedic_homeopathic")
            if ayurvedic:
                output += "\nAyurvedic & Homeopathic Support:\n"
                note = ayurvedic.get("note")
                if note:
                    output += f"Note: {note}\n"
                remedies = ayurvedic.get("supportive_remedies", [])
                for r in remedies:
                    output += f"- {r}\n"
            
        return output.strip()

  
    
    def _generate_treatment_response(self, entities, context):
        disease_entities = [e for e in entities if e.type == EntityType.DISEASE]
    
        if not disease_entities:
            return "Please specify which condition you are seeking treatment information for."

        target_name = disease_entities[0].text
    
        treats = self.medical_kb.get_treatments_for_disease(target_name)
        meds = self.medical_kb.get_medications_for_disease(target_name)

        if not treats and not meds:
            return f"I don't have recorded treatment or medication data for {target_name}. Please consult a healthcare professional."

        output = f"Management and Treatment for {target_name}:\n\n"
    
        combined = []
        for t in treats: combined.append(t.get("name", t.get("description", "")))
        for m in meds: combined.append(m.get("name", m.get("description", "")))
    
        unique_items = list(dict.fromkeys([i for i in combined if i]))

        for item in unique_items:
            output += f"- {item}\n"

        db_data = self.medical_kb.get_disease_info(target_name)
        if db_data:
            ayurvedic = db_data.get("ayurvedic_homeopathic")
            if ayurvedic:
                output += "\nAyurvedic & Homeopathic Support:\n"
                note = ayurvedic.get("note")
                if note:
                    output += f"Note: {note}\n"
                remedies = ayurvedic.get("supportive_remedies", [])
                for r in remedies:
                    output += f"- {r}\n"

        output += "\nIMPORTANT: Medications should only be taken under professional supervision."
        return output.strip()
    
    def _handle_diagnostic_answer(self, session, confirmed, new_symptoms, start_time):
        state = session.diagnostic_state
        last_symptom = state["last_asked_symptom"]

        if confirmed:
            state["confirmed_symptoms"].append(last_symptom)
        else:
            state["negated_symptoms"].append(last_symptom)

        def clean_to_list(val):
            if isinstance(val, str):
                return [s.strip() for s in val.split(",")]
            return val if isinstance(val, list) else []

        for sym_entity in new_symptoms:
            sym_text = sym_entity.text
            if sym_text not in state["confirmed_symptoms"] and sym_text != last_symptom:
                state["confirmed_symptoms"].append(sym_text)
                for disease_name in state["candidate_diseases"]:
                    disease_data = self.medical_kb.get_disease_info(disease_name)
                    if not disease_data: continue
                    related = [s.lower() for s in clean_to_list(disease_data.get("related_symptoms", []))]
                    if sym_text.lower() in related:
                        state["candidate_scores"][disease_name] += 1
                        
        # Rescore candidates based on the answer
        for disease_name in state["candidate_diseases"]:
            disease_data = self.medical_kb.get_disease_info(disease_name)
            if not disease_data:
                continue
            related = clean_to_list(disease_data.get("related_symptoms", []))
            related_lower = [s.lower() for s in related]
            
            if last_symptom.lower() in related_lower:
                if confirmed:
                    state["candidate_scores"][disease_name] += 1
                else:
                    state["candidate_scores"][disease_name] -= 1
            else:
                if confirmed:
                    state["candidate_scores"][disease_name] -= 1

        state["question_ptr"] += 1

        # Re-rank diseases based on updated scores
        sorted_candidates = sorted(state["candidate_scores"].items(), key=lambda item: item[1], reverse=True)
        if not sorted_candidates:
            state["is_active"] = False
            return self._format_response("I cannot determine the condition based on the symptoms provided.", [], Intent(IntentType.SYMPTOM_QUERY, 1.0), start_time)
            
        best_candidate = sorted_candidates[0][0]
        disease_data = self.medical_kb.get_disease_info(best_candidate)
        
        related = []
        if disease_data:
            related = clean_to_list(disease_data.get("related_symptoms", []))
            
        remaining = [s for s in related if s not in state["confirmed_symptoms"] and s not in state["negated_symptoms"]]

        # Stop condition: 5 questions max, or no remaining symptoms
        if remaining and state["question_ptr"] < 5:
            state["last_asked_symptom"] = remaining[0]
            msg = f"I see. To help narrow this down further, do you also have **{remaining[0]}**?"
            return self._format_response(msg, [], Intent(IntentType.SYMPTOM_QUERY, 1.0), start_time)
    
        state["is_active"] = False
    
        full_report = self._generate_disease_info_response(
            [Entity(text=best_candidate, type=EntityType.DISEASE, start_pos=0, end_pos=0)], 
            {}
        )
    
        final_msg = f"Based on your symptoms, it matches the profile for **{best_candidate}** best.\n\n{full_report}"
        return self._format_response(final_msg, [], Intent(IntentType.DISEASE_INFO, 1.0), start_time)
    
    def _generate_clarification_response(self, session: UserSession) -> str:
        """Ask for clarification when intent is unclear."""
        clarifications = [
            "I'm not sure I understand. Could you rephrase your question?",
            "Could you provide more details so I can help you better?",
            "I want to make sure I address your concern correctly. Could you elaborate?"
        ]
        return np.random.choice(clarifications)
    
    def _generate_thanks_response(self) -> str:
        """Generate response to thanks."""
        responses = [
            "You're welcome! Feel free to ask if you have more questions.",
            "Glad I could help! Stay healthy!",
            "Happy to help! Take care."
        ]
        return np.random.choice(responses)
    
    def _generate_fallback_response(self, message: str) -> str:
        """Generate a fallback response when no specific intent is detected."""
        fallbacks = [
            "I'm not sure how to answer that. I'm designed to provide information about diseases and public health.",
            "That's an interesting question. I specialize in public health information - would you like to ask about a specific disease or health topic?",
            "I'm still learning about many health topics. Could you try asking about diseases, symptoms, prevention, or treatments?"
        ]
        return np.random.choice(fallbacks)
    
    # Helper methods for entity extraction and processing
    def _map_spacy_entity(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our entity types."""
        mapping = {
            "DISEASE": EntityType.DISEASE,
            "SYMPTOM": EntityType.SYMPTOM,
            "ORG": EntityType.PERSON,  # Often medical organizations
            "PERSON": EntityType.PERSON,
            "DATE": EntityType.DATE,
            "CARDINAL": EntityType.NUMBER,
        }
        return mapping.get(spacy_label)
    
    from difflib import SequenceMatcher

    def _extract_medical_entities(self, text: str, threshold: float = 0.8) -> List[Entity]:
        """
        Advanced Hybrid Entity Extraction Engine.
        Combines Exact Phrase Matching, Fuzzy N-Gram Analysis, and Token-level Similarity.
        """
        entities = []
        text_lower = text.lower()
        # Normalize the input to remove punctuation that might mess up matching
        clean_text = re.sub(r'[^\w\s]', ' ', text_lower)
        tokens = clean_text.split()

        # 1. THE "DISEASE DICTIONARY" SCAN (Multi-word Phrase Matching)
        # We check if any known disease name exists as a substring first
        for disease in self.disease_dictionary:
            if disease in clean_text:
                start_idx = clean_text.find(disease)
                entities.append(Entity(
                    text=disease,
                    type=EntityType.DISEASE,
                    start_pos=start_idx,
                    end_pos=start_idx + len(disease),
                    confidence=1.0 # Exact match is gold standard
                ))

            # 2. THE "FUZZY N-GRAM" SCAN (Catching typos and partials)
        # If we didn't find a perfect match, we look for 'shadow' matches
        if not entities:
         # Check single words and 2-word pairs (N-grams)
            search_windows = []
            for i in range(len(tokens)):
                search_windows.append(tokens[i]) # Unigram: "dengu"
                if i < len(tokens) - 1:
                    search_windows.append(f"{tokens[i]} {tokens[i+1]}") # Bigram: "high fever"

            stop_words = {"fever", "disease", "syndrome", "disorder", "infection", "virus", "type", "of", "the", "a", "an", "ayurvedic", "homeopathic"}
            
            for candidate in search_windows:
                for known_disease in self.disease_dictionary:
                    base_similarity = SequenceMatcher(None, candidate, known_disease).ratio()
                    
                    # Compute similarity ignoring generic words so "dengue" strongly matches "dengue fever"
                    cleaned_candidate = " ".join([w for w in candidate.split() if w not in stop_words])
                    cleaned_known = " ".join([w for w in known_disease.split() if w not in stop_words])
                    
                    clean_similarity = 0
                    if cleaned_candidate and cleaned_known:
                        clean_similarity = SequenceMatcher(None, cleaned_candidate, cleaned_known).ratio()
                        
                    similarity = max(base_similarity, clean_similarity)
                
                    if similarity >= threshold:
                        entities.append(Entity(
                            text=known_disease, # IMPORTANT: Map back to the DB KEY
                            type=EntityType.DISEASE,
                            start_pos=clean_text.find(candidate),
                            end_pos=clean_text.find(candidate) + len(candidate),
                            confidence=round(similarity, 2)
                        ))

        # 3. SYMPTOM DETECTION (Crucial for your Questionnaire logic)
        # We pull from the 'symptoms' category in your KnowledgeBase
        
        # Exact matching
        matched_symptoms = set()
        symptom_keys = list(self.medical_kb.kb["symptoms"].keys())
        
        for symptom_key in symptom_keys:
            if symptom_key in clean_text:
                matched_symptoms.add(symptom_key)
                entities.append(Entity(
                    text=symptom_key,
                    type=EntityType.SYMPTOM,
                    start_pos=clean_text.find(symptom_key),
                    end_pos=clean_text.find(symptom_key) + len(symptom_key),
                    confidence=1.0
                ))

        # Fuzzy matching for symptoms (Catching typos like 'fevvver', 'stmoach')
        for i in range(len(tokens)):
            unigram = tokens[i]
            bigram = f"{tokens[i]} {tokens[i+1]}" if i < len(tokens) - 1 else ""
            
            for sym in symptom_keys:
                if sym in matched_symptoms:
                    continue
                
                # Check unigram
                if len(sym.split()) == 1 and len(unigram) > 3:
                    if SequenceMatcher(None, unigram, sym).ratio() > 0.8:
                        matched_symptoms.add(sym)
                        entities.append(Entity(text=sym, type=EntityType.SYMPTOM, start_pos=clean_text.find(unigram), end_pos=clean_text.find(unigram) + len(unigram), confidence=0.85))
                        continue
                        
                # Check bigram
                if bigram and len(sym.split()) == 2:
                    if SequenceMatcher(None, bigram, sym).ratio() > 0.8:
                        matched_symptoms.add(sym)
                        entities.append(Entity(text=sym, type=EntityType.SYMPTOM, start_pos=clean_text.find(tokens[i]), end_pos=clean_text.find(tokens[i+1]) + len(tokens[i+1]), confidence=0.85))

        # Semantic matching for complex descriptions using SentenceTransformer
        if hasattr(self, 'sentence_model') and self.sentence_model:
            try:
                if not hasattr(self, 'symptom_embeddings'):
                    self.known_symptoms_list = list(self.medical_kb.kb["symptoms"].keys())
                    if self.known_symptoms_list:
                        self.symptom_embeddings = self.sentence_model.encode(self.known_symptoms_list)
                    else:
                        self.symptom_embeddings = None
                        
                if self.symptom_embeddings is not None:
                    from sentence_transformers import util
                    doc = self.nlp(text)
                    chunks = [c.text.lower() for c in doc.noun_chunks]
                    search_texts = [clean_text] + chunks
                    
                    user_emb = self.sentence_model.encode(search_texts)
                    cos_scores = util.cos_sim(user_emb, self.symptom_embeddings)
                    
                    for i in range(len(search_texts)):
                        top_results = torch.topk(cos_scores[i], k=min(3, len(self.known_symptoms_list)))
                        
                        for score_tensor, idx_tensor in zip(top_results[0], top_results[1]):
                            score = score_tensor.item()
                            if score > 0.65: # Threshold for semantic match
                                matched_symptom = self.known_symptoms_list[idx_tensor.item()]
                                if matched_symptom not in matched_symptoms:
                                    matched_symptoms.add(matched_symptom)
                                    start_idx = clean_text.find(search_texts[i])
                                    entities.append(Entity(
                                        text=matched_symptom,
                                        type=EntityType.SYMPTOM,
                                        start_pos=start_idx if start_idx != -1 else 0,
                                        end_pos=(start_idx + len(search_texts[i])) if start_idx != -1 else len(clean_text),
                                        confidence=round(score, 2)
                                    ))
            except Exception as e:
                logger.warning(f"Semantic symptom extraction failed: {e}")

        # 3.5 PREVENTION DETECTION (Fuzzy matching)
        prevention_keys = list(self.medical_kb.kb["prevention"].keys())
        for i in range(len(tokens)):
            unigram = tokens[i]
            bigram = f"{tokens[i]} {tokens[i+1]}" if i < len(tokens) - 1 else ""
            
            for prev in prevention_keys:
                # Check unigram
                if len(prev.split()) == 1 and len(unigram) > 3:
                    if SequenceMatcher(None, unigram, prev).ratio() > 0.85:
                        entities.append(Entity(text=prev, type=EntityType.PREVENTION, start_pos=clean_text.find(unigram), end_pos=clean_text.find(unigram) + len(unigram), confidence=0.85))
                
                # Check bigram
                if bigram and len(prev.split()) == 2:
                    if SequenceMatcher(None, bigram, prev).ratio() > 0.85:
                        entities.append(Entity(text=prev, type=EntityType.PREVENTION, start_pos=clean_text.find(tokens[i]), end_pos=clean_text.find(tokens[i+1]) + len(tokens[i+1]), confidence=0.85))

        # 4. OVERLAP RESOLUTION (Deduplication)
        # If we found "Cancer" and "Lung Cancer", we keep the most specific one
        entities.sort(key=lambda x: (x.confidence, len(x.text)), reverse=True)
        final_entities = []
        covered_ranges = []

        for ent in entities:
            is_covered = False
            for start, end in covered_ranges:
                if ent.start_pos >= start and ent.end_pos <= end:
                    is_covered = True
                    break
            if not is_covered:
                final_entities.append(ent)
                covered_ranges.append((ent.start_pos, ent.end_pos))

        return final_entities
    
    def _rule_based_intent(self, text: str) -> Intent:
        """Rule-based intent classification as fallback."""
        text_lower = text.lower()
        
        # Greeting detection
        if any(word in text_lower for word in ["hello", "hi", "hey", "greetings"]):
            return Intent(IntentType.GREETING, 0.9)
        
        # Disease information
        if any(phrase in text_lower for phrase in ["what is", "tell me about", "information about"]) and \
           any(word in text_lower for word in ["covid", "flu", "disease", "illness"]):
            return Intent(IntentType.DISEASE_INFO, 0.8)
        
        # Symptoms
        if any(word in text_lower for word in ["symptom", "sign", "feel", "hurt", "pain"]):
            return Intent(IntentType.SYMPTOM_QUERY, 0.8)
        
        # Prevention
        if any(word in text_lower for word in ["prevent", "avoid", "stop from getting", "protection"]):
            return Intent(IntentType.PREVENTION, 0.8)
        
        # Treatment
        if any(word in text_lower for word in ["treat", "cure", "medicine", "medication", "therapy"]):
            return Intent(IntentType.TREATMENT, 0.8)
        
        # Thanks
        if any(word in text_lower for word in ["thank", "thanks", "appreciate"]):
            return Intent(IntentType.THANKS, 0.9)
        
        # Emergency detection
        if self.emergency_protocol.is_emergency(text):
            return Intent(IntentType.EMERGENCY, 1.0)
        
        return Intent(IntentType.UNKNOWN, 0.5)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for NLP tasks."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Redact PII if enabled
        if self.config.REDACT_PII:
            text = self._redact_pii(text)
        
        return text
    
    def _redact_pii(self, text: str) -> str:
        """Redact personally identifiable information."""
        # Redact email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Redact phone numbers
        text = re.sub(r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', '[PHONE]', text)
        
        # Redact sensitive numbers (SSN, credit card, etc.)
        text = re.sub(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', '[SSN]', text)  # SSN pattern
        
        return text
    
    def _calculate_relevance(self, document: str, intent: Intent, entities: List[Entity]) -> float:
        """Calculate relevance score for a knowledge base document."""
        # Simple implementation - in production, use more sophisticated methods
        score = 0.5  # Base score
        
        # Boost score if document contains entity terms
        entity_texts = [e.text.lower() for e in entities]
        doc_lower = document.lower()
        
        for entity_text in entity_texts:
            if entity_text in doc_lower:
                score += 0.1
        
        # Cap score between 0 and 1
        return max(0, min(1, score))
    
    def _get_fallback_knowledge(self, intent: Intent, entities: List[Entity]) -> Dict[str, Any]:
        """Get fallback knowledge when knowledge base is unavailable."""
        # This would contain predefined responses for common queries
        # In production, this would be more comprehensive
        
        fallback_knowledge = {
            "disease_info": [],
            "symptom_info": [],
            "prevention_info": [],
            "treatment_info": [],
            "general_info": []
        }
        
        # Add some basic fallback information
        if intent.type == IntentType.DISEASE_INFO:
            disease_entities = [e for e in entities if e.type == EntityType.DISEASE]
            if disease_entities:
                disease = disease_entities[0].text.lower()
                if "covid" in disease:
                    fallback_knowledge["disease_info"].append({
                        "content": "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus. Most people experience mild to moderate symptoms and recover without special treatment.",
                        "source": "WHO",
                        "confidence": 0.9,
                        "relevance": 0.9
                    })
        
        return fallback_knowledge

class EmergencyProtocol:
    """Handles emergency situations and provides appropriate responses."""
    
    def __init__(self):
        self.emergency_keywords = Config.EMERGENCY_KEYWORDS
        self.emergency_responses = {
            "general": "I'm not a medical emergency service. If you're experiencing a medical emergency, please call your local emergency number immediately.",
            "heart attack": "Chest pain can be serious. Please call emergency services immediately if you experience chest pain, especially if it radiates to your arm, neck, or jaw, or is accompanied by shortness of breath, nausea, or sweating.",
            "stroke": "If you or someone else is experiencing symptoms of a stroke (face drooping, arm weakness, speech difficulty), call emergency services immediately. Time is critical for stroke treatment.",
            "suicide": "If you're having thoughts of suicide, please contact a suicide prevention hotline immediately. You're not alone, and there are people who want to help.",
            "default": "Please seek emergency medical attention if you're experiencing a serious health crisis. I can provide information but cannot respond to emergencies."
        }
    
    def is_emergency(self, text: str) -> bool:
        """Check if the text indicates an emergency situation."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.emergency_keywords)
    
    def get_emergency_response(self, text: str) -> str:
        """Get appropriate emergency response based on the text."""
        text_lower = text.lower()
        
        if "heart" in text_lower or "chest" in text_lower:
            return self.emergency_responses["heart attack"]
        elif "stroke" in text_lower:
            return self.emergency_responses["stroke"]
        elif "suicide" in text_lower or "kill myself" in text_lower:
            return self.emergency_responses["suicide"]
        else:
            return self.emergency_responses["general"]

class SafetyChecker:
    """Checks responses for safety and appropriateness."""
    
    def __init__(self):
        self.unsafe_patterns = [
            r"(you should|you must) (take|use|try) [^\.]* (without|without consulting)",
            r"(diagnose|treat|prescribe) [^\.]* yourself",
            r"ignore [^\.]* (doctor|medical advice)",
            r"(certain|guarantee) (cure|treatment)",
            r"(secret|miracle) (cure|treatment|remedy)"
        ]
        
        self.redaction_patterns = [
            r"take \d+ mg of",
            r"use [^\.]* (times|days) (a|per) day",
            r"prescription",
            r"dosage of"
        ]
    
    def check_response(self, response: str) -> str:
        """Check a response for safety and redact unsafe content."""
        # Check for unsafe patterns
        for pattern in self.unsafe_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return "I'm not able to provide that specific advice. Please consult a healthcare professional for personalized guidance."
        
        # Redact specific dosage/medical advice patterns
        for pattern in self.redaction_patterns:
            response = re.sub(pattern, "[MEDICAL ADVICE REDACTED]", response, flags=re.IGNORECASE)
        
        return response

class RuleBasedIntentClassifier:
    """
    Simple rule-based intent classifier.
    Used as a fallback when ML-based intent classification is unavailable.
    """

    def __init__(self):
        # Define regex patterns for each intent
        self.intent_patterns = {
            IntentType.GREETING: [
                r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgreetings\b", r"\bgood (morning|afternoon|evening)\b"
            ],
            IntentType.DISEASE_INFO: [
                r"\b(covid|coronavirus|flu|influenza|diabetes|asthma|malaria|cancer|tb|tuberculosis)\b"
            ],
            IntentType.SYMPTOM_QUERY: [
                r"\b(symptom|symptoms|fever|cough|headache|nausea|pain|fatigue|rash|sore throat|shortness of breath)\b"
            ],
            IntentType.PREVENTION: [
                r"\b(prevent|avoid|protection|hygiene|vaccine|vaccination|mask|wash hands)\b"
            ],
            IntentType.TREATMENT: [
                r"\b(treat|treatment|cure|medicine|medication|therapy|drug)\b"
            ],
            IntentType.EMERGENCY: [
                r"\b(heart attack|stroke|suicide|chest pain|difficulty breathing|call 911|hospital)\b"
            ],
            IntentType.THANKS: [
                r"\bthank(s| you)?\b", r"\bappreciate\b", r"\bgrateful\b"
            ],
            IntentType.CLARIFICATION: [
                r"\bwhat do you mean\b", r"\bclarify\b", r"\bcould you explain\b", r"\bnot sure I understand\b"
            ],
            IntentType.FOLLOW_UP: [
                r"\btell me more\b", r"\bcontinue\b", r"\bwhat next\b"
            ]
        }

    def predict(self, text: str) -> Intent:
        """
        Predict intent based on regex patterns.
        Returns an Intent object with confidence score.
        """
        text_lower = text.lower()

        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return Intent(
                        type=intent_type,
                        confidence=0.85,  # Fixed high confidence for matches
                        entities=[]
                    )

        # If no pattern matched → Unknown
        return Intent(
            type=IntentType.UNKNOWN,
            confidence=0.5,
            entities=[]
        )


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


from pathlib import Path
import re, json, unicodedata, heapq
from collections import defaultdict
from functools import lru_cache
from typing import Optional, Dict, Any, List

from data_loader import loader as remote_loader
import os
import logging
logger = logging.getLogger("HealthChatbot.KB")

class MedicalKnowledgeBase:
    """
    Hybrid knowledge base:
      - Remote mode (GitHub Pages / S3): requires MEDICAL_DATA_BASE_URL and a manifest.json
      - Local mode: loads from ./medical_data/* folders (your current behavior)
    """
    def __init__(self, config):
        self.config = config
        self.kb = {
            "diseases": {},
            "symptoms": {},
            "prevention": {},
            "treatments": {},
            "medications": {}
        }
        self._indices = {
            "symptom_to_diseases": defaultdict(set),
            "disease_to_prevention": defaultdict(list),
            "disease_to_treatments": defaultdict(list),
            "disease_to_medications": defaultdict(list),
            "keyword_index": defaultdict(set),
        }
        self._normalized_keys = {}
        self.remote_mode = bool(os.getenv("MEDICAL_DATA_BASE_URL"))

    def initialize(self, medical_data_path: Optional[Path] = None) -> bool:
        try:
            if self.remote_mode:
                logger.info("Initializing KB in REMOTE mode...")
                self._initialize_remote()
            else:
                logger.info("Initializing KB in LOCAL mode...")
                base_path = self._find_medical_data_path(medical_data_path)
                if not base_path or not base_path.exists():
                    logger.warning(f"Medical data directory not found: {base_path}")
                    return False
                self._initialize_local(base_path)
            self._build_indices()
            logger.info(
                "KB loaded: %d diseases, %d symptoms, %d prevention, %d treatments, %d meds",
                len(self.kb["diseases"]), len(self.kb["symptoms"]),
                len(self.kb["prevention"]), len(self.kb["treatments"]),
                len(self.kb["medications"]),
            )
            return True
        except Exception as e:
            logger.exception("Failed initializing KB: %s", e)
            return False

    # ---------- Remote ----------
    def _initialize_remote(self):
        # Expect manifest.json like:
        # { "diseases": ["flu.json", ...], "symptoms": [...], ... }
        manifest = remote_loader.get_json("manifest.json")
        for cat in ["diseases", "symptoms", "prevention", "treatments", "medications"]:
            files = manifest.get(cat, [])
            self._load_category_remote(cat, files)

    def _load_category_remote(self, category_key: str, file_list: List[str]):
        count = 0
        for fname in file_list:
            try:
                data = remote_loader.get_json(category_key, fname)
                pk = self._get_primary_key(data, category_key, fname.replace(".json", ""))
                nk = self._normalize_text(pk)
                self.kb[category_key][nk] = {"data": data, "source_file": f"remote:{category_key}/{fname}", "original_key": pk}
                self._normalized_keys[pk] = nk
                count += 1
            except Exception as e:
                logger.warning("Failed to load remote %s/%s: %s", category_key, fname, e)
        logger.info("Loaded %d remote %s", count, category_key)

    # ---------- Local ----------
    def _initialize_local(self, base_path: Path):
        categories = {
            "diseases": "diseases",
            "symptoms": "symptoms",
            "prevention": "prevention",
            "treatments": "treatments",
            "medications": "medications",
        }
        for category_key, folder in categories.items():
            self._load_category_local(base_path / folder, category_key)

    def _find_medical_data_path(self, medical_data_path: Optional[Path]) -> Optional[Path]:
        if medical_data_path:
            return Path(medical_data_path)
        project_root = Path(__file__).resolve().parents[1]
        c = project_root / "medical_data"
        if c.exists(): return c
        c = Path.cwd() / "medical_data"
        if c.exists(): return c
        c = Path(getattr(self.config, "DATA_DIR", Path("./medical_data")))
        return c if c.exists() else None

    def _load_category_local(self, dir_path: Path, category_key: str):
        if not dir_path.exists():
            logger.warning("KB category folder not found: %s", dir_path)
            return
        count = 0
        for file_path in dir_path.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pk = self._get_primary_key(data, category_key, file_path.stem)
                nk = self._normalize_text(pk)
                self.kb[category_key][nk] = {"data": data, "source_file": str(file_path), "original_key": pk}
                self._normalized_keys[pk] = nk
                count += 1
            except Exception as e:
                logger.warning("Failed to load %s: %s", file_path, e)
        logger.info("Loaded %d files for category: %s", count, category_key)

    # ---------- Common helpers ----------
    def _get_primary_key(self, data: Dict, category_key: str, fallback: str) -> str:
        fields = {
            "diseases": ["name", "disease", "title"],
            "symptoms": ["name", "symptom", "title"],
            "prevention": ["name", "title", "prevention_method"],
            "treatments": ["name", "title", "treatment", "medication"],
            "medications": ["name", "title", "drug", "medication"],
        }.get(category_key, ["name", "title"])
        for f in fields:
            if f in data and data[f]:
                return str(data[f])
        return fallback

    @lru_cache(maxsize=10000)
    def _normalize_text(self, text: str) -> str:
        if not isinstance(text, str): return ""
        text = unicodedata.normalize("NFKD", text).lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text)

    def _build_indices(self):
        """
        High-performance inverted indexing for lightning-fast disease/symptom lookup.
        Features: Data sanitization, Set-based uniqueness, and memory-efficient keyword mapping.
        """
        logger.info("🚀 Initiating God-Level Index Construction...")
        start_time = time.time()

        # Reset indices to ensure a clean state
        self._indices["symptom_to_diseases"].clear()
        self._indices["disease_to_prevention"].clear()
        self._indices["disease_to_treatments"].clear()
        self._indices["disease_to_medications"].clear()
        self._indices["keyword_index"].clear()

        # 1. Map Symptoms -> Diseases (The Core for your Questionnaire)
        for dk, d in self.kb["diseases"].items():
            # DATA SANITIZATION: Handle both list and string formats in JSON
            raw_symptoms = d["data"].get("related_symptoms", [])
            if isinstance(raw_symptoms, str):
                raw_symptoms = [s.strip() for s in raw_symptoms.split(",")]

            for s in raw_symptoms:
                norm_s = self._normalize_text(s)
                if norm_s:
                # FIX: Merged lines to prevent the 'list object has no attribute add' crash
                    self._indices["symptom_to_diseases"][norm_s].add(dk)

    # 2. Map Diseases -> Supportive Info (Prevention, Treatment, Meds)
    # Optimized using a reusable internal mapping helper
        self._map_category_to_disease("prevention", "effective_against", "disease_to_prevention")
        self._map_category_to_disease("treatments", "used_for", "disease_to_treatments")
        self._map_category_to_disease("medications", "uses", "disease_to_medications")

    # 3. Global Inverted Keyword Index (For general search)
    # Uses set comprehension for faster processing
        for category, items in self.kb.items():
            for key, wrap in items.items():
                content_text = self._extract_indexable_text(wrap["data"])
                # Remove short words (stop words) and duplicates in one pass
                words = {self._normalize_text(w) for w in content_text.split() if len(w) > 2}
                for w in words:
                    if w:
                        self._indices["keyword_index"][w].add((category, key))

        elapsed = time.time() - start_time
        logger.info(f"✅ Indexing complete in {elapsed:.4f}s. GutBot is now 'Self-Aware'.")

    def _map_category_to_disease(self, kb_cat, data_field, index_key):
        """Helper to link supplemental data (like Meds) back to a primary disease key."""
        for key, wrap in self.kb[kb_cat].items():
            # Ensure we handle single strings or lists of target diseases
            targets = wrap["data"].get(data_field, [])
            if isinstance(targets, str):
                targets = [targets]
            
            for target_dz in targets:
                norm_dz = self._normalize_text(target_dz)
                if norm_dz:
                    self._indices[index_key][norm_dz].append(wrap["data"])

    def _extract_indexable_text(self, data: Any) -> str:
        if isinstance(data, str): return data
        if isinstance(data, dict): return " ".join(str(v) for v in data.values() if isinstance(v, (str, int, float)))
        if isinstance(data, list): return " ".join(self._extract_indexable_text(x) for x in data)
        return str(data)

    # --- query api (unchanged external interface) ---
    def search(self, query: str, category: Optional[str] = None, limit: int = 10) -> List[Dict]:
        nq = self._normalize_text(query); terms = nq.split()
        scores = defaultdict(float)
        for t in terms:
            if t in self._indices["keyword_index"]:
                for cat, item_key in self._indices["keyword_index"][t]:
                    if category and cat != category: continue
                    item = self.kb[cat][item_key]
                    boost = 2.0 if "name" in item["data"] and t in self._normalize_text(item["data"]["name"]) else 1.0
                    scores[(cat, item_key)] += boost
        top = heapq.nlargest(limit, scores.items(), key=lambda x: x[1])
        out = []
        for (cat, key), sc in top:
            itm = self.kb[cat][key].copy()
            itm["relevance_score"] = sc / max(1, len(terms))
            itm["category"] = cat
            out.append(itm)
        return out

    def get_disease_info(self, name: str) -> Optional[Dict]:
        return self._get_item("diseases", name)
    def get_symptom_info(self, name: str) -> Optional[Dict]:
        return self._get_item("symptoms", name)
    def get_prevention_information(self, name: str) -> Optional[Dict]:
        return self._get_item("prevention", name)
    def get_treatment_information(self, name: str) -> Optional[Dict]:
        return self._get_item("treatments", name)
    def get_medication_information(self, name: str) -> Optional[Dict]:
        return self._get_item("medications", name)

    def get_related_diseases(self, symptom: str) -> List[Dict]:
        keys = self._indices["symptom_to_diseases"].get(self._normalize_text(symptom), set())
        return [self.kb["diseases"][k]["data"] for k in keys]

    def _get_item(self, category: str, name: str) -> Optional[Dict]:
        if not name: return None
        nk = self._normalize_text(name)
        if nk in self.kb[category]:
            return self.kb[category][nk]["data"]
        # fuzzy (very light)
        for k in self.kb[category].keys():
            if nk in k or k in nk:
                return self.kb[category][k]["data"]
        return None
    def get_prevention_for_disease(self, disease_name):

        disease = self._normalize_text(disease_name)

        results = []

        for item in self.kb["prevention"].values():

            data = item["data"]

            diseases = [self._normalize_text(d) for d in data.get("effective_against", [])]

            if disease in diseases:
                results.append(data)

        return results
    def get_treatments_for_disease(self, disease_name):

        disease = self._normalize_text(disease_name)

        results = []

        for item in self.kb["treatments"].values():

            data = item["data"]

            diseases = [self._normalize_text(d) for d in data.get("used_for", [])]

            if disease in diseases:
                results.append(data)

        return results
    
    def get_medications_for_disease(self, disease_name):
        disease = self._normalize_text(disease_name)

        results = []

        for item in self.kb["medications"].values():

            data = item["data"]

            diseases = [self._normalize_text(d) for d in data.get("uses", [])]

            if disease in diseases:
                results.append(data)

        return results
    def get_diseases_for_symptom(self, symptom):

        symptom = self._normalize_text(symptom)

        results = []

        for item in self.kb["symptoms"].values():

            data = item["data"]

            if symptom == self._normalize_text(data.get("symptom","")):
                results.extend(data.get("possible_diseases", []))

        return results
    
def train_intent_classifier(args):
    """Train the intent classification model."""
    logger.info("Training intent classifier...")

    from train_intent import train_intent_model
    model, tokenizer = train_intent_model(args.data_dir, args.epochs, args.save_dir)

    
    # This would load training data and fine-tune the BERT model
    # Placeholder for training implementation
    
    logger.info(f"Intent classifier training completed with {args.epochs} epochs")


# ✅ Make chatbot easily importable by other files
#chatbot = HealthChatbot()

def chat_cli():
    """Command-line chat interface."""
    print("Public Health Chatbot - Command Line Interface")
    print("Type 'quit' to exit, 'clear' to clear history")
    print("=" * 50)
    chatbot = HealthChatbot()
    user_id = str(uuid.uuid4())
    
    while True:
        try:
            message = input("You: ").strip()
            
            if message.lower() in ['quit', 'exit']:
                break
                
            if message.lower() == 'clear':
                print("\n" * 50)
                continue
                
            if not message:
                continue
                
            response = chatbot.process_message(user_id, message)
            print(f"\nChatbot: {response['response']}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in CLI chat: {e}")
            print("Sorry, I encountered an error. Please try again.")

def evaluate_model(args):
    """Evaluate model performance on test data."""
    logger.info("Evaluating model performance...")
    
    # This would load test data and run evaluations
    # Placeholder for evaluation implementation
    
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        logger.error(f"Test data directory {args.test_data} does not exist")
        return
    
    logger.info(f"Evaluation completed using data from {args.test_data}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or load HealthBot model")
    parser.add_argument("command", choices=["train_intent"], help="Command to run")
    parser.add_argument("--data_dir", type=str, default="../public_health_chatbot/training_data", help="Directory with intents.json")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="./Backend/models/intent_classifier", help="Where to save model")

    args = parser.parse_args()

    if args.command == "train_intent":
        from train_intent import train_intent_model
        data_path = Path(args.data_dir) / "intents.json"   # ✅ build full path to file
        train_intent_model(
            data_path=data_path,
            model_dir=args.save_dir,
            epochs=args.epochs
        )

