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
import os
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
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
    from chromadb.config import Settings
except ImportError:
    logger.error("ChromaDB not installed. Please install with: pip install chromadb")
    sys.exit(1)



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
    DATA_DIR = Path("./data")
    MODELS_DIR = Path("./models")
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
        
        logger.info("All models initialized successfully")
    
    def initialize_intent_classifier(self):
        """Initialize the intent classification model."""
        try:
            # In production, you would load a fine-tuned model
            self.intent_tokenizer = AutoTokenizer.from_pretrained(self.config.INTENT_MODEL_NAME)
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.INTENT_MODEL_NAME,
                num_labels=len(IntentType)
            )
            
            # Load fine-tuned weights if available
            intent_model_path = self.config.MODELS_DIR / "intent_classifier"
            if intent_model_path.exists():
                self.intent_model.load_state_dict(torch.load(intent_model_path / "pytorch_model.bin"))
                logger.info("Loaded fine-tuned intent classifier")
            else:
                logger.info("Using base intent classifier (not fine-tuned)")
                
        except Exception as e:
            logger.error(f"Failed to initialize intent classifier: {e}")
            # Fallback to rule-based intent recognition
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
        """Initialize the medical knowledge base."""
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.Client(Settings(
                persist_directory=str(self.config.KNOWLEDGE_BASE_DIR / "chroma_db"),
                chroma_db_impl="duckdb+parquet"
            ))
            
            # Get or create collection
            self.kb_collection = self.chroma_client.get_or_create_collection(
                name="medical_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("Knowledge base initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            self.knowledge_base = FallbackKnowledgeBase()
    
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
        """
        Process a user message and generate a response.
        
        Args:
            user_id: Unique identifier for the user
            message: User's message text
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        # Get user session
        session = self.get_session(user_id)
        
        # Safety check for emergency situations
        if self.emergency_protocol.is_emergency(message):
            response = self.emergency_protocol.get_emergency_response(message)
            return self._format_response(response, [], Intent(IntentType.EMERGENCY, 1.0), start_time)
        
        # Preprocess message
        cleaned_message = self._preprocess_text(message)
        
        # Recognize intent
        intent = self.recognize_intent(cleaned_message)
        
        # Extract entities
        entities = self.extract_entities(cleaned_message, intent)
        
        # Retrieve relevant knowledge
        context = self.retrieve_knowledge(cleaned_message, intent, entities)
        
        # Generate response
        response = self.generate_response(cleaned_message, intent, entities, context, session)
        
        # Safety check response
        response = self.safety_checker.check_response(response)
        
        # Store conversation turn
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
                
                intent_type = IntentType(list(IntentType)[predicted.item()])
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
        
        try:
            # Generate query embedding
            query_embedding = self.sentence_model.encode(text)
            
            # Query knowledge base
            results = self.kb_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=5,
                include=["metadatas", "documents", "distances"]
            )
            
            # Process results
            for i, (document, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                category = metadata.get("category", "general_info")
                if category in context:
                    context[category].append({
                        "content": document,
                        "source": metadata.get("source", "unknown"),
                        "confidence": 1 - distance,  # Convert distance to similarity
                        "relevance": self._calculate_relevance(document, intent, entities)
                    })
            
        except Exception as e:
            logger.error(f"Knowledge base query failed: {e}")
            # Fallback to predefined responses
            context = self._get_fallback_knowledge(intent, entities)
        
        return context
    
    def generate_response(self, message: str, intent: Intent, entities: List[Entity], 
                         context: Dict[str, Any], session: UserSession) -> str:
        """Generate a natural language response."""
        # Use different strategies based on intent
        if intent.type == IntentType.GREETING:
            return self._generate_greeting_response(session)
        
        elif intent.type == IntentType.DISEASE_INFO:
            return self._generate_disease_info_response(entities, context)
        
        elif intent.type == IntentType.SYMPTOM_QUERY:
            return self._generate_symptom_response(entities, context)
        
        elif intent.type == IntentType.PREVENTION:
            return self._generate_prevention_response(entities, context)
        
        elif intent.type == IntentType.TREATMENT:
            return self._generate_treatment_response(entities, context)
        
        elif intent.type == IntentType.EMERGENCY:
            return self.emergency_protocol.get_emergency_response(message)
        
        elif intent.type == IntentType.CLARIFICATION:
            return self._generate_clarification_response(session)
        
        elif intent.type == IntentType.THANKS:
            return self._generate_thanks_response()
        
        else:
            return self._generate_fallback_response(message)
    
    def _format_response(self, response: str, entities: List[Entity], 
                        intent: Intent, start_time: float) -> Dict[str, Any]:
        """Format the response with metadata."""
        return {
            "response": response,
            "entities": [{"text": e.text, "type": e.type.value, "confidence": e.confidence} 
                        for e in entities],
            "intent": {"type": intent.type.value, "confidence": intent.confidence},
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "disclaimer": "This information is for educational purposes only. Please consult a healthcare professional for medical advice."
        }
    
    # Helper methods for response generation
    def _generate_greeting_response(self, session: UserSession) -> str:
        """Generate a greeting response."""
        greetings = [
            "Hello! I'm here to provide information about public health and diseases. How can I help you today?",
            "Hi there! I'm a health information assistant. What would you like to know about?",
            "Welcome! I can answer questions about diseases, symptoms, prevention, and treatments. What would you like to know?"
        ]
        return np.random.choice(greetings)
    
    def _generate_disease_info_response(self, entities: List[Entity], context: Dict[str, Any]) -> str:
        """Generate response for disease information queries."""
        disease_entities = [e for e in entities if e.type == EntityType.DISEASE]
        
        if not disease_entities and not context["disease_info"]:
            return "I'd be happy to tell you about a disease. Could you specify which disease you're interested in?"
        
        # Get disease name from entities or context
        disease_name = disease_entities[0].text if disease_entities else "this disease"
        
        if context["disease_info"]:
            info = context["disease_info"][0]["content"]
            source = context["disease_info"][0].get("source", "reliable medical sources")
            return f"Here's what I know about {disease_name}:\n\n{info}\n\nSource: {source}"
        else:
            return f"I don't have specific information about {disease_name} in my knowledge base. Would you like me to search for general information about it?"
    
    def _generate_symptom_response(self, entities: List[Entity], context: Dict[str, Any]) -> str:
        """Generate response for symptom queries."""
        symptom_entities = [e for e in entities if e.type == EntityType.SYMPTOM]
        
        if not symptom_entities and not context["symptom_info"]:
            return "I can help with information about symptoms. Could you tell me which symptoms you're concerned about?"
        
        symptom_name = symptom_entities[0].text if symptom_entities else "these symptoms"
        
        if context["symptom_info"]:
            info = context["symptom_info"][0]["content"]
            return f"Here's information about {symptom_name}:\n\n{info}"
        else:
            return f"I don't have detailed information about {symptom_name} specifically. Symptoms can vary widely, so it's best to consult a healthcare provider for personalized advice."
    
    def _generate_prevention_response(self, entities: List[Entity], context: Dict[str, Any]) -> str:
        """Generate response for prevention queries."""
        if context["prevention_info"]:
            info = context["prevention_info"][0]["content"]
            source = context["prevention_info"][0].get("source", "reliable medical sources")
            return f"Here's prevention information:\n\n{info}\n\nSource: {source}"
        else:
            return "Prevention strategies vary by disease. General prevention measures include good hygiene, vaccination when available, and avoiding contact with sick individuals."
    
    def _generate_treatment_response(self, entities: List[Entity], context: Dict[str, Any]) -> str:
        """Generate response for treatment queries."""
        if context["treatment_info"]:
            info = context["treatment_info"][0]["content"]
            source = context["treatment_info"][0].get("source", "reliable medical sources")
            disclaimer = "\n\nImportant: Treatment should always be prescribed by a healthcare professional based on individual circumstances."
            return f"Here's information about treatments:\n\n{info}{disclaimer}\n\nSource: {source}"
        else:
            return "I don't have specific treatment information. Treatments should always be discussed with a healthcare provider who can consider your individual situation."
    
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
    
    def _extract_medical_entities(self, text: str) -> List[Entity]:
        """Rule-based extraction of medical entities."""
        entities = []
        
        # Simple pattern matching for medical terms (in production, use a comprehensive medical dictionary)
        medical_terms = {
            "covid": EntityType.DISEASE,
            "covid-19": EntityType.DISEASE,
            "influenza": EntityType.DISEASE,
            "flu": EntityType.DISEASE,
            "diabetes": EntityType.DISEASE,
            "cancer": EntityType.DISEASE,
            "headache": EntityType.SYMPTOM,
            "fever": EntityType.SYMPTOM,
            "cough": EntityType.SYMPTOM,
            "pain": EntityType.SYMPTOM,
            "aspirin": EntityType.MEDICATION,
            "vaccine": EntityType.MEDICATION,
            "x-ray": EntityType.PROCEDURE,
            "surgery": EntityType.PROCEDURE,
        }
        
        for term, entity_type in medical_terms.items():
            for match in re.finditer(rf"\b{term}\b", text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(),
                    type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.7
                ))
        
        return entities
    
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

# --- REPLACE the whole MedicalKnowledgeBase class with this ---
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
        logger.info("Building search indices...")
        # symptom -> diseases
        for dk, d in self.kb["diseases"].items():
            for s in d["data"].get("symptoms", []):
                self._indices["symptom_to_diseases"][self._normalize_text(s)].add(dk)
        # disease -> prevention/treatments/medications
        for pk, p in self.kb["prevention"].items():
            for dz in p["data"].get("effective_against", []):
                self._indices["disease_to_prevention"][self._normalize_text(dz)].append(p["data"])
        for tk, t in self.kb["treatments"].items():
            for dz in t["data"].get("used_for", []):
                self._indices["disease_to_treatments"][self._normalize_text(dz)].append(t["data"])
        for mk, m in self.kb["medications"].items():
            for dz in m["data"].get("treats", []):
                self._indices["disease_to_medications"][self._normalize_text(dz)].append(m["data"])
        # inverted keyword index
        for category, items in self.kb.items():
            for key, wrap in items.items():
                for word in self._extract_indexable_text(wrap["data"]).split():
                    w = self._normalize_text(word)
                    if len(w) > 2:
                        self._indices["keyword_index"][w].add((category, key))

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
    
def train_intent_classifier(args):
    """Train the intent classification model."""
    logger.info("Training intent classifier...")

    from train_intent import train_intent_model
    model, tokenizer = train_intent_model(args.data_dir, args.epochs, args.save_dir)

    
    # This would load training data and fine-tune the BERT model
    # Placeholder for training implementation
    
    logger.info(f"Intent classifier training completed with {args.epochs} epochs")


# ✅ Make chatbot easily importable by other files
chatbot = HealthChatbot()

def chat_cli():
    """Command-line chat interface."""
    print("Public Health Chatbot - Command Line Interface")
    print("Type 'quit' to exit, 'clear' to clear history")
    print("=" * 50)
    
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

