from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

# HuggingFace imports
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB setup
try:
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    # Trigger a server selection to check connection
    client.server_info()
    db = client["complaints_db"]
    complaints_collection = db["complaints"]
    logger.info("Connected to MongoDB at %s", MONGODB_URI)
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    import sys
    sys.exit(1)

# FastAPI app
app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ComplaintIn(BaseModel):
    problem: str
    location: str
    latitude: float
    longitude: float
    timestamp: Optional[datetime] = None
    topic: Optional[str] = None
    topic_confidence: Optional[float] = None
    urgency: Optional[str] = None
    urgency_probabilities: Optional[dict] = None

class ComplaintOut(ComplaintIn):
    id: str = Field(..., alias="_id")

# --- Classification endpoint models ---
class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    topic: str
    topic_confidence: float
    urgency: str
    urgency_confidence: float

# --- Load HuggingFace models at startup ---
TOPIC_MODEL_REPO = os.getenv("TOPIC_MODEL_REPO", "valterjpcaldeira/iVedrasQueixas")
URGENCY_MODEL_REPO = os.getenv("URGENCY_MODEL_REPO", "valterjpcaldeira/iVedrasUrgencia")
HF_TOKEN = os.getenv("HF_API_TOKEN")

print("Loading HuggingFace models...")
try:
    topic_model = AutoModelForSequenceClassification.from_pretrained(
        TOPIC_MODEL_REPO,
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.float32
    )
    topic_tokenizer = AutoTokenizer.from_pretrained(
        TOPIC_MODEL_REPO,
        use_auth_token=HF_TOKEN
    )
    urgency_model = AutoModelForSequenceClassification.from_pretrained(
        URGENCY_MODEL_REPO,
        use_auth_token=HF_TOKEN,
        torch_dtype=torch.float32
    )
    urgency_tokenizer = AutoTokenizer.from_pretrained(
        URGENCY_MODEL_REPO,
        use_auth_token=HF_TOKEN
    )
    logger.info("HuggingFace models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load HuggingFace models: {e}")
    import sys
    sys.exit(1)

TOPIC_LABELS = [
    "Animais e Ambiente",
    "Comércio e Atividades Económicas",
    "Infraestruturas e Obras",
    "Limpeza e Resíduos",
    "Outros",
    "Segurança e Ordem Pública",
    "Serviços Sociais e Comunitários",
    "Trânsito e Mobilidade",
    "Água e Saneamento",
    "Áreas Verdes e Espaços Públicos"
]
URGENCY_LABELS = ["Não Urgente", "Urgente"]

@app.get("/")
def read_root():
    return {"status": "ok"}

# --- Classification endpoint ---
@app.post("/classify", response_model=ClassifyResponse)
def classify_text(req: ClassifyRequest):
    # Topic
    inputs = topic_tokenizer(req.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = topic_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        topic_score = probs[0, pred].item()
    topic = TOPIC_LABELS[pred] if 0 <= pred < len(TOPIC_LABELS) else str(pred)
    # Urgency
    inputs_u = urgency_tokenizer(req.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits_u = urgency_model(**inputs_u).logits
        probs_u = torch.softmax(logits_u, dim=1)
        pred_u = torch.argmax(probs_u, dim=1).item()
        urgency_score = probs_u[0, pred_u].item()
    urgency = URGENCY_LABELS[pred_u] if 0 <= pred_u < len(URGENCY_LABELS) else str(pred_u)
    return ClassifyResponse(
        topic=topic,
        topic_confidence=topic_score,
        urgency=urgency,
        urgency_confidence=urgency_score
    ) 

from bson import ObjectId

# Helper to convert MongoDB documents to dicts with string IDs
def complaint_to_dict(complaint):
    complaint = dict(complaint)
    complaint["_id"] = str(complaint["_id"])
    # Optionally convert datetime to isoformat for frontend
    if "timestamp" in complaint and complaint["timestamp"]:
        complaint["timestamp"] = complaint["timestamp"].isoformat()
    return complaint

@app.get("/complaints", response_model=List[ComplaintOut])
def get_complaints():
    complaints = list(complaints_collection.find().sort("timestamp", -1))
    return [complaint_to_dict(c) for c in complaints]

@app.post("/complaints", response_model=ComplaintOut)
def create_complaint(complaint: ComplaintIn):
    data = complaint.dict()
    # Set timestamp if not provided
    if not data.get("timestamp"):
        data["timestamp"] = datetime.utcnow()
    result = complaints_collection.insert_one(data)
    data["_id"] = str(result.inserted_id)
    # Return the stored complaint (with string _id and iso timestamp)
    return complaint_to_dict(data)