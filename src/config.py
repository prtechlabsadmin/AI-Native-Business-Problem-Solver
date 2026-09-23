"""
Configuration and Environment Settings for Support Triage System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TICKETS_FILE = DATA_DIR / "tickets.json"
KB_FILE = DATA_DIR / "knowledge_base.json"
PROMPTS_FILE = BASE_DIR / "prompts" / "prompts.md"

# Load .env if present
load_dotenv(BASE_DIR / ".env")

# LLM Configurations
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()  # "openai", "anthropic", "gemini", "mock", "auto"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Triage & Routing Thresholds
AUTO_SEND_CONFIDENCE_THRESHOLD = float(os.getenv("AUTO_SEND_CONFIDENCE_THRESHOLD", "0.85"))
TOP_K_RAG_RESULTS = int(os.getenv("TOP_K_RAG_RESULTS", "2"))

CATEGORIES = [
    "Billing",
    "Technical Support",
    "Feature Request",
    "Account Access",
    "Complaint",
]

URGENCIES = [
    "P1 - Critical",
    "P2 - High",
    "P3 - Medium",
    "P4 - Low",
]

SENTIMENTS = [
    "Positive",
    "Neutral",
    "Frustrated",
    "Angry",
]

ACTIONS = [
    "auto_send",
    "human_review",
    "refund_approval",
    "engineer_escalation",
    "escalate_tier3",
]
