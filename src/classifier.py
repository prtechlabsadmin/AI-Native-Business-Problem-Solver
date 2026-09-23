"""
Structured Support Ticket Classifier Module
"""

import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from src.llm import LLMClient
from src import config

logger = logging.getLogger(__name__)

class ClassificationResult(BaseModel):
    category: str = Field(description="Category of the ticket")
    urgency: str = Field(description="Urgency priority level (P1-P4)")
    sentiment: str = Field(description="Customer sentiment")
    key_issue: str = Field(description="One-sentence summary of the core issue")
    confidence_score: float = Field(description="Confidence between 0.0 and 1.0")
    reasoning: str = Field(description="Explanation for the classification")

CLASSIFICATION_SYSTEM_PROMPT = """You are an expert AI Support Triage Dispatcher for an enterprise SaaS platform.
Analyze incoming customer support tickets and output a structured JSON response.

Taxonomy:
1. Category (pick exactly one):
   - "Billing": Payments, invoices, subscriptions, refunds, pricing, VAT, credit cards.
   - "Technical Support": API errors (500s, 429s), bugs, downtime, SAML/SSO, webhooks, crashes, rate limits.
   - "Feature Request": Enhancements, roadmap requests, new integrations, UI requests.
   - "Account Access": Password resets, 2FA recovery, team permissions, workspace transfers, deletion.
   - "Complaint": Severe customer dissatisfaction, SLA breach claims, slow support complaints, churn threats.

2. Urgency Priority:
   - "P1 - Critical": Production outages, massive API failures, direct SLA violations, revenue halted.
   - "P2 - High": Broken workflows, duplicate billing, lockout of admin, high customer frustration.
   - "P3 - Medium": Non-critical bugs, routine questions, standard permission updates, invoices.
   - "P4 - Low": Feature requests, cosmetic suggestions, minor feedback.

3. Customer Sentiment:
   - "Positive", "Neutral", "Frustrated", "Angry".

Return ONLY valid JSON with keys:
{
  "category": "...",
  "urgency": "...",
  "sentiment": "...",
  "key_issue": "...",
  "confidence_score": 0.95,
  "reasoning": "..."
}"""

class TicketClassifier:
    """Classifies customer tickets into categories, urgency, sentiment, and confidence."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def classify(self, ticket: Dict[str, Any]) -> ClassificationResult:
        """Runs classification on a single ticket dictionary."""
        prompt = (
            f"Customer Tier: {ticket.get('customer_tier', 'Starter')}\n"
            f"Subject: {ticket.get('subject', '')}\n"
            f"Body:\n{ticket.get('body', '')}\n"
        )

        try:
            raw_response = self.llm.generate_json(
                prompt=prompt,
                system_prompt=CLASSIFICATION_SYSTEM_PROMPT
            )

            # Normalization and guardrails
            category = raw_response.get("category", "Technical Support")
            if category not in config.CATEGORIES:
                category = "Technical Support"

            urgency = raw_response.get("urgency", "P3 - Medium")
            if urgency not in config.URGENCIES:
                urgency = "P3 - Medium"

            sentiment = raw_response.get("sentiment", "Neutral")
            if sentiment not in config.SENTIMENTS:
                sentiment = "Neutral"

            confidence = float(raw_response.get("confidence_score", 0.85))
            confidence = max(0.0, min(1.0, confidence))

            return ClassificationResult(
                category=category,
                urgency=urgency,
                sentiment=sentiment,
                key_issue=raw_response.get("key_issue", ticket.get("subject", "Customer query")),
                confidence_score=confidence,
                reasoning=raw_response.get("reasoning", "Classified based on contextual relevance."),
            )

        except Exception as e:
            logger.error(f"Classification failed: {e}. Falling back to default.")
            return ClassificationResult(
                category="Technical Support",
                urgency="P3 - Medium",
                sentiment="Neutral",
                key_issue=ticket.get("subject", "Support request"),
                confidence_score=0.70,
                reasoning="Fallback due to processing exception.",
            )
