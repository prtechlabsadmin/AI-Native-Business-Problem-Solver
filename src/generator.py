"""
Context-Aware Response Generation & Next Action Routing Module
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from src.llm import LLMClient
from src import config

logger = logging.getLogger(__name__)

class GenerationResult(BaseModel):
    draft_response: str = Field(description="The formulated customer response email")
    suggested_action: str = Field(description="Action: auto_send, human_review, refund_approval, engineer_escalation, escalate_tier3")
    action_reasoning: str = Field(description="Rationale for the suggested action")

GENERATION_SYSTEM_PROMPT = """You are a senior, empathetic Customer Support Specialist for an enterprise SaaS platform.
Your objective is to craft a professional, helpful draft reply and designate the appropriate workflow action.

Guidelines:
1. Tone Tuning:
   - For Enterprise / P1 / Angry / Frustrated users: Sincere empathy, high accountability, zero defensiveness.
   - For Standard / Positive / Neutral users: Helpful, courteous, direct, and concise.
2. Grounding:
   - Base technical steps and policies strictly on the provided Knowledge Base context.
   - If information is missing, ask polite clarifying questions rather than hallucinating answers.
3. Action Decision Rules:
   - "auto_send": High confidence (>=0.85), standard routine solution, zero financial or security risk.
   - "human_review": Account security (2FA/password lockout), ambiguous inquiries, or confidence < 0.85.
   - "refund_approval": Involves duplicate charges, plan downgrades with refund requests, or billing disputes.
   - "engineer_escalation": Critical service outages, 500 API errors, database locks, Kafka stream halts.
   - "escalate_tier3": SLA breach complaints, churn threats, or severe escalations.

Return pure JSON only in the format:
{
  "draft_response": "...",
  "suggested_action": "auto_send | human_review | refund_approval | engineer_escalation | escalate_tier3",
  "action_reasoning": "..."
}"""

class ResponseGenerator:
    """Generates customer response drafts and suggested workflow actions."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def generate(
        self,
        ticket: Dict[str, Any],
        classification: Any,
        retrieved_context: str,
    ) -> GenerationResult:
        """Generates draft response grounded in retrieved context and classification metadata."""
        prompt = (
            f"CUSTOMER PROFILE:\n"
            f"- Name: {ticket.get('customer_name', 'Customer')}\n"
            f"- Email: {ticket.get('customer_email', 'N/A')}\n"
            f"- Tier: {ticket.get('customer_tier', 'Starter')}\n\n"
            f"TRIAGE CLASSIFICATION:\n"
            f"- Category: {classification.category}\n"
            f"- Urgency: {classification.urgency}\n"
            f"- Sentiment: {classification.sentiment}\n"
            f"- Key Issue: {classification.key_issue}\n"
            f"- Confidence Score: {classification.confidence_score:.2f}\n\n"
            f"RETRIEVED KNOWLEDGE BASE CONTEXT:\n{retrieved_context}\n\n"
            f"CUSTOMER TICKET:\n"
            f"Subject: {ticket.get('subject', '')}\n"
            f"Body:\n{ticket.get('body', '')}\n"
        )

        try:
            raw_response = self.llm.generate_json(
                prompt=prompt,
                system_prompt=GENERATION_SYSTEM_PROMPT
            )

            draft = raw_response.get("draft_response", "")
            action = raw_response.get("suggested_action", "human_review")
            reasoning = raw_response.get("action_reasoning", "Standard routing applied.")

            # Rule-based safety guardrails & overrides
            if classification.urgency == "P1 - Critical" and classification.category == "Technical Support":
                action = "engineer_escalation"
                reasoning = "P1 Critical technical incident overrides to Engineering Escalation."
            elif classification.urgency == "P1 - Critical" and classification.category == "Complaint":
                action = "escalate_tier3"
                reasoning = "P1 SLA breach / complaint overrides to Tier 3 Escalation."
            elif "refund" in ticket.get("subject", "").lower() or "charge" in ticket.get("subject", "").lower():
                if action == "auto_send":
                    action = "refund_approval"
                    reasoning = "Billing financial adjustments require supervisor refund approval."
            elif classification.confidence_score < config.AUTO_SEND_CONFIDENCE_THRESHOLD and action == "auto_send":
                action = "human_review"
                reasoning = f"Confidence score ({classification.confidence_score:.2f}) below auto-send threshold ({config.AUTO_SEND_CONFIDENCE_THRESHOLD})."

            if action not in config.ACTIONS:
                action = "human_review"

            return GenerationResult(
                draft_response=draft or self._fallback_draft(ticket, classification),
                suggested_action=action,
                action_reasoning=reasoning,
            )

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return GenerationResult(
                draft_response=self._fallback_draft(ticket, classification),
                suggested_action="human_review",
                action_reasoning="System exception encountered during response drafting.",
            )

    def _fallback_draft(self, ticket: Dict[str, Any], classification: Any) -> str:
        name = ticket.get("customer_name", "Valued Customer")
        return (
            f"Hello {name},\n\n"
            f"Thank you for contacting our support team regarding '{ticket.get('subject', '')}'.\n\n"
            f"We have registered your ticket under {classification.category} priority {classification.urgency}. "
            f"Our team is currently reviewing your workspace diagnostics to provide a resolution.\n\n"
            f"Best regards,\nCustomer Support Operations"
        )
