"""
End-to-End Customer Support Ticket Triage Pipeline Orchestrator
"""

import time
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from src.classifier import TicketClassifier, ClassificationResult
from src.rag import KnowledgeBaseRetriever
from src.generator import ResponseGenerator, GenerationResult
from src.llm import LLMClient
from src import config

logger = logging.getLogger(__name__)

class TriageResponse(BaseModel):
    ticket_id: Optional[str] = None
    customer_name: str
    customer_email: str
    customer_tier: str
    subject: str
    body: str

    # Classification
    category: str
    urgency: str
    sentiment: str
    key_issue: str
    confidence_score: float
    classification_reasoning: str

    # RAG Context
    retrieved_articles: List[Dict[str, Any]]

    # Response & Next Action
    draft_response: str
    suggested_action: str
    action_reasoning: str

    # Performance
    latency_ms: float

class TicketTriagePipeline:
    """
    Main pipeline coordinating Classification -> RAG Retrieval -> Response Generation -> Triage Decision.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        kb_path: Optional[str] = None,
    ):
        self.llm = LLMClient(provider=llm_provider)
        self.retriever = KnowledgeBaseRetriever(kb_path=kb_path)
        self.classifier = TicketClassifier(llm_client=self.llm)
        self.generator = ResponseGenerator(llm_client=self.llm)

    def process_ticket(self, ticket: Dict[str, Any]) -> TriageResponse:
        """
        Executes end-to-end triage on a single incoming ticket.
        """
        start_time = time.time()

        # Step 1: Classify ticket
        classification: ClassificationResult = self.classifier.classify(ticket)

        # Step 2: Retrieve relevant RAG Knowledge Base articles
        query_text = f"{ticket.get('subject', '')} {ticket.get('body', '')} {classification.key_issue}"
        retrieved_docs = self.retriever.retrieve(
            query=query_text,
            top_k=config.TOP_K_RAG_RESULTS,
            category_filter=classification.category,
        )
        context_block = self.retriever.format_context_for_prompt(retrieved_docs)

        # Step 3: Generate Draft Response and Action Recommendation
        generation: GenerationResult = self.generator.generate(
            ticket=ticket,
            classification=classification,
            retrieved_context=context_block,
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return TriageResponse(
            ticket_id=ticket.get("ticket_id"),
            customer_name=ticket.get("customer_name", "Valued Customer"),
            customer_email=ticket.get("customer_email", "support@customer.com"),
            customer_tier=ticket.get("customer_tier", "Starter"),
            subject=ticket.get("subject", ""),
            body=ticket.get("body", ""),
            category=classification.category,
            urgency=classification.urgency,
            sentiment=classification.sentiment,
            key_issue=classification.key_issue,
            confidence_score=classification.confidence_score,
            classification_reasoning=classification.reasoning,
            retrieved_articles=retrieved_docs,
            draft_response=generation.draft_response,
            suggested_action=generation.suggested_action,
            action_reasoning=generation.action_reasoning,
            latency_ms=latency_ms,
        )

    def process_batch(self, tickets: List[Dict[str, Any]]) -> List[TriageResponse]:
        """Processes a list of tickets sequentially."""
        results = []
        for idx, ticket in enumerate(tickets, 1):
            logger.info(f"Processing ticket {idx}/{len(tickets)}: {ticket.get('ticket_id', 'N/A')}")
            res = self.process_ticket(ticket)
            results.append(res)
        return results
