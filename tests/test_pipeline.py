"""
Unit and Integration Tests for End-to-End Ticket Triage Pipeline
"""

import pytest
from src.pipeline import TicketTriagePipeline, TriageResponse

@pytest.fixture
def pipeline():
    return TicketTriagePipeline()

def test_pipeline_end_to_end(pipeline):
    ticket = {
        "ticket_id": "TEST-101",
        "customer_name": "Sarah Jenkins",
        "customer_email": "s.jenkins@acme.com",
        "customer_tier": "Enterprise",
        "subject": "Charged twice on invoice #INV-98231",
        "body": "Hi team, we were billed twice for our Enterprise subscription. Please refund the extra charge."
    }

    response: TriageResponse = pipeline.process_ticket(ticket)

    assert response.ticket_id == "TEST-101"
    assert response.category == "Billing"
    assert response.customer_tier == "Enterprise"
    assert len(response.draft_response) > 20
    assert response.suggested_action in ["refund_approval", "human_review", "auto_send"]
    assert len(response.retrieved_articles) > 0
    assert response.latency_ms > 0

def test_pipeline_batch_processing(pipeline):
    tickets = [
        {
            "ticket_id": "T1",
            "customer_name": "Alex",
            "customer_email": "alex@test.com",
            "customer_tier": "Starter",
            "subject": "Dark mode feature request",
            "body": "Can you please add dark mode?"
        },
        {
            "ticket_id": "T2",
            "customer_name": "Taylor",
            "customer_email": "taylor@test.com",
            "customer_tier": "Enterprise",
            "subject": "Production 500 error on API",
            "body": "Our production transactions endpoint is throwing 500 error!"
        }
    ]

    results = pipeline.process_batch(tickets)
    assert len(results) == 2
    assert results[0].category == "Feature Request"
    assert results[1].category == "Technical Support"
