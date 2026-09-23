"""
Unit Tests for Ticket Classifier
"""

import pytest
from src.classifier import TicketClassifier

@pytest.fixture
def classifier():
    return TicketClassifier()

def test_classify_billing_ticket(classifier):
    ticket = {
        "subject": "Charged twice on invoice #INV-98231",
        "body": "We were billed $499 twice on September 19th. Please refund immediately.",
        "customer_tier": "Enterprise"
    }
    res = classifier.classify(ticket)
    assert res.category == "Billing"
    assert res.urgency in ["P1 - Critical", "P2 - High"]
    assert res.confidence_score >= 0.80

def test_classify_p1_technical_outage(classifier):
    ticket = {
        "subject": "API throwing 500 Internal Server Error on /v2/transactions endpoint",
        "body": "Our production payment ingestion is completely halted!",
        "customer_tier": "Pro"
    }
    res = classifier.classify(ticket)
    assert res.category == "Technical Support"
    assert res.urgency == "P1 - Critical"

def test_classify_feature_request(classifier):
    ticket = {
        "subject": "Would love dark mode support and custom color themes",
        "body": "Our design team spends 8+ hours a day in the analytics dashboard and we'd really appreciate a native Dark Mode option.",
        "customer_tier": "Starter"
    }
    res = classifier.classify(ticket)
    assert res.category == "Feature Request"
    assert res.urgency == "P4 - Low"
    assert res.sentiment == "Positive"
