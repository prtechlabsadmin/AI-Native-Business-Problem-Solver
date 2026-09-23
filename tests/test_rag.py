"""
Unit Tests for RAG Knowledge Base Retriever
"""

import pytest
from src.rag import KnowledgeBaseRetriever

@pytest.fixture
def retriever():
    return KnowledgeBaseRetriever()

def test_retriever_initialization(retriever):
    assert len(retriever.documents) > 0
    assert retriever.doc_vectors is not None

def test_retriever_billing_query(retriever):
    query = "Customer charged twice on invoice and needs a refund"
    results = retriever.retrieve(query, top_k=2)
    assert len(results) <= 2
    assert len(results) > 0
    top_doc = results[0]
    assert "similarity_score" in top_doc
    assert top_doc["similarity_score"] > 0
    assert top_doc["category"] == "Billing" or "Double Charges" in top_doc["title"]

def test_retriever_technical_query(retriever):
    query = "500 internal server error on transactions API endpoint"
    results = retriever.retrieve(query, top_k=2)
    assert len(results) > 0
    top_doc = results[0]
    assert "500" in top_doc["title"] or top_doc["category"] == "Technical Support"

def test_context_formatting(retriever):
    results = retriever.retrieve("password reset", top_k=2)
    formatted = retriever.format_context_for_prompt(results)
    assert isinstance(formatted, str)
    assert "[Doc 1]" in formatted
