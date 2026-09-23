"""
RAG Knowledge Base Retriever using Semantic TF-IDF Vector Search
"""

import json
import logging
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src import config

logger = logging.getLogger(__name__)

class KnowledgeBaseRetriever:
    """
    RAG retriever that indexes past resolution templates and documentation,
    computing similarity against incoming tickets.
    """

    def __init__(self, kb_path: str = None):
        self.kb_path = kb_path or str(config.KB_FILE)
        self.documents: List[Dict[str, Any]] = []
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
        )
        self.doc_vectors = None
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """Loads and indexes KB documents from JSON."""
        try:
            with open(self.kb_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            self._build_index()
            logger.info(f"Loaded and indexed {len(self.documents)} knowledge base articles.")
        except Exception as e:
            logger.error(f"Failed to load knowledge base from {self.kb_path}: {e}")
            self.documents = []

    def _build_index(self):
        """Builds TF-IDF vector index over document corpus."""
        if not self.documents:
            return

        corpus = []
        for doc in self.documents:
            keywords_str = " ".join(doc.get("keywords", []))
            text = f"{doc.get('title', '')} {doc.get('category', '')} {keywords_str} {doc.get('content', '')}"
            corpus.append(text)

        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = None, category_filter: str = None) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant KB articles matching the query.
        Returns a list of dicts with document data + 'similarity_score'.
        """
        top_k = top_k or config.TOP_K_RAG_RESULTS
        if not self.documents or self.doc_vectors is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()

        # Rank documents by score
        ranked_indices = similarities.argsort()[::-1]

        results = []
        for idx in ranked_indices:
            doc = self.documents[idx]
            score = float(similarities[idx])

            # Optional category filter
            if category_filter and doc.get("category", "").lower() != category_filter.lower():
                # Allow cross-category if score is high, but prefer matched category
                score *= 0.85

            doc_result = dict(doc)
            doc_result["similarity_score"] = round(score, 4)
            results.append(doc_result)

            if len(results) >= top_k:
                break

        return results

    def format_context_for_prompt(self, matched_docs: List[Dict[str, Any]]) -> str:
        """Formats retrieved documents into a clean context block for LLM prompt."""
        if not matched_docs:
            return "No specific knowledge base article matched. Use standard support best practices."

        context_lines = []
        for i, doc in enumerate(matched_docs, 1):
            context_lines.append(
                f"[Doc {i}] ID: {doc.get('id')} | Title: {doc.get('title')} (Relevance: {doc.get('similarity_score', 0.0):.2f})\n"
                f"Category: {doc.get('category')}\n"
                f"Guidance: {doc.get('content')}\n"
                f"Resolution Pattern: {doc.get('resolution_template', '')}\n"
            )
        return "\n---\n".join(context_lines)
