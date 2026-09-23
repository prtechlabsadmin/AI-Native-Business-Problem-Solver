"""
Evaluation Engine for Support Ticket Triage Pipeline
Computes Accuracy, Precision, Recall, F1, Latency, and Routing Analytics.
"""

import json
import logging
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from src.pipeline import TicketTriagePipeline, TriageResponse
from src import config

logger = logging.getLogger(__name__)

class PipelineEvaluator:
    """Evaluates pipeline accuracy and operational efficiency on benchmark datasets."""

    def __init__(self, pipeline: TicketTriagePipeline = None):
        self.pipeline = pipeline or TicketTriagePipeline()

    def run_evaluation(self, test_tickets: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes evaluation over benchmark tickets and calculates statistical metrics.
        """
        if test_tickets is None:
            with open(config.TICKETS_FILE, "r", encoding="utf-8") as f:
                test_tickets = json.load(f)

        y_true_category = []
        y_pred_category = []
        y_true_urgency = []
        y_pred_urgency = []
        y_true_sentiment = []
        y_pred_sentiment = []
        y_true_action = []
        y_pred_action = []

        confidences = []
        latencies = []
        results = []

        for ticket in test_tickets:
            triage_res: TriageResponse = self.pipeline.process_ticket(ticket)
            results.append(triage_res)

            # Category
            if "ground_truth_category" in ticket:
                y_true_category.append(ticket["ground_truth_category"])
                y_pred_category.append(triage_res.category)

            # Urgency
            if "ground_truth_urgency" in ticket:
                y_true_urgency.append(ticket["ground_truth_urgency"])
                y_pred_urgency.append(triage_res.urgency)

            # Sentiment
            if "ground_truth_sentiment" in ticket:
                y_true_sentiment.append(ticket["ground_truth_sentiment"])
                y_pred_sentiment.append(triage_res.sentiment)

            # Action
            if "expected_action" in ticket:
                y_true_action.append(ticket["expected_action"])
                y_pred_action.append(triage_res.suggested_action)

            confidences.append(triage_res.confidence_score)
            latencies.append(triage_res.latency_ms)

        # Calculate metrics
        metrics = {
            "total_samples": len(test_tickets),
            "avg_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        }

        if y_true_category:
            cat_acc = accuracy_score(y_true_category, y_pred_category)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true_category, y_pred_category, average="macro", zero_division=0
            )
            metrics["category_accuracy"] = round(float(cat_acc), 4)
            metrics["category_precision_macro"] = round(float(prec), 4)
            metrics["category_recall_macro"] = round(float(rec), 4)
            metrics["category_f1_macro"] = round(float(f1), 4)
            metrics["classification_report"] = classification_report(
                y_true_category, y_pred_category, output_dict=True, zero_division=0
            )
            labels = sorted(list(set(y_true_category + y_pred_category)))
            cm = confusion_matrix(y_true_category, y_pred_category, labels=labels)
            metrics["confusion_matrix"] = {
                "labels": labels,
                "matrix": cm.tolist()
            }

        if y_true_urgency:
            metrics["urgency_accuracy"] = round(float(accuracy_score(y_true_urgency, y_pred_urgency)), 4)

        if y_true_sentiment:
            metrics["sentiment_accuracy"] = round(float(accuracy_score(y_true_sentiment, y_pred_sentiment)), 4)

        if y_true_action:
            metrics["action_routing_accuracy"] = round(float(accuracy_score(y_true_action, y_pred_action)), 4)

        # Action breakdown
        action_counts = {}
        for r in results:
            action_counts[r.suggested_action] = action_counts.get(r.suggested_action, 0) + 1
        metrics["action_distribution"] = action_counts

        return {
            "metrics": metrics,
            "results": results
        }
