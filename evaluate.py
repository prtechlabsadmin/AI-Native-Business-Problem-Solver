#!/usr/bin/env python3
"""
CLI Evaluation Script for Support Ticket Triage Pipeline
Run: python evaluate.py
"""

import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from src.evaluator import PipelineEvaluator
from src.pipeline import TicketTriagePipeline

console = Console()

def main():
    console.print(Panel.fit(
        "[bold cyan]AI-Native Customer Support Ticket Triage Pipeline[/bold cyan]\n"
        "[dim]System Benchmark & Evaluation Suite[/dim]",
        border_style="cyan"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Evaluating pipeline against benchmark dataset...", total=None)
        pipeline = TicketTriagePipeline()
        evaluator = PipelineEvaluator(pipeline=pipeline)
        eval_output = evaluator.run_evaluation()

    metrics = eval_output["metrics"]

    # Overview Metrics Table
    summary_table = Table(title="Overall Benchmark Summary", border_style="bright_blue")
    summary_table.add_column("Metric", style="bold white")
    summary_table.add_column("Score / Value", style="bold green")

    summary_table.add_row("Total Evaluation Tickets", str(metrics.get("total_samples", 0)))
    summary_table.add_row("Category Classification Accuracy", f"{metrics.get('category_accuracy', 0.0) * 100:.2f}%")
    summary_table.add_row("Macro F1-Score", f"{metrics.get('category_f1_macro', 0.0):.4f}")
    summary_table.add_row("Macro Precision", f"{metrics.get('category_precision_macro', 0.0):.4f}")
    summary_table.add_row("Macro Recall", f"{metrics.get('category_recall_macro', 0.0):.4f}")
    summary_table.add_row("Urgency Prediction Accuracy", f"{metrics.get('urgency_accuracy', 0.0) * 100:.2f}%")
    summary_table.add_row("Sentiment Detection Accuracy", f"{metrics.get('sentiment_accuracy', 0.0) * 100:.2f}%")
    summary_table.add_row("Action Routing Accuracy", f"{metrics.get('action_routing_accuracy', 0.0) * 100:.2f}%")
    summary_table.add_row("Average Confidence Score", f"{metrics.get('avg_confidence', 0.0) * 100:.2f}%")
    summary_table.add_row("Average Latency per Ticket", f"{metrics.get('avg_latency_ms', 0.0):.2f} ms")

    console.print(summary_table)

    # Per-Category Breakdown
    if "classification_report" in metrics:
        cat_table = Table(title="Per-Category Performance Breakdown", border_style="magenta")
        cat_table.add_column("Category", style="bold cyan")
        cat_table.add_column("Precision", style="white")
        cat_table.add_column("Recall", style="white")
        cat_table.add_column("F1-Score", style="white")
        cat_table.add_column("Support", style="white")

        for cat, scores in metrics["classification_report"].items():
            if isinstance(scores, dict) and cat not in ["accuracy", "macro avg", "weighted avg"]:
                cat_table.add_row(
                    cat,
                    f"{scores.get('precision', 0):.2f}",
                    f"{scores.get('recall', 0):.2f}",
                    f"{scores.get('f1-score', 0):.2f}",
                    str(int(scores.get('support', 0)))
                )

        console.print(cat_table)

    # Action Routing Distribution
    action_table = Table(title="Triage Action Routing Distribution", border_style="yellow")
    action_table.add_column("Suggested Action", style="bold yellow")
    action_table.add_column("Ticket Count", style="bold white")
    action_table.add_column("Percentage", style="bold green")

    total_tickets = metrics.get("total_samples", 1)
    for action, count in metrics.get("action_distribution", {}).items():
        pct = (count / total_tickets) * 100
        action_table.add_row(action, str(count), f"{pct:.1f}%")

    console.print(action_table)
    console.print("[bold green][SUCCESS] Evaluation completed successfully![/bold green]\n")

if __name__ == "__main__":
    main()
