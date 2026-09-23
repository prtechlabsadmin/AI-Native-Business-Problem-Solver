#!/usr/bin/env python3
"""
CLI Application for AI-Native Customer Support Ticket Triage & Auto-Response System
Usage:
    python main.py demo
    python main.py triage --subject "..." --body "..." --tier "Pro"
    python main.py interactive
    python main.py batch [--file data/tickets.json]
    python main.py eval
"""

import sys
import json
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from src.pipeline import TicketTriagePipeline, TriageResponse
from src.evaluator import PipelineEvaluator
from src import config

console = Console()

def display_triage_result(res: TriageResponse):
    """Renders a visually structured triage card."""
    action_colors = {
        "auto_send": "green",
        "human_review": "yellow",
        "refund_approval": "magenta",
        "engineer_escalation": "red",
        "escalate_tier3": "bold red",
    }
    action_color = action_colors.get(res.suggested_action, "cyan")

    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="left", ratio=1)

    grid.add_row(
        f"[bold]Customer:[/bold] {res.customer_name} ({res.customer_email})",
        f"[bold]Tier:[/bold] [cyan]{res.customer_tier}[/cyan]"
    )
    grid.add_row(
        f"[bold]Category:[/bold] [bright_blue]{res.category}[/bright_blue]",
        f"[bold]Urgency:[/bold] [bold red]{res.urgency}[/bold red]"
    )
    grid.add_row(
        f"[bold]Sentiment:[/bold] {res.sentiment}",
        f"[bold]Confidence Score:[/bold] {res.confidence_score * 100:.1f}%"
    )
    grid.add_row(
        f"[bold]Suggested Action:[/bold] [{action_color}]{res.suggested_action.upper()}[/{action_color}]",
        f"[bold]Latency:[/bold] {res.latency_ms:.1f} ms"
    )

    console.print(Panel(
        grid,
        title=f"[bold]Ticket Triage Analysis: {res.subject}[/bold]",
        border_style="blue"
    ))

    # Action Reasoning & RAG Articles
    info_table = Table(box=None, show_header=False, expand=True)
    info_table.add_column("Field", style="bold dim", width=18)
    info_table.add_column("Value")
    info_table.add_row("Key Issue:", res.key_issue)
    info_table.add_row("Action Rationale:", f"[{action_color}]{res.action_reasoning}[/{action_color}]")

    if res.retrieved_articles:
        kb_refs = ", ".join([f"{doc.get('id')} ({doc.get('title')})" for doc in res.retrieved_articles])
        info_table.add_row("Matched KB SOPs:", f"[dim]{kb_refs}[/dim]")

    console.print(Panel(info_table, title="Triage Diagnostics", border_style="dim"))

    # Draft Response
    console.print(Panel(
        res.draft_response,
        title=f"[bold green]AI Drafted Customer Response ({res.suggested_action})[/bold green]",
        border_style="green"
    ))
    console.print()

def run_demo():
    """Runs an automated showcase of diverse support tickets."""
    console.print(Panel.fit(
        "[bold cyan]AI-Native Support Ticket Triage Pipeline - Live Demo[/bold cyan]\n"
        "[dim]Simulating automated classification, RAG retrieval, response drafting, and routing.[/dim]",
        border_style="cyan"
    ))

    pipeline = TicketTriagePipeline()

    sample_tickets = [
        {
            "ticket_id": "DEMO-01",
            "customer_name": "Sarah Jenkins",
            "customer_email": "s.jenkins@acmecorp.io",
            "customer_tier": "Enterprise",
            "subject": "Charged twice on invoice #INV-98231",
            "body": "Hi team, I noticed this morning that our company credit card was billed $499 twice on September 19th for invoice #INV-98231. We only have one active Enterprise workspace. Could you please look into this and refund the extra $499 immediately?"
        },
        {
            "ticket_id": "DEMO-02",
            "customer_name": "David Chen",
            "customer_email": "dchen@fintechflow.com",
            "customer_tier": "Pro",
            "subject": "API throwing 500 Internal Server Error on /v2/transactions endpoint",
            "body": "Starting around 08:00 UTC today, all our POST requests to /v2/transactions are failing with HTTP 500. Request ID: req_9a8f21c0b3. Our production payment ingestion is completely halted!"
        },
        {
            "ticket_id": "DEMO-03",
            "customer_name": "Elena Rostova",
            "customer_email": "elena@designstudio.co",
            "customer_tier": "Starter",
            "subject": "Would love dark mode support and custom color themes",
            "body": "Hello! Love using the tool every day. Our design team spends 8+ hours a day in the analytics dashboard and we'd really appreciate a native Dark Mode option. Is this on the roadmap?"
        }
    ]

    for i, t in enumerate(sample_tickets, 1):
        console.print(f"[bold yellow]=== Case Study {i} of {len(sample_tickets)} ===[/bold yellow]")
        result = pipeline.process_ticket(t)
        display_triage_result(result)

def run_interactive():
    """Interactive CLI mode to triage tickets on the fly."""
    console.print(Panel("[bold cyan]Interactive Support Ticket Triage Terminal[/bold cyan]", border_style="cyan"))
    pipeline = TicketTriagePipeline()

    while True:
        console.print("[dim]Enter ticket details below (or type 'exit' as subject to quit):[/dim]")
        subject = Prompt.ask("[bold white]Subject[/bold white]")
        if subject.strip().lower() in ["exit", "quit", "q"]:
            console.print("[yellow]Exiting interactive session.[/yellow]")
            break

        body = Prompt.ask("[bold white]Body[/bold white]")
        tier = Prompt.ask("[bold white]Customer Tier[/bold white]", choices=["Enterprise", "Pro", "Starter", "Free"], default="Pro")
        name = Prompt.ask("[bold white]Customer Name[/bold white]", default="Customer")
        email = Prompt.ask("[bold white]Customer Email[/bold white]", default="customer@domain.com")

        ticket = {
            "customer_name": name,
            "customer_email": email,
            "customer_tier": tier,
            "subject": subject,
            "body": body,
        }

        with console.status("[bold green]Processing through AI Triage Pipeline...[/bold green]"):
            res = pipeline.process_ticket(ticket)

        display_triage_result(res)

def run_single_triage(args):
    """Processes a single ticket from CLI arguments."""
    pipeline = TicketTriagePipeline()
    ticket = {
        "customer_name": args.name or "Customer",
        "customer_email": args.email or "customer@example.com",
        "customer_tier": args.tier or "Pro",
        "subject": args.subject,
        "body": args.body,
    }
    res = pipeline.process_ticket(ticket)
    display_triage_result(res)

def run_batch(file_path: str = None):
    """Processes batch file of tickets and displays summary table."""
    path = file_path or str(config.TICKETS_FILE)
    console.print(f"[bold cyan]Loading batch tickets from:[/bold cyan] {path}")

    with open(path, "r", encoding="utf-8") as f:
        tickets = json.load(f)

    pipeline = TicketTriagePipeline()
    console.print(f"[dim]Processing {len(tickets)} tickets...[/dim]")
    results = pipeline.process_batch(tickets)

    table = Table(title=f"Batch Triage Results ({len(results)} Tickets)", border_style="bright_blue")
    table.add_column("ID", style="bold white", width=10)
    table.add_column("Customer", style="white", width=16)
    table.add_column("Tier", style="cyan", width=10)
    table.add_column("Category", style="magenta", width=16)
    table.add_column("Urgency", style="red", width=14)
    table.add_column("Sentiment", style="yellow", width=10)
    table.add_column("Action", style="bold green", width=18)
    table.add_column("Latency", style="dim", width=10)

    for r in results:
        table.add_row(
            r.ticket_id or "N/A",
            r.customer_name[:14],
            r.customer_tier,
            r.category,
            r.urgency,
            r.sentiment,
            r.suggested_action,
            f"{r.latency_ms:.0f} ms",
        )

    console.print(table)
    console.print(f"[bold green][SUCCESS] Successfully processed {len(results)} tickets![/bold green]\n")

def main():
    parser = argparse.ArgumentParser(description="AI-Native Customer Support Ticket Triage System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo
    subparsers.add_parser("demo", help="Run automated demonstration showcase")

    # Interactive
    subparsers.add_parser("interactive", help="Start interactive live triage shell")

    # Triage single
    triage_parser = subparsers.add_parser("triage", help="Triage a single ticket")
    triage_parser.add_argument("--subject", required=True, help="Ticket subject")
    triage_parser.add_argument("--body", required=True, help="Ticket message body")
    triage_parser.add_argument("--tier", default="Pro", help="Customer tier (Enterprise/Pro/Starter/Free)")
    triage_parser.add_argument("--name", default="Customer", help="Customer Name")
    triage_parser.add_argument("--email", default="customer@domain.com", help="Customer Email")

    # Batch
    batch_parser = subparsers.add_parser("batch", help="Batch process tickets from JSON file")
    batch_parser.add_argument("--file", default=str(config.TICKETS_FILE), help="Path to tickets JSON file")

    # Eval
    subparsers.add_parser("eval", help="Run benchmark evaluation suite")

    args = parser.parse_args()

    if args.command == "demo" or not args.command:
        run_demo()
    elif args.command == "interactive":
        run_interactive()
    elif args.command == "triage":
        run_single_triage(args)
    elif args.command == "batch":
        run_batch(args.file)
    elif args.command == "eval":
        from evaluate import main as eval_main
        eval_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
