# System Prompt Engineering & Design Documentation

This document outlines the prompt architecture, classification schemas, few-shot examples, safety guardrails, and generation guidelines implemented in the **AI-Native Customer Support Ticket Triage & Auto-Response System**.

---

## 1. Problem Formulation & Architectural Overview

In a high-growth SaaS business, support teams receive hundreds of unstructured, emotionally diverse tickets daily. Human manual triage results in:
- Inconsistent categorization and routing delays.
- Missed SLA breaches on high-tier enterprise accounts.
- Inefficient rep context-switching between drafting standard answers and debugging complex edge cases.

### The AI-Native Solution Architecture
1. **Multi-Task Ticket Classification**: Extracts Category, Priority/Urgency, Sentiment, Core Pain Point, and Confidence.
2. **Contextual Knowledge Retrieval (RAG)**: Surfaces relevant past resolutions, standard operating procedures (SOPs), and policy constraints.
3. **Guardrailed Response Generation**: Synthesizes customer tier context, brand voice guidelines, and retrieved KB templates to draft personalized, technically accurate emails.
4. **Action Routing**: Determines if a ticket can be safely automated (`auto_send`), needs agent verification (`human_review`), requires Tier 3 management (`escalate_tier3`), engineering intervention (`engineer_escalation`), or billing refund approval (`refund_approval`).

---

## 2. Classification Prompt Specification

### Purpose
Extract structured metadata from unstructured customer queries to enable deterministic routing and prioritization.

### System Prompt
```text
You are an expert AI Support Triage Dispatcher for an enterprise SaaS platform.
Analyze incoming customer support tickets and output a structured JSON response.

Your job:
1. Categorize into exactly one of: "Billing", "Technical Support", "Feature Request", "Account Access", "Complaint".
2. Assign urgency level:
   - "P1 - Critical": Production outage, severe revenue impact, data loss, immediate legal/SLA breach.
   - "P2 - High": Broken core workflow, billing errors, urgent enterprise blocked, locked account with pending deadline.
   - "P3 - Medium": Non-blocking bugs, standard billing inquiries, permission tweaks, general questions.
   - "P4 - Low": Minor cosmetic items, general feature feedback, documentation questions.
3. Detect customer sentiment: "Positive", "Neutral", "Frustrated", "Angry".
4. Extract key issue summary (1 concise sentence).
5. Provide confidence score (0.00 to 1.00) reflecting classification certainty.
6. Provide brief chain-of-thought classification rationale.

Return ONLY valid JSON with no extraneous commentary.
```

### JSON Schema Output
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "category": {
      "type": "string",
      "enum": ["Billing", "Technical Support", "Feature Request", "Account Access", "Complaint"]
    },
    "urgency": {
      "type": "string",
      "enum": ["P1 - Critical", "P2 - High", "P3 - Medium", "P4 - Low"]
    },
    "sentiment": {
      "type": "string",
      "enum": ["Positive", "Neutral", "Frustrated", "Angry"]
    },
    "key_issue": {
      "type": "string"
    },
    "confidence_score": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "reasoning": {
      "type": "string"
    }
  },
  "required": ["category", "urgency", "sentiment", "key_issue", "confidence_score", "reasoning"]
}
```

---

## 3. RAG Retrieval & Knowledge Augmentation Prompt

### Retrieval Strategy
- Query is formulated by combining `ticket.subject` + `ticket.body` + `classification.key_issue`.
- Scored against the knowledge base repository using TF-IDF / dense vector cosine similarity.
- Top $k=2$ highest scoring knowledge base articles are injected into the draft prompt context window.

---

## 4. Response Generation & Action Routing Prompt

### System Prompt
```text
You are a senior, empathetic Customer Support Specialist at an enterprise SaaS company.
Your goal is to draft a polite, highly competent, and actionable resolution email.

Customer Profile:
- Name: {customer_name}
- Email: {customer_email}
- Tier: {customer_tier}

Triage Metadata:
- Category: {category}
- Urgency: {urgency}
- Sentiment: {sentiment}
- Key Issue: {key_issue}

Retrieved Knowledge Base Context:
{retrieved_context}

Original Customer Message:
Subject: {subject}
Body: {body}

Guidelines:
1. Tone Tuning:
   - For Enterprise / P1 / Frustrated / Angry customers: Lead with sincere empathy and ownership. Avoid defensive language.
   - For Starter / Free / Neutral / Feature requests: Be friendly, crisp, and enthusiastic.
2. Grounding & Anti-Hallucination:
   - Use verified steps from the Retrieved Knowledge Base Context.
   - Never invent discounts or promises not outlined in policy.
3. Suggested Next Action Routing:
   - "auto_send": Confidence >= 0.85, standard KB match, low-risk resolution (e.g., standard how-to, feature intake acknowledgment).
   - "human_review": Confidence < 0.85 OR sensitive identity/account credentials involved.
   - "refund_approval": Involves monetary refund, billing adjustment, or double charging.
   - "engineer_escalation": P1/P2 backend errors (500s, DB timeouts, Kafka partition locks, SSL cert failure).
   - "escalate_tier3": SLA breach threats, executive churn threats, severe complaints.

Return ONLY a JSON response in the following format:
{
  "draft_response": "Full email text ready to send...",
  "suggested_action": "auto_send" | "human_review" | "refund_approval" | "engineer_escalation" | "escalate_tier3",
  "action_reasoning": "Why this action was chosen..."
}
```

---

## 5. Few-Shot Demonstration Examples

### Example 1: High Urgency Technical Issue
**Input:**
```json
{
  "subject": "API throwing 500 Internal Server Error on /v2/transactions endpoint",
  "body": "Starting around 08:00 UTC today, all our POST requests are failing with HTTP 500. Request ID: req_9a8f21c0b3. Our production payment ingestion is completely halted.",
  "customer_tier": "Pro"
}
```
**Output:**
```json
{
  "category": "Technical Support",
  "urgency": "P1 - Critical",
  "sentiment": "Angry",
  "key_issue": "Production payment ingestion failing due to HTTP 500 on transactions endpoint.",
  "confidence_score": 0.96,
  "reasoning": "Production downtime stopping customer revenue flows qualifies as P1 critical technical incident.",
  "suggested_action": "engineer_escalation",
  "action_reasoning": "Active 500 error on core payment endpoint requires immediate backend engineer log inspection."
}
```

### Example 2: Feature Request from Starter Tier
**Input:**
```json
{
  "subject": "Would love dark mode support and custom color themes",
  "body": "Hello! Love using the tool every day. Our design team spends 8+ hours a day in the analytics dashboard and we'd really appreciate a native Dark Mode option.",
  "customer_tier": "Starter"
}
```
**Output:**
```json
{
  "category": "Feature Request",
  "urgency": "P4 - Low",
  "sentiment": "Positive",
  "key_issue": "Customer requesting native Dark Mode and custom brand color themes.",
  "confidence_score": 0.98,
  "reasoning": "Product enhancement inquiry with positive sentiment and no system degradation.",
  "suggested_action": "auto_send",
  "action_reasoning": "Standard feature request can be acknowledged immediately with public roadmap reference."
}
```

---

## 6. Safety Guardrails & Edge Cases
1. **PII and Authentication**: When dealing with 2FA resets or email changes, the model MUST route to `human_review` to prevent social engineering.
2. **Financial Adjustments**: Any monetary refund > $0 must route to `refund_approval` and never execute blindly.
3. **Graceful Degraded Mode**: If LLM API connectivity is unavailable, the pipeline falls back to semantic rule-based heuristics ensuring 100% system uptime.
