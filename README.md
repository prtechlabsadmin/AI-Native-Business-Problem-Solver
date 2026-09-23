# AI-Native Business Problem Solver
### Customer Support Ticket Triage & Auto-Response System

> Portfolio project demonstrating AI-native problem solving for a real business vertical.

## Business Problem
Customer support teams are overwhelmed by high ticket volume. Manual triage and first-response drafting is slow and inconsistent, leading to longer resolution times and lower CSAT.

## Solution Overview
An AI-native pipeline that:
1. Classifies incoming tickets
2. Retrieves similar past resolutions (lightweight RAG)
3. Generates a professional draft response with confidence score
4. Suggests next human action

## Tech Stack
- Python 3.10+
- OpenAI / Anthropic API (or local alternative)
- pandas, scikit-learn (for baseline evaluation)
- Optional: Streamlit for demo

## Project Structure.
├── data/
│   └── tickets.csv
├── src/
│   ├── classifier.py
│   ├── rag_pipeline.py
│   └── main.py
├── app.py                 # Streamlit demo
├── prompts.md
├── requirements.txt
└── README.md
text## Quick Start
```bash
git clone <repo-url>
cd ai-native-business-problem-solver
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# Add your API key to .env
python src/main.py
# or
streamlit run app.py
How It Works

Load ticket
Classify category + urgency
Retrieve top-k similar historical tickets
Generate draft response using structured prompt
Return structured output (category, confidence, draft, suggested action)

Evaluation

Classification accuracy on held-out set
Manual review of response quality (examples included)
Latency and cost notes

Key Learnings Demonstrated

Problem framing from ambiguous business need
Prompt engineering & structured output
Lightweight RAG
End-to-end ownership of an AI solution
Clear documentation for stakeholders

Future Improvements

Multi-agent workflow
Human-in-the-loop feedback loop
Integration with Zendesk / Intercom
Continuous evaluation dashboard

Author
