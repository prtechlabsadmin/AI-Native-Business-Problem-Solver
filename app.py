"""
Streamlit Web Application: AI-Native Customer Support Ticket Triage & Auto-Response System
"""

import json
import pandas as pd
import streamlit as st
from src.pipeline import TicketTriagePipeline, TriageResponse
from src.evaluator import PipelineEvaluator
from src.rag import KnowledgeBaseRetriever
from src import config

# Page configuration
st.set_page_config(
    page_title="AI Support Triage & Auto-Response System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .badge-p1 { background-color: #FEE2E2; color: #991B1B; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-p2 { background-color: #FEF3C7; color: #92400E; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-p3 { background-color: #E0E7FF; color: #3730A3; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-p4 { background-color: #F3F4F6; color: #374151; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    return TicketTriagePipeline()

@st.cache_data
def get_sample_tickets():
    with open(config.TICKETS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def get_kb_data():
    with open(config.KB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

pipeline = get_pipeline()
sample_tickets = get_sample_tickets()
kb_docs = get_kb_data()

# Sidebar
st.sidebar.title("🎫 Triage Control Plane")
st.sidebar.markdown("**AI-Native Support Dispatcher**")

provider_option = st.sidebar.selectbox(
    "Active LLM Provider",
    ["Mock Heuristic Engine (Offline / 0-Cost)", "OpenAI (GPT-4o-mini)", "Anthropic (Claude 3.5 Haiku)", "Google Gemini"],
    index=0
)

st.sidebar.divider()
st.sidebar.markdown("### System Architecture")
st.sidebar.info(
    "**Pipeline Flow:**\n"
    "1. Multi-Task Classifier (Zero-shot / Few-shot)\n"
    "2. Vector RAG Retriever (TF-IDF & Cosine Similarity)\n"
    "3. Context-Augmented Response Generator\n"
    "4. Deterministic Action Router with Guardrails"
)

# Navigation
tabs = st.tabs([
    "🎯 Live Triage Studio",
    "📊 Batch Triage & Operations",
    "📚 RAG Knowledge Base Explorer",
    "📈 Benchmark & Model Evaluation",
])

# -------------------------------------------------------------
# TAB 1: Live Triage Studio
# -------------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="main-header">Real-Time Ticket Triage Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Classify, retrieve past SOPs, formulate drafted responses, and route incoming tickets.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.subheader("📥 Incoming Ticket")

        template_choice = st.selectbox(
            "Load Preloaded Customer Scenario:",
            ["(Custom Ticket Entry)"] + [f"{t['ticket_id']}: {t['subject']}" for t in sample_tickets[:10]]
        )

        selected_ticket = None
        if template_choice != "(Custom Ticket Entry)":
            t_id = template_choice.split(":")[0].strip()
            selected_ticket = next((t for t in sample_tickets if t.get("ticket_id") == t_id), None)

        with st.form("triage_form"):
            c1, c2 = st.columns(2)
            with c1:
                cust_name = st.text_input("Customer Name", value=selected_ticket["customer_name"] if selected_ticket else "Sarah Jenkins")
                cust_tier = st.selectbox("Customer Tier", ["Enterprise", "Pro", "Starter", "Free"], index=0 if not selected_ticket else ["Enterprise", "Pro", "Starter", "Free"].index(selected_ticket.get("customer_tier", "Pro")))
            with c2:
                cust_email = st.text_input("Customer Email", value=selected_ticket["customer_email"] if selected_ticket else "s.jenkins@acmecorp.io")

            subject = st.text_input("Subject", value=selected_ticket["subject"] if selected_ticket else "Charged twice on invoice #INV-98231")
            body = st.text_area("Ticket Body", value=selected_ticket["body"] if selected_ticket else "Hi team, I noticed this morning that our company credit card was billed $499 twice on September 19th for invoice #INV-98231. We only have one active Enterprise workspace. Could you please look into this and refund the extra $499 immediately?", height=140)

            submitted = st.form_submit_button("⚡ Run AI Triage & Response Pipeline", use_container_width=True)

    with col2:
        st.subheader("🔍 AI Triage Results & Draft")

        ticket_payload = {
            "customer_name": cust_name,
            "customer_email": cust_email,
            "customer_tier": cust_tier,
            "subject": subject,
            "body": body,
        }

        # Run triage
        result: TriageResponse = pipeline.process_ticket(ticket_payload)

        # Classification Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Category", result.category)
        with m2:
            st.metric("Urgency", result.urgency)
        with m3:
            st.metric("Sentiment", result.sentiment)
        with m4:
            st.metric("Confidence", f"{result.confidence_score * 100:.1f}%")

        # Action Routing Alert
        action_colors = {
            "auto_send": "success",
            "human_review": "warning",
            "refund_approval": "info",
            "engineer_escalation": "error",
            "escalate_tier3": "error",
        }
        status_fn = getattr(st, action_colors.get(result.suggested_action, "info"))
        status_fn(f"**Suggested Workflow Action:** `{result.suggested_action.upper()}`\n\n*Rationale:* {result.action_reasoning}")

        # Drafted Response Card
        st.markdown("#### ✉️ Grounded AI Draft Response")
        st.text_area("Generated Reply", value=result.draft_response, height=180)

        # RAG Context Viewer
        with st.expander("📚 Inspect Retrieved RAG Knowledge Base Articles", expanded=True):
            if result.retrieved_articles:
                for doc in result.retrieved_articles:
                    st.markdown(f"**[{doc.get('id')}] {doc.get('title')}** (Similarity Score: `{doc.get('similarity_score', 0):.2f}`)")
                    st.markdown(f"*Category:* `{doc.get('category')}`")
                    st.caption(doc.get('content'))
                    st.divider()
            else:
                st.write("No matching articles found.")

        st.caption(f"⚡ Processing Latency: **{result.latency_ms:.1f} ms**")

# -------------------------------------------------------------
# TAB 2: Batch Triage & Operations Dashboard
# -------------------------------------------------------------
with tabs[1]:
    st.markdown('<div class="main-header">Batch Triage & Support Operations Queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">High-throughput automated triage and queue analytics across customer support tickets.</div>', unsafe_allow_html=True)

    if st.button("🚀 Process Full Benchmark Batch (60 Tickets)", type="primary"):
        with st.spinner("Processing batch queue through AI pipeline..."):
            batch_results = pipeline.process_batch(sample_tickets)
            st.session_state["batch_results"] = batch_results

    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        data_rows = []
        for r in results:
            data_rows.append({
                "Ticket ID": r.ticket_id,
                "Customer": r.customer_name,
                "Tier": r.customer_tier,
                "Subject": r.subject,
                "Category": r.category,
                "Urgency": r.urgency,
                "Sentiment": r.sentiment,
                "Action": r.suggested_action,
                "Confidence": f"{r.confidence_score * 100:.1f}%",
                "Latency (ms)": r.latency_ms,
            })
        df = pd.DataFrame(data_rows)

        # Operational KPIs
        k1, k2, k3, k4 = st.columns(4)
        total = len(df)
        auto_count = len(df[df["Action"] == "auto_send"])
        refund_count = len(df[df["Action"] == "refund_approval"])
        eng_count = len(df[df["Action"] == "engineer_escalation"])

        k1.metric("Total Tickets Processed", total)
        k2.metric("Auto-Resolution Rate", f"{(auto_count / total) * 100:.1f}%", f"{auto_count} tickets")
        k3.metric("Billing Approval Queue", f"{refund_count} tickets", "Requires supervisor")
        k4.metric("Eng Escalations", f"{eng_count} tickets", "P1/P2 Incidents")

        st.markdown("### 📋 Triage Queue")

        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            cat_filter = st.multiselect("Filter by Category", options=df["Category"].unique())
        with f2:
            urg_filter = st.multiselect("Filter by Urgency", options=df["Urgency"].unique())
        with f3:
            act_filter = st.multiselect("Filter by Action", options=df["Action"].unique())

        filtered_df = df.copy()
        if cat_filter:
            filtered_df = filtered_df[filtered_df["Category"].isin(cat_filter)]
        if urg_filter:
            filtered_df = filtered_df[filtered_df["Urgency"].isin(urg_filter)]
        if act_filter:
            filtered_df = filtered_df[filtered_df["Action"].isin(act_filter)]

        st.dataframe(filtered_df, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: Knowledge Base Explorer
# -------------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="main-header">RAG Knowledge Base & Resolution Templates</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Vector search index powering accurate retrieval-augmented response generation.</div>', unsafe_allow_html=True)

    search_query = st.text_input("🔍 Test Vector Retrieval with Custom Query:", value="how to recover lost 2fa authenticator")

    if search_query:
        retriever = KnowledgeBaseRetriever()
        matches = retriever.retrieve(search_query, top_k=3)

        st.markdown(f"**Top {len(matches)} Retrieved SOPs:**")
        for m in matches:
            with st.container():
                st.markdown(f"### {m.get('title')} (`{m.get('id')}`)")
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.info(f"**Category:** {m.get('category')}\n\n**Similarity Score:** `{m.get('similarity_score', 0):.3f}`")
                with c2:
                    st.markdown(f"**Documentation Content:**\n{m.get('content')}")
                    st.markdown(f"**Standard Resolution Pattern:**\n```\n{m.get('resolution_template')}\n```")
                st.divider()

# -------------------------------------------------------------
# TAB 4: Benchmark & Evaluation
# -------------------------------------------------------------
with tabs[3]:
    st.markdown('<div class="main-header">System Benchmark & Quality Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Statistical performance evaluation against held-out ground truth support tickets.</div>', unsafe_allow_html=True)

    if st.button("📊 Run Full Pipeline Benchmark", type="primary"):
        with st.spinner("Computing precision, recall, F1, and action calibration..."):
            evaluator = PipelineEvaluator(pipeline=pipeline)
            eval_res = evaluator.run_evaluation()
            st.session_state["eval_res"] = eval_res

    if "eval_res" in st.session_state:
        metrics = st.session_state["eval_res"]["metrics"]

        # KPI Metrics
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Category Accuracy", f"{metrics.get('category_accuracy', 0) * 100:.1f}%")
        e2.metric("Macro F1-Score", f"{metrics.get('category_f1_macro', 0):.3f}")
        e3.metric("Urgency Accuracy", f"{metrics.get('urgency_accuracy', 0) * 100:.1f}%")
        e4.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.1f} ms")

        # Classification Report Table
        if "classification_report" in metrics:
            st.markdown("### 📊 Per-Category Precision, Recall & F1-Score")
            report_data = []
            for cat, scores in metrics["classification_report"].items():
                if isinstance(scores, dict) and cat not in ["accuracy", "macro avg", "weighted avg"]:
                    report_data.append({
                        "Category": cat,
                        "Precision": f"{scores.get('precision', 0):.2f}",
                        "Recall": f"{scores.get('recall', 0):.2f}",
                        "F1-Score": f"{scores.get('f1-score', 0):.2f}",
                        "Support (Samples)": int(scores.get('support', 0)),
                    })
            st.table(pd.DataFrame(report_data))

        # Action distribution
        st.markdown("### 🔀 Workflow Action Routing Breakdown")
        act_df = pd.DataFrame(
            list(metrics.get("action_distribution", {}).items()),
            columns=["Action", "Ticket Count"]
        )
        st.bar_chart(act_df.set_index("Action"))
