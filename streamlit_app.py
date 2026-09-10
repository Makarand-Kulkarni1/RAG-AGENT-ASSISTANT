import json
from pathlib import Path

import streamlit as st
import pandas as pd

from app.config import DATA_DIR
from app.rag import build_index, answer_question
from eval.evaluate import evaluate

st.set_page_config(page_title="RAG Agent Assistant", layout="wide", page_icon="🤖")

# ---------------------------------------------------------------
# Custom CSS — gradient background, card-style boxes, styled tabs
# ---------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    h1 {
        background: linear-gradient(90deg, #f857a6, #ff5858);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 18px;
        backdrop-filter: blur(6px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(248,87,166,0.15), rgba(255,88,88,0.10));
        border: 1px solid rgba(248,87,166,0.35);
        border-radius: 14px;
        padding: 16px 20px;
        text-align: center;
    }

    .metric-card .value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ff85b3;
    }

    .metric-card .label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.65);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        color: rgba(255,255,255,0.7);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #f857a6, #ff5858) !important;
        color: white !important;
        font-weight: 600;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #f857a6, #ff5858);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 4px 16px rgba(248,87,166,0.4);
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        color: white;
    }

    .answer-box {
        background: rgba(255,255,255,0.06);
        border-left: 4px solid #f857a6;
        border-radius: 8px;
        padding: 18px 22px;
        margin-top: 16px;
    }

    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_index():
    return build_index(DATA_DIR)


index, chunks, metadata = get_index()
log_path = Path("logs/queries.jsonl")

st.title("🤖 RAG Agent Assistant — Dashboard")

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ System Stats")

    total_queries = 0
    if log_path.exists():
        total_queries = len(log_path.read_text().strip().split("\n"))

    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:14px;">
        <div class="value">{len(chunks)}</div>
        <div class="label">Chunks Indexed</div>
    </div>
    <div class="metric-card">
        <div class="value">{total_queries}</div>
        <div class="label">Queries Logged</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🧬 **Embedding:** all-MiniLM-L6-v2")
    st.caption("🧠 **Chat model:** openai/gpt-oss-20b (Groq)")
    st.caption("🔍 **Vector store:** FAISS (cosine sim)")


tab1, tab2, tab3 = st.tabs(["💬  Ask a Question", "📜  Query Log", "📊  Evaluation Results"])

# --- Tab 1 ---
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Ask something about the indexed documents")
    question = st.text_input("Your question:", placeholder="e.g. What ML projects has this person built?")

    if st.button("Ask →") and question:
        with st.spinner("Thinking..."):
            answer = answer_question(question, index, chunks, metadata)
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 2 ---
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Recent Queries")
    if log_path.exists():
        lines = log_path.read_text().strip().split("\n")
        st.caption(f"Showing the {min(20, len(lines))} most recent of {len(lines)} total queries")
        for line in reversed(lines[-20:]):
            entry = json.loads(line)
            with st.expander(f"🕐 {entry['timestamp']} — {entry['question']}"):
                st.markdown(entry["answer"])
    else:
        st.info("No queries logged yet — ask something in the first tab.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 3 ---
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Retrieval Evaluation")

    if st.button("▶ Run Evaluation"):
        with st.spinner("Running evaluation suite..."):
            results, accuracy = evaluate("eval/eval_set.json", DATA_DIR)

        c1, c2, c3 = st.columns(3)
        for col, value, label in [
            (c1, f"{accuracy:.0%}", "Accuracy"),
            (c2, sum(r["hit"] for r in results), "Passed"),
            (c3, len(results), "Total Questions"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
                <div class="value">{value}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        df = pd.DataFrame([
            {
                "Question": r["question"],
                "Status": "✅ PASS" if r["hit"] else "❌ FAIL",
                "Keywords Found": ", ".join(r["keywords_found"]) or "(none)",
            }
            for r in results
        ])
        st.dataframe(df, use_container_width=True)

        with st.expander("ℹ️ Why this matters / known limitations"):
            st.markdown(
                "Retrieval accuracy is measured by checking whether expected "
                "keywords appear in the retrieved context for each question. "
                "Failures here typically point to either chunking issues "
                "(a fact split across chunk boundaries) or embedding-model "
                "limitations (a lightweight model not strongly associating "
                "a paraphrased question with the right chunk)."
            )
    else:
        st.info("Click **Run Evaluation** to test retrieval accuracy against the eval set.")
    st.markdown('</div>', unsafe_allow_html=True)
