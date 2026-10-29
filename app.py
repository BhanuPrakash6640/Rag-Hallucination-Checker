"""Demo UI.  Run:  streamlit run app.py"""

import html

import streamlit as st

from hallcheck.config import Settings
from hallcheck.verifier import CONTRADICTED, SKIPPED, SUPPORTED, UNSUPPORTED, HallucinationChecker

st.set_page_config(page_title="RAG Hallucination Checker", page_icon="🔎", layout="wide")

STYLE = {
    SUPPORTED: ("#d1f2dc", "#0f5132", "✅ Supported"),
    CONTRADICTED: ("#f8d7da", "#842029", "❌ Contradicted"),
    UNSUPPORTED: ("#fff3cd", "#664d03", "⚠️ Not in sources"),
    SKIPPED: ("#e9ecef", "#495057", "➖ Skipped"),
}

EXAMPLE_SOURCE = (
    "Full-time employees receive 18 days of paid leave per calendar year. Unused paid leave does not "
    "carry over to the next year. Up to 6 unused days can instead be encashed at the end of December. "
    "Employees receive 10 days of sick leave per year."
)
EXAMPLE_ANSWER = (
    "You get 18 paid leave days per year, and unused days carry over to next year. "
    "You also get 12 days of sick leave. Up to 6 unused days can be encashed in December."
)


@st.cache_resource(show_spinner="Loading the NLI model (first run downloads it)...")
def get_checker(model_name: str) -> HallucinationChecker:
    s = Settings()
    s.nli_model = model_name
    return HallucinationChecker.from_settings(s)


@st.cache_resource(show_spinner="Loading the document index and LLM client...")
def get_pipeline(model_name: str):
    from hallcheck.pipeline import RAGPipeline

    return RAGPipeline(checker=get_checker(model_name))


def render(result) -> None:
    counts = result.counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", "⚠️ Check needed" if result.hallucinated else "✅ Grounded")
    c2.metric("Supported claims", f"{counts[SUPPORTED]}/{len(result.checked)}")
    c3.metric("Risk score", f"{result.risk_score:.2f}")
    c4.metric("Check time", f"{result.latency_ms:.0f} ms")

    chips = []
    for c in result.claims:
        bg, fg, label = STYLE[c.verdict]
        chips.append(
            f'<span title="{html.escape(label)}: {html.escape(c.reason)}" style="background:{bg};color:{fg};'
            f'padding:2px 6px;border-radius:6px;margin:2px;line-height:2;display:inline">{html.escape(c.claim)}</span>'
        )
    st.markdown("**Answer, claim by claim** (hover a claim for the reason)")
    st.markdown(" ".join(chips), unsafe_allow_html=True)

    st.markdown("**Details**")
    for c in result.claims:
        _, _, label = STYLE[c.verdict]
        with st.expander(f"{label} — {c.claim}", expanded=c.verdict in (CONTRADICTED, UNSUPPORTED)):
            if c.verdict == SKIPPED:
                st.write(f"Skipped: {c.reason}")
                continue
            st.write(f"**Why:** {c.reason}")
            st.write(f"Entailment {c.entailment:.2f} · Contradiction {c.contradiction:.2f}")
            if c.evidence:
                src = f" ({c.evidence_source})" if c.evidence_source else ""
                st.info(f"Closest evidence{src}: {c.evidence}")


st.title("🔎 RAG Hallucination Checker")
st.caption("Splits an answer into claims and checks each one against the source documents with a small NLI model.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "NLI model",
        [
            Settings().nli_model,
            "cross-encoder/nli-deberta-v3-xsmall",
            "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        ],
        help="Bigger models are more accurate but slower.",
    )
    checker = get_checker(model_name)
    checker.entail_threshold = st.slider("Support threshold", 0.1, 0.95, checker.entail_threshold, 0.05)
    checker.check_numbers = st.checkbox("Number check", value=checker.check_numbers)
    checker.split_compound = st.checkbox("Split compound sentences", value=checker.split_compound)

tab_check, tab_rag = st.tabs(["Check any answer", "Ask the documents"])

with tab_check:
    st.write("Paste a source text and an AI answer. No LLM needed.")
    left, right = st.columns(2)
    source = left.text_area("Source text", EXAMPLE_SOURCE, height=220)
    answer = right.text_area("AI answer to check", EXAMPLE_ANSWER, height=220)
    if st.button("Check answer", type="primary"):
        render(checker.check(answer, [source]))

with tab_rag:
    st.write("Ask about the files in `data/docs`. Needs the index (`python build_index.py`) and an LLM (see `.env`).")
    question = st.text_input("Question", "How many paid leave days do I get, and do unused days carry over?")
    if st.button("Ask", type="primary"):
        try:
            pipeline = get_pipeline(model_name)
        except Exception as e:
            st.error(f"Could not start the RAG pipeline: {e}")
            st.stop()
        with st.spinner("Retrieving, answering and checking..."):
            try:
                out = pipeline.ask(question)
            except Exception as e:
                st.error(f"Request failed. Is your LLM running and set in .env? Details: {e}")
                st.stop()
        t = out["timing_ms"]
        st.caption(f"Retrieval {t['retrieval']:.0f} ms · Generation {t['generation']:.0f} ms · Checking {t['checking']:.0f} ms")
        render(out["check"])
        with st.expander("Retrieved passages"):
            for i, c in enumerate(out["chunks"], 1):
                st.markdown(f"**[{i}] {c['source']}** (score {c['score']:.2f})")
                st.write(c["text"])
