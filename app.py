import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import os
import io
import re
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai  # pip install openai  (works with OpenAI or Groq via base_url)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Resume Screener",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # fast, free, no API key needed

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF file (bytes)."""
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_chunks.append(text)
    return "\n".join(text_chunks).strip()


def clean_text(text: str) -> str:
    """Basic cleanup — collapse whitespace, remove special chars."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
    return text.strip()


def embed_texts(model, texts: list[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def score_resumes(jd_text: str, resume_texts: list[str], model) -> list[float]:
    all_texts = [jd_text] + resume_texts
    embeddings = embed_texts(model, all_texts)
    jd_emb = embeddings[0:1]
    resume_embs = embeddings[1:]
    scores = cosine_similarity(jd_emb, resume_embs)[0]
    return (scores * 100).tolist()   # return as percentage


def get_llm_explanation(jd_text: str, resume_text: str, score: float, api_key: str, use_groq: bool) -> str:
    """
    Ask an LLM to explain the match between the JD and a resume.
    Falls back to a rule-based summary if no API key is provided.
    """
    if not api_key:
        return _fallback_explanation(jd_text, resume_text, score)

    prompt = f"""You are an expert HR recruiter.

JOB DESCRIPTION:
{jd_text[:1500]}

CANDIDATE RESUME:
{resume_text[:1500]}

MATCH SCORE: {score:.1f}%

In 3-5 bullet points, briefly explain:
1. Key skills / experience that MATCH the job description
2. Notable GAPS or missing requirements
3. One overall recommendation sentence

Be concise. Use bullet points starting with ✅ for matches and ❌ for gaps."""

    try:
        if use_groq:
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            model_id = "llama3-8b-8192"
        else:
            client = openai.OpenAI(api_key=api_key)
            model_id = "gpt-3.5-turbo"

        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ LLM explanation unavailable: {e}\n\n" + _fallback_explanation(jd_text, resume_text, score)


def _fallback_explanation(jd_text: str, resume_text: str, score: float) -> str:
    """Simple keyword overlap explanation when no LLM is configured."""
    jd_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", jd_text.lower()))
    res_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", resume_text.lower()))
    common = jd_words & res_words
    top_matches = sorted(common, key=lambda w: len(w), reverse=True)[:10]
    missing = sorted(jd_words - res_words, key=lambda w: len(w), reverse=True)[:5]

    lines = [f"**Match score: {score:.1f}%** (semantic similarity)"]
    if top_matches:
        lines.append("✅ Shared keywords: " + ", ".join(top_matches))
    if missing:
        lines.append("❌ Potentially missing: " + ", ".join(missing))
    lines.append("_(Add an API key in the sidebar for a detailed LLM explanation.)_")
    return "\n\n".join(lines)


def rank_color(score: float) -> str:
    if score >= 70:
        return "🟢"
    elif score >= 45:
        return "🟡"
    return "🔴"


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    st.subheader("LLM Explanation (optional)")
    llm_provider = st.radio("Provider", ["None (keyword fallback)", "OpenAI", "Groq (free tier)"])
    api_key = ""
    use_groq = False

    if llm_provider != "None (keyword fallback)":
        use_groq = llm_provider.startswith("Groq")
        placeholder = "gsk_..." if use_groq else "sk-..."
        api_key = st.text_input("API Key", type="password", placeholder=placeholder)
        if use_groq:
            st.caption("Get a free key at [console.groq.com](https://console.groq.com)")
        else:
            st.caption("Get a key at [platform.openai.com](https://platform.openai.com)")

    st.markdown("---")
    st.subheader("Scoring")
    min_score = st.slider("Minimum score to show (%)", 0, 80, 0)

    st.markdown("---")
    st.markdown("**Embed model:** `all-MiniLM-L6-v2`")
    st.markdown("Runs locally — no data leaves your machine for scoring.")


# ─────────────────────────────────────────────
#  MAIN UI
# ─────────────────────────────────────────────
st.title("🔍 Smart Resume Screener")
st.caption("Upload resumes and a job description — get ranked candidates with AI explanations.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Job Description")
    jd_input = st.text_area(
        "Paste the job description here",
        height=280,
        placeholder="We are looking for a Machine Learning Engineer with 2+ years of experience in Python, PyTorch, and NLP...",
        label_visibility="collapsed",
    )

with col2:
    st.subheader("Resumes (PDF)")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF resumes",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

st.markdown("---")

run_btn = st.button("🚀 Screen Resumes", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
#  PROCESSING
# ─────────────────────────────────────────────
if run_btn:
    if not jd_input.strip():
        st.error("Please paste a job description before screening.")
        st.stop()
    if not uploaded_files:
        st.error("Please upload at least one PDF resume.")
        st.stop()

    with st.spinner("Loading embedding model…"):
        model = load_embed_model()

    # Parse PDFs
    resume_data = []
    parse_errors = []
    progress = st.progress(0, text="Parsing resumes…")

    for i, f in enumerate(uploaded_files):
        try:
            raw = extract_text_from_pdf(f.read())
            cleaned = clean_text(raw)
            if len(cleaned) < 50:
                parse_errors.append(f"{f.name}: too little text extracted (possibly scanned image PDF)")
            else:
                resume_data.append({"name": f.name, "text": cleaned})
        except Exception as e:
            parse_errors.append(f"{f.name}: {e}")
        progress.progress((i + 1) / len(uploaded_files), text=f"Parsed {i+1}/{len(uploaded_files)} resumes…")

    progress.empty()

    if parse_errors:
        for err in parse_errors:
            st.warning(f"⚠️ {err}")

    if not resume_data:
        st.error("No resumes could be parsed. Please check your files.")
        st.stop()

    # Score
    with st.spinner("Computing similarity scores…"):
        jd_clean = clean_text(jd_input)
        scores = score_resumes(jd_clean, [r["text"] for r in resume_data], model)
        for i, r in enumerate(resume_data):
            r["score"] = round(scores[i], 1)

    # Filter & rank
    resume_data = [r for r in resume_data if r["score"] >= min_score]
    resume_data.sort(key=lambda x: x["score"], reverse=True)

    if not resume_data:
        st.info(f"No resumes scored above {min_score}%. Try lowering the minimum score in the sidebar.")
        st.stop()

    # ── Results header ──
    st.subheader(f"Results — {len(resume_data)} candidate(s) ranked")

    # Summary table
    table_data = [
        {
            "Rank": i + 1,
            "Candidate": r["name"].replace(".pdf", ""),
            "Match Score": f"{rank_color(r['score'])} {r['score']}%",
        }
        for i, r in enumerate(resume_data)
    ]
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # CSV download
    csv_df = pd.DataFrame([
        {"Rank": i+1, "Candidate": r["name"], "Score (%)": r["score"]}
        for i, r in enumerate(resume_data)
    ])
    st.download_button(
        "⬇️ Download rankings as CSV",
        csv_df.to_csv(index=False),
        file_name="resume_rankings.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── Detailed cards ──
    st.subheader("Candidate Breakdown")

    explain_progress = st.progress(0, text="Generating explanations…")

    for i, r in enumerate(resume_data):
        with st.expander(
            f"#{i+1}  {r['name'].replace('.pdf','')}  —  {rank_color(r['score'])} {r['score']}%",
            expanded=(i == 0),
        ):
            explanation = get_llm_explanation(jd_clean, r["text"], r["score"], api_key, use_groq)
            st.markdown(explanation)

            st.caption("**Resume text preview (first 500 chars):**")
            st.code(r["text"][:500] + "…", language=None)

        explain_progress.progress((i + 1) / len(resume_data), text=f"Explained {i+1}/{len(resume_data)}…")
        time.sleep(0.05)  # tiny pause so progress bar is visible

    explain_progress.empty()
    st.success("✅ Screening complete!")