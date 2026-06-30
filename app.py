import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import io
import re
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Resume Screener",
    page_icon="🔍",
    layout="wide",
)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_chunks.append(text)
    return "\n".join(text_chunks).strip()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def embed_texts(model, texts):
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def score_resumes(jd_text, resume_texts, model):
    all_texts = [jd_text] + resume_texts
    embeddings = embed_texts(model, all_texts)
    jd_emb = embeddings[0:1]
    resume_embs = embeddings[1:]
    scores = cosine_similarity(jd_emb, resume_embs)[0]
    return (scores * 100).tolist()


def get_llm_client(api_key, use_groq):
    if use_groq:
        return openai.OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1"), "llama-3.3-70b-versatile"
    return openai.OpenAI(api_key=api_key), "gpt-3.5-turbo"


def get_llm_explanation(jd_text, resume_text, score, api_key, use_groq):
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
        client, model_id = get_llm_client(api_key, use_groq)
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ LLM explanation unavailable: {e}\n\n" + _fallback_explanation(jd_text, resume_text, score)


def get_resume_improver(jd_text, resume_text, score, api_key, use_groq):
    """NEW FEATURE: AI suggests specific improvements to boost resume score."""
    if not api_key:
        return _fallback_improver(jd_text, resume_text)

    prompt = f"""You are an expert resume coach and career consultant.

JOB DESCRIPTION:
{jd_text[:1500]}

CANDIDATE RESUME:
{resume_text[:1500]}

CURRENT MATCH SCORE: {score:.1f}%

The candidate wants to improve their resume to better match this job description.
Give them a concrete, actionable improvement plan:

1. **Top 5 Keywords to Add** — list exact keywords/phrases from the JD missing in their resume
2. **Skills to Highlight** — which existing skills they should make more prominent
3. **Experience Rewrites** — suggest 2-3 specific bullet point rewrites using stronger action verbs and JD keywords
4. **Sections to Add** — any missing sections (e.g. certifications, projects, tools)
5. **Predicted Score After Changes** — estimate the new score if they follow your advice

Be specific. Use the exact words from the JD. Format clearly with headers."""

    try:
        client, model_id = get_llm_client(api_key, use_groq)
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Improver unavailable: {e}\n\n" + _fallback_improver(jd_text, resume_text)


def _fallback_improver(jd_text, resume_text):
    jd_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", jd_text.lower()))
    res_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", resume_text.lower()))
    missing = sorted(jd_words - res_words, key=lambda w: len(w), reverse=True)[:10]
    lines = [
        "**📝 Keywords to add to your resume:**",
        ", ".join(missing) if missing else "No obvious missing keywords found.",
        "",
        "_(Add a Groq or OpenAI API key in the sidebar for detailed AI-powered rewrite suggestions.)_"
    ]
    return "\n\n".join(lines)


def _fallback_explanation(jd_text, resume_text, score):
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


def rank_color(score):
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
#  TABS
# ─────────────────────────────────────────────
st.title("🔍 Smart Resume Screener")
st.caption("Upload resumes and a job description — get ranked candidates with AI explanations.")

tab1, tab2 = st.tabs(["📋 Screen Resumes", "✏️ Resume Score Improver"])

# ══════════════════════════════════════════════
#  TAB 1 — SCREEN RESUMES (original feature)
# ══════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Job Description")
        jd_input = st.text_area(
            "Paste the job description here",
            height=280,
            placeholder="We are looking for a Machine Learning Engineer with 2+ years of experience in Python, PyTorch, and NLP...",
            label_visibility="collapsed",
            key="jd_screen",
        )

    with col2:
        st.subheader("Resumes (PDF)")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF resumes",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="resumes_screen",
        )

    st.markdown("---")
    run_btn = st.button("🚀 Screen Resumes", type="primary", use_container_width=True)

    if run_btn:
        if not jd_input.strip():
            st.error("Please paste a job description before screening.")
            st.stop()
        if not uploaded_files:
            st.error("Please upload at least one PDF resume.")
            st.stop()

        with st.spinner("Loading embedding model…"):
            model = load_embed_model()

        resume_data = []
        parse_errors = []
        progress = st.progress(0, text="Parsing resumes…")

        for i, f in enumerate(uploaded_files):
            try:
                raw = extract_text_from_pdf(f.read())
                cleaned = clean_text(raw)
                if len(cleaned) < 50:
                    parse_errors.append(f"{f.name}: too little text extracted")
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
            st.error("No resumes could be parsed.")
            st.stop()

        with st.spinner("Computing similarity scores…"):
            jd_clean = clean_text(jd_input)
            scores = score_resumes(jd_clean, [r["text"] for r in resume_data], model)
            for i, r in enumerate(resume_data):
                r["score"] = round(scores[i], 1)

        resume_data = [r for r in resume_data if r["score"] >= min_score]
        resume_data.sort(key=lambda x: x["score"], reverse=True)

        if not resume_data:
            st.info(f"No resumes scored above {min_score}%.")
            st.stop()

        st.subheader(f"Results — {len(resume_data)} candidate(s) ranked")

        table_data = [
            {"Rank": i+1, "Candidate": r["name"].replace(".pdf",""), "Match Score": f"{rank_color(r['score'])} {r['score']}%"}
            for i, r in enumerate(resume_data)
        ]
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        csv_df = pd.DataFrame([{"Rank": i+1, "Candidate": r["name"], "Score (%)": r["score"]} for i, r in enumerate(resume_data)])
        st.download_button("⬇️ Download rankings as CSV", csv_df.to_csv(index=False), file_name="resume_rankings.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Candidate Breakdown")
        explain_progress = st.progress(0, text="Generating explanations…")

        for i, r in enumerate(resume_data):
            with st.expander(f"#{i+1}  {r['name'].replace('.pdf','')}  —  {rank_color(r['score'])} {r['score']}%", expanded=(i == 0)):
                explanation = get_llm_explanation(jd_clean, r["text"], r["score"], api_key, use_groq)
                st.markdown(explanation)
                st.caption("**Resume text preview (first 500 chars):**")
                st.code(r["text"][:500] + "…", language=None)
            explain_progress.progress((i + 1) / len(resume_data), text=f"Explained {i+1}/{len(resume_data)}…")
            time.sleep(0.05)

        explain_progress.empty()
        st.success("✅ Screening complete!")


# ══════════════════════════════════════════════
#  TAB 2 — RESUME SCORE IMPROVER (new feature)
# ══════════════════════════════════════════════
with tab2:
    st.subheader("✏️ Resume Score Improver")
    st.caption("Upload YOUR resume + a job description — get AI-powered suggestions to rewrite and boost your score.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Job Description**")
        jd_improve = st.text_area(
            "Paste the job description",
            height=250,
            placeholder="Paste the job description you are applying for...",
            label_visibility="collapsed",
            key="jd_improve",
        )

    with col2:
        st.markdown("**Your Resume (PDF)**")
        my_resume = st.file_uploader(
            "Upload your resume PDF",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="my_resume",
        )

    st.markdown("---")
    improve_btn = st.button("🎯 Analyse & Improve My Resume", type="primary", use_container_width=True)

    if improve_btn:
        if not jd_improve.strip():
            st.error("Please paste a job description.")
            st.stop()
        if not my_resume:
            st.error("Please upload your resume PDF.")
            st.stop()

        with st.spinner("Loading embedding model…"):
            model = load_embed_model()

        with st.spinner("Analysing your resume…"):
            raw = extract_text_from_pdf(my_resume.read())
            resume_clean = clean_text(raw)
            jd_clean2 = clean_text(jd_improve)

            if len(resume_clean) < 50:
                st.error("Could not extract enough text from your resume. Make sure it's not a scanned image PDF.")
                st.stop()

            scores = score_resumes(jd_clean2, [resume_clean], model)
            current_score = round(scores[0], 1)

        # Score display
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Current Match Score", f"{current_score}%", delta=None)
        with col_b:
            st.metric("Rating", rank_color(current_score) + (" Strong" if current_score >= 70 else " Average" if current_score >= 45 else " Weak"))
        with col_c:
            st.metric("Target Score", "70%+", delta=f"+{max(0, 70 - current_score):.0f}% needed")

        st.markdown("---")

        # Improvement suggestions
        st.subheader("🤖 AI Improvement Suggestions")

        if not api_key:
            st.info("💡 Add a Groq or OpenAI API key in the sidebar for detailed AI rewrite suggestions. Showing keyword analysis below.")

        with st.spinner("Generating improvement plan…"):
            suggestions = get_resume_improver(jd_clean2, resume_clean, current_score, api_key, use_groq)

        st.markdown(suggestions)

        st.markdown("---")

        # Quick keyword gap analysis (always shown)
        st.subheader("🔍 Keyword Gap Analysis")
        jd_words = set(re.findall(r"\b[a-zA-Z]{5,}\b", jd_clean2.lower()))
        res_words = set(re.findall(r"\b[a-zA-Z]{5,}\b", resume_clean.lower()))
        missing_words = sorted(jd_words - res_words, key=lambda w: len(w), reverse=True)[:15]
        matched_words = sorted(jd_words & res_words, key=lambda w: len(w), reverse=True)[:15]

        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown("**✅ Keywords already in your resume:**")
            if matched_words:
                st.success("  |  ".join(matched_words))
            else:
                st.warning("No strong keyword matches found.")

        with col_y:
            st.markdown("**❌ Keywords missing from your resume:**")
            if missing_words:
                st.error("  |  ".join(missing_words))
            else:
                st.success("Great — no major keywords missing!")

        st.markdown("---")
        st.info("💡 **Tip:** Copy the missing keywords above and naturally incorporate them into your resume's skills, experience, and summary sections. Then re-run this tool to see your improved score!")