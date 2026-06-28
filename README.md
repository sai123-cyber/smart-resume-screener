\# Smart Resume Screener 🔍



An AI-powered tool that ranks candidates against a job description using \*\*semantic embeddings\*\* and generates plain-English explanations for each match score — no keyword matching, no black boxes.



!\[Python](https://img.shields.io/badge/Python-3.10%2B-blue)

!\[Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

!\[License](https://img.shields.io/badge/license-MIT-green)



\---



\## Demo



> Upload a job description + a folder of PDF resumes → get a ranked table with match scores and AI explanations in seconds.



```

\#1  Alice\_Chen.pdf      🟢 84.2%   ✅ Matches: Python, NLP, PyTorch ...

\#2  Bob\_Sharma.pdf      🟡 61.7%   ✅ Matches: Python  ❌ Gaps: PyTorch, LLMs ...

\#3  Carol\_Wu.pdf        🔴 38.1%   ❌ Gaps: ML experience, relevant degree ...

```



\---



\## Features



\- \*\*Semantic ranking\*\* — uses `sentence-transformers` (`all-MiniLM-L6-v2`) for meaning-aware scoring, not just keyword overlap

\- \*\*Explainable results\*\* — optional LLM integration (OpenAI GPT-3.5 or Groq Llama 3, free tier) to explain each match

\- \*\*Keyword fallback\*\* — works fully offline with no API key; falls back to a keyword overlap summary

\- \*\*Batch PDF upload\*\* — drag and drop multiple resumes at once

\- \*\*CSV export\*\* — download the ranked list for sharing

\- \*\*Privacy-first\*\* — embeddings run locally; resumes never leave your machine unless you enable LLM explanations



\---



\## How it works



```

Job Description ──┐

&#x20;                 ├─► sentence-transformers (embed) ──► cosine similarity ──► ranked scores

PDF Resumes ──────┘

&#x20;                                                              │

&#x20;                                                              ▼

&#x20;                                                   LLM (optional) ──► per-candidate explanation

```



1\. \*\*Parse\*\* — `pdfplumber` extracts text from each uploaded PDF

2\. \*\*Embed\*\* — both the JD and resumes are encoded into 384-dim vectors using `all-MiniLM-L6-v2`

3\. \*\*Score\*\* — cosine similarity between JD vector and each resume vector → percentage score

4\. \*\*Explain\*\* — an LLM (GPT-3.5 or Groq Llama 3) reads the JD + resume and lists matched skills and gaps



\---



\## Quickstart



\### 1. Clone the repo



```bash

git clone https://github.com/YOUR\_USERNAME/smart-resume-screener.git

cd smart-resume-screener

```



\### 2. Create a virtual environment



```bash

python -m venv venv

source venv/bin/activate        # Windows: venv\\Scripts\\activate

```



\### 3. Install dependencies



```bash

pip install -r requirements.txt

```



> First run downloads the `all-MiniLM-L6-v2` model (\~90 MB) automatically.



\### 4. Run the app



```bash

streamlit run app.py

```



Open \[http://localhost:8501](http://localhost:8501) in your browser.



\---



\## Configuration



Everything is controlled from the \*\*sidebar\*\* inside the app:



| Setting | Default | Description |

|---|---|---|

| LLM Provider | None | Keyword fallback — no API key needed |

| OpenAI API key | — | Enables GPT-3.5 explanations |

| Groq API key | — | Free-tier Llama 3 explanations |

| Minimum score | 0% | Hide candidates below this threshold |



\### Getting a free Groq API key



1\. Sign up at \[console.groq.com](https://console.groq.com) (free, no credit card)

2\. Create an API key

3\. Paste it into the sidebar → select \*\*Groq\*\*



\---



\## Project structure



```

smart-resume-screener/

├── app.py              # main Streamlit application

├── requirements.txt    # Python dependencies

└── README.md           # this file

```



\---



\## Tech stack



| Layer | Library |

|---|---|

| UI | Streamlit |

| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |

| Similarity | scikit-learn (cosine similarity) |

| PDF parsing | pdfplumber |

| LLM explanations | OpenAI SDK (OpenAI / Groq) |

| Data | pandas, numpy |



\---



\## Possible extensions



\- \[ ] Keyword heatmap — highlight which resume terms matched the JD

\- \[ ] Multi-JD mode — compare one resume against several job descriptions

\- \[ ] Bias detection — flag if certain demographic keywords consistently affect scores

\- \[ ] Resume section extraction — parse education, skills, experience separately

\- \[ ] Vector DB — store embeddings in FAISS for large-scale screening



\---



\## License



MIT — free to use, modify, and distribute.



\---



\## Author



Built as a portfolio project to demonstrate practical NLP, retrieval, and LLM integration skills.

