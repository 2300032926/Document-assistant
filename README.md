# HR Document Assistant

A multilingual RAG (Retrieval-Augmented Generation) chatbot that lets employees ask natural-language questions about company HR policies — Leave Policy, HR Policy, Employee Handbook, and any other PDF uploaded — and get accurate, source-cited answers in English, Telugu, Hindi, or Tamil, with optional voice output.

**Live demo:** https://document-assistant-kd23e9qyrtb2marozeysj6.streamlit.app/

---

## Overview

Most internal HR FAQ bots either hallucinate answers or require employees to dig through long PDFs themselves. This project solves that by combining semantic document retrieval with a large language model: every answer is grounded in the actual HR documents, shows which document and page it came from, and includes a confidence score so users know how reliable the match is.

## Features

- **Authentication** — registration and login with SHA-256 password hashing, SQLite-backed user store
- **Retrieval-Augmented Generation** — answers are generated only from retrieved document context, not the model's general knowledge
- **Source citations** — every answer shows which PDF (and page) it was pulled from
- **Confidence scoring** — a relevance score (from FAISS similarity) shown as a visual indicator per answer
- **Multilingual** — questions can be answered in English, Telugu, Hindi, or Tamil
- **Voice output** — answers can be read aloud via text-to-speech
- **Dynamic document management** — upload new PDFs through the UI, or delete previously uploaded ones, without redeploying
- **Persistent, per-user chat history** — each user only ever sees their own past questions, stored in SQLite and available across sessions
- **Admin analytics dashboard** — a dedicated admin view showing total users, most active users, most common questions, and language usage breakdown
- **Rate limiting** — a daily question cap per user to control API usage
- **Streaming-style responses** — answers render progressively rather than appearing all at once

## Architecture

```
User -> Auth (SQLite) -> Question
                            |
HR PDFs -> HuggingFace embeddings -> FAISS vector search (top-k chunks)
                            |
                    Gemini 2.5 Flash (answer generation)
                            |
        Answer + source citations + confidence score + audio
                            |
                    Saved to per-user chat history
```

## Tech stack

| Layer | Technology |
|---|---|
| UI / app framework | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Orchestration | LangChain |
| Database | SQLite (users + chat history) |
| Text-to-speech | gTTS |
| Deployment | Streamlit Community Cloud |

## Running locally

```bash
git clone https://github.com/2300032926/Document-assistant.git
cd Document-assistant
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` in the project root with your Gemini API key:

```toml
GOOGLE_API_KEY = "your_gemini_api_key_here"
```

Then run:

```bash
streamlit run app.py
```

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). The API key is stored as a Streamlit secret (never committed to the repo) and loaded at runtime via `st.secrets`.

## Project structure

```
.
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── .gitignore
├── *.pdf                   # Default HR documents (Leave Policy, HR Policy, Handbook, Recruitment Policy)
└── uploaded_pdfs/           # User-uploaded documents (created at runtime)
```

## Future improvements

- Multi-file diff / versioning for HR policy updates
- Export individual answers to PDF
- Slack/Teams integration for in-chat HR queries
- Configurable per-role document access (e.g. managers see different policies than employees)