import streamlit as st
import sqlite3
import hashlib
import os
import io
import csv
import time
from datetime import datetime, date
from gtts import gTTS

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="HR Assistant", page_icon="🏢", layout="wide", initial_sidebar_state="expanded")

# Load API key from Streamlit secrets (for deployment) or fallback to env var (for local dev)
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    os.environ.setdefault("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DAILY_QUESTION_LIMIT = 20

TECH_INFO = {
    "Gemini 2.5 Flash": "Google's fast multimodal LLM used here to generate answers from retrieved HR document context, in multiple languages.",
    "FAISS Vector DB": "Facebook AI Similarity Search — stores document embeddings and quickly finds the most relevant text chunks for a question.",
    "HuggingFace": "Provides the sentence-transformer embedding model (all-MiniLM-L6-v2) that converts text into numerical vectors for semantic search.",
    "LangChain": "Framework that connects the PDF loader, text splitter, vector store, and LLM together into one RAG pipeline.",
}

# ==========================================================
# CSS
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*, *::before, *::after { font-family:'Inter',sans-serif; box-sizing:border-box; }
.stApp { background:#eef2f7; }
#MainMenu, footer { visibility:hidden; }
.block-container { max-width:1100px !important; padding-top:3rem !important; padding-left:2rem !important; padding-right:2rem !important; }

.auth-wrap { max-width:560px; margin:0 auto; border-radius:22px; overflow:hidden; box-shadow:0 20px 60px rgba(30,58,95,0.18); }
.auth-top { background:linear-gradient(145deg,#1a3560 0%,#1e50c8 55%,#4f8ef7 100%); padding:44px 48px 36px; text-align:center; position:relative; }
.auth-top::after { content:''; position:absolute; bottom:-1px; left:0; right:0; height:30px; background:#fff; border-radius:50% 50% 0 0/100% 100% 0 0; }
.auth-top-icon { font-size:58px; line-height:1; margin-bottom:16px; display:block; }
.auth-top-title { color:#fff; font-size:26px; font-weight:800; margin-bottom:6px; letter-spacing:-0.3px; }
.auth-top-sub { color:rgba(255,255,255,0.70); font-size:14px; }
.auth-bottom { background:#fff; padding:8px 48px 44px; }

.success-wrap { max-width:560px; margin:0 auto; background:#fff; border-radius:22px; padding:60px 48px; text-align:center; box-shadow:0 20px 60px rgba(30,58,95,0.18); }
.success-icon { font-size:72px; margin-bottom:18px; display:block; }
.success-title { color:#166534; font-size:24px; font-weight:800; margin-bottom:10px; }
.success-msg { color:#64748b; font-size:15px; line-height:1.7; margin-bottom:32px; }

.stTabs [data-baseweb="tab-list"] { background:#f1f5fb !important; border-radius:12px !important; padding:5px !important; gap:6px !important; border:none !important; margin-bottom:6px !important; }
.stTabs [data-baseweb="tab"] { color:#64748b !important; border-radius:9px !important; font-weight:600 !important; font-size:14px !important; padding:8px 20px !important; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#1e50c8,#4f8ef7) !important; color:#fff !important; box-shadow:0 3px 10px rgba(30,80,200,0.25) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:10px !important; }

.stTextInput > label { color:#374151 !important; font-size:13px !important; font-weight:600 !important; margin-bottom:4px !important; }
.stTextInput > div > div > input { background:#f8fafd !important; border:1.5px solid #d1dae8 !important; border-radius:10px !important; color:#1e293b !important; font-size:15px !important; padding:11px 15px !important; }
.stTextInput > div > div > input:focus { border-color:#1e50c8 !important; background:#fff !important; box-shadow:0 0 0 3px rgba(30,80,200,0.13) !important; }
.stTextInput > div > div > input::placeholder { color:#aab4c4 !important; }

.stSelectbox > label { color:#374151 !important; font-size:13px !important; font-weight:600 !important; }
.stSelectbox > div > div { background:#f8fafd !important; border:1.5px solid #d1dae8 !important; border-radius:10px !important; color:#1e293b !important; }

.stButton > button { background:linear-gradient(135deg,#1e50c8 0%,#4f8ef7 100%) !important; color:#fff !important; border:none !important; border-radius:10px !important; height:46px !important; font-weight:700 !important; font-size:15px !important; width:100% !important; box-shadow:0 4px 16px rgba(30,80,200,0.22) !important; }
.stButton > button:hover { opacity:0.88 !important; }
.stDownloadButton > button { background:#16a34a !important; color:#fff !important; border:none !important; border-radius:10px !important; font-weight:600 !important; }

.hero { background:linear-gradient(135deg,#1a3560 0%,#1e50c8 55%,#4f8ef7 100%); border-radius:20px; padding:36px 40px; margin-bottom:24px; display:flex; align-items:center; gap:28px; box-shadow:0 8px 32px rgba(30,80,200,0.20); position:relative; overflow:hidden; }
.hero::before { content:''; position:absolute; top:-50px; right:-50px; width:240px; height:240px; background:rgba(255,255,255,0.05); border-radius:50%; }
.hero::after { content:''; position:absolute; bottom:-70px; right:100px; width:180px; height:180px; background:rgba(255,255,255,0.04); border-radius:50%; }
.hero-icon-box { background:rgba(255,255,255,0.15); border-radius:18px; padding:18px 22px; font-size:52px; line-height:1; flex-shrink:0; }
.hero-text h1 { color:#fff; font-size:26px; font-weight:800; margin:0 0 6px; letter-spacing:-0.3px; }
.hero-text p  { color:rgba(255,255,255,0.72); font-size:14px; margin:0 0 14px; }
.hero-pills { display:flex; flex-wrap:wrap; gap:8px; }
.hero-pill { background:rgba(255,255,255,0.16); border-radius:20px; padding:4px 14px; font-size:12px; color:#fff; font-weight:500; }

.stat-card { background:#fff; border:1px solid #dde4ef; border-radius:14px; padding:20px; text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.04); }
.stat-val { color:#1e50c8; font-size:32px; font-weight:800; line-height:1; }
.stat-lbl { color:#64748b; font-size:11px; margin-top:6px; text-transform:uppercase; letter-spacing:0.06em; }

.answer-box { background:#fff; border:1px solid #dde4ef; border-left:5px solid #1e50c8; border-radius:12px; padding:24px 28px; color:#1e293b; font-size:15px; line-height:1.78; margin-top:14px; box-shadow:0 2px 10px rgba(0,0,0,0.04); }
.answer-lbl { color:#1e50c8; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px; }
.section-title { color:#1a3560; font-size:15px; font-weight:700; margin-bottom:10px; }

.source-chip { display:inline-flex; align-items:center; gap:6px; background:#eef2ff; border:1px solid #c7d2fe; border-radius:8px; padding:5px 12px; font-size:12px; color:#3730a3; font-weight:600; margin:3px 4px 0 0; }
.confidence-wrap { display:flex; align-items:center; gap:10px; margin-top:14px; }
.confidence-bar-bg { flex:1; height:8px; background:#e5e9f2; border-radius:6px; overflow:hidden; }
.confidence-bar-fill { height:100%; border-radius:6px; }
.confidence-text { font-size:12px; font-weight:700; color:#1a3560; white-space:nowrap; }

.usage-card { background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.13); border-radius:12px; padding:12px 14px; margin-bottom:14px; }
.usage-bar-bg { height:6px; background:rgba(255,255,255,0.15); border-radius:5px; overflow:hidden; margin-top:6px; }
.usage-bar-fill { height:100%; background:linear-gradient(90deg,#4f8ef7,#93c5fd); border-radius:5px; }

[data-testid="stSidebar"] {
    background:#1a3560 !important;
    border-right:1px solid #163054 !important;
}
[data-testid="stSidebar"] * { color:#e2e8f0 !important; }

/* Ensure the sidebar collapse/expand control is always visible and clickable */
[data-testid="collapsedControl"] {
    display:flex !important;
    visibility:visible !important;
    opacity:1 !important;
    background:#1a3560 !important;
    border-radius:8px !important;
    z-index:999999 !important;
}
[data-testid="collapsedControl"] svg { fill:#ffffff !important; color:#ffffff !important; }
button[kind="header"] { display:flex !important; visibility:visible !important; }
.sb-user { background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); border-radius:14px; padding:18px; margin-bottom:22px; text-align:center; }
.sb-avatar { font-size:36px; margin-bottom:8px; }
.sb-name { font-weight:700; font-size:15px; color:#fff !important; }
.sb-role { font-size:12px; color:rgba(255,255,255,0.5) !important; margin-top:2px; }
.sb-badge { display:inline-flex; align-items:center; gap:6px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); border-radius:20px; padding:4px 12px; font-size:12px; color:#93c5fd !important; margin:3px; }
.sb-doc-row { display:flex; align-items:center; justify-content:space-between; background:rgba(255,255,255,0.06); border-radius:8px; padding:5px 10px; margin:4px 0; }

[data-testid="stSidebar"] .streamlit-expanderHeader { background:rgba(255,255,255,0.08) !important; border:1px solid rgba(255,255,255,0.12) !important; border-radius:20px !important; color:#93c5fd !important; font-size:12px !important; font-weight:500 !important; padding:4px 12px !important; min-height:0 !important; }
[data-testid="stSidebar"] .streamlit-expanderContent { background:rgba(255,255,255,0.05) !important; border-radius:8px !important; color:#cbd5e1 !important; font-size:12px !important; padding:8px 10px !important; }

[data-testid="stSidebar"] .stFileUploader label { color:#e2e8f0 !important; font-size:13px !important; font-weight:600 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background:rgba(255,255,255,0.06) !important; border:1.5px dashed rgba(255,255,255,0.25) !important; border-radius:10px !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color:#cbd5e1 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { background:rgba(255,255,255,0.12) !important; color:#fff !important; border:1px solid rgba(255,255,255,0.2) !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg { fill:#93c5fd !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] { background:rgba(255,255,255,0.08) !important; border-radius:8px !important; color:#e2e8f0 !important; }
[data-testid="stSidebar"] .stButton > button { background:rgba(220,38,38,0.85) !important; height:30px !important; font-size:11px !important; padding:0 8px !important; box-shadow:none !important; }

.stFileUploader label { color:#374151 !important; font-size:13px !important; font-weight:600 !important; }
.stFileUploader > div { background:#f8fafd !important; border:1.5px dashed #d1dae8 !important; border-radius:10px !important; }

.stSuccess>div{background:#f0fdf4!important;color:#166534!important;border-radius:10px!important;}
.stError>div  {background:#fef2f2!important;color:#991b1b!important;border-radius:10px!important;}
.stWarning>div{background:#fffbeb!important;color:#92400e!important;border-radius:10px!important;}
.stInfo>div   {background:#eff6ff!important;color:#1e3a8a!important;border-radius:10px!important;}
.streamlit-expanderHeader{background:#fff!important;border-radius:10px!important;color:#1e293b!important;font-weight:600!important;}

.admin-table { width:100%; border-collapse:collapse; margin-top:10px; }
.admin-table th { background:#1a3560; color:#fff; padding:10px 14px; text-align:left; font-size:12px; text-transform:uppercase; letter-spacing:0.04em; }
.admin-table td { padding:10px 14px; border-bottom:1px solid #e5e9f2; font-size:14px; color:#1e293b; }
.admin-table tr:nth-child(even) { background:#f8fafd; }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# DATABASE
# ==========================================================
def init_db():
    conn = sqlite3.connect("users.db", check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE, email TEXT UNIQUE,
        password TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chat_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        question TEXT,
        answer TEXT,
        language TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

    # Migration: add new columns if this is an older database
    cur.execute("PRAGMA table_info(chat_history)")
    existing_cols = [row[1] for row in cur.fetchall()]
    if "sources" not in existing_cols:
        cur.execute("ALTER TABLE chat_history ADD COLUMN sources TEXT")
    if "confidence" not in existing_cols:
        cur.execute("ALTER TABLE chat_history ADD COLUMN confidence REAL")

    conn.commit()
    return conn

def hp(p): return hashlib.sha256(p.encode()).hexdigest()

def register_user(conn, u, e, p):
    try:
        conn.cursor().execute("INSERT INTO users(username,email,password) VALUES(?,?,?)",(u,e,hp(p)))
        conn.commit(); return True,"ok"
    except sqlite3.IntegrityError as ex:
        return False,("Username already exists." if "username" in str(ex) else "Email already registered.")

def login_user(conn, u, p):
    c=conn.cursor(); c.execute("SELECT * FROM users WHERE username=? AND password=?",(u,hp(p))); return c.fetchone()

def save_chat(conn, username, question, answer, language, sources, confidence):
    conn.cursor().execute(
        "INSERT INTO chat_history(username,question,answer,language,sources,confidence) VALUES(?,?,?,?,?,?)",
        (username, question, answer, language, sources, confidence))
    conn.commit()

def get_chat_history(conn, username, limit=5):
    cur = conn.cursor()
    cur.execute(
        "SELECT question,answer,language,sources,confidence,created_at FROM chat_history WHERE username=? ORDER BY id DESC LIMIT ?",
        (username, limit))
    return cur.fetchall()

def get_all_chat_history(conn, username):
    cur = conn.cursor()
    cur.execute(
        "SELECT question,answer,language,sources,confidence,created_at FROM chat_history WHERE username=? ORDER BY id DESC",
        (username,))
    return cur.fetchall()

def get_today_question_count(conn, username):
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM chat_history WHERE username=? AND date(created_at)=date('now')",
        (username,))
    return cur.fetchone()[0]

def get_admin_stats(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM chat_history")
    total_questions = cur.fetchone()[0]
    cur.execute("""SELECT username, COUNT(*) as cnt FROM chat_history
                   GROUP BY username ORDER BY cnt DESC LIMIT 5""")
    top_users = cur.fetchall()
    cur.execute("""SELECT question, COUNT(*) as cnt FROM chat_history
                   GROUP BY question ORDER BY cnt DESC LIMIT 5""")
    top_questions = cur.fetchall()
    cur.execute("""SELECT language, COUNT(*) FROM chat_history GROUP BY language""")
    lang_breakdown = cur.fetchall()
    return total_users, total_questions, top_users, top_questions, lang_breakdown


# ==========================================================
# SESSION STATE
# ==========================================================
for k,v in [("logged_in",False),("username",""),("registered",False),("reg_username","")]:
    if k not in st.session_state: st.session_state[k]=v

conn = init_db()


# ==========================================================
# AUTH PAGE
# ==========================================================
def show_auth():
    _, col, _ = st.columns([1, 1.6, 1])
    with col:

        if st.session_state.registered:
            st.markdown(f"""
            <div class='success-wrap'>
                <span class='success-icon'>🎉</span>
                <div class='success-title'>Account Created Successfully!</div>
                <div class='success-msg'>
                    Welcome <strong>{st.session_state.reg_username}</strong>!<br>
                    Your HR Assistant account is ready.<br>Please sign in to continue.
                </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("→  Go to Sign In", key="goto"):
                st.session_state.registered = False
                st.rerun()
            return

        st.markdown("""
        <div class='auth-wrap'>
          <div class='auth-top'>
            <span class='auth-top-icon'>🏢</span>
            <div class='auth-top-title'>HR Assistant Portal</div>
            <div class='auth-top-sub'>Sign in to access company HR documents</div>
          </div>
          <div class='auth-bottom'>
        """, unsafe_allow_html=True)

        t1, t2 = st.tabs(["  🔑  Sign In  ", "  📝  Register  "])

        with t1:
            st.markdown("<br>", unsafe_allow_html=True)
            u = st.text_input("Username", placeholder="Enter your username", key="li_u")
            p = st.text_input("Password", type="password", placeholder="Enter your password", key="li_p")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In →", key="btn_li"):
                if u.strip() and p:
                    user = login_user(conn, u.strip(), p)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.username  = u.strip()
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.warning("Please fill in all fields.")

        with t2:
            st.markdown("<br>", unsafe_allow_html=True)
            ru = st.text_input("Username", placeholder="Choose a username", key="reg_u")
            re = st.text_input("Email",    placeholder="your@email.com",   key="reg_e")
            rp = st.text_input("Password", type="password", placeholder="Min 6 characters", key="reg_p")
            rc = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="reg_c")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", key="btn_reg"):
                if not all([ru.strip(),re.strip(),rp,rc]):
                    st.warning("Please fill in all fields.")
                elif rp != rc:
                    st.error("Passwords do not match.")
                elif len(rp) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    ok,msg = register_user(conn, ru.strip(), re.strip(), rp)
                    if ok:
                        st.session_state.registered   = True
                        st.session_state.reg_username = ru.strip()
                        st.rerun()
                    else:
                        st.error("❌ " + msg)

        st.markdown("</div></div>", unsafe_allow_html=True)


# ==========================================================
# RAG LOADING
# ==========================================================
@st.cache_resource(show_spinner=False)
def load_rag(pdf_paths_tuple):
    pdf_paths = list(pdf_paths_tuple)
    docs = []
    for pdf in pdf_paths:
        try:
            loaded = PyPDFLoader(pdf).load()
            for d in loaded:
                d.metadata["source_file"] = os.path.basename(pdf)
            docs.extend(loaded)
        except Exception:
            continue
    if not docs:
        return None, None, 0, 0
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db  = FAISS.from_documents(chunks, emb)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return db, llm, len(docs), len(chunks)


def get_all_pdf_paths():
    paths = []
    for f in ["Employee-Handbook.pdf", "Leave_Policy.pdf", "HR_Policy.pdf", "Recruitment Policy.pdf"]:
        if os.path.exists(f):
            paths.append(f)
    if os.path.isdir(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            if f.lower().endswith(".pdf"):
                paths.append(os.path.join(UPLOAD_DIR, f))
    return paths


def delete_uploaded_pdf(filename):
    path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ==========================================================
# MAIN APP
# ==========================================================
def show_app():

    is_admin = st.session_state.username.lower() == "admin"

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.markdown(f"""
        <div class='sb-user'>
            <div class='sb-avatar'>👤</div>
            <div class='sb-name'>{st.session_state.username}</div>
            <div class='sb-role'>{"Administrator" if is_admin else "Employee"}</div>
        </div>""", unsafe_allow_html=True)

        # Usage tracking
        today_count = get_today_question_count(conn, st.session_state.username)
        pct = min(100, int((today_count / DAILY_QUESTION_LIMIT) * 100))
        bar_color = "#16a34a" if pct < 70 else ("#f59e0b" if pct < 100 else "#dc2626")
        st.markdown(f"""
        <div class='usage-card'>
            <div style='font-size:12px; font-weight:600;'>📊 Today's Usage</div>
            <div style='font-size:11px; color:rgba(255,255,255,0.6); margin-top:2px;'>{today_count} / {DAILY_QUESTION_LIMIT} questions asked</div>
            <div class='usage-bar-bg'><div class='usage-bar-fill' style='width:{pct}%; background:{bar_color};'></div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**📂 Documents**")
        pdf_paths = get_all_pdf_paths()
        uploaded_names = set()
        if os.path.isdir(UPLOAD_DIR):
            uploaded_names = set(os.listdir(UPLOAD_DIR))

        if pdf_paths:
            for p in pdf_paths:
                name = os.path.basename(p)
                is_user_uploaded = name in uploaded_names
                col_a, col_b = st.columns([4,1])
                with col_a:
                    st.markdown(f"<div class='sb-badge'>✅ {name}</div>", unsafe_allow_html=True)
                with col_b:
                    if is_user_uploaded:
                        if st.button("🗑️", key=f"del_{name}", help=f"Delete {name}"):
                            delete_uploaded_pdf(name)
                            st.cache_resource.clear()
                            st.rerun()
        else:
            st.markdown("<div class='sb-badge'>⚠️ No PDFs found</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📤 Upload More PDFs**")
        uploaded = st.file_uploader("Add HR documents", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
        if uploaded:
            for file in uploaded:
                save_path = os.path.join(UPLOAD_DIR, file.name)
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())
            st.success(f"✅ {len(uploaded)} file(s) uploaded!")
            st.cache_resource.clear()
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**⚡ Tech Stack** _(click to learn more)_")
        for tech, info in TECH_INFO.items():
            with st.expander(f"🔹 {tech}"):
                st.write(info)

        # Export chat history
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📥 Export History**")
        all_history = get_all_chat_history(conn, st.session_state.username)
        if all_history:
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["Question", "Answer", "Language", "Sources", "Confidence", "Date"])
            for row in all_history:
                writer.writerow(row)
            st.download_button(
                "⬇️ Download as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{st.session_state.username}_chat_history.csv",
                mime="text/csv"
            )
        else:
            st.caption("No history yet to export.")

        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🚪 Sign Out", key="signout_btn"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.cache_resource.clear()
            st.rerun()

    # ---------- HERO ----------
    st.markdown(f"""
    <div class='hero'>
        <div class='hero-icon-box'>🏢</div>
        <div class='hero-text'>
            <h1>HR Document Assistant</h1>
            <p>Welcome back, <strong>{st.session_state.username}</strong> · Ask anything from your company's HR documents</p>
            <div class='hero-pills'>
                <span class='hero-pill'>📄 Leave Policy</span>
                <span class='hero-pill'>📋 HR Policy</span>
                <span class='hero-pill'>📘 Employee Handbook</span>
                <span class='hero-pill'>🌐 Multilingual</span>
                <span class='hero-pill'>🔊 Voice Output</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ---------- ADMIN DASHBOARD ----------
    if is_admin:
        with st.expander("📊 Admin Analytics Dashboard", expanded=False):
            total_users, total_questions, top_users, top_questions, lang_breakdown = get_admin_stats(conn)
            ac1, ac2, ac3 = st.columns(3)
            with ac1: st.markdown(f"<div class='stat-card'><div class='stat-val'>{total_users}</div><div class='stat-lbl'>Total Users</div></div>", unsafe_allow_html=True)
            with ac2: st.markdown(f"<div class='stat-card'><div class='stat-val'>{total_questions}</div><div class='stat-lbl'>Total Questions</div></div>", unsafe_allow_html=True)
            with ac3:
                avg = round(total_questions/total_users, 1) if total_users else 0
                st.markdown(f"<div class='stat-card'><div class='stat-val'>{avg}</div><div class='stat-lbl'>Avg Q/User</div></div>", unsafe_allow_html=True)

            st.markdown("<br>**🏆 Most Active Users**", unsafe_allow_html=True)
            if top_users:
                rows = "".join([f"<tr><td>{u}</td><td>{c}</td></tr>" for u,c in top_users])
                st.markdown(f"<table class='admin-table'><tr><th>Username</th><th>Questions Asked</th></tr>{rows}</table>", unsafe_allow_html=True)

            st.markdown("<br>**🔥 Most Common Questions**", unsafe_allow_html=True)
            if top_questions:
                rows = "".join([f"<tr><td>{q[:60]}</td><td>{c}</td></tr>" for q,c in top_questions])
                st.markdown(f"<table class='admin-table'><tr><th>Question</th><th>Times Asked</th></tr>{rows}</table>", unsafe_allow_html=True)

            st.markdown("<br>**🌐 Language Usage**", unsafe_allow_html=True)
            if lang_breakdown:
                rows = "".join([f"<tr><td>{l}</td><td>{c}</td></tr>" for l,c in lang_breakdown])
                st.markdown(f"<table class='admin-table'><tr><th>Language</th><th>Count</th></tr>{rows}</table>", unsafe_allow_html=True)

    # ---------- LOAD PDFS ----------
    pdf_paths = get_all_pdf_paths()
    if not pdf_paths:
        st.warning("No PDF documents found. Upload PDFs from the sidebar to get started.")
        return

    try:
        with st.spinner("Loading knowledge base..."):
            db, llm, n_pages, n_chunks = load_rag(tuple(sorted(pdf_paths)))
    except Exception as e:
        st.error(f"⚠️ Failed to load documents: {e}")
        if st.button("🔄 Retry"):
            st.cache_resource.clear()
            st.rerun()
        return

    if db is None:
        st.error("Could not load any documents. They may be corrupted — try re-uploading.")
        return

    c1,c2,c3 = st.columns(3)
    for col,val,lbl in [(c1,n_pages,"Pages Indexed"),(c2,n_chunks,"Knowledge Chunks"),(c3,4,"Languages")]:
        with col:
            st.markdown(f"<div class='stat-card'><div class='stat-val'>{val}</div><div class='stat-lbl'>{lbl}</div></div>",unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- RATE LIMIT CHECK ----------
    today_count = get_today_question_count(conn, st.session_state.username)
    limit_reached = today_count >= DAILY_QUESTION_LIMIT

    cl,cq = st.columns([1,3])
    with cl: language = st.selectbox("🌐 Language",["English","Telugu","Hindi","Tamil"])
    with cq: question = st.text_input("💬 Your Question", placeholder="e.g. How many leave days per year?", disabled=limit_reached)

    if limit_reached:
        st.warning(f"⚠️ You've reached your daily limit of {DAILY_QUESTION_LIMIT} questions. Please try again tomorrow.")

    ask_clicked = st.button("🔍 Ask HR Assistant", disabled=limit_reached)

    if ask_clicked:
        if question.strip():
            try:
                results_with_scores = db.similarity_search_with_relevance_scores(question, k=3)
            except Exception:
                results_with_scores = [(d, 0.0) for d in db.similarity_search(question, k=3)]

            docs_found = [r[0] for r in results_with_scores]
            scores = [max(0.0, min(1.0, r[1])) for r in results_with_scores]
            avg_confidence = round((sum(scores)/len(scores))*100, 1) if scores else 0.0

            ctx = "\n".join([d.page_content for d in docs_found])
            source_files = sorted(set(d.metadata.get("source_file", "Unknown") for d in docs_found))
            source_pages = sorted(set(
                f"{d.metadata.get('source_file','Unknown')} (p.{d.metadata.get('page', '?')+1 if isinstance(d.metadata.get('page'), int) else '?'})"
                for d in docs_found
            ))

            prompt = f"""You are a professional HR assistant.
Answer ONLY from the context. If not found say: "This information is not available in the HR documents."
Be clear and concise. Respond in {language} language.
Context:\n{ctx}\nQuestion: {question}"""

            with st.spinner("Searching HR documents..."):
                answer = llm.invoke(prompt).content

            sources_str = ", ".join(source_pages)
            save_chat(conn, st.session_state.username, question, answer, language, sources_str, avg_confidence)

            # Streaming-style display
            placeholder = st.empty()
            displayed = ""
            words = answer.split(" ")
            chunk_size = max(1, len(words)//40) if len(words) > 40 else 1
            for i in range(0, len(words), chunk_size):
                displayed += " ".join(words[i:i+chunk_size]) + " "
                placeholder.markdown(f"""<div class='answer-box'><div class='answer-lbl'>Answer · {language}</div>{displayed}▌</div>""", unsafe_allow_html=True)
                time.sleep(0.02)
            placeholder.markdown(f"""<div class='answer-box'><div class='answer-lbl'>Answer · {language}</div>{answer}</div>""", unsafe_allow_html=True)

            # Confidence bar
            try:
                avg_confidence = float(avg_confidence)
            except (TypeError, ValueError):
                avg_confidence = 0.0
            conf_color = "#16a34a" if avg_confidence >= 70 else ("#f59e0b" if avg_confidence >= 40 else "#dc2626")
            st.markdown(f"""
            <div class='confidence-wrap'>
                <span class='confidence-text'>Match Confidence:</span>
                <div class='confidence-bar-bg'><div class='confidence-bar-fill' style='width:{avg_confidence}%; background:{conf_color};'></div></div>
                <span class='confidence-text'>{avg_confidence}%</span>
            </div>
            """, unsafe_allow_html=True)

            # Source citations
            if source_files:
                chips = "".join([f"<span class='source-chip'>📄 {s}</span>" for s in source_files])
                st.markdown(f"<div style='margin-top:10px;'>{chips}</div>", unsafe_allow_html=True)

            # Voice
            try:
                lmap={"English":"en","Telugu":"te","Hindi":"hi","Tamil":"ta"}
                tts=gTTS(text=answer,lang=lmap[language],slow=False); tts.save("answer.mp3")
                st.markdown("<br>**🔊 Listen to Answer**")
                with open("answer.mp3","rb") as f: st.audio(f.read(),format="audio/mp3")
            except Exception as e:
                st.warning(f"Voice unavailable: {e}")
        else:
            st.warning("Please enter a question.")

    # ---------- CHAT HISTORY ----------
    history = get_chat_history(conn, st.session_state.username, limit=5)
    if history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📜 Your Recent Questions</div>", unsafe_allow_html=True)
        for q, a, lang, src, conf, created in history:
            with st.expander(f"Q: {q[:65]}  [{lang}]"):
                st.markdown(f"""<div class='answer-box'><div class='answer-lbl'>{lang}</div>{a}</div>""",unsafe_allow_html=True)
                try:
                    conf = float(conf) if conf is not None else None
                except (TypeError, ValueError):
                    conf = None
                if conf is not None:
                    conf_color = "#16a34a" if conf >= 70 else ("#f59e0b" if conf >= 40 else "#dc2626")
                    st.markdown(f"""
                    <div class='confidence-wrap'>
                        <span class='confidence-text'>Confidence:</span>
                        <div class='confidence-bar-bg'><div class='confidence-bar-fill' style='width:{conf}%; background:{conf_color};'></div></div>
                        <span class='confidence-text'>{conf}%</span>
                    </div>""", unsafe_allow_html=True)
                if src:
                    chips = "".join([f"<span class='source-chip'>📄 {s}</span>" for s in src.split(", ")])
                    st.markdown(f"<div style='margin-top:8px;'>{chips}</div>", unsafe_allow_html=True)


# ==========================================================
# ROUTER
# ==========================================================
if st.session_state.logged_in:
    show_app()
else:
    show_auth()