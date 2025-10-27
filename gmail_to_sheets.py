# streamlit_app.py
import os
import time
import json
import base64
import requests
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime, time as dt_time, timezone, timedelta
from zoneinfo import ZoneInfo
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from dotenv import load_dotenv
from litellm import completion  # liteLLM

if "GMAIL_CREDENTIALS" in st.secrets:
    creds_json = json.loads(st.secrets["GMAIL_CREDENTIALS"])
    with open("credentials.json", "w") as f:
        json.dump(creds_json, f)

if "GMAIL_TOKEN" in st.secrets:
    token_json = json.loads(st.secrets["GMAIL_TOKEN"])
    with open("token.json", "w") as f:
        json.dump(token_json, f)
# ---------------------------
# CONFIG / LOAD SECRETS
# ---------------------------
# Streamlit Cloud: set secrets in app settings and access via st.secrets
# Locally: create a .env with keys (not committed)
load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Prefer Streamlit secrets, fallback to env
def get_secret(name):
    # Streamlit secrets (secure on Streamlit Cloud)
    if st.runtime.exists() and hasattr(st, "secrets") and name in st.secrets:
        return st.secrets[name]
    return os.getenv(name)

TRELLO_KEY = get_secret("TRELLO_KEY")
TRELLO_TOKEN = get_secret("TRELLO_TOKEN")
TRELLO_LIST_ID = get_secret("TRELLO_LIST_ID")
GROQ_API_KEY = get_secret("GROQ_API_KEY")
LLM_MODEL = get_secret("LLM_MODEL") or "groq/llama-3.1-8b-instant"

# Put Groq key into env for litellm to pick it up
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ---------------------------
# UTIL: Validate config
# ---------------------------
def validate_config(allow_missing_llm=False):
    missing = []
    for k in ["TRELLO_KEY", "TRELLO_TOKEN", "TRELLO_LIST_ID"]:
        if not globals().get(k):
            missing.append(k)
    if not allow_missing_llm and not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        raise EnvironmentError(f"Missing required secrets/env: {', '.join(missing)}")

# ---------------------------
# GMAIL: Auth and fetch
# ---------------------------
def get_credentials_local():
    """Get Gmail credentials. This will open the OAuth browser flow the first time locally.
       (On Streamlit Cloud this local browser flow won't work — see deployment note.)"""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                # Force re-auth
                try:
                    os.remove("token.json")
                except Exception:
                    pass
                creds = None
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
            with open("token.json", "w") as f:
                f.write(creds.to_json())
    return creds

def build_gmail_service():
    creds = get_credentials_local()
    return build("gmail", "v1", credentials=creds)

def start_of_today_unix_utc(tz_name="Asia/Karachi"):
    # Compute start of day in user's timezone then convert to UNIX seconds (Gmail uses 'after' as seconds)
    tz = ZoneInfo(tz_name)
    local_now = datetime.now(tz)
    local_midnight = datetime(local_now.year, local_now.month, local_now.day, 0, 0, 0, tzinfo=tz)
    # convert to UTC timestamp (seconds)
    utc_ts = int(local_midnight.astimezone(timezone.utc).timestamp())
    return utc_ts

def fetch_todays_emails(max_results=50):
    service = build_gmail_service()
    after_ts = start_of_today_unix_utc()
    query = f"after:{after_ts}"
    resp = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    messages = resp.get("messages", [])
    out = []
    for m in messages:
        msg = service.users().messages().get(userId="me", id=m["id"], format="full").execute()
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        subject = ""
        sender = ""
        for h in headers:
            name = h.get("name", "").lower()
            if name == "subject":
                subject = h.get("value", "")
            elif name == "from":
                sender = h.get("value", "")
        # decode body
        body = ""
        # gmail parts can be nested; handle simple two-level case
        parts = payload.get("parts", [])
        if parts:
            for p in parts:
                data = (p.get("body") or {}).get("data")
                if data:
                    decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                    if p.get("mimeType","").lower() == "text/html":
                        body += BeautifulSoup(decoded, "html.parser").get_text()
                    else:
                        body += decoded
        else:
            data = (payload.get("body") or {}).get("data")
            if data:
                body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        out.append({
            "id": m["id"],
            "subject": subject,
            "from": sender,
            "body": body.strip()
        })
    return out

# ---------------------------
# AI PARSING (LiteLLM/Groq)
# ---------------------------
def analyze_with_ai(email_text):
    prompt = f"""
You are an AI email parser.
Decide if this email describes a project request or assignment. If yes, extract:
- project_name
- client_name
- description
- due_date (ISO or natural language or empty)
- dropbox_links (comma separated)

Return **valid JSON only** with keys:
{{"is_project": bool, "project_name": str, "client_name": str, "description": str, "due_date": str, "dropbox_links": str}}
Email:
{email_text}
"""
    try:
        resp = completion(model=LLM_MODEL, messages=[{"role":"user","content":prompt}])
        text = resp["choices"][0]["message"]["content"]
        parsed = json.loads(text)
        return parsed
    except Exception as e:
        return {"is_project": False, "error": str(e)}

# ---------------------------
# TRELLO: create card
# ---------------------------
def create_trello_card(project_obj):
    url = "https://api.trello.com/1/cards"
    desc = (
        f"Client: {project_obj.get('client_name','N/A')}\n\n"
        f"Description: {project_obj.get('description','N/A')}\n\n"
        f"Due Date: {project_obj.get('due_date','N/A')}\n\n"
        f"Links: {project_obj.get('dropbox_links','None')}"
    )
    params = {
        "key": TRELLO_KEY,
        "token": TRELLO_TOKEN,
        "idList": TRELLO_LIST_ID,
        "name": project_obj.get("project_name", "Unnamed Project"),
        "desc": desc
    }
    r = requests.post(url, params=params)
    return r.status_code in (200, 201), r

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Gmail → Trello AI", page_icon="📧", layout="wide")
st.title("📧 Gmail → Trello (AI)")

st.markdown(
    "Fetch today's Gmail, use AI to detect project emails, and insert them into Trello with a configurable delay to handle rate limits."
)

# Show config and quick validate
with st.expander("🔧 Configuration / Secrets (hidden)"):
    st.write("Use Streamlit Secrets or local .env (do NOT commit keys).")
try:
    validate_config(allow_missing_llm=True)
    st.success("Configuration loaded (Trello keys present).")
except Exception as e:
    st.error(f"Configuration error: {e}")

col1, col2 = st.columns([2,1])
with col1:
    if st.button("📬 Fetch Today's Emails"):
        with st.spinner("Fetching emails from Gmail..."):
            try:
                emails = fetch_todays_emails(max_results=50)
                st.session_state["emails"] = emails
                st.success(f"Fetched {len(emails)} emails from today.")
            except Exception as ex:
                st.error(f"Failed to fetch emails: {ex}")

    if "emails" in st.session_state:
        emails = st.session_state["emails"]
        st.write(f"### Today's emails ({len(emails)})")
        for idx, e in enumerate(emails):
            with st.expander(f"{idx+1}. {e['subject'] or '(no subject)'}"):
                st.write(f"**From:** {e['from']}")
                st.write(e["body"][:1000] + ("..." if len(e["body"])>1000 else ""))
                if st.button(f"Parse with AI (preview) — #{idx+1}", key=f"parse_{idx}"):
                    with st.spinner("Parsing email..."):
                        ai = analyze_with_ai(e["body"])
                        st.json(ai)

with col2:
    st.sidebar.header("Insertion settings")
    delay_seconds = st.sidebar.number_input("Delay between Trello inserts (seconds)", min_value=1, max_value=300, value=20)
    max_process = st.sidebar.number_input("Max emails to process in one run", min_value=1, max_value=50, value=10)
    run_insert = st.sidebar.button("🚀 Insert Project Emails to Trello")

# Insert flow
if run_insert:
    if "emails" not in st.session_state or not st.session_state["emails"]:
        st.warning("No emails fetched yet. Click 'Fetch Today's Emails' first.")
    else:
        # validate config (ensure Groq can be missing; but TRELLO needed)
        try:
            validate_config(allow_missing_llm=False)
        except Exception as ex:
            st.error(f"Config problem: {ex}")
            st.stop()

        emails = st.session_state["emails"][:max_process]
        total = len(emails)
        progress = st.progress(0)
        log_area = st.empty()
        inserted = 0
        for i, e in enumerate(emails):
            log_area.text(f"Parsing email {i+1}/{total}: {e['subject']}")
            ai_result = analyze_with_ai(e["body"])
            # show AI error if any
            if ai_result.get("error"):
                log_area.text(f"AI error for email {i+1}: {ai_result['error']}")
                # if rate limit, show friendly message and retry after short sleep
                if "rate limit" in ai_result["error"].lower() or "RateLimitError".lower() in ai_result["error"].lower():
                    st.warning("Rate limit hit. Waiting one delay cycle before retry...")
                    time.sleep(delay_seconds)
                    # try once more
                    ai_result = analyze_with_ai(e["body"])
            if ai_result.get("is_project"):
                ok, resp = create_trello_card(ai_result)
                if ok:
                    inserted += 1
                    log_area.text(f"✅ Created Trello card for: {ai_result.get('project_name','(unnamed)')}")
                else:
                    log_area.text(f"❌ Trello error for {ai_result.get('project_name','(unnamed)')}: {resp.status_code} {resp.text}")
            else:
                log_area.text(f"⏭️ Skipped: Not a project or AI said no. Subject: {e['subject']}")
            progress.progress((i+1)/total)
            # countdown UI for delay
            for s in range(delay_seconds, 0, -1):
                st.sidebar.markdown(f"Waiting **{s}s** before next insert...")
                time.sleep(1)
        st.success(f"Done. Inserted {inserted} cards out of {total} processed.")

st.markdown("---")
st.caption("Built with LiteLLM (Groq) + Gmail API + Trello API. Keep keys in secrets; don't commit them.")


