# streamlit_app.py
import os
import time
import json
import base64
import threading
import requests
import streamlit as st
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from litellm import completion

# -------------------------
# Configuration / Secrets
# -------------------------
st.set_page_config(page_title="Gmail → Trello (AI)", layout="wide")
st.title("Gmail → Trello (AI)")

st.markdown("Fetch Inbox emails (last 24 hours), detect project-related messages using AI, and create Trello cards with delay and an Auto Mode option.")

# Sidebar: settings
st.sidebar.header("Settings")
delay_seconds = st.sidebar.number_input("Delay between Trello inserts (seconds)", min_value=5, max_value=300, value=60)
check_interval_minutes = st.sidebar.number_input("Auto-check interval (minutes)", min_value=1, max_value=180, value=15)
max_emails_per_run = st.sidebar.number_input("Max emails to process per run", min_value=1, max_value=100, value=10)
st.sidebar.markdown("---")

# Secrets: prefer st.secrets (Streamlit Cloud). For local, fall back to environment variables.
def secret(name):
    return st.secrets.get(name) if hasattr(st, "secrets") else os.getenv(name)

TRELLO_KEY = secret("TRELLO_KEY") or os.getenv("TRELLO_KEY")
TRELLO_TOKEN = secret("TRELLO_TOKEN") or os.getenv("TRELLO_TOKEN")
TRELLO_LIST_ID = secret("TRELLO_LIST_ID") or os.getenv("TRELLO_LIST_ID")
GROQ_API_KEY = secret("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
LLM_MODEL = secret("LLM_MODEL") or os.getenv("LLM_MODEL") or "groq/llama-3.1-8b-instant"

# Recreate credentials.json and token.json from secrets if present
if hasattr(st, "secrets"):
    if "GMAIL_CREDENTIALS" in st.secrets:
        try:
            creds_json = json.loads(st.secrets["GMAIL_CREDENTIALS"])
            with open("credentials.json", "w", encoding="utf-8") as f:
                json.dump(creds_json, f)
        except Exception:
            pass
    if "GMAIL_TOKEN" in st.secrets:
        try:
            token_json = json.loads(st.secrets["GMAIL_TOKEN"])
            with open("token.json", "w", encoding="utf-8") as f:
                json.dump(token_json, f)
        except Exception:
            pass

# Put GROQ key into environment for liteLLM
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Validate critical secrets
def validate_required():
    missing = []
    for name, val in [("TRELLO_KEY", TRELLO_KEY), ("TRELLO_TOKEN", TRELLO_TOKEN), ("TRELLO_LIST_ID", TRELLO_LIST_ID)]:
        if not val:
            missing.append(name)
    if missing:
        raise EnvironmentError(f"Missing required secrets: {', '.join(missing)}")

try:
    validate_required()
    st.sidebar.success("Trello keys loaded.")
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

# -------------------------
# Gmail helpers
# -------------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def get_gmail_credentials():
    """Return google oauth credentials, using token.json or credentials.json.
       On first local run, will open local browser for consent.
       In deployed mode, token.json can be provided via st.secrets['GMAIL_TOKEN']."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there is no token.json but credentials.json exists, use local flow (only works locally)
    if not creds and os.path.exists("credentials.json"):
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    # Refresh if necessary
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open("token.json", "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        except Exception:
            # If refresh fails, remove token to force reauth locally later
            try:
                os.remove("token.json")
            except Exception:
                pass
            creds = None
    return creds

def build_gmail_service():
    creds = get_gmail_credentials()
    if not creds:
        raise RuntimeError("No Google credentials available. Run locally to authorize or provide token.json via secrets.")
    return build("gmail", "v1", credentials=creds)

def fetch_inbox_last_24h(max_results=50):
    service = build_gmail_service()
    query = "in:inbox newer_than:1d"
    resp = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    msgs = resp.get("messages", [])
    emails = []
    for m in msgs:
        msg = service.users().messages().get(userId="me", id=m["id"], format="full").execute()
        headers = msg.get("payload", {}).get("headers", [])
        hmap = {h["name"].lower(): h["value"] for h in headers}
        subject = hmap.get("subject", "(No Subject)")
        sender = hmap.get("from", "Unknown Sender")
        snippet = msg.get("snippet", "")
        # build a lightweight content string for AI
        content = f"From: {sender}\nSubject: {subject}\n\n{snippet}"
        emails.append({"id": m["id"], "subject": subject, "from": sender, "snippet": snippet, "content": content})
    return emails

# -------------------------
# AI parsing (LiteLLM / Groq)
# -------------------------
def analyze_is_project(email_content):
    """Return tuple (is_project: bool, ai_error: str or None). Handles rate-limit errors by raising them to caller."""
    prompt = (
        "Decide if this email is project-related (work, assignment, request for deliverable). "
        "Answer only with 'yes' or 'no'.\n\nEmail:\n" + email_content
    )
    try:
        resp = completion(model=LLM_MODEL, messages=[{"role":"user","content":prompt}])
        text = resp["choices"][0]["message"]["content"].strip().lower()
        return ("yes" in text, None)
    except Exception as e:
        # propagate errors (caller will handle retries for rate limits)
        return (False, str(e))

# -------------------------
# Trello helpers
# -------------------------
def create_trello_card(title, desc):
    url = "https://api.trello.com/1/cards"
    params = {
        "key": TRELLO_KEY,
        "token": TRELLO_TOKEN,
        "idList": TRELLO_LIST_ID,
        "name": title,
        "desc": desc
    }
    r = requests.post(url, params=params)
    return r.status_code in (200, 201), r.status_code, r.text

# -------------------------
# Processing logic
# -------------------------
# Keep a simple in-memory set of processed message ids to avoid duplicates across cycles.
if "processed_ids" not in st.session_state:
    st.session_state["processed_ids"] = set()
if "auto_mode" not in st.session_state:
    st.session_state["auto_mode"] = False
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None
if "log_lines" not in st.session_state:
    st.session_state["log_lines"] = []

def log(msg):
    timestamp = datetime.now(timezone.utc).astimezone().isoformat()
    line = f"[{timestamp}] {msg}"
    st.session_state["log_lines"].append(line)
    # keep size reasonable
    st.session_state["log_lines"] = st.session_state["log_lines"][-500:]

def process_once(max_emails):
    """Fetch recent inbox emails (24h), filter out already processed, analyze, insert to Trello with delay where configured.
       Returns summary dict."""
    summary = {"fetched": 0, "processed": 0, "inserted": 0, "skipped": 0, "errors": 0}
    try:
        emails = fetch_inbox_last_24h(max_results=max_emails)
    except Exception as e:
        log(f"Failed to fetch emails: {e}")
        summary["errors"] += 1
        return summary

    summary["fetched"] = len(emails)
    for i, e in enumerate(emails, 1):
        mid = e["id"]
        if mid in st.session_state["processed_ids"]:
            log(f"Skipping already processed message: {e['subject']}")
            summary["skipped"] += 1
            continue
        st.session_state["processed_ids"].add(mid)
        summary["processed"] += 1
        log(f"Analyzing ({i}/{len(emails)}): {e['subject']}")
        is_project, ai_error = analyze_is_project(e["content"])
        if ai_error:
            # check for rate-limit or other provider hints; if rate limit, wait and retry once
            err_lower = ai_error.lower()
            log(f"AI error: {ai_error}")
            summary["errors"] += 1
            if "rate limit" in err_lower or "rate_limit" in err_lower or "ratelimit" in err_lower:
                log("AI rate limit detected. Waiting one delay cycle then retrying.")
                time.sleep(delay_seconds)
                is_project, ai_error = analyze_is_project(e["content"])
                if ai_error:
                    log(f"AI retry failed: {ai_error}")
                    summary["errors"] += 1
                    continue
            else:
                continue
        if is_project:
            title = e["subject"] or "Unnamed project"
            desc = f"From: {e['from']}\n\nSnippet:\n{e['snippet']}"
            ok, code, text = create_trello_card(title, desc)
            if ok:
                log(f"Inserted card for: {title}")
                summary["inserted"] += 1
            else:
                log(f"Trello error ({code}): {text}")
                summary["errors"] += 1
            # delay between inserts
            log(f"Waiting {delay_seconds} seconds before next action.")
            time.sleep(delay_seconds)
        else:
            log(f"Skipped (not project): {e['subject']}")
            summary["skipped"] += 1
    return summary

# -------------------------
# Auto mode thread
# -------------------------
auto_thread = None

def auto_worker(stop_event):
    log("Auto mode started.")
    while not stop_event.is_set():
        log("Starting processing cycle.")
        summary = process_once(max_emails_per_run)
        log(f"Cycle summary: fetched={summary['fetched']}, processed={summary['processed']}, inserted={summary['inserted']}, skipped={summary['skipped']}, errors={summary['errors']}")
        st.session_state["last_run"] = datetime.now().isoformat()
        # wait for next check interval, but break early if stop_event set
        for _ in range(int(check_interval_minutes * 60)):
            if stop_event.is_set():
                break
            time.sleep(1)
    log("Auto mode stopped.")

# Controls UI
col1, col2 = st.columns([3,1])
with col1:
    st.subheader("Manual run")
    if st.button("Run now"):
        st.session_state["last_run"] = None
        st.session_state["log_lines"] = st.session_state.get("log_lines", [])
        summary = process_once(max_emails_per_run)
        st.success(f"Run complete. Inserted {summary['inserted']} cards. Processed {summary['processed']} emails.")
with col2:
    st.subheader("Auto Mode")
    if "auto_stop_event" not in st.session_state:
        st.session_state["auto_stop_event"] = None
    auto_toggle = st.checkbox("Enable Auto Mode", value=st.session_state["auto_mode"])
    st.session_state["auto_mode"] = auto_toggle
    if auto_toggle and (st.session_state.get("auto_stop_event") is None or st.session_state["auto_stop_event"].is_set()):
        # start thread
        stop_event = threading.Event()
        st.session_state["auto_stop_event"] = stop_event
        thread = threading.Thread(target=auto_worker, args=(stop_event,), daemon=True)
        st.session_state["auto_thread"] = thread
        thread.start()
        st.success("Auto Mode enabled.")
    if (not auto_toggle) and st.session_state.get("auto_stop_event"):
        stop_event = st.session_state["auto_stop_event"]
        stop_event.set()
        st.session_state["auto_stop_event"] = None
        st.success("Auto Mode disabled.")

# Logs and status
st.markdown("## Status")
last_run = st.session_state.get("last_run")
st.write(f"Last run: {last_run or 'Never'}")
st.write(f"Processed message ids in this session: {len(st.session_state['processed_ids'])}")

st.markdown("## Logs")
log_area = st.empty()
log_lines = st.session_state.get("log_lines", [])
if log_lines:
    log_area.text("\n".join(log_lines[-200:]))
else:
    log_area.text("No logs yet.")

st.caption("Notes: Auto Mode will run in the background of the Streamlit session. For Gmail OAuth, provide token.json via Streamlit secrets as GMAIL_TOKEN or run locally once to generate token.json and then store it in secrets. Do not commit credentials to Git.")
