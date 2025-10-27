# streamlit_app.py
import os
import time
import json
import base64
import re
import requests
import threading
import streamlit as st
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from litellm import completion

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="Gmail → Trello (AI)", layout="wide")
st.title("Gmail → Trello (AI)")
st.write("Fetch Inbox emails (last 24 hours), ask Groq to format them for Trello, and create cards. Attachments are uploaded to Trello.")

# -------------------------
# Sidebar settings
# -------------------------
st.sidebar.header("Settings")
delay_seconds = st.sidebar.number_input("Delay between Trello inserts (seconds)", min_value=5, max_value=300, value=60)
max_emails_per_run = st.sidebar.number_input("Max emails to process per run", min_value=1, max_value=100, value=10)
auto_mode_toggle = st.sidebar.checkbox("Enable Auto Mode", value=False)
st.sidebar.markdown("---")
st.sidebar.write("Provide secrets in Streamlit Secrets (recommended) or env vars locally.")

# -------------------------
# Secrets helper
# -------------------------
def secret(key):
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key)

TRELLO_KEY = secret("TRELLO_KEY")
TRELLO_TOKEN = secret("TRELLO_TOKEN")
TRELLO_LIST_ID = secret("TRELLO_LIST_ID")
GROQ_API_KEY = secret("GROQ_API_KEY")
LLM_MODEL = secret("LLM_MODEL") or os.getenv("LLM_MODEL") or "groq/llama-3.1-8b-instant"

# Recreate Gmail credentials/token files if provided in secrets
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

# Put Groq key into env for LiteLLM
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Validate Trello secrets
missing = [k for k, v in (("TRELLO_KEY", TRELLO_KEY), ("TRELLO_TOKEN", TRELLO_TOKEN), ("TRELLO_LIST_ID", TRELLO_LIST_ID)) if not v]
if missing:
    st.error(f"Missing Trello secrets: {', '.join(missing)}. Add them to Streamlit Secrets or env.")
    st.stop()

# -------------------------
# Gmail helpers
# -------------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def get_gmail_creds():
    creds = None
    if os.path.exists("token.json"):
        try:
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        except Exception:
            creds = None
    if not creds and os.path.exists("credentials.json"):
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open("token.json", "w", encoding="utf-8") as f:
                f.write(creds.to_json())
        except Exception:
            try:
                os.remove("token.json")
            except Exception:
                pass
            creds = None
    return creds

def build_gmail_service():
    creds = get_gmail_creds()
    if not creds:
        raise RuntimeError("No Gmail credentials available. Provide token.json via secrets or run locally to authorize.")
    return build("gmail", "v1", credentials=creds)

def fetch_inbox_last_24h(max_results=50):
    service = build_gmail_service()
    query = "category:primary newer_than:1d"
    resp = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    msgs = resp.get("messages", []) or []
    emails = []
    for m in msgs:
        msg = service.users().messages().get(userId="me", id=m["id"], format="full").execute()
        headers = msg.get("payload", {}).get("headers", [])
        hmap = {h["name"].lower(): h["value"] for h in headers}
        subject = hmap.get("subject", "(No Subject)")
        sender = hmap.get("from", "Unknown Sender")
        date = hmap.get("date", "")
        snippet = msg.get("snippet", "")
        payload = msg.get("payload", {})
        emails.append({"id": m["id"], "subject": subject, "from": sender, "date": date, "snippet": snippet, "payload": payload})
    return emails

# Download attachment via Gmail attachment ID
def download_gmail_attachment(service, message_id, attachment_id, filename, project_folder):
    try:
        att = service.users().messages().attachments().get(userId="me", messageId=message_id, id=attachment_id).execute()
        data = att.get("data")
        if not data:
            return False, "no data"
        file_bytes = base64.urlsafe_b64decode(data.encode("utf-8"))
        os.makedirs(project_folder, exist_ok=True)
        path = os.path.join(project_folder, filename)
        with open(path, "wb") as f:
            f.write(file_bytes)
        return True, path
    except Exception as e:
        return False, str(e)

# -------------------------
# Groq prompt & call
# -------------------------
PROMPT_TEMPLATE = """You are a professional assistant that extracts project information and produces a Trello-ready description.
Read the email below. Produce a compact structured text suitable for directly pasting into a Trello card description.
Include a clear first line: "Project Name: <name>".
Then include: Client:, Overview:, Due Date: (if any), Links:, Attachments: (list names or 'none').
Do NOT return JSON. Return only human readable text with those labels.

Email:
Subject: {subject}
From: {sender}
Date: {date}

{body}
"""

def ask_groq_for_trello_text(subject, sender, date, body_text):
    prompt = PROMPT_TEMPLATE.format(subject=subject, sender=sender, date=date, body=body_text)
    try:
        resp = completion(model=LLM_MODEL, messages=[{"role":"user","content":prompt}])
        text = resp["choices"][0]["message"]["content"]
        return text, None
    except Exception as e:
        return None, str(e)

# -------------------------
# Trello helpers
# -------------------------
TRELLO_BASE = "https://api.trello.com/1"

def create_trello_card(title, description):
    url = f"{TRELLO_BASE}/cards"
    params = {"key": TRELLO_KEY, "token": TRELLO_TOKEN, "idList": TRELLO_LIST_ID, "name": title, "desc": description}
    r = requests.post(url, params=params)
    if r.status_code in (200, 201):
        return True, r.json().get("id")
    return False, r.text

def attach_file_to_trello_card(card_id, file_path, name=None):
    url = f"{TRELLO_BASE}/cards/{card_id}/attachments"
    params = {"key": TRELLO_KEY, "token": TRELLO_TOKEN}
    with open(file_path, "rb") as f:
        files = {"file": (name or os.path.basename(file_path), f)}
        r = requests.post(url, params=params, files=files)
    return r.status_code in (200, 201), r.text

# -------------------------
# Persistence & logs
# -------------------------
if "processed_ids" not in st.session_state:
    st.session_state["processed_ids"] = set()
if "logs" not in st.session_state:
    st.session_state["logs"] = []

def log(msg):
    ts = datetime.now(timezone.utc).astimezone().isoformat()
    st.session_state["logs"].append(f"[{ts}] {msg}")
    st.session_state["logs"] = st.session_state["logs"][-500:]

# -------------------------
# Core run logic (single pass)
# -------------------------
def process_once(max_emails, delay_seconds):
    summary = {"fetched":0, "uploaded":0, "skipped":0, "errors":0}
    try:
        emails = fetch_inbox_last_24h(max_results=max_emails)
    except Exception as e:
        log(f"Failed to fetch emails: {e}")
        summary["errors"] += 1
        return summary

    summary["fetched"] = len(emails)
    # prepare Gmail service for attachments if available
    try:
        gmail_service = build_gmail_service()
    except Exception:
        gmail_service = None

    for i, e in enumerate(emails, start=1):
        mid = e["id"]
        if mid in st.session_state["processed_ids"]:
            log(f"Skipping already processed message: {e['subject']}")
            summary["skipped"] += 1
            continue

        log(f"Processing ({i}/{len(emails)}): {e['subject']}")
        body_text = e.get("snippet","")
        # try to include text parts from payload to give Groq more context
        payload = e.get("payload") or {}
        parts = payload.get("parts", []) or []
        for p in parts:
            mime = p.get("mimeType","")
            if mime in ("text/plain","text/html"):
                data = (p.get("body") or {}).get("data")
                if data:
                    try:
                        decoded = base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="ignore")
                        if mime == "text/html":
                            decoded = re.sub(r"<[^>]+>", "", decoded)
                        body_text += "\n" + decoded
                    except Exception:
                        pass

        # Ask Groq to produce trello-ready text; retry once on failure
        trello_text, err = ask_groq_for_trello_text(e["subject"], e["from"], e["date"], body_text)
        if err:
            log(f"Groq error: {err}. Will wait and retry once.")
            time.sleep(delay_seconds)
            trello_text, err = ask_groq_for_trello_text(e["subject"], e["from"], e["date"], body_text)
            if err:
                log(f"Groq retry failed: {err}. Skipping email.")
                summary["errors"] += 1
                st.session_state["processed_ids"].add(mid)
                continue

        # Extract "Project Name:" line for title if present
        project_name_match = re.search(r"Project\s*Name\s*:\s*(.+)", trello_text, flags=re.IGNORECASE)
        if project_name_match:
            card_title = project_name_match.group(1).strip()
        else:
            # fallback to subject cleaned
            card_title = re.sub(r"^(fwd:|fw:|re:)\s*", "", e["subject"], flags=re.IGNORECASE).strip() or "Unnamed Project"

        # Create Trello card (retry once on failure)
        ok, res = create_trello_card(card_title, trello_text)
        if not ok:
            log(f"Trello create error: {res}. Retrying once after delay.")
            time.sleep(delay_seconds)
            ok, res = create_trello_card(card_title, trello_text)
            if not ok:
                log(f"Trello retry failed: {res}. Skipping.")
                summary["errors"] += 1
                st.session_state["processed_ids"].add(mid)
                continue

        card_id = res
        log(f"Created Trello card '{card_title}' (id={card_id})")

        # Try to upload attachments from Gmail to Trello
        attached_any = False
        try:
            if gmail_service:
                for p in parts:
                    fn = p.get("filename")
                    body = p.get("body", {}) or {}
                    if fn and body.get("attachmentId"):
                        att_id = body["attachmentId"]
                        ok_dl, path_or_err = download_gmail_attachment(gmail_service, mid, att_id, fn, project_folder := os.path.join("data","attachments"))
                        if ok_dl:
                            attached_any = True
                            ok_att, att_resp = attach_file_to_trello_card(card_id, path_or_err, name=fn)
                            if ok_att:
                                log(f"Attached file to Trello: {fn}")
                            else:
                                log(f"Failed to attach file to Trello: {att_resp}")
                        else:
                            log(f"Attachment download failed: {path_or_err}")
        except Exception as ex:
            log(f"Attachment handling error: {ex}")

        st.session_state["processed_ids"].add(mid)
        summary["uploaded"] += 1

        log(f"Waiting {delay_seconds} seconds before next email.")
        time.sleep(delay_seconds)

    return summary

# -------------------------
# UI controls and auto mode
# -------------------------
if "auto_worker" not in st.session_state:
    st.session_state["auto_worker"] = None
if "auto_stop" not in st.session_state:
    st.session_state["auto_stop"] = None

def auto_loop(stop_event):
    log("Auto Mode started.")
    while not stop_event.is_set():
        summary = process_once(max_emails_per_run, delay_seconds)
        log(f"Auto cycle result: {summary}")
        # wait check interval but break when stop_event set
        for _ in range(60):  # sleep 60 seconds increments for responsiveness (default 1 minute)
            if stop_event.is_set():
                break
            time.sleep(1)
    log("Auto Mode stopped.")

col1, col2 = st.columns([3,1])
with col1:
    if st.button("Fetch & Upload Projects"):
        st.session_state.setdefault("logs", [])
        st.write("Running fetch & upload...")
        result = process_once(max_emails_per_run, delay_seconds)
        st.write(f"Done. Uploaded: {result['uploaded']}, Skipped: {result['skipped']}, Errors: {result['errors']}.")
with col2:
    st.write("Auto Mode")
    if auto_mode_toggle and not st.session_state.get("auto_running", False):
        stop_event = threading.Event()
        st.session_state["auto_stop"] = stop_event
        th = threading.Thread(target=auto_loop, args=(stop_event,), daemon=True)
        st.session_state["auto_worker"] = th
        st.session_state["auto_running"] = True
        th.start()
        st.success("Auto Mode enabled.")
    if (not auto_mode_toggle) and st.session_state.get("auto_running", False):
        st.session_state["auto_stop"].set()
        st.session_state["auto_running"] = False
        st.success("Auto Mode disabled.")

# Logs + status
st.markdown("## Status")
st.write(f"Processed ids (this session): {len(st.session_state.get('processed_ids', set()))}")
last_logs = st.session_state.get("logs", [])[-200:]
if last_logs:
    st.markdown("## Logs (latest)")
    st.text("\n".join(last_logs))
else:
    st.info("No logs yet. Click 'Fetch & Upload Projects' to run.")
st.caption("Keep credentials in Streamlit Secrets. Token.json can be pasted as GMAIL_TOKEN if deploying to Streamlit Cloud.")

