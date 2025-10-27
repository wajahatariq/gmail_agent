# streamlit_app.py
import os
import io
import time
import json
import re
import base64
import requests
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
st.markdown("Fetch Inbox emails (last 24 hours), extract project details with AI, save attachments, and create Trello cards.")

# -------------------------
# Sidebar settings
# -------------------------
st.sidebar.header("Settings")
delay_seconds = st.sidebar.number_input("Delay between Trello inserts (seconds)", min_value=5, max_value=300, value=30)
check_interval_minutes = st.sidebar.number_input("Auto-check interval (minutes)", min_value=1, max_value=180, value=15)
max_emails_per_run = st.sidebar.number_input("Max emails to process per run", min_value=1, max_value=100, value=10)
auto_mode_toggle = st.sidebar.checkbox("Enable Auto Mode", value=False)
st.sidebar.markdown("---")
st.sidebar.write("Secrets should be set in Streamlit secrets (recommended) or environment variables for local testing.")

# -------------------------
# Secrets / env helper
# -------------------------
def secret(name):
    if hasattr(st, "secrets") and name in st.secrets:
        return st.secrets[name]
    return os.getenv(name)

TRELLO_KEY = secret("TRELLO_KEY")
TRELLO_TOKEN = secret("TRELLO_TOKEN")
TRELLO_LIST_ID = secret("TRELLO_LIST_ID")
GROQ_API_KEY = secret("GROQ_API_KEY")
LLM_MODEL = secret("LLM_MODEL") or os.getenv("LLM_MODEL") or "groq/llama-3.1-8b-instant"

# Recreate credentials/token files from secrets if provided
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

# Put Groq key in env for LiteLLM
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Validate Trello keys
missing = [k for k, v in (("TRELLO_KEY", TRELLO_KEY), ("TRELLO_TOKEN", TRELLO_TOKEN), ("TRELLO_LIST_ID", TRELLO_LIST_ID)) if not v]
if missing:
    st.error(f"Missing required Trello secrets: {', '.join(missing)}. Add them to Streamlit Secrets or env.")
    st.stop()

# -------------------------
# Gmail helpers
# -------------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/gmail.modify"]

def get_gmail_creds():
    creds = None
    if os.path.exists("token.json"):
        try:
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        except Exception:
            creds = None
    if not creds and os.path.exists("credentials.json"):
        # Only works locally (opens a local browser)
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
        raise RuntimeError("No Gmail credentials. Run locally to create token.json or provide token in secrets.")
    return build("gmail", "v1", credentials=creds)

def fetch_inbox_last_24h(max_results=50):
    service = build_gmail_service()
    query = "in:inbox newer_than:1d"
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
        # We'll need full payload when downloading attachments
        emails.append({"id": m["id"], "subject": subject, "from": sender, "date": date, "snippet": snippet, "payload": msg.get("payload", {})})
    return emails

# Download attachment via Gmail attachment ID
def download_gmail_attachment(service, message_id, attachment_id, filename, project_folder):
    try:
        att = service.users().messages().attachments().get(userId="me", messageId=message_id, id=attachment_id).execute()
        data = att.get("data")
        if not data:
            return False, "no data"
        file_bytes = base64.urlsafe_b64decode(data.encode("utf-8"))
        # ensure folder
        os.makedirs(project_folder, exist_ok=True)
        path = os.path.join(project_folder, filename)
        with open(path, "wb") as f:
            f.write(file_bytes)
        return True, path
    except Exception as e:
        return False, str(e)

# Try to download a URL (direct file link) into project folder
def download_url_to_project(url, project_folder):
    try:
        # polite headers
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            # try to detect filename
            cd = r.headers.get("content-disposition", "")
            if "filename=" in cd:
                filename = re.findall(r'filename="?([^";]+)"?', cd)[0]
            else:
                filename = os.path.basename(url.split("?")[0]) or f"file_{int(time.time())}"
            os.makedirs(project_folder, exist_ok=True)
            fp = os.path.join(project_folder, filename)
            with open(fp, "wb") as f:
                f.write(r.content)
            return True, fp
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)

# -------------------------
# AI parsing & extraction
# -------------------------
def clean_subject(subject):
    return re.sub(r"^(fwd:|fw:|re:)\s*", "", subject, flags=re.IGNORECASE).strip()

# Prompt that asks for structured JSON with fields we need
EXTRACTION_PROMPT_TEMPLATE = """
You are an assistant that extracts structured project information from an email message.
Return JSON only with keys:
{
  "is_project": true/false,
  "project_name": "string (project title or subject)",
  "client_name": "string (client name or sender name/email)",
  "instructions": "string (cleaned main request/instructions)",
  "due_date": "string or empty",
  "file_links": ["list","of","urls"]
}

Email:
{email}
"""

def analyze_and_extract(email_text):
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(email=email_text)
    try:
        resp = completion(model=LLM_MODEL, messages=[{"role":"user","content":prompt}])
        text = resp["choices"][0]["message"]["content"]
        parsed = json.loads(text)
        return parsed, None
    except Exception as e:
        return None, str(e)

# -------------------------
# Trello helpers (create card, attach file)
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
    files = {"file": (name or os.path.basename(file_path), open(file_path, "rb"))}
    r = requests.post(url, params=params, files=files)
    return r.status_code in (200, 201), r.text

def attach_url_to_trello_card(card_id, url_link, name=None):
    url = f"{TRELLO_BASE}/cards/{card_id}/attachments"
    params = {"key": TRELLO_KEY, "token": TRELLO_TOKEN, "url": url_link, "name": name or url_link}
    r = requests.post(url, params=params)
    return r.status_code in (200, 201), r.text

# -------------------------
# Persistence: session state
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
# Core processing: single run
# -------------------------
def process_emails_once(max_emails):
    summary = {"fetched": 0, "parsed": 0, "inserted": 0, "skipped": 0, "errors": 0}
    try:
        emails = fetch_inbox_last_24h(max_results=max_emails)
    except Exception as e:
        log(f"Failed to fetch emails: {e}")
        summary["errors"] += 1
        return summary

    summary["fetched"] = len(emails)
    service = None
    try:
        service = build_gmail_service()
    except Exception:
        # service is only necessary for attachments download; if missing we'll still try link downloads and Trello uploads
        service = None

    for e in emails:
        mid = e["id"]
        if mid in st.session_state["processed_ids"]:
            log(f"Already processed message id {mid}: skipping")
            summary["skipped"] += 1
            continue

        st.session_state["processed_ids"].add(mid)
        summary["parsed"] += 1
        log(f"Parsing message: {e['subject']}")

        # Build full email text for AI: subject, sender, date, snippet and payload snippet
        payload_text = e.get("snippet", "")
        # include payload body parts text if present (concatenate simple text parts)
        parts = e.get("payload", {}).get("parts", [])
        body_text = payload_text
        if parts:
            for part in parts:
                mime = part.get("mimeType","")
                if mime in ("text/plain","text/html"):
                    data = (part.get("body") or {}).get("data")
                    if data:
                        try:
                            decoded = base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="ignore")
                            if mime == "text/html":
                                # strip tags quickly
                                decoded = re.sub(r"<[^>]+>", "", decoded)
                            body_text += "\n" + decoded
                        except Exception:
                            pass

        email_text = f"Subject: {e['subject']}\nFrom: {e['from']}\nDate: {e['date']}\n\n{body_text}"

        # Ask AI to extract project details
        parsed, err = analyze_and_extract(email_text)
        if err:
            log(f"AI extraction error: {err}")
            summary["errors"] += 1
            # If rate-limited, wait and retry a single time
            if "rate" in err.lower():
                log("Rate limit detected. Waiting then retrying extraction.")
                time.sleep(delay_seconds)
                parsed, err = analyze_and_extract(email_text)
                if err:
                    log(f"AI retry failed: {err}")
                    summary["errors"] += 1
                    continue
            else:
                continue

        if not parsed or not parsed.get("is_project"):
            log(f"Not a project: {e['subject']}")
            summary["skipped"] += 1
            continue

        # Clean project name
        project_name = parsed.get("project_name") or clean_subject(e["subject"])
        project_name_clean = re.sub(r"[^\w\s\-_.]", "", project_name).strip().replace(" ", "_")[:120] or "project"

        # Create project folder
        base_project_folder = os.path.join("data", "projects", project_name_clean)
        attachments_folder = os.path.join(base_project_folder, "attachments")
        os.makedirs(attachments_folder, exist_ok=True)

        # Gather file paths or URLs to attach to trello
        attached_files = []
        attached_links = []

        # 1) Download attachments from Gmail payload (if any)
        try:
            payload = e.get("payload", {})
            parts = payload.get("parts", [])
            for part in parts:
                filename = part.get("filename")
                body = part.get("body", {}) or {}
                if filename and body.get("attachmentId") and service:
                    attachment_id = body["attachmentId"]
                    ok, res = download_gmail_attachment(service, e["id"], attachment_id, filename, attachments_folder)
                    if ok:
                        attached_files.append(res)
                        log(f"Saved attachment: {res}")
                    else:
                        log(f"Attachment download failed: {res}")
        except Exception as ex:
            log(f"Attachment extraction error: {ex}")

        # 2) Try to download file links returned by AI or present in the email body
        file_links = parsed.get("file_links", []) or []
        # Also extract links from body_text
        url_candidates = re.findall(r"https?://[^\s'\"<>]+", body_text)
        for u in url_candidates + file_links:
            # Only attempt to download if it looks like a direct file or dropbox link
            if any(ext in u.lower() for ext in [".pdf", ".png", ".jpg", ".jpeg", ".zip", ".doc", ".docx", ".xls", ".xlsx"]) or "dropbox.com" in u.lower():
                ok, res = download_url_to_project(u, attachments_folder)
                if ok:
                    attached_files.append(res)
                    log(f"Downloaded link to: {res}")
                else:
                    log(f"Link download failed ({u}): {res}")
                    attached_links.append(u)
            else:
                # Keep as link to attach or put into description
                attached_links.append(u)

        # Compose Trello description
        desc_lines = [
            f"Client: {parsed.get('client_name') or e['from']}",
            f"Instructions: {parsed.get('instructions','').strip()}",
            f"Due Date: {parsed.get('due_date','')}",
        ]
        if attached_links:
            desc_lines.append("Links:")
            for l in attached_links:
                desc_lines.append(l)
        desc_lines.append(f"Stored files folder: {os.path.abspath(base_project_folder)}")
        description = "\n\n".join(desc_lines)

        # Create Trello card
        title = clean_subject(project_name) or "Unnamed Project"
        ok, card_id_or_err = create_trello_card(title, description)
        if not ok:
            log(f"Failed to create Trello card: {card_id_or_err}")
            summary["errors"] += 1
            continue

        card_id = card_id_or_err
        log(f"Created Trello card: {title} (id={card_id})")

        # Attach files to card if present; else attach links (as URLs)
        if attached_files:
            for fp in attached_files:
                try:
                    ok_attach, attach_resp = attach_file_to_trello_card(card_id, fp, name=os.path.basename(fp))
                    if ok_attach:
                        log(f"Attached file to Trello: {fp}")
                    else:
                        log(f"Failed to attach file to Trello: {attach_resp}")
                except Exception as ex:
                    log(f"Error uploading attachment to Trello: {ex}")
        else:
            # if no saved files but links, attach links
            for l in attached_links:
                try:
                    ok_link, resp_text = attach_url_to_trello_card(card_id, l)
                    if ok_link:
                        log(f"Attached URL to Trello card: {l}")
                    else:
                        log(f"Failed to attach URL to Trello: {resp_text}")
                except Exception as ex:
                    log(f"Error attaching URL to Trello: {ex}")

        summary["inserted"] += 1
        # Delay between inserts
        log(f"Waiting {delay_seconds} seconds before next item.")
        time.sleep(delay_seconds)

    return summary

# -------------------------
# Auto Mode background worker
# -------------------------
import threading

stop_event = None
worker_thread = None

def auto_worker():
    log("Auto worker started.")
    while True:
        if stop_event and stop_event.is_set():
            log("Auto worker stopped by event.")
            break
        log("Auto cycle: starting processing run.")
        summary = process_emails_once(max_emails_per_run)
        log(f"Auto cycle summary: {summary}")
        # Sleep for check_interval_minutes with early stop
        for _ in range(check_interval_minutes * 60):
            if stop_event and stop_event.is_set():
                break
            time.sleep(1)
    log("Auto worker exiting.")

# -------------------------
# UI Controls & flow
# -------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.subheader("Manual Run")
    if st.button("Run Now"):
        st.session_state.setdefault("logs", [])
        st.session_state.setdefault("processed_ids", set())
        st.rerun()
    st.write("To perform a run without auto mode, press 'Run Now' above. Auto Mode will run in background when enabled.")
with col2:
    st.subheader("Auto Mode")
    # Manage toggle and thread lifecycle
    if auto_mode_toggle and not st.session_state.get("auto_running"):
        # start
        st.session_state["auto_running"] = True
        st.session_state["auto_stop_flag"] = threading.Event()
        stop_event = st.session_state["auto_stop_flag"]
        worker_thread = threading.Thread(target=auto_worker, daemon=True)
        worker_thread.start()
        st.success("Auto Mode enabled.")
    if (not auto_mode_toggle) and st.session_state.get("auto_running"):
        # stop
        st.session_state["auto_stop_flag"].set()
        st.session_state["auto_running"] = False
        st.success("Auto Mode disabled.")

# If Run Now pressed -> execute once directly (synchronous)
if st.button("Execute Run Now (synchronous)"):
    st.write("Processing...")
    result = process_emails_once(max_emails_per_run)
    st.write(f"Run finished. Inserted: {result['inserted']}, Skipped: {result['skipped']}, Errors: {result['errors']}")

# Logs & status
st.markdown("## Status")
st.write(f"Processed message ids in this session: {len(st.session_state.get('processed_ids', set()))}")
last_logs = st.session_state.get("logs", [])[-200:]
if last_logs:
    st.markdown("## Logs (latest)")
    st.text("\n".join(last_logs))
else:
    st.info("No logs yet. Click 'Execute Run Now' to run a synchronous extraction.")
st.caption("Notes: store credentials/token securely in Streamlit secrets. If running locally, keep credentials.json and run once to generate token.json.")

