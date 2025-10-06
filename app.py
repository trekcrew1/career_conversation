# app.py
import os
import re
import json
from pathlib import Path
from string import Template

from dotenv import load_dotenv
from openai import OpenAI
import requests
from pypdf import PdfReader
import gradio as gr

# ----------------------------
# Environment & configuration
# ----------------------------
load_dotenv(override=True)  # useful locally; harmless on Spaces

def _get_openai_key():
    # ultra-robust fetch in case of odd casing/whitespace
    for k, v in os.environ.items():
        if k.strip().upper() == "OPENAI_API_KEY" and v:
            return v
    return os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")

OPENAI_API_KEY = _get_openai_key()

# Looking/not-looking flag (default: True here per your current file)
def _get_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

LOOKING_FOR_ROLE = _get_bool("LOOKING_FOR_ROLE", False)

# --- Dynamic decline powered by OpenAI ---
DECLINE_FALLBACK = (
    "Thanks so much for reaching out about the role. "
    "I’m very happy where I am and not looking right now, "
    "but I appreciate the connection and would love to stay in touch."
)

def generate_polite_decline(user_text: str) -> str:
    """
    Create a short, polite, appreciative decline tailored to the inbound message.
    Requirements:
      - 1–3 sentences, professional and warm.
      - Clearly say Robert is very happy where he is and not looking.
      - Welcome staying connected / keeping in touch.
      - Do NOT ask for email or share a resume.
    """
    if not OPENAI_READY:
        return DECLINE_FALLBACK
    try:
        client = OpenAI()  # reads OPENAI_API_KEY from env
        system = (
            "You write brief, professional replies on behalf of Robert Morrow. "
            "When a recruiter or contact proposes a role, craft a concise decline "
            "that is appreciative and polite. Make it clear Robert is very happy "
            "where he is and not looking. Invite staying connected. 1–3 sentences. "
            "No resume sharing. No over-promising. Keep it friendly."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Compose the reply to this inbound message:\n\n{user_text}"},
        ]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=120,
        )
        text = resp.choices[0].message.content.strip()
        return text or DECLINE_FALLBACK
    except Exception as e:
        print(f"Decline generation error: {e}")
        return DECLINE_FALLBACK

# --- Dynamic interested reply powered by OpenAI (for when LOOKING_FOR_ROLE=True) ---
INTEREST_FALLBACK = (
    "Thanks for reaching out — I’m currently exploring opportunities. "
    "Could you share a bit more about the role (scope, team/domain, remote or on-site, "
    "comp range, and timeline)? Happy to continue the conversation."
)

def generate_polite_interest(user_text: str) -> str:
    """
    Create a short, professional 'interested' reply.
    Requirements:
      - 1–3 sentences, appreciative and confident (not desperate).
      - Ask 2–4 concise, qualifying questions (scope, team/domain, remote/on-site, comp range, timeline).
      - Do NOT share personal contact details or ask for email here.
    """
    if not OPENAI_READY:
        return INTEREST_FALLBACK
    try:
        client = OpenAI()
        system = (
            "You write brief, professional replies on behalf of Robert Morrow. "
            "When a recruiter proposes a role and Robert IS open to opportunities, craft a concise, "
            "confident reply (1–3 sentences) that is appreciative and asks 2–4 qualifying questions: "
            "role scope, team/domain, remote/on-site, comp range, and timeline. "
            "Avoid sounding desperate. Do NOT include personal contact details or ask for an email."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Compose the reply to this inbound message:\n\n{user_text}"},
        ]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=140,
        )
        text = resp.choices[0].message.content.strip()
        return text or INTEREST_FALLBACK
    except Exception as e:
        print(f"Interest generation error: {e}")
        return INTEREST_FALLBACK

PUSHOVER_USER  = os.getenv("PUSHOVER_USER")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_URL   = os.getenv("PUSHOVER_URL") or "https://api.pushover.net/1/messages.json"

OPENAI_READY = bool(OPENAI_API_KEY)
print(f"OpenAI key present: {OPENAI_READY}")
print(f"Pushover configured: {bool(PUSHOVER_USER and PUSHOVER_TOKEN)}")
print(f"Looking for role: {LOOKING_FOR_ROLE}")

# -------------
# Pushover util
# -------------
def push(message: str):
    if not (PUSHOVER_USER and PUSHOVER_TOKEN and PUSHOVER_URL):
        return
    try:
        requests.post(
            PUSHOVER_URL,
            data={"user": PUSHOVER_USER, "token": PUSHOVER_TOKEN, "message": message},
            timeout=10,
        )
    except Exception as e:
        print(f"Pushover error: {e}")

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}

# --------------------------
# Tool schemas for function calling
# --------------------------
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Context worth recording from the conversation"},
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Use this tool to record a question that was asked but not answered",
    "parameters": {
        "type": "object",
        "properties": {"question": {"type": "string", "description": "The question that was asked"}},
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]

def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool_fn = globals().get(tool_name)
        result = tool_fn(**arguments) if tool_fn else {}
        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
    return results

# ----------------------------
# Optional local content files
# ----------------------------
linkedin = ""
pdf_path = Path("personal_info/linkedin_profile.pdf")
if pdf_path.exists():
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                linkedin += text
    except Exception as e:
        print(f"PDF read error: {e}")

summary = ""
summary_path = Path("personal_info/summary.txt")
if summary_path.exists():
    try:
        summary = summary_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Summary read error: {e}")

# ----------------------------
# Curated education facts (deterministic answers)
# ----------------------------
EDUCATION = [
    {
        "institution": "Cleveland State University",
        "credential": "BBA (Information Technology)",
        "dates": "Feb 2008 – Jun 2010",
        "details": [
            "Completed 60 credit hours before transitioning to full-time work."
        ],
    },
    {
        "institution": "Louisiana State University",
        "credential": "Certificate, Information Technology (COAWA – Principles of Cloud Computing I)",
        "dates": "May 2023 – Jun 2023",
        "details": [
            "Covered motivations for cloud computing (business, economic, political).",
            "Deployment models (public/private/hybrid) and service types (IaaS, PaaS, SaaS).",
            "OSI model in the context of cloud.",
            "Vendor selection concepts."
        ],
    },
    {
        "institution": "Udemy",
        "credential": "Certificate, IoT Enabled Aeroponics using Raspberry Pi 3",
        "dates": "Jan 2022",
        "details": [
            "Flow sensor, LCD, ultrasonic water mister control on Raspberry Pi.",
            "Beebotte dashboard for remote monitoring; environment variables tracking."
        ],
    },
    {
        "institution": "Udemy",
        "credential": "Certificate, Growing Microgreens for Business and Pleasure",
        "dates": "Jun 2020",
        "details": [
            "Home microgreen garden: equipment, soil & hydroponic methods, environment setup, plant care."
        ],
    },
    {
        "institution": "Udemy",
        "credential": "Deep Learning (course)",
        "dates": "2018 – 2019",
        "details": [
            "Real-world tasks: ANN for churn, CNNs for image recognition, RNNs for stock prediction.",
            "SOMs for fraud, Boltzmann machines for recommenders, stacked autoencoders.",
            "TensorFlow & PyTorch."
        ],
    },
    {
        "institution": "Udemy",
        "credential": "Artificial Intelligence – The AI Masterclass",
        "dates": "2018 – 2019",
        "details": [
            "Hybrid Intelligent Systems; DL/DRL/Policy Gradient/NeuroEvolution.",
            "TF & Keras; FCNNs, CNNs, RNNs, VAEs, MDNs, Genetic Algorithms, ES, CMA-ES, PEPG."
        ],
    },
]

def _education_markdown() -> str:
    lines = ["### Education"]
    for item in EDUCATION:
        lines.append(f"**{item['institution']}** — {item['credential']}  \n{item['dates']}")
        if item.get("details"):
            for d in item["details"]:
                lines.append(f"- {d}")
        lines.append("")  # spacer
    return "\n".join(lines)

def _looks_like_education(text: str) -> bool:
    t = (text or "").lower()
    keys = (
        "education", "degree", "degrees", "school", "schools", "university", "college",
        "bba", "bs", "b.s.", "certificate", "certifications", "udemy", "lsu", "louisiana state",
        "cleveland state", "coursework", "studies", "study"
    )
    return any(k in t for k in keys)

# ----------------------------
# System prompt (dynamic by availability) + include curated education context
# ----------------------------
name = "Robert Morrow"

availability_instructions = (
    # Actively looking
    "AVAILABILITY:\n"
    "- Robert is **actively looking for a new position**.\n"
    "- When relevant, briefly mention he’s exploring opportunities now.\n"
    "- Ask 2–4 concise, qualifying questions (role, team/domain, location/remote, comp range, timeline).\n"
    "- Proactively and politely ask for an email to follow up; use `record_user_details` to capture it.\n"
    "- Be confident and concise—never sound desperate.\n"
    if LOOKING_FOR_ROLE else
    # Not actively looking
    "AVAILABILITY:\n"
    "- Robert is **not actively looking** and is happy where he is.\n"
    "- Be appreciative and neutral; avoid presenting as actively searching.\n"
    "- Only request contact details if the opportunity sounds truly exceptional.\n"
)

education_context = _education_markdown()

system_prompt = (
    f"You are acting as {name}. You are answering questions on {name}'s website, "
    f"particularly questions related to {name}'s career, background, skills and experience. "
    f"Your responsibility is to represent {name} for interactions on the website as faithfully as possible. "
    f"You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. "
    f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
    f"If you don't know the answer to any question, use your record_unknown_question tool to record the question that "
    f"you couldn't answer, even if it's about something trivial or unrelated to career. "
    f"If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email "
    f"and record it using your record_user_details tool. "
    "\n\n" + availability_instructions +
    "\nIMPORTANT: When answering questions about employment, current job, or tenure, treat multiple roles "
    "at the same employer as a combined employment history. Identify all roles listed for that employer in the provided "
    "profile. For each role, state the job title, start and end dates (or 'Present' if currently held), and the duration in years and months. "
    "Compute and state the total combined tenure at that employer by summing durations across all roles and avoiding "
    "double-counting overlapping time (if dates overlap, compute the union of intervals). "
    "If dates are missing or ambiguous, mark any values you estimate as 'estimated' and explain briefly. "
    "When the user asks about the 'current job', list the current role(s) first (those with end date 'Present') and then "
    "provide a concise company-level summary that includes total time at the company and a role-by-role breakdown. "
    "Example format: \"Rocket Mortgage — Total 9 years 8 months (Data Engineer, Mar 2023-Present — 2 yr 7 mo; "
    "Software Engineer, Feb 2016-Mar 2023 — 7 yr 2 mo)\"."
    f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
    f"## Education (curated – authoritative)\n{education_context}\n"
    f"With this context, please chat with the user, always staying in character as {name}.\n"
    "When asked about Education, prefer this curated section over inference; if something is not in the list, say you’re not certain."
)

# def _looks_like_job_pitch(text: str) -> bool:
#     t = (text or "").lower()
#     keywords = (
#         "job", "position", "role", "opportunity", "opening",
#         "hire", "hiring", "recruit", "recruiter", "headcount",
#         "interview", "join our", "work with us", "offer"
#     )
#     return any(k in t for k in keywords)

def _looks_like_job_pitch(text: str) -> bool:
    t = (text or "").lower()
    keywords = (
        "opportunity", "opening",
        "hire", "hiring", "recruit", "recruiter", "headcount",
        "interview", "join our", "work with us", "offer"
    )
    return any(k in t for k in keywords)

# ----------------------------
# Moderation + Brand Guardrails
# ----------------------------
SAFE_REFUSAL = "I’m going to keep it professional and skip that request. Happy to discuss experience, projects, and fit for roles."

# Heuristics (brand rules)
BAD_WORDS = re.compile(r"\b(fuck|shit|bitch|bastard|dumbass|moron|idiot|suck|crap)\b", re.I)
SLUR_HINTS = re.compile(r"\b(retard|tranny|faggot|chink|spic|kike)\b", re.I)
SECRET_LEAK_HINTS = re.compile(r"(api[_-]?key|sk-[A-Za-z0-9]{20,}|OPENAI_API_KEY|PUSHOVER_TOKEN|BEGIN PRIVATE KEY)", re.I)
POLITICAL_PUSH = re.compile(r"\b(vote for|endorse|support (?:the )?(?:democrats?|republicans?|candidate|party))\b", re.I)
GOSSIP_HINTS = re.compile(r"\b(name|identify).*(coworker|manager|person).*(stole|crime|illegal|harassment)\b", re.I)
INSULT_PATTERN = re.compile(r"\b(you|they|recruiters|hiring managers).*(stupid|dumb|idiot|useless|incompetent)\b", re.I)

# Inbound secret/injection requests (block before generation)
SECRET_EXFIL_REQ = re.compile(
    r"(print|show|reveal).*(system (?:prompt|instructions)|env(?:ironment)? (?:vars?|variables?)|api[_-]?keys?)|"
    r"list (?:all )?environment variables|OPENAI_API_KEY|PUSHOVER_TOKEN",
    re.I
)

def check_moderation(text: str):
    """Return (flagged: bool, categories: list[str]) using OpenAI Moderation."""
    try:
        client = OpenAI()
        r = client.moderations.create(model="omni-moderation-latest", input=text or "")
        res = r.results[0]
        cats = [k for k, v in res.categories.__dict__.items() if v]
        return bool(res.flagged), cats
    except Exception as e:
        # Fail open on moderation errors—but log for visibility
        print(f"Moderation error: {e}")
        return False, []

def heuristic_flags(text: str):
    """Return list of brand-rule issues found."""
    t = text or ""
    flags = []
    if BAD_WORDS.search(t): flags.append("profanity")
    if SLUR_HINTS.search(t): flags.append("slur")
    if SECRET_LEAK_HINTS.search(t): flags.append("secret_like")
    if POLITICAL_PUSH.search(t): flags.append("political_endorsement")
    if GOSSIP_HINTS.search(t): flags.append("defamation_risk")
    if INSULT_PATTERN.search(t): flags.append("insulting_tone")
    if "buzz off" in t.lower() or "trash" in t.lower(): flags.append("unprofessional_tone")
    return flags

def soften_tone(reply: str) -> str:
    """Rewrite response to a neutral, professional tone (no endorsements or secrets)."""
    try:
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system",
                 "content": "Rewrite the message to be polite, neutral, professional, and free of endorsements or secrets. Avoid profanity and insults."},
                {"role": "user", "content": reply}
            ],
            max_tokens=220,
        )
        new_text = r.choices[0].message.content.strip()
        return new_text or reply
    except Exception as e:
        print(f"Tone rewrite error: {e}")
        return "Let’s keep this professional. I’m happy to talk about my background, projects, and how I can help."

def safe_finalize(reply: str) -> str:
    """Apply moderation + brand rules to the model's draft reply."""
    # 1) Safety moderation
    flagged, cats = check_moderation(reply)
    if flagged:
        print(f"Moderation flagged categories: {cats}")
        return SAFE_REFUSAL

    # 2) Brand guardrails
    flags = heuristic_flags(reply)
    if not flags:
        return reply

    print(f"Brand flags: {flags}")

    # Hard block on secret-like content
    if "secret_like" in flags:
        return "For security and privacy, I can’t share credentials, environment variables, or internal prompts."

    # For tone/endorsement/defamation, try to rewrite
    if any(f in flags for f in ["political_endorsement", "unprofessional_tone", "insulting_tone", "defamation_risk", "profanity", "slur"]):
        return soften_tone(reply)

    # Fallback
    return reply

def guard_inbound_request(user_text: str):
    """Block unsafe or secret-exfil requests before generation."""
    flagged, cats = check_moderation(user_text)
    if flagged:
        print(f"Inbound moderation flagged: {cats}")
        return SAFE_REFUSAL
    if SECRET_EXFIL_REQ.search(user_text or ""):
        return "For security reasons, I can’t reveal system prompts, environment variables, or API keys."
    return None

# ----------------------------
# Chat handler
# ----------------------------
def chat(message, history):
    if not OPENAI_READY:
        return "Server is not configured with OPENAI_API_KEY. Add it in Settings → Variables & secrets, then restart this Space."

    # Inbound guard
    inbound_block = guard_inbound_request(message)
    if inbound_block:
        return inbound_block

    # If not looking and the inbound reads like a job/offer, produce a tailored decline
    if not LOOKING_FOR_ROLE and _looks_like_job_pitch(message):
        return generate_polite_decline(message)    
    
    # Education queries: deterministic, no LLM
    if _looks_like_education(message):
        return _education_markdown()
    
    # Sanitize message before sending to OpenAI.
    message = safe_finalize(message)    

    # LLM path
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    while True:
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
            )
        except Exception as e:
            return f"OpenAI error: {e}"

        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":
            msg = choice.message
            tool_calls = msg.tool_calls or []
            messages.append(msg)
            messages.extend(handle_tool_calls(tool_calls))
            # loop so the model can use tool outputs
        else:
            draft = choice.message.content
            return safe_finalize(draft)

# ----------------------------
# UI: modern light with gradient frame & enhanced input row
# ----------------------------

# Palette
G1 = "#2563EB"        # royal blue
G2 = "#06B6D4"        # cyan
TEXT_DARK = "#0F172A" # slate-900

# Badge styles depend on availability
if LOOKING_FOR_ROLE:
    BADGE_BG = "linear-gradient(135deg, #10b981, #22d3ee)"  # emerald -> cyan
    BADGE_COLOR = "#ffffff"
    BADGE_SHADOW = "0 8px 16px rgba(16,185,129,.28)"
    BADGE_TEXT = "Open to opportunities"
else:
    BADGE_BG = "#e2e8f0"   # slate-200
    BADGE_COLOR = "#0f172a"
    BADGE_SHADOW = "inset 0 0 0 1px #cbd5e1"
    BADGE_TEXT = "Not actively looking"

# Theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif"],
    font_mono=[gr.themes.GoogleFont("Fira Code"), "ui-monospace"],
)

# Header markup (includes availability badge + notice)
header_html = f"""
<div class="hero">
  <div class="hero-left">
    <div class="hero-avatar"></div>
    <div class="hero-text">
      <div class="hero-title">Chat with Robert's profile AI</div>
      <div class="notice" style="margin-top:20px;">
        <span class="dot"></span>
        <span class="text"><i>I’m an AI version of Robert. I’m still learning and may be wrong sometimes—please verify important info.</i></span>
      </div>
    </div>
  </div>
  <!-- Badge pinned to the top-right with 5px "border" inside the header -->
  <div class="hero-badge">{BADGE_TEXT}</div>
</div>
"""

# CSS via Template so normal braces don't break Python
_css_tpl = Template(r"""
:root {
  --g1: $G1;
  --g2: $G2;
  --text-dark: $TEXT_DARK;
  --badge-bg: $BADGE_BG;
  --badge-color: $BADGE_COLOR;
  --badge-shadow: $BADGE_SHADOW;

  /* Keep the page in light mode (prevents some mobile auto-darking) */
  color-scheme: light;

  /* Shared var for chat area background */
  --chat-bg: #ffffff;
}
html, body { background: #f3f6fb; color: var(--text-dark); }

#chat-shell {
  position: relative;
  max-width: 760px;
  margin: 32px auto;
  padding: 2px;                          /* gradient "frame" thickness */
  border-radius: 18px;
  background: linear-gradient(135deg, var(--g1), var(--g2));
  box-shadow: 0 20px 50px rgba(2, 6, 23, 0.14);
}
#chat-inner {
  background: var(--chat-bg) !important;
  border-radius: 16px;
  overflow: hidden;
}
.hero {
  position: relative;                        /* allow absolute child positioning */
  display: flex; align-items: center;        /* keep the avatar/title row aligned */
  justify-content: flex-start;               /* badge is now absolutely positioned */
  background: linear-gradient(135deg, var(--g1), var(--g2));
  color: #fff;
  padding: 16px 18px;
}
.hero-left { display:flex; align-items:center; gap:12px; }
.hero-avatar {
  width: 44px; height: 44px; border-radius: 50%;
  background: radial-gradient(ellipse at 30% 30%, #ffffff 0%, #dbeafe 35%, transparent 60%),
              linear-gradient(135deg, #60a5fa, #38bdf8);
  box-shadow: 0 6px 14px rgba(0,0,0,.18), inset 0 0 0 2px rgba(255,255,255,.35);
}
.hero-title { font-weight: 700; font-size: 18px; line-height: 1.2; }
.hero-subtitle { opacity: .9; font-size: 13px; }
.hero-badge {
  position: absolute;
  top: 5px;
  right: 5px;
  z-index: 1;

  font-size: 12px; font-weight: 700; letter-spacing: .2px;
  padding: 8px 10px; border-radius: 9999px;
  background: var(--badge-bg); color: var(--badge-color);
  box-shadow: var(--badge-shadow);
  white-space: nowrap;
}

/* Ensure wrapper uses dark text by default */
#ci {
  background: var(--chat-bg) !important;
  border-radius: 0 0 16px 16px;
  padding: 8px 10px 12px;
  color: var(--text-dark);
}

/* Chat area — force light backgrounds even on mobile dark mode */
#ci .gr-chatbot,
#ci .chatbot,
#ci .gr-chatbot > div,
#ci .chatbot > div,
#ci .gr-box,
#ci .gr-panel {
  background: var(--chat-bg) !important;
}

/* ---- Always readable colors for bot messages ---- */
#ci .message.bot {
  background: #eef2f7 !important;     /* soft light bubble */
  color: var(--text-dark) !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 14px !important;
}
#ci .message.bot *,
#ci .message.bot .prose,
#ci .message.bot .prose *,
#ci .message.bot .markdown-body,
#ci .message.bot .markdown-body * {
  color: var(--text-dark) !important;
}
#ci .message.bot a { color: #1d4ed8 !important; }
#ci .message.bot code, #ci .message.bot pre {
  background: #f1f5f9 !important;      /* slate-100 */
  color: #111827 !important;           /* gray-900 */
  border-radius: 6px;
  padding: 0.1rem 0.3rem;
}

/* User bubble remains high contrast */
#ci .message.user {
  background: linear-gradient(135deg, var(--g1), var(--g2)) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 14px !important;
  box-shadow: 0 6px 16px rgba(37,99,235,.25);
}

/* Padding & slight global contrast bump */
#ci .gr-chatbot { padding: 12px; }
#ci .gr-chatbot, #ci .chatbot {
  filter: saturate(1.05) contrast(1.03);
}

/* ===== Input row: clearer box + prominent send arrow ===== */
#ci .gr-textbox {
  background: #f8fafc !important;
  border: 1.6px solid #cbd5e1 !important;
  border-radius: 14px !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.6);
}
#ci .gr-textbox textarea {
  padding: 14px 16px !important;
  min-height: 56px !important;
  line-height: 1.4 !important;
  resize: none !important;
  color: var(--text-dark) !important;
}
#ci .gr-textbox textarea::placeholder {
  color: #64748b !important;
  opacity: .95 !important;
}
#ci .gr-textbox:focus-within {
  border-color: transparent !important;
  box-shadow:
    0 0 0 2px rgba(37,99,235,.35),
    0 0 0 5px rgba(6,182,212,.20),
    inset 0 1px 0 rgba(255,255,255,.6) !important;
}
#ci .gr-form, #ci .gradio-row { gap: 10px; }
#ci button.gr-button.primary {
  width: 46px; min-width: 46px; height: 46px;
  padding: 0 !important;
  border-radius: 9999px !important;
  background: linear-gradient(135deg, var(--g1), var(--g2)) !important;
  border: 0 !important;
  color: #fff !important;
  box-shadow: 0 10px 22px rgba(37,99,235,.28);
  display: inline-flex; align-items: center; justify-content: center;
}
#ci button.gr-button.primary svg {
  width: 22px; height: 22px;
  color: #ffffff !important;
  stroke: #ffffff !important; fill: #ffffff !important;
  filter: drop-shadow(0 1px 1px rgba(0,0,0,.18));
}
#ci button.gr-button.primary:hover { filter: brightness(1.06); transform: translateY(-1px); }
#ci button.gr-button.primary:active { transform: translateY(0); filter: brightness(0.98); }
#ci button.gr-button { border-radius: 12px !important; }

/* ---- Defensive overrides for devices that auto-apply dark mode or invert colors ---- */
@media (prefers-color-scheme: dark) {
  html, body { background: #f3f6fb !important; color: var(--text-dark) !important; }
  #chat-inner, #ci, #ci .gr-chatbot, #ci .chatbot, #ci .gr-box, #ci .gr-panel, .gradio-container {
    background: var(--chat-bg) !important;
  }
  #ci .message.bot { background: #e5e7eb !important; }
  #ci .message.bot *, #ci .gr-markdown * { color: var(--text-dark) !important; }
}

/* Extra safety on narrow/mobile viewports */
@media screen and (max-width: 480px) {
  #chat-inner, #ci, #ci .gr-chatbot, #ci .chatbot { background: var(--chat-bg) !important; }
}

/* Keep layout tidy on wide screens */
.gradio-container { padding: 12px; }
""")
css = _css_tpl.substitute(
    G1=G1, G2=G2, TEXT_DARK=TEXT_DARK,
    BADGE_BG=BADGE_BG, BADGE_COLOR=BADGE_COLOR, BADGE_SHADOW=BADGE_SHADOW
)

# Chatbot area
chatbot = gr.Chatbot(
    type="messages",
    height=580,
    show_copy_button=True,
)

# Build the UI (Spaces will look for `demo`)
with gr.Blocks(theme=theme, css=css) as demo:
    with gr.Group(elem_id="chat-shell"):
        with gr.Column(elem_id="chat-inner"):
            gr.HTML(header_html, elem_id="hero")
            with gr.Group(elem_id="ci"):
                gr.ChatInterface(
                    chat,
                    chatbot=chatbot,
                    title=None,
                    description=None,
                )

if __name__ == "__main__":
    demo.launch()
