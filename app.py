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
            return choice.message.content

# Expose top-level Gradio app for Spaces
demo = gr.ChatInterface(chat, type="messages")

if __name__ == "__main__":
    demo.launch()
