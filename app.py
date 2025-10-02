import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import requests
from pypdf import PdfReader
import gradio as gr

# Load .env for local dev; on Spaces, env comes from "Variables & secrets"
load_dotenv(override=True)

# --- Config / env ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
PUSHOVER_USER  = os.getenv("PUSHOVER_USER")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_URL   = os.getenv("PUSHOVER_URL") or "https://api.pushover.net/1/messages.json"

OPENAI_READY = bool(OPENAI_API_KEY)  # don't instantiate client yet
print(f"OpenAI key present: {OPENAI_READY}")
print(f"Pushover configured: {bool(PUSHOVER_USER and PUSHOVER_TOKEN)}")

# --- Pushover helper (no-op if not configured) ---
def push(message: str):
    if not (PUSHOVER_USER and PUSHOVER_TOKEN and PUSHOVER_URL):
        return
    try:
        requests.post(PUSHOVER_URL, data={
            "user": PUSHOVER_USER, "token": PUSHOVER_TOKEN, "message": message
        }, timeout=10)
    except Exception as e:
        print(f"Pushover error: {e}")

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}

# --- Tool schemas for function calling ---
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Context worth recording from the conversation"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}
record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Use this tool to record a question that was asked but not answered",
    "parameters": {
        "type": "object",
        "properties": {"question": {"type": "string", "description": "The question that was asked"}},
        "required": ["question"],
        "additionalProperties": False
    }
}
tools = [{"type": "function", "function": record_user_details_json},
         {"type": "function", "function": record_unknown_question_json}]

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

# --- Load optional local files safely ---
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

# --- Prompt ---
name = "Robert Morrow"
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
    "\n\nIMPORTANT: When answering questions about employment, current job, or tenure, treat multiple roles "
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
    f"With this context, please chat with the user, always staying in character as {name}."
)

# --- Chat handler ---
def chat(message, history):
    if not OPENAI_READY:
        return "Server is not configured with OPENAI_API_KEY. Add it in Settings → Variables & secrets and restart this Space."

    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    while True:
        try:
            # instantiate client at call time; SDK reads OPENAI_API_KEY from env
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools
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
# demo = gr.ChatInterface(
#     chat,
#     type="messages",
#     theme="freddyaboulton/dracula_revamped",  # high-contrast dark
#     title="Career Conversation",
#     description="Ask Robert about his background, projects, and experience."
# )

# theme = gr.themes.Soft(
#     primary_hue="indigo",      # accent color
#     secondary_hue="violet",
#     neutral_hue="slate",
#     font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif"]
# ).set(
#     body_background_fill="#0b1220",    # deep background
#     block_background_fill="#0f172a",
#     body_text_color="#e5e7eb",
#     link_text_color="#93c5fd",
#     # radius_size="12px",
#     shadow_spread="2px",
# )
# demo = gr.ChatInterface(chat, type="messages", theme=theme,
#                         title="Career Conversation",
#                         description="Ask Robert about his background, projects, and experience.")

# chatbot = gr.Chatbot(
#     type="messages",
#     height=640,
#     bubble_full_width=False,         # narrower bubbles feel more “app-like”
#     show_copy_button=True,
#     avatar_images=(None, None)       # or ("assets/user.png", "assets/bot.png")
# )
# demo = gr.ChatInterface(chat, chatbot=chatbot,
#                         title="Career Conversation",
#                         description="Ask Robert about his background, projects, and experience.",
#                         theme="freddyaboulton/dracula_revamped")


# 1) Higher-contrast light theme with modern fonts & accents
theme = gr.themes.Soft(
    primary_hue="indigo",      # buttons/links/accent
    secondary_hue="emerald",   # secondary accent
    neutral_hue="slate",       # text & borders
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif"],
    font_mono=[gr.themes.GoogleFont("Fira Code"), "ui-monospace"]
)

# 2) A few tasteful CSS tweaks for pop and readability (still light)
css = """
/* Max width, centered layout */
.gradio-container { max-width: 980px; margin: 0 auto; }

/* Subtle light background and card elevation */
body { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 60%); }
.gr-box, .gr-panel, .gr-card, .gr-form, .gr-column, .gr-group {
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;         /* slate-200 */
  box-shadow: 0 10px 30px rgba(2, 6, 23, 0.06);  /* soft elevation */
  border-radius: 12px !important;
}

/* Buttons: bold primary + subtle hover lift */
button.gr-button {
  font-weight: 600;
  border-radius: 10px !important;
}
button.gr-button.primary {
  background: #4f46e5 !important;   /* indigo-600 */
  border-color: #4f46e5 !important;
  color: white !important;
}
button.gr-button.primary:hover { filter: brightness(1.06); transform: translateY(-1px); }

/* Inputs: slightly larger & rounded */
textarea, input, .gr-textbox, .gr-text-input { 
  border-radius: 10px !important;
}

/* Chat area: crisp bubbles with gentle color cues */
.chatbot, .gr-chatbot { background: #ffffff !important; }
.chatbot .message.user, .gr-chatbot .message.user {
  background: #eef2ff !important;    /* indigo-50 */
  border: 1px solid #c7d2fe !important; /* indigo-200 */
}
.chatbot .message.bot, .gr-chatbot .message.bot {
  background: #ecfeff !important;    /* cyan-50 */
  border: 1px solid #a5f3fc !important; /* cyan-200 */
}

/* Links: visible but not shouting */
a { color: #2563eb; }                 /* blue-600 */
a:hover { color: #1d4ed8; }
"""

# 3) Chatbot config (narrower bubbles feel more "app-like")
chatbot = gr.Chatbot(
    type="messages",
    height=640,
    bubble_full_width=False,
    show_copy_button=True
)

# 4) Final app object for Spaces (replace your existing demo=... line with this)
demo = gr.ChatInterface(
    chat,
    chatbot=chatbot,
    theme=theme,
    css=css,
    title="Career Conversation",
    description="Chat with Robert about his background, projects, and experience."
)



if __name__ == "__main__":
    demo.launch()
