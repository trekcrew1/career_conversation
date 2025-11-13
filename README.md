---
title: career_conversation
app_file: app.py
sdk: gradio
sdk_version: 5.47.0
---

# Career Conversation Bot

An AI-powered chatbot that represents your professional profile, built with OpenAI and Gradio. Deploy it as a HuggingFace Space to let recruiters and contacts learn about your background, skills, and career interests through natural conversation.

## What This Does

This repository creates a personalized AI assistant that:
- **Answers questions** about your professional background, skills, and experience
- **Intelligently responds to recruiters** based on whether you're actively looking for opportunities
- **Notifies you** via Pushover when someone shows interest or asks questions
- **Provides curated responses** about your availability and career goals
- **Runs as a web app** on HuggingFace Spaces with a polished, modern UI

The bot uses GPT-4o-mini to have natural conversations while staying grounded in your professional summary and LinkedIn profile.

## Features

- ✅ **Automated recruiter responses** - Generates polite, context-aware replies (interested or declining)
- ✅ **Dynamic availability status** - Displays whether you're open to opportunities
- ✅ **Notification system** - Pushover integration to alert you about conversations
- ✅ **LinkedIn profile integration** - Can extract info from LinkedIn PDF exports
- ✅ **Professional moderation** - Filters inappropriate content
- ✅ **Function calling** - Records user contact details and unanswered questions
- ✅ **Beautiful UI** - Gradient-styled chat interface with availability badges

## See It In Action

Want to see what this looks like? Check out my live career bot:

- **🤖 Chat with my AI:** [https://huggingface.co/spaces/4robmorrow/career-conversation](https://huggingface.co/spaces/4robmorrow/career-conversation)
- **👔 My LinkedIn Profile:** [https://www.linkedin.com/in/robert-morrow-5408a08/](https://www.linkedin.com/in/robert-morrow-5408a08/)

Try asking about my experience, skills, or current projects - the AI will answer based on my professional summary and background!

## Quick Start: Deploy Your Own Bot

### Prerequisites

1. **Python 3.8+** installed on your machine
2. **HuggingFace account** (free) - [Sign up here](https://huggingface.co/join)
3. **OpenAI API key** - [Get one here](https://platform.openai.com/api-keys)
4. **(Optional)** Pushover account for notifications - [pushover.net](https://pushover.net/)

### Step 1: Set Up HuggingFace

1. **Create a HuggingFace account** at [huggingface.co](https://huggingface.co/join)
2. **Generate an access token:**
   - Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Give it a name (e.g., "career-bot-deploy")
   - Select **"Write"** access
   - Click "Generate token"
   - **Copy and save this token** - you'll need it shortly

### Step 2: Get Your OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to [API Keys](https://platform.openai.com/api-keys)
4. Click "Create new secret key"
5. **Copy and save the key** - you won't be able to see it again

### Step 3: Prepare Your Information

Before running the builder, gather:
- **Your full name** (how you want to be addressed)
- **Your LinkedIn profile URL** (e.g., `https://www.linkedin.com/in/yourname/`)
- **A professional summary** (2-4 paragraphs about your background, skills, and interests)
- **Current job search status** (are you looking for opportunities?)

### Step 4: Run the Space Builder

1. **Clone this repository:**
   ```bash
   git clone https://github.com/trekcrew1/career_conversation.git
   cd career_conversation
   ```

2. **Install dependencies:**
   ```bash
   pip install huggingface_hub
   ```

3. **Run the builder:**
   ```bash
   python space_builder.py
   ```

4. **Follow the prompts:**
   - Enter your **HuggingFace token**
   - Enter your **OpenAI API key**
   - Paste your **professional summary**
   - Enter your **LinkedIn URL**
   - Enter your **display name**
   - Choose a **Space name** (e.g., "career-conversation")
   - Select if you're **open to opportunities** (Yes/No)
   - Choose **public or private** visibility

5. **Done!** The script will:
   - Create a new HuggingFace Space
   - Upload all necessary files
   - Configure your OpenAI API key as a secret
   - Provide you with the live URL

Your bot will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

## Local Development

To run the bot locally for testing:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/trekcrew1/career_conversation.git
   cd career_conversation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file:**
   ```env
   OPENAI_API_KEY=your_openai_key_here
   LOOKING_FOR_ROLE=false
   PUSHOVER_USER=your_pushover_user (optional)
   PUSHOVER_TOKEN=your_pushover_token (optional)
   ```

4. **Add your personal info** to `personal_info/`:
   - `summary.txt` - Your professional summary
   - `name.txt` - Your display name
   - `linkedin_url.txt` - Your LinkedIn URL
   - `looking.json` - `{"looking": true}` or `{"looking": false}`
   - *(Optional)* `linkedin_profile.pdf` - LinkedIn PDF export

5. **Run the app:**
   ```bash
   python app.py
   ```

6. **Open your browser** to the local URL shown in the terminal (typically `http://127.0.0.1:7860`)

## Customization

### Changing Your Availability Status

**On HuggingFace Spaces:**
1. Go to your Space's settings
2. Navigate to "Variables and secrets"
3. Add/update environment variable: `LOOKING_FOR_ROLE=true` (or `false`)
4. Restart the Space

**Locally:**
Update the `.env` file with `LOOKING_FOR_ROLE=true` or `false`

### Adding Pushover Notifications

To get notified when people interact with your bot:

1. Sign up at [pushover.net](https://pushover.net/)
2. Create an application/API token
3. Add to HuggingFace Space secrets:
   - `PUSHOVER_USER` - Your user key
   - `PUSHOVER_TOKEN` - Your application token

### Updating Your Summary

Edit `personal_info/summary.txt` with your latest information and re-upload to HuggingFace.

## File Structure

```
career_conversation/
├── app.py                          # Main Gradio application
├── space_builder.py                # Automated Space deployment script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── personal_info/
│   ├── summary.txt                # Your professional summary
│   ├── name.txt                   # Your display name
│   ├── linkedin_url.txt           # Your LinkedIn URL
│   ├── looking.json               # Job search status
│   └── linkedin_profile.pdf       # (Optional) LinkedIn PDF export
└── _*.py                          # Utility scripts for Space management
```

## How It Works

1. **User sends a message** through the Gradio chat interface
2. **Content is moderated** for professionalism
3. **OpenAI GPT-4o-mini processes** the message with context from your summary
4. **Function calling** can record contact details or unanswered questions
5. **Pushover sends notifications** (if configured) about interactions
6. **Recruiter-specific logic** generates appropriate responses based on availability

## Privacy & Security

- Your **OpenAI API key** is stored as a HuggingFace secret (not in code)
- The bot only shares information you provide in your summary
- Make your Space **private** if you want to control access
- All conversations are processed through OpenAI's API

## Cost Considerations

- **HuggingFace Spaces**: Free tier available
- **OpenAI API**: Pay-per-use (~$0.0005 per conversation with GPT-4o-mini)
- **Pushover**: One-time $5 payment (optional)

Expected cost: **~$1-5/month** depending on traffic

## Troubleshooting

**Space won't start:**
- Check that `OPENAI_API_KEY` is set in Space secrets
- Verify the secret value doesn't have extra whitespace

**Bot gives generic responses:**
- Make sure your `summary.txt` has substantial content
- Check that the OpenAI API key is valid and has credits

**Not receiving notifications:**
- Verify `PUSHOVER_USER` and `PUSHOVER_TOKEN` are correctly set
- Check Pushover app quota limits

## Credits

Created by Robert Morrow. Feel free to fork and customize for your own career bot!

## License

See [LICENSE](LICENSE) file for details.
