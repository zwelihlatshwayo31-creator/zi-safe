# Zi Safe — SDS Safety Summarizer (Streamlit + SerpAPI + PubChem)

Zi Safe reads a chemistry procedure, fetches Safety Data Sheets (SDS) for the chemicals involved,
and generates a consolidated safety brief (hazards, PPE, handling/storage, first aid, etc.).
It uses **SerpAPI** to find vendor SDS PDFs and **PubChem** to normalize names & cross-check GHS codes.

## Features
- Detect chemicals from free-text procedures (editable list)
- Fetch SDS PDFs from supplier domains via **SerpAPI**
- Parse key SDS sections (Hazards, PPE, First Aid, Handling/Storage, Exposure Controls, etc.)
- PubChem normalization (canonical name, CID, synonyms) + GHS H-code cross-check
- Procedure-aware cautions (e.g., heating, acids/bases, oxidizers, cryogens)
- One-click Markdown export of the safety summary

## Folder layout (suggested)

```
zi-safe/
├─ app.py
├─ setup_and_run.bat
├─ requirements.txt
├─ .gitignore
├─ .env.example
└─ README.md
```

## Quick start (Windows)

1. Install Python 3.10+ from https://www.python.org (check “Add Python to PATH”).
2. Clone or unzip this repo, then open **Command Prompt** in the folder.
3. Create a venv and install deps:
   ```bat
   python -m venv venv
   call venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Create `.env` with your SerpAPI key:
   ```
   SERPAPI_KEY=your_serpapi_key_here
   ```
5. Run the app:
   ```bat
   streamlit run app.py
   ```
   Your browser should open at http://localhost:8501

> You can also double-click `setup_and_run.bat` if your app package includes it.

## Environment variables

- `SERPAPI_KEY` — **required** to search for SDS PDFs via SerpAPI

Optional (if your code supports them):
- `BING_SEARCH_KEY` — if you later enable Bing Web Search
- `SERPAPI_ENGINE` — override engine (defaults to Google via `engine=google`)

Create a template file for teammates:
```
# .env.example
SERPAPI_KEY=your_key_here
```

## .gitignore (recommended)

```
# Python
venv/
__pycache__/
*.pyc

# Streamlit cache/artifacts
sds_cache/
.streamlit/

# OS/editor
.DS_Store
Thumbs.db

# Secrets
.env
*.key
*.pem
```

## GitHub: create & push

```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: Zi Safe app"
git remote add origin https://github.com/<your-username>/zi-safe.git
git push -u origin main
```

If the remote already has files, pull first:
```bash
git pull --rebase origin main
git push -u origin main
```

## Deploy options

### Streamlit Community Cloud (fastest)
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and connect your repo.
3. In **Advanced settings**, add the secret:
   - `SERPAPI_KEY = <your key>`
4. Deploy. You’ll get an HTTPS URL you can share or wrap in a mobile app.

### Other hosts
- **Render** / **Railway** / **Fly.io** / **Azure App Service**: build a Python app, install `requirements.txt`,
  and set `SERPAPI_KEY` in environment variables.

## Mobile wrapper (optional)
If you want presence on the Play Store / App Store, wrap your deployed URL with **Capacitor** (WebView shell).
I can provide a ready-made wrapper — just share your production Streamlit URL.

## Security & safety
- Never commit `.env` (contains your API key).
- Always verify summaries against original SDS and your local EHS policies.
- This tool **does not replace** institutional safety review.
