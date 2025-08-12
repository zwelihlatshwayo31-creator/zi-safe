SDS Safety Summarizer – SerpAPI-only (Fixed) – Windows Quick Start

1) Double-click: setup_and_run.bat
   - The first time, it will create a virtual environment and install dependencies.
   - Edit the created .env file to put your SerpAPI key:
       SERPAPI_KEY=YOUR_REAL_KEY

2) After install, your browser should open at http://localhost:8501

3) Usage
   - Paste a chemistry procedure into the text box.
   - Review/edit the detected chemical list.
   - Click "Fetch SDS & Summarize" to generate the safety brief.
   - Download the Markdown summary if desired.

Notes
- This build fixes the guess_supplier_from_url runtime issue and supports Python 3.8+.
- Always verify SDS-derived guidance with original documents and your local EHS policies.
