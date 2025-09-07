# AI Text → Slides (Streamlit, no Hugging Face)

Turn a paragraph into a slide deck outline and download as PPTX. Uses OpenAI API.

## Deploy on Streamlit Cloud
1) Fork this repo to your GitHub account.
2) Go to https://share.streamlit.io → "New app" → select this repo and branch → Deploy.
3) In your deployed app, open "Settings → Secrets" and add:
   OPENAI_API_KEY = sk-...
4) Click "Rerun" if needed, then generate decks.

## Local run (optional)
pip install -r requirements.txt
streamlit run app.py
