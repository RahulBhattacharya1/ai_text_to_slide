import json, io, textwrap
import streamlit as st
from dataclasses import dataclass
from typing import List, Optional
from openai import OpenAI
from utils_pptx import deck_to_pptx_bytes

# Read API key from Streamlit secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    WARN = "Missing OPENAI_API_KEY secret. Add it in Settings → Secrets after deploy."
else:
    WARN = None

# Initialize client
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="AI Text → Slides", page_icon="📑", layout="wide")

# ------------------------------- Data Model -------------------------------
@dataclass
class Slide:
    title: str
    bullets: List[str]
    notes: Optional[str] = None

@dataclass
class Deck:
    topic: str
    brand_color: str
    slides: List[Slide]

SYSTEM_INSTRUCTIONS = """You are a slide-writing assistant.
Given a topic or paragraph, produce a concise slide deck outline in JSON.

Requirements:
- 1 title slide + 4 to 8 content slides.
- Each content slide: short title (<= 7 words) + 3-5 bullets, each bullet <= 14 words.
- Include optional speaker notes (1-2 sentences) per slide.
- Clear, professional language. No fluff.

Output JSON schema exactly:
{
  "slides": [
    {
      "title": "string",
      "bullets": ["string", "string", ...],
      "notes": "string (optional)"
    }
  ]
}
Return ONLY valid JSON with no extra text.
"""

def generate_json(topic: str, model: str = "gpt-4o-mini", max_tokens: int = 800, temperature: float = 0.4) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": f"Topic:\n{topic}\n\nJSON:"}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        n=1
    )
    return resp.choices[0].message.content

def coerce_json(s: str) -> dict:
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    return json.loads(s)

def json_to_deck(topic: str, brand_color: str, data: dict) -> Deck:
    slides = []
    for item in data.get("slides", []):
        title = str(item.get("title", "")).strip() or "Slide"
        bullets = [str(b).strip() for b in item.get("bullets", []) if str(b).strip()]
        notes = item.get("notes")
        slides.append(Slide(title=title, bullets=bullets, notes=notes))
    if slides and slides[0].bullets:
        slides.insert(0, Slide(title=topic.strip()[:60], bullets=[]))
    elif not slides:
        slides = [Slide(title=topic.strip()[:60], bullets=[])]
    return Deck(topic=topic, brand_color=brand_color, slides=slides)

# ------------------------------- UI -------------------------------
st.title("AI Text → Slide Deck")

with st.sidebar:
    st.subheader("Options")
    brand = st.color_picker("Brand color", value="#0F62FE")
    temp = st.slider("Creativity", 0.0, 1.0, 0.4, 0.05)
    max_tokens = st.slider("Max tokens", 128, 2048, 800, 64)
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    if WARN:
        st.warning(WARN)

topic = st.text_area("Enter topic or paragraph", height=140, placeholder="Example: Generative AI for small retail: use cases, risks, pilot plan, KPIs.")

col1, col2 = st.columns([1,1])
with col1:
    gen = st.button("Generate Slides", type="primary", disabled=not topic.strip())

deck_state = st.session_state.get("deck")

if gen:
    if not OPENAI_API_KEY:
        st.error("Please add OPENAI_API_KEY in Streamlit Secrets.")
        deck = None
    else:
        try:
            raw = generate_json(topic.strip(), model=model, max_tokens=max_tokens, temperature=temp)
            data = coerce_json(raw)
            deck = json_to_deck(topic.strip(), brand, data)
            st.session_state["deck"] = deck
        except Exception as e:
            st.error(f"Generation failed: {e}")
            deck = None
else:
    deck = deck_state

# ------------------------------- Preview + Export -------------------------------
if deck:
    st.subheader("Live Preview")
    for idx, sld in enumerate(deck.slides):
        with st.container(border=True):
            st.markdown(f"### {idx+1}. {sld.title}")
            if sld.bullets:
                for b in sld.bullets:
                    st.markdown(f"- {b}")
            if sld.notes:
                with st.expander("Speaker notes"):
                    st.write(textwrap.fill(sld.notes, width=100))

    pptx_bytes = deck_to_pptx_bytes(deck)
    st.download_button(
        label="Download PPTX",
        data=pptx_bytes,
        file_name="ai_slides.pptx",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        type="primary"
    )
