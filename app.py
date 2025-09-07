import time
import datetime as dt
import math
import streamlit as st

# Rate-limit knobs
COOLDOWN_SECONDS = 30        # one call every 30 seconds per session
DAILY_LIMIT      = 40        # max generations per session per day
HOURLY_SHARED_CAP = 250      # optional global cap across all users; set 0 to disable

def init_rate_limit_state():
    ss = st.session_state
    today = dt.date.today().isoformat()
    if "rl_date" not in ss or ss["rl_date"] != today:
        ss["rl_date"] = today
        ss["rl_calls_today"] = 0
        ss["rl_last_ts"] = 0.0
    if "rl_last_ts" not in ss:
        ss["rl_last_ts"] = 0.0
    if "rl_calls_today" not in ss:
        ss["rl_calls_today"] = 0

def can_call_now():
    """Returns (allowed: bool, reason: str, seconds_left: int)"""
    init_rate_limit_state()
    ss = st.session_state
    now = time.time()

    # Cooldown check
    remaining = int(max(0, ss["rl_last_ts"] + COOLDOWN_SECONDS - now))
    if remaining > 0:
        return (False, f"Please wait {remaining}s before the next generation.", remaining)

    # Daily session cap
    if ss["rl_calls_today"] >= DAILY_LIMIT:
        return (False, f"Daily limit reached ({DAILY_LIMIT} generations). Try again tomorrow.", 0)

    # Shared hourly cap (optional)
    if HOURLY_SHARED_CAP > 0:
        bucket = _hour_bucket()
        counters = _shared_hourly_counters()
        used = counters.get(bucket, 0)
        if used >= HOURLY_SHARED_CAP:
            return (False, "Hourly capacity reached for this app. Please try in a little while.", 0)

    return (True, "", 0)

def record_successful_call():
    ss = st.session_state
    ss["rl_last_ts"] = time.time()
    ss["rl_calls_today"] += 1

    if HOURLY_SHARED_CAP > 0:
        bucket = _hour_bucket()
        counters = _shared_hourly_counters()
        counters[bucket] = counters.get(bucket, 0) + 1

@st.cache_resource
def _shared_hourly_counters():
    # In-memory dict shared by all sessions in this Streamlit process
    # key: "YYYY-MM-DD-HH", value: int count
    return {}

def _hour_bucket(now=None):
    now = now or dt.datetime.utcnow()
    return now.strftime("%Y-%m-%d-%H")


import json
import re
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Tuple

import streamlit as st
from pydantic import BaseModel, Field, ValidationError, conlist, constr

# Optional: only needed if provider == "openai"
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from utils_pptx import deck_to_pptx_bytes

# ======================= App Config =======================
st.set_page_config(page_title="AI Text → Slides", page_icon="📑", layout="wide")

# ======================= Data Models =======================
# Hard constraints for portfolio rigor
MAX_TITLE_WORDS = 7
MIN_CONTENT_SLIDES = 4
MAX_CONTENT_SLIDES = 8
MIN_BULLETS = 3
MAX_BULLETS = 5
MAX_BULLET_WORDS = 14

from typing import List, Optional, Annotated
from pydantic import BaseModel, Field, constr, field_validator

MAX_TITLE_WORDS = 7
MIN_CONTENT_SLIDES = 4
MAX_CONTENT_SLIDES = 8
MIN_BULLETS = 3
MAX_BULLETS = 5
MAX_BULLET_WORDS = 14

class SlideModel(BaseModel):
    title: constr(strip_whitespace=True, min_length=1)
    bullets: List[constr(strip_whitespace=True, min_length=1)] = Field(default_factory=list)
    notes: Optional[str] = None

    @field_validator("bullets")
    @classmethod
    def clean_bullets(cls, v):
        # normalize and drop empties
        return [b.strip() for b in v if str(b).strip()]

SlidesType = Annotated[List[SlideModel], Field(min_length=1, max_length=1 + MAX_CONTENT_SLIDES)]

class DeckJSON(BaseModel):
    slides: SlidesType

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

# ======================= Prompting (LLM Guardrails) =======================
SYSTEM_INSTRUCTIONS = f"""
You are a slide-writing assistant. Return ONLY JSON conforming to this schema:
{{
  "slides": [
    {{
      "title": "string (<= {MAX_TITLE_WORDS} words)",
      "bullets": ["3 to 5 bullets, each <= {MAX_BULLET_WORDS} words"],
      "notes": "optional string with 1-2 sentences"
    }}
  ]
}}

Requirements:
- Create 1 title-only slide first (no bullets).
- Then {MIN_CONTENT_SLIDES} to {MAX_CONTENT_SLIDES} content slides.
- Bullet style: concise, no trailing punctuation, no numbering.
- Professional tone, clear language, no fluff.
- No extra keys in JSON. No markdown, no prose, no code fences.
- Output must be valid JSON and nothing else.
"""

USER_TEMPLATE = """Topic:
{topic}

Constraints:
- Title words ≤ {max_title_words}
- Content slides: {min_slides}–{max_slides}
- Bullets per slide: {min_bullets}–{max_bullets}, ≤ {max_bullet_words} words each

JSON only:
"""

# ======================= Provider Abstraction =======================
def call_openai(topic: str, model: str, temperature: float, max_tokens: int) -> str:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in Streamlit Secrets.")

    if OpenAI is None:
        raise RuntimeError("openai package not available. Add openai to requirements.txt.")

    client = OpenAI(api_key=api_key)
    prompt_user = USER_TEMPLATE.format(
        topic=topic,
        max_title_words=MAX_TITLE_WORDS,
        min_slides=MIN_CONTENT_SLIDES, max_slides=MAX_CONTENT_SLIDES,
        min_bullets=MIN_BULLETS, max_bullets=MAX_BULLETS,
        max_bullet_words=MAX_BULLET_WORDS
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS.strip()},
            {"role": "user", "content": prompt_user.strip()}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        n=1
    )
    return resp.choices[0].message.content

# ======================= Offline Fallback (Rule-based) =======================
def naive_sentence_split(text: str) -> List[str]:
    segs = re.split(r"[.!?]\s+", text.strip())
    return [s.strip() for s in segs if s.strip()]

def extract_keywords(text: str, k: int = 8) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text.lower())
    stop = set("""
        the a an and or of for with in on to from by as at is are be was were has have had this that these those it its their
        using into about over under between within across via per not no more less
    """.split())
    freq = {}
    for w in words:
        if w in stop or len(w) < 3:
            continue
        freq[w] = freq.get(w, 0) + 1
    sorted_terms = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in sorted_terms[:k]]

def chunk_list(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def generate_offline_deck_json(topic: str) -> dict:
    # Heuristic outline generator when no LLM is used.
    sents = naive_sentence_split(topic)
    if not sents:
        sents = [topic]
    terms = extract_keywords(topic, k=12)
    groups = chunk_list(terms, MAX_BULLETS)

    slides = []
    # Title slide
    title = " ".join(terms[:MAX_TITLE_WORDS]).title() if terms else topic.strip()[:60]
    slides.append({"title": title, "bullets": []})

    # Content slides
    for i, grp in enumerate(groups[:MAX_CONTENT_SLIDES]):
        if i >= MIN_CONTENT_SLIDES and len(slides) - 1 >= MIN_CONTENT_SLIDES:
            break
        slide_title = " ".join(grp[:MAX_TITLE_WORDS]).title() if grp else f"Topic {i+1}"
        bullets = []
        for g in grp[:MAX_BULLETS]:
            bullets.append(f"{g.title()} insights and considerations")
        # Ensure min bullets
        while len(bullets) < MIN_BULLETS:
            bullets.append("Key point to elaborate")
        slides.append({
            "title": slide_title,
            "bullets": bullets[:MAX_BULLETS],
            "notes": "Summarize rationale and actionable next steps."
        })

    # Ensure minimum slide count
    while len(slides) - 1 < MIN_CONTENT_SLIDES:
        slides.append({
            "title": "Additional Considerations",
            "bullets": ["Context", "Constraints", "Next steps"],
            "notes": "Fill with specifics relevant to the topic."
        })

    return {"slides": slides[:1 + MAX_CONTENT_SLIDES]}

# ======================= JSON Handling / Validation =======================
def coerce_json_block(s: str) -> str:
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return s

def validate_json_strict(json_str: str) -> DeckJSON:
    data = json.loads(json_str)
    return DeckJSON.model_validate(data)

def enforce_constraints(deck: DeckJSON) -> DeckJSON:
    fixed = []
    seen_titles = set()
    for idx, s in enumerate(deck.slides):
        # Title length constraint by words
        title_words = s.title.split()
        if len(title_words) > MAX_TITLE_WORDS:
            s.title = " ".join(title_words[:MAX_TITLE_WORDS])

        # Deduplicate titles softly
        key = s.title.lower()
        if key in seen_titles:
            s.title = f"{s.title} ({idx})"
        seen_titles.add(s.title.lower())

        # Bullets constraints
        clean_bullets = []
        for b in s.bullets:
            b = b.strip()
            b = re.sub(r"[.;:,\-–—]\s*$", "", b)  # trim trailing punctuation
            words = b.split()
            if len(words) > MAX_BULLET_WORDS:
                b = " ".join(words[:MAX_BULLET_WORDS])
            clean_bullets.append(b)
        # Enforce bullet count
        if len(clean_bullets) < MIN_BULLETS and idx != 0:  # not title slide
            clean_bullets = clean_bullets + ["Add detail"] * (MIN_BULLETS - len(clean_bullets))
        if idx == 0:
            clean_bullets = []  # Title slide must be bulletless
        s.bullets = clean_bullets[:MAX_BULLETS]

        fixed.append(s)

    # Enforce min/max slide counts (beyond title)
    title = fixed[0]
    content = fixed[1:]
    if len(content) < MIN_CONTENT_SLIDES:
        need = MIN_CONTENT_SLIDES - len(content)
        for i in range(need):
            content.append(SlideModel(title=f"Additional Slide {i+1}", bullets=["Point 1", "Point 2", "Point 3"]))
    content = content[:MAX_CONTENT_SLIDES]
    return DeckJSON(slides=[title, *content])

def deckjson_to_deck(topic: str, brand_color: str, dj: DeckJSON) -> Deck:
    return Deck(
        topic=topic,
        brand_color=brand_color,
        slides=[Slide(title=s.title, bullets=list(s.bullets or []), notes=s.notes) for s in dj.slides]
    )

# ======================= Evaluation / Scoring =======================
def score_deck(dj: DeckJSON) -> Tuple[float, dict]:
    issues = []
    scores = []

    # Title slide
    if len(dj.slides[0].bullets) != 0:
        issues.append("Title slide contains bullets.")
    if len(dj.slides[0].title.split()) > MAX_TITLE_WORDS:
        issues.append("Title too long.")

    # Content slides
    content = dj.slides[1:]
    if not (MIN_CONTENT_SLIDES <= len(content) <= MAX_CONTENT_SLIDES):
        issues.append("Invalid content slide count.")

    title_set = set()
    for i, s in enumerate(content):
        tw = len(s.title.split())
        if tw == 0 or tw > MAX_TITLE_WORDS:
            issues.append(f"Slide {i+2} title length issue.")
        if s.title.lower() in title_set:
            issues.append(f"Duplicate title on slide {i+2}.")
        title_set.add(s.title.lower())

        if not (MIN_BULLETS <= len(s.bullets) <= MAX_BULLETS):
            issues.append(f"Slide {i+2} bullet count issue.")
        for b in s.bullets:
            if len(b.split()) > MAX_BULLET_WORDS:
                issues.append(f"Slide {i+2} bullet too long.")

    # Heuristic density score
    total_bullets = sum(len(s.bullets) for s in content)
    avg_bullet_len = 0.0
    if total_bullets > 0:
        avg_bullet_len = sum(len(" ".join(s.bullets).split()) for s in content) / total_bullets
    density_score = max(0.0, 1.0 - abs(avg_bullet_len - 9) / 9)  # prefer around 9 words per bullet
    scores.append(density_score)

    # Penalty for issues
    base = 0.9
    penalty = min(0.7, 0.05 * len(issues))
    final = max(0.0, base * density_score - penalty)
    details = {
        "issues": issues,
        "avg_bullet_words": round(avg_bullet_len, 2),
        "total_slides": len(dj.slides),
        "total_bullets": total_bullets,
        "density_score": round(density_score, 2)
    }
    return round(final, 2), details

# ======================= UI =======================
# ======= UI (with rate limiting) =======
st.title("AI Text → Slide Deck")

with st.sidebar:
    st.subheader("Generator")
    provider = st.selectbox("Provider", ["OpenAI", "Offline (rule-based)"])
    model = st.selectbox("Model (OpenAI)", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    brand = st.color_picker("Brand color", value="#0F62FE")
    temp = st.slider("Creativity (OpenAI)", 0.0, 1.0, 0.4, 0.05)
    max_tokens = st.slider("Max tokens (OpenAI)", 256, 2048, 900, 32)
    st.caption("Tip: Use Offline to demo your engineering logic without any API.")

    # Optional: show per-session usage and shared hourly capacity
    init_rate_limit_state()
    ss = st.session_state
    st.markdown("**Usage limits**")
    st.write(f"Today: {ss['rl_calls_today']} / {DAILY_LIMIT} generations")
    if HOURLY_SHARED_CAP > 0:
        counters = _shared_hourly_counters()
        used = counters.get(_hour_bucket(), 0)
        st.write(f"Hour capacity: {used} / {HOURLY_SHARED_CAP}")
    remaining = int(max(0, ss["rl_last_ts"] + COOLDOWN_SECONDS - time.time()))
    if remaining > 0:
        st.progress(min(1.0, (COOLDOWN_SECONDS - remaining) / COOLDOWN_SECONDS))
        st.caption(f"Cooldown: {remaining}s")

topic = st.text_area(
    "Enter topic or paragraph",
    height=160,
    placeholder="Generative AI for small retail: opportunities, risks, pilot plan, KPIs."
)

# Rate-limit check before enabling the button
allowed, reason, _wait = can_call_now()

col1, col2 = st.columns([1,1])
with col1:
    gen = st.button(
        "Generate Slides",
        type="primary",
        disabled=(not topic.strip()) or (not allowed)
    )
with col2:
    if not allowed:
        st.caption(reason)

deck: Optional[Deck] = st.session_state.get("deck")

if gen:
    # Double-check right before calling the provider
    allowed, reason, _ = can_call_now()
    if not allowed:
        st.warning(reason)
    else:
        raw_json = None
        try:
            if provider == "OpenAI":
                content = call_openai(
                    topic.strip(),
                    model=model,
                    temperature=temp,
                    max_tokens=max_tokens
                )
            else:
                # Offline generator returns a Python dict; serialize for uniform handling
                content = json.dumps(generate_offline_deck_json(topic.strip()))

            block = coerce_json_block(content)
            try:
                dj = validate_json_strict(block)
            except ValidationError:
                # Attempt one auto-repair pass: parse loosely then coerce into schema
                dj = DeckJSON.model_validate(json.loads(block))

            # Enforce portfolio constraints regardless of provider
            dj = enforce_constraints(dj)
            deck = deckjson_to_deck(topic.strip(), brand, dj)
            st.session_state["deck"] = deck

            # Evaluation
            score, details = score_deck(dj)
            st.session_state["deck_eval"] = (score, details)

            # Mark this call as successful for rate limits
            record_successful_call()

        except Exception as e:
            st.error(f"Generation failed: {e}")
            deck = None


# ======================= Preview, QA, Export =======================
eval_state = st.session_state.get("deck_eval")

if deck:
    left, right = st.columns([2,1])

    with left:
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

    with right:
        st.subheader("Quality Report")
        if eval_state:
            score, details = eval_state
            st.metric("Deck Score", score)
            st.write(f"Slides: {details['total_slides']} | Bullets: {details['total_bullets']}")
            st.write(f"Avg words per bullet: {details['avg_bullet_words']}")
            if details["issues"]:
                st.warning("Issues found:")
                for it in details["issues"]:
                    st.write(f"- {it}")
            else:
                st.success("No structural issues detected.")
        else:
            st.info("Generate to see quality metrics.")

        st.subheader("Export")
        pptx_bytes = deck_to_pptx_bytes(deck)
        st.download_button(
            label="Download PPTX",
            data=pptx_bytes,
            file_name="ai_slides_portfolio.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            type="primary"
        )

else:
    st.info("Enter a topic and click Generate to create a deck.")

init_rate_limit_state()
ss = st.session_state

st.markdown("**Usage limits**")
st.write(f"Today: {ss['rl_calls_today']} / {DAILY_LIMIT} generations")
if HOURLY_SHARED_CAP > 0:
    counters = _shared_hourly_counters()
    used = counters.get(_hour_bucket(), 0)
    st.write(f"Hour capacity: {used} / {HOURLY_SHARED_CAP}")

# Optional: show remaining cooldown visually
remaining = int(max(0, ss["rl_last_ts"] + COOLDOWN_SECONDS - time.time()))
if remaining > 0:
    st.progress(min(1.0, (COOLDOWN_SECONDS - remaining) / COOLDOWN_SECONDS))
    st.caption(f"Cooldown: {remaining}s")

