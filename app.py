import re
from collections import Counter
from datetime import datetime, timezone

import emoji
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="EmojiSense | Sentiment Studio", page_icon="💬", layout="wide")

EMOTION_MAP = {
    "joy": ("joy", 1.0), "smile": ("joy", 0.8), "grin": ("joy", 1.0),
    "laugh": ("joy", 1.0), "happy": ("joy", 1.0), "heart": ("love", 0.9),
    "love": ("love", 1.0), "kiss": ("love", 0.8), "thumbs_up": ("approval", 0.8),
    "clap": ("approval", 0.8), "party": ("joy", 1.0), "fire": ("excitement", 0.7),
    "star": ("approval", 0.6), "sad": ("sadness", -0.8), "cry": ("sadness", -1.0),
    "sob": ("sadness", -1.0), "broken_heart": ("sadness", -1.0),
    "angry": ("anger", -1.0), "rage": ("anger", -1.0),
    "disappoint": ("sadness", -0.8), "fear": ("fear", -0.8),
    "scream": ("fear", -0.9), "worried": ("fear", -0.7),
    "confus": ("confusion", -0.3), "thinking": ("confusion", 0.0),
    "neutral": ("neutral", 0.0), "expressionless": ("neutral", 0.0),
    "eyes": ("curiosity", 0.0), "surpris": ("surprise", 0.1),
    "wow": ("surprise", 0.3), "poop": ("disgust", -0.8),
    "thumbs_down": ("disapproval", -0.8), "skull": ("negative / humor", -0.3),
}

EXACT_EMOJIS = {
    "❤️": ("love", 1.0), "❤": ("love", 1.0), "😍": ("love", 1.0),
    "🥰": ("love", 1.0), "😘": ("love", 0.8), "😊": ("joy", 0.8),
    "😀": ("joy", 0.9), "😄": ("joy", 1.0), "😂": ("joy / laughter", 0.8),
    "🤣": ("joy / laughter", 0.8), "🥳": ("celebration", 1.0),
    "👍": ("approval", 0.8), "👏": ("approval", 0.8), "🔥": ("excitement", 0.7),
    "😭": ("sadness / intense emotion", -0.8), "😢": ("sadness", -0.9),
    "😞": ("sadness", -0.8), "😡": ("anger", -1.0), "😠": ("anger", -0.9),
    "💔": ("heartbreak", -1.0), "👎": ("disapproval", -0.8),
    "😱": ("fear / surprise", -0.7), "😨": ("fear", -0.8),
    "😐": ("neutral", 0.0), "🤔": ("thinking", 0.0),
    "😕": ("confusion", -0.3), "🙄": ("annoyance", -0.5),
    "😎": ("confidence", 0.5), "🙏": ("gratitude / request", 0.3),
    "✨": ("excitement", 0.6), "💯": ("approval", 0.8),
    "😅": ("nervous laughter", 0.2), "🥲": ("bittersweet", 0.0),
    "😬": ("discomfort", -0.4), "💀": ("negative / humor", -0.3),
}

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1120px;}
.hero {padding: 30px 34px; background:linear-gradient(120deg,#25255c,#563a9b 65%,#805ad5);
       color:white; border-radius:20px; margin-bottom:22px;}
.hero h1 {color:white; font-size:2.5rem; margin:0 0 6px;}
.hero p {color:#eee8ff; font-size:1.05rem; margin:0;}
.note {font-size:.9rem; opacity:.82;}
[data-testid="stMetricValue"] {font-size:1.8rem;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>💬 EmojiSense</h1><p>Understand the feelings behind words and emojis. Instantly, privately, and without an API key.</p></div>', unsafe_allow_html=True)

@st.cache_resource
def get_analyzer():
    return SentimentIntensityAnalyzer()


def emoji_details(symbol):
    if symbol in EXACT_EMOJIS:
        return EXACT_EMOJIS[symbol]
    name = emoji.demojize(symbol, language="en").strip(":").lower()
    for token, result in EMOTION_MAP.items():
        if token in name:
            return result
    return ("context dependent", 0.0)


def analyze(value):
    analyzer = get_analyzer()
    found = emoji.emoji_list(value)
    symbols = [item["emoji"] for item in found]
    text_only = emoji.replace_emoji(value, replace=" ").strip()
    text_score = analyzer.polarity_scores(text_only) if text_only else {"neg":0.0,"neu":1.0,"pos":0.0,"compound":0.0}
    detail_rows = []
    for sym in symbols:
        label, score = emoji_details(sym)
        detail_rows.append({"Emoji": sym, "Meaning": emoji.demojize(sym).strip(":").replace("_", " "),
                            "Emotion": label, "Polarity": round(score, 2)})
    emoji_score = sum(x["Polarity"] for x in detail_rows) / len(detail_rows) if detail_rows else 0.0
    # Emojis influence the output, but never overpower a substantial written message.
    if text_only and symbols:
        weight = min(0.40, 0.15 + 0.05 * len(symbols))
        combined = (1-weight) * text_score["compound"] + weight * emoji_score
    elif symbols:
        combined = emoji_score
    else:
        combined = text_score["compound"]
    combined = max(-1.0, min(1.0, combined))
    label = "Positive" if combined >= 0.05 else "Negative" if combined <= -0.05 else "Neutral"
    mood = {"Positive":"😊", "Negative":"😔", "Neutral":"😐"}[label]
    if text_only and symbols:
        explanation = "This result combines the emotional tone of the words with the likely meanings of the emojis."
        if text_score["compound"] * emoji_score < 0:
            explanation += " The text and emojis point in different directions, suggesting mixed feelings or possible irony."
    elif symbols:
        explanation = "This result is based on emoji meanings; the intended emotion may change with context."
    else:
        explanation = "This result is based on the emotional language in the text."
    explanation += {"Positive":" Overall the message leans encouraging, happy, or approving.",
                    "Negative":" Overall the message leans unhappy, critical, or upset.",
                    "Neutral":" Overall the message is balanced, factual, or emotionally ambiguous."}[label]
    # These are descriptive index values, not calibrated probabilities.
    intensity = abs(combined) * 100
    distribution = {"Positive":max(combined, 0), "Negative":max(-combined,0),
                    "Neutral":max(0, 1-abs(combined))}
    return {"label":label, "mood":mood, "compound":combined, "intensity":intensity,
            "text_compound":text_score["compound"], "emoji_compound":emoji_score,
            "details":detail_rows, "explanation":explanation, "distribution":distribution,
            "characters":len(value), "words":len(text_only.split()), "emoji_count":len(symbols)}

with st.sidebar:
    st.header("✨ About EmojiSense")
    st.write("A lightweight sentiment explorer built with Streamlit, VADER sentiment rules, and emoji interpretation.")
    st.info("No sign-in, no external inference API, and no model download required.")
    st.subheader("Try a sample")
    samples = {
        "Choose a sample...":"",
        "Happy feedback":"I absolutely love this update! It's amazing 😍🎉❤️",
        "Frustrated customer":"This service is terrible. I'm really disappointed 😡💔",
        "Mixed feelings":"The presentation went well, but I am exhausted 😅😢",
        "Neutral update":"The meeting is scheduled for Monday at 10 AM.",
        "Emoji only":"🥳❤️🔥✨",
    }
    chosen = st.selectbox("Example messages", list(samples))
    st.caption("Language: best suited to English; emojis are interpreted approximately.")

if "draft" not in st.session_state:
    st.session_state.draft = ""
if "last_sample" not in st.session_state:
    st.session_state.last_sample = None
if chosen != st.session_state.last_sample:
    st.session_state.draft = samples[chosen]
    st.session_state.last_sample = chosen

st.subheader("✍️ Your message")
with st.form("sentiment_form"):
    message = st.text_area("Enter a review, comment, caption, or emojis", value=st.session_state.draft,
        height=155, max_chars=4000, placeholder="Example: I finally finished my project! So proud 😍🎉")
    submitted = st.form_submit_button("🔎 Analyze sentiment", type="primary", use_container_width=True)

if submitted:
    if not message.strip():
        st.warning("Please enter some text or emojis to analyze.")
        st.session_state.pop("result", None)
    else:
        st.session_state.result = (message, analyze(message))

if "result" in st.session_state:
    original, result = st.session_state.result
    st.divider()
    st.subheader("📊 Sentiment dashboard")
    cols = st.columns(4)
    cols[0].metric("Overall sentiment", f"{result['mood']} {result['label']}")
    cols[1].metric("Sentiment index", f"{result['compound']:+.2f}")
    cols[2].metric("Emotion intensity", f"{result['intensity']:.0f}/100")
    cols[3].metric("Emojis detected", str(result["emoji_count"]))
    st.caption("Sentiment index: -1 (most negative) to +1 (most positive). Intensity is a heuristic, not a confidence probability.")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Sentiment profile")
        chart = pd.DataFrame([{"Dimension":k, "Score":round(v*100,1)} for k,v in result["distribution"].items()])
        st.bar_chart(chart.set_index("Dimension"), y="Score", horizontal=True, height=245)
        st.caption("Heuristic profile visualization; bars are not class probabilities.")
    with right:
        st.markdown("#### 💡 What this means")
        st.info(result["explanation"])
        st.write(f"**Words:** {result['words']} · **Characters:** {result['characters']}")
        st.write(f"**Text polarity:** {result['text_compound']:+.2f} · **Emoji polarity:** {result['emoji_compound']:+.2f}")
    st.markdown("#### 😀 Emoji breakdown")
    if result["details"]:
        st.dataframe(pd.DataFrame(result["details"]), hide_index=True, use_container_width=True)
        moods = Counter(row["Emotion"] for row in result["details"])
        st.caption("Detected moods: " + ", ".join(f"{k} ({v})" for k,v in moods.items()))
    else:
        st.caption("No emojis were found in this message.")
    export = pd.DataFrame([{"timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "message":original, "sentiment":result["label"], "sentiment_index":round(result["compound"],3),
        "intensity_index":round(result["intensity"],1), "emojis_detected":result["emoji_count"],
        "explanation":result["explanation"]}])
    st.download_button("⬇️ Download analysis as CSV", export.to_csv(index=False).encode("utf-8-sig"),
                       "emojisense_analysis.csv", "text/csv", use_container_width=True)

st.divider()
st.caption("EmojiSense · Educational sentiment analysis · Emoji meanings and sarcasm are context dependent; predictions are not ground truth.")
