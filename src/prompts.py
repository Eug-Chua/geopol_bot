import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def briefing_prompt(top_topic_headlines: list) -> str:
    prompt = f"""
    You are a geopolitical analyst writing high-clarity, high-signal briefings for readers who want to quickly understand the key developments shaping today's global power landscape.

    Your job is to analyze a set of recent headlines and convert them into a structured mini-briefing in the following format — compact, clear, and strategic. Use emoji headers for clarity and hierarchy. Avoid long paragraphs. Prioritize readability and insight.

    TODAY'S FOCAL POINT: "GEOPOLITICS"

    INTELLIGENCE HEADLINES:
    {chr(10).join(f"• {h}" for h in top_topic_headlines)}

    ============================

    📌 1. EXECUTIVE SUMMARY  
    🔹 Give 3–4 short bullet points.  
    🔹 Capture the core developments across regions or actors.  
    🔹 Highlight who’s involved, what’s shifting, and why it matters.

    ============================

    📊 2. SIGNAL ANALYSIS  
    🔺 Power dynamics — Who is gaining or losing strategic influence?  
    💰 Economic indicators — What financial or trade signals stand out?  
    🕊 Diplomatic positioning — Are alliances shifting? Who's negotiating or escalating?  
    ⚔️ Military/security developments — Any signs of force posturing, conflict, or deterrence?

    ❗ Meaningful Signals vs. Noise — What’s real movement vs. PR or political distraction?

    🔀 Conflicting Trends — Are words and actions mismatched? Any strategic ambiguity?

    ============================

    📈 3. IMPACT ASSESSMENT  
    🔹 Primary stakeholders affected and how  
    🔹 Geographic spread: regional vs global consequences  
    🔹 Sectoral impact: tech, defense, energy, diplomacy, etc.  
    🔹 Timeframe: Immediate risks vs longer-term trends  
    🔹 Severity: **Low / Moderate / High / Critical**, with a one-line justification

    ============================

    STYLE GUIDELINES:
    - Use bullet points and emoji headers
    - Avoid hedging language (no “maybe,” “could be”)
    - Keep it readable: no dense jargon or unnecessary complexity
    - Don’t speculate — only summarize what the signals reasonably suggest
    - Focus on strategic insight, not event-by-event reporting

    Return ONLY the structured briefing with no explanation or commentary.

    Limit your output to 250 words.
    """

    response = client.chat.completions.create(
        model=os.getenv('OPENAI_MODEL'),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content

def lesson_prompt(briefing_text: str) -> str:
    prompt = f"""You are a geopolitical analyst trained to extract key themes from complex international developments and translate them into accessible one-way lessons. Your job is to take in a structured geopolitical intelligence briefing (with executive summary, signal analysis, and impact assessment) and generate a concise, one-way, mini-lesson for the reader.

This lesson should:
- Explain **one key concept in geopolitics**
- Use **examples pulled directly from the briefing**
- Offer a clear, digestible **definition and context**
- Unpack how the concept plays out in the real world (based on the signals and players mentioned)
- End with a brief **"food for thought" section** to spark deeper reflection

### Format:
1. 🧭 Mini Lesson Title
   *Choose a relevant geopolitics concept and frame it as a lens on the briefing*

2. 📌 What It Means  
   *Define the concept in 2–4 sentences in plain terms*

3. 🌐 How It Plays Out in This Briefing
   *Extract key examples from the provided summary to show this concept in action. Quote specific phrases where helpful.*

4. 🧠 Why It Matters
   *Explain the broader stakes — how this concept shapes global stability, alliances, or economic security*

5. 🧩 Food for Thought
   *End with 2–3 thought-provoking questions related to the example and concept*

Tone: Clear, professional, non-academic, slightly strategic — like a smart newsletter for readers who want to understand what’s *really going on beneath the headlines.* Keep it readable: no dense jargon or unnecessary complexity

Avoid: Overloading with jargon, listing too many actors, giving overly abstract theory.

---

### Briefing to Analyze:
```
{briefing_text.strip()}
```

Limit your output to 250 words.
""".strip()
    response = client.chat.completions.create(
    model="gpt-4.1-nano",  # or another model you prefer
    messages=[{"role": "user", "content": prompt}],
    temperature=0.5,
)
    return response.choices[0].message.content