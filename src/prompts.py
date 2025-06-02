import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def briefing_prompt(top_topic_headlines):
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
    """


    response = client.chat.completions.create(
        model=os.getenv('OPENAI_MODEL'),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content



