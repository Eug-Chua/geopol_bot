import os
import yaml
from openai import OpenAI
from collections import defaultdict, Counter

from src.utils import scrape_rss_and_analyze, send_to_telegram_sync

with open("geopolitics_schema.yaml", "r") as f:
    geopolitics_schema = yaml.safe_load(f)

# Core pipeline
def run_briefing():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Scrape and analyze
    feeds = {
        "zerohedge.com": "https://cms.zerohedge.com/fullrss2.xml"
    }
    all_results = []
    for site, feed_url in feeds.items():
        all_results.extend(scrape_rss_and_analyze(feed_url, site))

    # Sentiment by topic
    topic_sentiment = defaultdict(list)
    for r in all_results:
        for topic in r["topics"]:
            topic_sentiment[topic].append(r["sentiment"]["label"])

    top_topic = 'geopolitics'
    top_topic_headlines = [r["headline"] for r in all_results if top_topic in r["topics"]]

    # Step 1: Generate Briefing
    briefing_prompt = f"""
    You are a geopolitical analyst writing high-clarity, high-signal briefings... (trimmed)
    {chr(10).join(f"\u2022 {h}" for h in top_topic_headlines)}
    """

    briefing_response = client.chat.completions.create(
        model=os.getenv('OPENAI_MODEL'),
        messages=[{"role": "user", "content": briefing_prompt}],
        temperature=0.5,
    )
    briefing_text = briefing_response.choices[0].message.content

    # Step 2: Generate Mini Lesson
    lesson_prompt = f"""
    You are a geopolitical analyst trained to extract key themes... (trimmed)
    ---\n\n### Briefing to Analyze:\n{briefing_text.strip()}
    """
    lesson_response = client.chat.completions.create(
        model=os.getenv('OPENAI_MODEL'),
        messages=[{"role": "user", "content": lesson_prompt}],
        temperature=0.5,
    )
    mini_lesson = lesson_response.choices[0].message.content

    # Telegram delivery
    telegram_message = f"<b>📊 Daily Intelligence Brief: {top_topic.upper()}</b>\n{briefing_text}\n\n<b>🧭 Mini Lesson</b>\n{mini_lesson}"
    send_to_telegram_sync(
        message=telegram_message,
        token=os.getenv("TELEGRAM_BOT_TOKEN"),
        chat_id=os.getenv("TELEGRAM_CHAT_ID")
    )

if __name__ == "__main__":
    run_briefing()
