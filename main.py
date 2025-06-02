import os
import yaml
from openai import OpenAI
from collections import defaultdict
from src.utils import scrape_rss_and_analyze, send_to_telegram_sync, extract_topic_keywords, tag_topic
from src.prompts import briefing_prompt, lesson_prompt

# Load schema
with open("geopolitics_schema.yaml", "r") as f:
    geopolitics_schema = yaml.safe_load(f)

# Extract flat topic → keywords mapping
topic_keywords = extract_topic_keywords(geopolitics_schema)

def run_briefing():
    feeds = {
        "zerohedge.com": "https://cms.zerohedge.com/fullrss2.xml"
    }

    all_results = []
    for site, feed_url in feeds.items():
        scraped = scrape_rss_and_analyze(feed_url, site)

        # Dynamically tag topics from schema
        for r in scraped:
            r["topics"] = tag_topic(r["headline"], topic_keywords)
        all_results.extend(scraped)

    # Collect sentiment by topic
    topic_sentiment = defaultdict(list)
    for r in all_results:
        for topic in r["topics"]:
            topic_sentiment[topic].append(r["sentiment"]["label"])

    top_topic = "geopolitics"
    top_topic_headlines = [r["headline"] for r in all_results if top_topic in r["topics"]]

    if not top_topic_headlines:
        print("No headlines found for topic:", top_topic)
        return

    # Step 1: Generate Briefing
    briefing_text = briefing_prompt(top_topic_headlines)

    # Step 2: Generate Mini Lesson
    mini_lesson = lesson_prompt(briefing_text, geopolitics_schema)

    # Step 3: Telegram Delivery
    telegram_message = f"<b>📊 Daily Intelligence Brief: {top_topic.upper()}</b>\n{briefing_text}\n\n<b>🧭 Mini Lesson</b>\n{mini_lesson}"
    send_to_telegram_sync(
        message=telegram_message,
        token=os.getenv("TELEGRAM_BOT_TOKEN"),
        chat_id=os.getenv("TELEGRAM_CHAT_ID")
    )

if __name__ == "__main__":
    run_briefing()
