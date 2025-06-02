import re
from dotenv import load_dotenv
import torch
import datetime
import feedparser
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
labels = ['positive', 'negative', 'neutral']

# Utility functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits.numpy()[0])
    top_label = labels[probs.argmax()]
    return {
        "label": top_label,
        "confidence": float(probs.max()),
        "scores": dict(zip(labels, map(float, probs)))
    }

def tag_topic(headline):
    headline_lower = headline.lower()
    matched_topics = [
        topic for topic, keywords in TOPIC_KEYWORDS.items()
        if any(keyword in headline_lower for keyword in keywords)
    ]
    return matched_topics if matched_topics else ["other"]

def scrape_rss_and_analyze(feed_url, source_name):
    headlines = []
    feed = feedparser.parse(feed_url)
    for entry in feed.entries:
        headline = entry.title
        sentiment = get_finbert_sentiment(headline)
        topic_tags = tag_topic(headline)
        headlines.append({
            "source": source_name,
            "headline": headline,
            "sentiment": sentiment,
            "published": entry.get("published", str(datetime.now())),
            "topics": topic_tags
        })
    return headlines