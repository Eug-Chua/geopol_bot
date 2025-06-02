from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
labels = ['positive', 'negative', 'neutral']

# Tag keywords
TOPIC_KEYWORDS = {
    "geopolitics": ["india", "pakistan", "china", "israel", "gaza", "trump", "ukraine", "russia", "nato", 
                    "iran", "usa", "north korea", "united nations", "sanctions", "diplomacy", "eu", "brexit", 
                    "middle east", "taiwan", "south china sea", "border", "military", "defense", "g7", "summit"],
    
    "finance": ["market", "stocks", "rates", "tariff", "bank", "inflation", "trade", "futures", "economy",
                "recession", "fed", "federal reserve", "treasury", "bond", "crypto", "bitcoin", "portfolio", 
                "investment", "debt", "deficit", "gdp", "dollar", "currency", "hedge fund", "bear market", 
                "bull market", "earnings", "volatility", "ipo", "housing", "mortgage", "retail", "consumer", "supply chain", "unemployment", "jobs", "wage", "minimum wage"],
    
    "technology": ["ai", "openai", "neurodiversity", "chip", "robot", "software", "satoshi", 
                  "cybersecurity", "blockchain", "cloud", "data", "privacy", "algorithm", "semiconductor", 
                  "quantum", "startup", "silicon valley", "neural network", "machine learning", "web3", 
                  "autonomous", "spacex", "tesla", "meta", "apple", "google", "microsoft", "nvidia"],
    
    "energy": ["oil", "gas", "nuclear", "energy", "drilling", "saudi", 
              "renewable", "solar", "wind", "carbon", "emissions", "climate", "opec", "fracking", 
              "coal", "electric", "grid", "battery", "hydrogen", "biofuel", "sustainability"],
    
    "social": ["education", "diversity", "gender", "vaccine", "nypd", "college", 
              "healthcare", "inequality", "poverty", "immigration", "religion", "misinformation", 
              "civil rights", "protest", "abortion", "gun control", "lgbtq", "housing", "mental health", 
              "pandemic", "remote work", "crime", "justice", "policy"],
    
    "health": ["covid", "virus", "medical", "pharmaceutical", "hospital", "treatment", "disease", 
              "research", "who", "pandemic", "epidemic", "obesity", "aging", "longevity", "medicare", 
              "insurance", "wellness", "drug", "biotech", "telemedicine"],
}

