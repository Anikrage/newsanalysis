import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import numpy as np
from gtts import gTTS
from googletrans import Translator
import math
from gnews import GNews
import requests
import logging
from collections import defaultdict

# Initialize pipelines
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_news(company_name):
    """Search for news articles using NewsAPI"""
    try:
        api_key = "ceb40821cbb544a4be6619f0875a76c1"  # Replace with your actual NewsAPI key
        url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey={api_key}"
        
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'ok':
            articles = data['articles']
            article_urls = [article['url'] for article in articles]
            return article_urls[:10]  # Return top 10 articles
        else:
            logger.error(f"Failed to retrieve news: {data['message']}")
            return []
    
    except Exception as e:
        logger.error(f"Error searching for news: {str(e)}")
        return []
    
def manual_news_search(query):
    """Fallback manual search using Google News"""
    try:
        url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        
        # Find all article elements
        for article in soup.find_all('article'):
            link = article.find('a', href=True)
            if link and 'href' in link.attrs:
                article_url = f"https://news.google.com{link['href'][1:]}"
                articles.append(article_url)
        
        return articles[:10]
    
    except Exception as e:
        logger.error(f"Manual search failed: {str(e)}")
        return default_fallback_articles()
    
    except Exception as e:
        print(f"Error searching for news: {e}")
        # Return a few default news sites as fallback
        return [
            "https://www.reuters.com/",
            "https://www.bloomberg.com/",
            "https://www.cnbc.com/",
            "https://www.bbc.com/news"
        ]
def default_fallback_articles():
    """Return reliable default news sources"""
    return [
        "https://www.reuters.com",
        "https://www.bloomberg.com",
        "https://www.cnbc.com",
        "https://apnews.com",
        "https://www.bbc.com/news"
    ]

def extract_article_content(url):
    """Improved article content extraction with multiple fallback methods"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try multiple content extraction strategies
        content = ""
        selectors = [
            {'name': 'article', 'selectors': ['article']},
            {'name': 'main content', 'selectors': ['main', '.article-body']},
            {'name': 'generic content', 'selectors': ['body']}
        ]
        
        for strategy in selectors:
            for selector in strategy['selectors']:
                element = soup.select_one(selector)
                if element:
                    paragraphs = element.find_all(['p', 'h1', 'h2', 'h3'])
                    content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    if len(content) > 500:  # Only return if substantial content
                        return {
                            'title': soup.title.get_text(strip=True) if soup.title else "News Article",
                            'content': content,
                            'url': url
                        }
        
        # Fallback to simple text extraction
        return {
            'title': soup.title.get_text(strip=True) if soup.title else "News Article",
            'content': ' '.join([p.get_text(strip=True) for p in soup.find_all('p')]),
            'url': url
        }
    
    except Exception as e:
        logger.error(f"Failed to extract content from {url}: {str(e)}")
        return None

def extract_article_content(url):
    """Extract content from a news article URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Improved content extraction
        article_body = soup.find('article') or soup.find('main') or soup.body
        paragraphs = article_body.find_all(['p', 'h1', 'h2', 'h3']) if article_body else []
        
        content = ' '.join([p.text.strip() for p in paragraphs if p.text.strip()])
        return {
            'title': soup.title.text.strip() if soup.title else "No title",
            'content': content,
            'url': url
        }
    except Exception as e:
        print(f"Error extracting from {url}: {e}")
        return None


def summarize_text(text, max_length=150):
    """Generate a concise summary of the given text."""
    if len(text) < 50:  # Skip summarization for very short texts
        return text[:max_length] + "..." if len(text) > max_length else text
    
    # Calculate dynamic max_length (50-70% of text length)
    text_length = len(text.split())
    dynamic_max = max(30, min(int(text_length * 0.7), max_length))
    
    try:
        # Use a summarization model to get a concise summary
        summary = summarizer(
            text[:1024],  # Truncate to model's max input length
            max_length=dynamic_max,
            min_length=int(dynamic_max * 0.5),
            do_sample=False
        )[0]['summary_text']
        
        return summary
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        # Fallback to simple text truncation
        return text[:dynamic_max] + "..."




nltk.download('stopwords')
logger = logging.getLogger(__name__)

def analyze_sentiment(text):
    """Robust sentiment analysis with error handling"""
    try:
        if len(text) < 20:  # Minimum text length for meaningful analysis
            return {'sentiment': 'Neutral', 'score': 0.0}
            
        result = sentiment_analyzer(text[:512])[0]
        return {
            'sentiment': result['label'].capitalize(),
            'score': float(result['score'])
        }
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        return {'sentiment': 'Neutral', 'score': 0.0}

def extract_topics(text, num_topics=5):
    """Improved topic extraction with better preprocessing"""
    try:
        if len(text) < 100:  # Minimum text length for topic extraction
            return []
            
        vectorizer = CountVectorizer(
            stop_words=list(stopwords.words('english')),
            ngram_range=(1, 2),
            min_df=1,
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'  # Only words with 3+ letters
        )
        
        X = vectorizer.fit_transform([text])
        words = vectorizer.get_feature_names_out()
        counts = X.toarray()[0]
        
        return [word for word, _ in sorted(zip(words, counts), 
                key=lambda x: x[1], reverse=True)[:num_topics]]
        
    except Exception as e:
        logger.error(f"Topic extraction failed: {str(e)}")
        return []

def perform_comparative_analysis(articles_data):
    """Fault-tolerant comparative analysis"""
    analysis = {
        "sentiment_distribution": defaultdict(int),
        "average_sentiment_score": 0.0,
        "common_topics": [],
        "total_articles": len(articles_data)
    }

    try:
        # Sentiment analysis
        sentiment_scores = []
        for article in articles_data:
            sentiment = article.get('sentiment', {}).get('sentiment', 'Neutral')
            analysis["sentiment_distribution"][sentiment] += 1
            if 'score' in article.get('sentiment', {}):
                sentiment_scores.append(article['sentiment']['score'])
        
        # Convert defaultdict to regular dict
        analysis["sentiment_distribution"] = dict(analysis["sentiment_distribution"])
        
        # Calculate average score
        if sentiment_scores:
            analysis["average_sentiment_score"] = round(np.nanmean(sentiment_scores), 2)
        
        # Topic analysis
        topic_counts = defaultdict(int)
        for article in articles_data:
            for topic in article.get('topics', []):
                topic_counts[topic] += 1
        analysis["common_topics"] = sorted(topic_counts.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]

    except Exception as e:
        logger.error(f"Comparative analysis failed: {str(e)}")

    return analysis

def _get_common_topics(articles_data):
    """Helper function to extract common topics"""
    try:
        topic_counts = defaultdict(int)
        for article in articles_data:
            for topic in article.get("topics", []):
                topic_counts[topic] += 1
        return sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    except:
        return []



def translate_to_hindi(text):
    """Translates English text to Hindi."""
    translator = Translator()
    hindi_text = translator.translate(text, src='en', dest='hi').text
    return hindi_text

def generate_hindi_speech(text, output_file="output.mp3"):
    """Generate Hindi speech from text and save to file."""
    try:
        hindi_text = translate_to_hindi(text)  # Translate English text to Hindi
        tts = gTTS(text=hindi_text, lang='hi')  # Generate speech using gTTS
        tts.save(output_file)  # Save as MP3 file
        return output_file, hindi_text
    except Exception as e:
        logger.error(f"Failed to generate Hindi speech: {str(e)}")
        return None, None

def preprocess_data(data):
    """
    Recursively preprocesses the data structure to replace NaN and infinite values.
    This ensures that the JSON serialization (and plotting) does not fail.
    """
    if isinstance(data, dict):
        return {k: preprocess_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [preprocess_data(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return 0
    return data
