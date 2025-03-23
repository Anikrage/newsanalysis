from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import utils
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class CompanyRequest(BaseModel):
    company_name: str

@app.post("/analyze-company")
async def analyze_company(request: CompanyRequest):
    try:
        # Search for news articles
        articles = utils.search_news(request.company_name)
        valid_articles = []
        
        # Process each article
        for url in articles:
            article = utils.extract_article_content(url)
            if article and len(article['content']) > 100:  # Filter short content
                try:
                    # Process the article
                    summary = utils.summarize_text(article['content'])
                    sentiment = utils.analyze_sentiment(article['content'])
                    topics = utils.extract_topics(article['content'])
                    
                    # Add to valid articles
                    valid_articles.append({
                        'title': article['title'],
                        'url': article['url'],
                        'summary': summary,
                        'sentiment': sentiment,
                        'topics': topics
                    })
                except Exception as e:
                    logger.error(f"Error processing {url}: {str(e)}")
        
        # Check if we have any valid articles
        if not valid_articles:
            return JSONResponse(content={
                "message": "No valid articles found",
                "fallback": True,
                "articles": [],
                "company": request.company_name
            })
        
        # Perform comparative analysis
        comparative_analysis = utils.perform_comparative_analysis(valid_articles)
        
        # Generate detailed summary text
        sentiment_distribution = comparative_analysis["sentiment_distribution"]
        common_topics = comparative_analysis["common_topics"]
        
        summary_text = f"Analysis for {request.company_name}:\n"
        summary_text += f"A total of {comparative_analysis['total_articles']} articles were analyzed.\n"
        
        # Discuss sentiment distribution
        summary_text += f"Sentiment distribution: "
        summary_text += f"{sentiment_distribution.get('Positive', 0)} positive articles, "
        summary_text += f"{sentiment_distribution.get('Negative', 0)} negative articles, and "
        summary_text += f"{sentiment_distribution.get('Neutral', 0)} neutral articles.\n"
        
        # Discuss common topics
        if common_topics:
            topic_list = ', '.join([topic for topic, _ in common_topics[:5]])
            summary_text += f"The most frequently discussed topics were: {topic_list}.\n"
        
        # Add overall sentiment score
        avg_score = comparative_analysis["average_sentiment_score"]
        summary_text += f"The average sentiment score across all articles was {avg_score:.2f}.\n"
        
        # Generate Hindi translation and speech
        speech_file, hindi_text = utils.generate_hindi_speech(summary_text)
        
        result = {
            "company": request.company_name,
            "articles": valid_articles,
            "comparative_analysis": comparative_analysis,
            "summary_text": summary_text,
            "hindi_text": hindi_text,
            "speech_file": speech_file
        }
        
        # Preprocess the result to remove any problematic float values
        processed_result = utils.preprocess_data(result)
        return JSONResponse(content=processed_result)
    
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Company News Analyzer API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
