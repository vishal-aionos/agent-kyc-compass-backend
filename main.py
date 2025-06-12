from fastapi import FastAPI, HTTPException
from concurrent.futures import ThreadPoolExecutor
import asyncio
import httpx
from typing import Dict, Any
import uvicorn
from company_overview import generate_company_snapshot
from news import search_news, scrape_articles, summarize_articles, generate_themes_sync, get_news_snapshot
from challenges import get_challenges_and_solutions, battle_challenges

app = FastAPI(title="Company Analysis API")

# Create a thread pool for parallel execution
thread_pool = ThreadPoolExecutor(max_workers=3)

async def run_in_threadpool(func, *args):
    """Run a function in the thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, func, *args)

async def get_company_overview_analysis(company_name: str) -> Dict[str, Any]:
    """Get company overview analysis."""
    try:
        # Create a client for the company overview
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Call generate_company_snapshot directly since it's already async
            result = await generate_company_snapshot(company_name)
            if isinstance(result, dict):
                return result
            return {"error": "Invalid company overview result"}
    except Exception as e:
        return {"error": f"Company overview error: {str(e)}"}

async def get_news_analysis(company_name: str) -> Dict[str, Any]:
    """Get news analysis for a company."""
    try:
        # Search for news
        news_links = await search_news(company_name)
        if not news_links:
            return {"message": f"No news found for {company_name}"}
        
        # Scrape articles
        articles = await scrape_articles(news_links)
        if not articles:
            return {"message": f"Could not extract content from news articles for {company_name}"}
        
        # Summarize articles
        summaries = await summarize_articles(articles, company_name)
        if not summaries:
            return {"message": f"Could not generate summaries for {company_name}"}
        
        # Generate themes
        theme_texts = [summary["summary"] for summary in summaries]
        themes = generate_themes_sync(theme_texts)
        
        # Generate themes summary
        news_snapshot = await get_news_snapshot(themes)
        
        return {
            "themes": themes,
            "articles": summaries,
            "news_snapshot": news_snapshot["snapshot"]
        }
    except Exception as e:
        return {"error": f"News analysis error: {str(e)}"}

async def get_challenges_analysis(client: httpx.AsyncClient, company_name: str) -> Dict[str, Any]:
    """Get challenges analysis for a company."""
    try:
        result = await get_challenges_and_solutions(client, company_name)
        if isinstance(result, dict):
            # Create battle-ready summary
            battle_summary = battle_challenges(result)
            return {
                "challenges": result,
                "battle_summary": battle_summary.get("battle_summary", {"error": "Failed to generate battle summary"})
            }
        return {"error": "Invalid challenges result"}
    except Exception as e:
        return {"error": f"Challenges analysis error: {str(e)}"}

@app.get("/analyze/{company_name}")
async def analyze_company(company_name: str) -> Dict[str, Any]:
    """
    Analyze a company by running challenges analysis.
    
    Args:
        company_name (str): Name of the company to analyze
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    try:
        # Create an HTTP client for making requests
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Run challenges analysis
            challenges = await get_challenges_analysis(client, company_name)
                        # Run analyses in parallel
            company_overview_task = get_company_overview_analysis(company_name)
            news_task = get_news_analysis(company_name)
            challenges_task = get_challenges_analysis(client, company_name)
            
            # Wait for all tasks to complete
            company_overview, news, challenges = await asyncio.gather(
                company_overview_task,
                news_task,
                challenges_task,
                return_exceptions=True  # This will prevent one task from failing the entire request
            )
            
            # Handle any exceptions that occurred during task execution
            if isinstance(company_overview, Exception):
                company_overview = {"error": f"Company overview error: {str(company_overview)}"}
            if isinstance(news, Exception):
                news = {"error": f"News analysis error: {str(news)}"}
            if isinstance(challenges, Exception):
                challenges = {"error": f"Challenges analysis error: {str(challenges)}"}

            # Combine the results
            result = {
                "company_overview": company_overview,
                "news": news,
                "challenges": challenges
            }
            
            return result
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Company Analysis API",
        "endpoints": {
            "/analyze/{company_name}": "Analyze a company (company overview, news, and challenges)"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
