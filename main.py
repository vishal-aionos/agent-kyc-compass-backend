from fastapi import FastAPI, HTTPException
from concurrent.futures import ThreadPoolExecutor
import asyncio
import httpx
from typing import Dict, Any, List
import uvicorn
from company_overview import generate_company_snapshot
from news import search_news, scrape_articles, summarize_articles, generate_themes_sync, get_news_snapshot
from challenges import get_challenges_and_solutions, battle_challenges
import os
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import time
import random
from bs4 import BeautifulSoup
import ssl
import re
import platform

# Configure Gemini
GEMINI_API_KEY = "AIzaSyAt_c0xgaXGg9H4oFX0YUqsQuhnV4gi7BY"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-lite")

# Add Windows-specific event loop policy
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(title="Company Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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

async def analyze_company(company_name: str) -> Dict[str, Any]:
    """Analyze a company by running all analyses in parallel."""
    try:
        start_time = time.time()
        
        # Create an HTTP client for making requests
        async with httpx.AsyncClient(timeout=30.0) as client:
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

            # Calculate total processing time
            processing_time = time.time() - start_time
            
            # Extract battle summary from challenges if available
            battle_summary = None
            if isinstance(challenges, dict) and "battle_summary" in challenges:
                battle_summary = challenges["battle_summary"]
                # Remove battle summary from challenges to avoid duplication
                challenges.pop("battle_summary", None)
            
            # Combine the results
            result = {
                "company_overview": company_overview,
                "news": news,
                "challenges": challenges,
                "battle_summary": battle_summary or {"error": "Failed to generate battle summary"},
                "processing_time": f"{processing_time:.2f} seconds"
            }
            
            return result
            
    except Exception as e:
        return {"error": f"Analysis error: {str(e)}"}

@app.get("/analyze/{company_name}")
async def analyze_company_endpoint(company_name: str) -> Dict[str, Any]:
    """
    Analyze a company by running all analyses in parallel.
    
    Args:
        company_name (str): Name of the company to analyze
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    try:
        result = await analyze_company(company_name)
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

async def main():
    """Main function to run the company analysis."""
    print("\n=== Company Analysis Tool ===\n")
    company_name = input("Enter company name: ").strip()
    
    if not company_name:
        print("Error: Please provide a valid company name.")
        return
    
    print(f"\n🔍 Analyzing {company_name}...\n")
    
    try:
        result = await analyze_company(company_name)
        
        print("\n📊 ANALYSIS RESULTS")
        print("=" * 50)
        
        # Print company overview
        if "company_overview" in result:
            print("\n🏢 COMPANY OVERVIEW")
            print("-" * 30)
            print(result["company_overview"].get("snapshot", "No overview available"))
        
        # Print news analysis
        if "news_analysis" in result:
            print("\n📰 NEWS ANALYSIS")
            print("-" * 30)
            if "themes" in result["news_analysis"]:
                for theme, content in result["news_analysis"]["themes"].items():
                    print(f"\n{theme}:")
                    print(content)
            print(f"\nNews Snapshot:\n{result['news_analysis'].get('news_snapshot', 'No news snapshot available')}")
        
        # Print challenges analysis
        if "challenges_analysis" in result:
            print("\n🎯 CHALLENGES & SOLUTIONS")
            print("-" * 30)
            if "challenges" in result["challenges_analysis"]:
                print(result["challenges_analysis"]["challenges"].get("analysis", "No challenges analysis available"))
        
        # Print battle summary
        if "battle_summary" in result:
            print("\n⚔️ BATTLE SUMMARY")
            print("-" * 30)
            if isinstance(result["battle_summary"], dict) and "battle_summary" in result["battle_summary"]:
                print(result["battle_summary"]["battle_summary"])
            else:
                print(result["battle_summary"].get("error", "No battle summary available"))
        
        print(f"\n⏱️ Total processing time: {result['processing_time']}")
        
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
    finally:
        # Clean up resources
        thread_pool.shutdown(wait=False)

def run_async_main():
    """Run the async main function."""
    asyncio.run(main())

if __name__ == "__main__":
    run_async_main()
