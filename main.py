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
from industry import get_industry_analysis

# Configure Gemini
GEMINI_API_KEY = "AIzaSyAt_c0xgaXGg9H4oFX0YUqsQuhnV4gi7BY"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

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

# Prompt template for industry analysis
PROMPT_TEMPLATE ="""You are an expert market analyst. Analyze the company "{company}" by identifying its country of origin and primary operating geography automatically, based on publicly available data (e.g., country of registration, headquarters, dominant market).

Your task is to deliver a detailed industry overview that is structured into three main sections: Market Structure & Dynamics, Competitive Landscape, and Macro Environment & Forward Outlook.

For each subsection, provide exactly 4 concise bullet points (except PESTLE which requires 6), each being one valid sentence that presents a specific, current, and geography-relevant insight.

Use verifiable, real-world data and insights wherever possible (e.g., from Statista, Gartner, IMF, IDC, World Bank). Do not speculate or include vague statements like “it is believed that.” Each insight should be rooted in observable market trends or dynamics that directly relate to the company’s industry and its operating geography.

Your final output must be a JSON object that strictly follows this structure:

Sections to include in your analysis (output as shown in JSON):

Market Structure & Dynamics

Industry Definition & Scope: Define the industry the company operates in, its key sub-sectors, and how the value chain is structured.
Provide 4 clear, one-sentence bullet points.

Market Size & Growth: Provide figures for total market size (TAM), historical growth, future CAGR, and key growth drivers.
Provide 4 bullet points with precise data.

Geographic & Segment Breakdown: Break down the market by geography, customer segments, or industry verticals.
Provide 4 bullet points with regional or segment-level insights.

Competitive Landscape

Competitive Landscape: Identify leading players, challengers, market shares, and whether the structure is fragmented or consolidated.
Provide 4 bullet points with comparative or strategic insights.

Value Chain & Ecosystem: Describe the supply chain, distribution channels, partnerships, and where value is created or leaked.
Provide 4 bullet points with supply chain or ecosystem insights.

Technology & Innovation: Highlight major technology trends, R&D investments, and innovation themes.
Provide 4 bullet points related to innovation, digital transformation, or emerging tech.

Macro Environment & Forward Outlook

PESTLE Analysis: Cover 6 macro factors affecting the industry (Political, Economic, Social, Technological, Legal, Environmental).
Provide 1 bullet per factor, totaling 6.

Risks & Challenges: Identify operational, structural, regulatory, or market risks relevant to the industry.
Provide 4 bullet points describing major friction points or constraints.

Opportunities & Outlook: Present white space opportunities, growth areas, and future projections.
Provide 4 forward-looking bullet points based on trends or analyst views.

Important rules:

Each bullet must be 1 short, data-rich sentence.

All insights must be industry-specific, geography-specific, and timely.

Your final answer must be formatted as a structured JSON object exactly as shown above—do not include an introduction, summary, or any extra content.
"""



async def get_industry_analysis(company_name: str) -> Dict[str, Any]:
    """Get industry overview for a company."""
    try:
        # Detect company geography
        
        # Generate industry overview using Gemini
        prompt = PROMPT_TEMPLATE.format(
            company=company_name
        )
        
        response = model.generate_content(prompt)
        analysis_text = response.text.strip()
        
        # Try to parse the JSON response
        try:
            import json
            import re
            analysis_text = re.findall(r"```json\s*(\{.*?\})\s*```", analysis_text, re.DOTALL)[-1] 
            analysis = json.loads(analysis_text)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {str(e)}")
            # If JSON parsing fails, return the raw text
            analysis = {"raw_analysis": analysis_text}
        
        return {
            "company_name": company_name,
            "analysis": analysis
        }
            
    except Exception as e:
        print(f"Error in get_industry_overview: {str(e)}")
        return {
            "error": f"Error generating industry overview: {str(e)}"
        }

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
            industry_task = get_industry_analysis(company_name)
            
            # Wait for all tasks to complete
            company_overview, news, challenges, industry = await asyncio.gather(
                company_overview_task,
                news_task,
                challenges_task,
                industry_task,
                return_exceptions=True  # This will prevent one task from failing the entire request
            )
            
            # Handle any exceptions that occurred during task execution
            if isinstance(company_overview, Exception):
                company_overview = {"error": f"Company overview error: {str(company_overview)}"}
            if isinstance(news, Exception):
                news = {"error": f"News analysis error: {str(news)}"}
            if isinstance(challenges, Exception):
                challenges = {"error": f"Challenges analysis error: {str(challenges)}"}
            if isinstance(industry, Exception):
                industry = {"error": f"Industry analysis error: {str(industry)}"}

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
                "industry": industry,
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
        
        # Print industry analysis
        if "industry" in result:
            print("\n🌐 INDUSTRY ANALYSIS")
            print("-" * 30)
            print(result["industry"])
        
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
