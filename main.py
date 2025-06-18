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
import json


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
PROMPT_TEMPLATE = """You are an expert market analyst. Analyze the company "{company}" by automatically determining its country of origin and primary operating geography using publicly available data.If the company operates primarily in India, all market insights, figures, and regulatory or macroeconomic context should reference India-specific standards, trends, and authorities (e.g., Indian government policy, RBI, SEBI, TRAI, NITI Aayog, Indian market data from Statista or McKinsey, etc.). Otherwise, use the relevant geographic standards accordingly.
provide the data in the following JSON format. Return ONLY the JSON object, no other text:
provide each point with one sentence with more effective information with bullet mark
{{
"market_structure_dynamics": {{
"industry_definition": "Provide 4 brief bullet points defining the industry’s boundaries,core sub-sectors and standard taxonomy..",
"market_size": "Provide 4 bullet points covering market size and growth rate and Quantifies current market value (revenue/volume), historical trends and 3–5 year forecasts..",
"geographic_&_segment_breakdown": "Provide 4 points describing Splits the market by region and customer/end-market segments, showing relative contributions.."
}},
"competitive_landscape": {{
"major_competitors": "List 4 main competitors and Profiles major incumbents and challengers, plus market concentration metrics in the industry.",
"Value_chain_ecosystem": "List 4 point that defines Maps upstream suppliers, downstream channels and where value is created (P2 priority - ok if not executable).",
"Technology & Innovation": "Highlight 4 current trends shaping the industry."
}},
"macro_environment": {{
"PESTLE_analysis": {{
"political": "List 4 political factors affecting the industry in one sentence.",
"economic": "Provide 4 key economic conditions influencing the industry in one sentence..",
"social": "List 4 relevant social or cultural factors in one sentence..",
"technological": "Mention 4 technological developments impacting the sector in one sentence..",
"legal": "Outline 4 legal or regulatory aspects.",
"environmental": "List 4 environmental considerations for the industry."
}},
"risks_challenges": "list 4 points that summarises principal headwinds, bottlenecks and regulatory/policy risks that could derail growth.",
"opportunities_outlook": "list 4 points that Identifies white-space markets, adjacent expansion levers and consensus analyst forecasts for the next 1–5 years."
}}"""

async def get_industry_analysis(company_name: str) -> Dict[str, Any]:
    """Get industry overview for a company."""
    try:
        # Generate industry overview using Gemini
        prompt = PROMPT_TEMPLATE.format(company=company_name)
        
        # Get response from model with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                raw_output = response.text.strip()
                
                # # Debug print
                # print(f"Attempt {attempt + 1} - Raw model output:", raw_output)
                
                # Try to parse the output as JSON
                try:
                    # First try direct JSON parsing
                    analysis_data = json.loads(raw_output)
                except json.JSONDecodeError:
                    # If that fails, try to clean and parse
                    # Remove any markdown formatting
                    cleaned_output = raw_output.replace('```json', '').replace('```', '').strip()
                    # Remove any leading/trailing quotes
                    cleaned_output = cleaned_output.strip('"\'')
                    analysis_data = json.loads(cleaned_output)
                
                # Validate the structure
                required_sections = ["market_structure_dynamics", "competitive_landscape", "macro_environment"]
                for section in required_sections:
                    if section not in analysis_data:
                        analysis_data[section] = {"error": f"Missing {section} data"}
                
                return {
                    "company_name": company_name,
                    "analysis": analysis_data
                }
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                continue
        
    except Exception as e:
        print(f"All attempts failed: {str(e)}")
        # Return a structured error response
        return {
            "company_name": company_name,
            "analysis": {
                "error": f"Industry analysis error: {str(e)}",
                "market_structure_dynamics": {
                    "industry_definition": "Analysis failed",
                    "market_size": "Analysis failed",
                    "key_segments": "Analysis failed",
                    "geographic_presence": "Analysis failed"
                },
                "competitive_landscape": {
                    "major_competitors": "Analysis failed",
                    "market_position": "Analysis failed",
                    "competitive_advantages": "Analysis failed",
                    "industry_trends": "Analysis failed"
                },
                "macro_environment": {
                    "political": "Analysis failed",
                    "economic": "Analysis failed",
                    "social": "Analysis failed",
                    "technological": "Analysis failed",
                    "legal": "Analysis failed",
                    "environmental": "Analysis failed"
                }
            }
        }

async def analyze_company(company_name: str) -> Dict[str, Any]:
    """Analyze a company by running all analyses in parallel."""
    try:
        start_time = time.time()
        
        # Create an HTTP client for making requests
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Run analyses in parallel
            # company_overview_task = get_company_overview_analysis(company_name)
            # news_task = get_news_analysis(company_name)
            # challenges_task = get_challenges_analysis(client, company_name)
            industry_task = get_industry_analysis(company_name)
            
            # Wait for all tasks to complete
            # company_overview, news, challenges, 
            industry = await asyncio.gather(
                # company_overview_task,
                # news_task,
                # challenges_task,
                industry_task,
                return_exceptions=True  # This will prevent one task from failing the entire request
            )
            
            # Handle any exceptions that occurred during task execution
            # if isinstance(company_overview, Exception):
            #     company_overview = {"error": f"Company overview error: {str(company_overview)}"}
            # if isinstance(news, Exception):
            #     news = {"error": f"News analysis error: {str(news)}"}
            # if isinstance(challenges, Exception):
            #     challenges = {"error": f"Challenges analysis error: {str(challenges)}"}
            if isinstance(industry, Exception):
                industry = {"error": f"Industry analysis error: {str(industry)}"}

            # Calculate total processing time
            processing_time = time.time() - start_time
            
            # Extract battle summary from challenges if available
            # battle_summary = None
            # if isinstance(challenges, dict) and "battle_summary" in challenges:
            #     battle_summary = challenges["battle_summary"]
            #     # Remove battle summary from challenges to avoid duplication
            #     challenges.pop("battle_summary", None)
            
            # Combine the results
            result = {
                # "company_overview": company_overview,
                # "news": news,
                # "challenges": challenges,
                "industry": industry,
                # "battle_summary": battle_summary or {"error": "Failed to generate battle summary"},
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
