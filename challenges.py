import requests
import trafilatura
import google.generativeai as genai
import asyncio
import httpx
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
import io
import aiohttp
import sys
import platform
import os
import re
from bs4 import BeautifulSoup
import json

# API Keys
SERPER_API_KEY = "d38fff68cf3c2e994f15273fb1f8dc5743535d2b"
GEMINI_API_KEY = "AIzaSyAt_c0xgaXGg9H4oFX0YUqsQuhnV4gi7BY"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Create thread pool and semaphore
thread_pool = ThreadPoolExecutor(max_workers=8)
API_SEMAPHORE = asyncio.Semaphore(5)

# AIonOS Capabilities
AIonOS_CAPABILITIES = (
     """Agentic Assistants & Customer Experience
ConciergeAgent/Mate – Real-time voice/text assistant for end-to-end journey support and disruption management.

IntelliConverse – Multilingual AI interface for seamless support across voice, chat, and messaging channels.

IntelliEmail – Automated email engine for confirmations, updates, and customer notifications.

IntelliVoice – Natural language assistant for bookings, changes, and cancellations.

IntelliSurvey – Conversational feedback tool to capture post-service insights.

IntelliSocial – Real-time social engagement tool for responding to customer signals and brand mentions.

AI & Analytics Engines
IntelliPulse – Dashboard monitoring sentiment, compliance, and agent-customer interactions.

IntelliRAG – Retrieval-Augmented Generation system delivering accurate, policy-compliant agent responses.

IntelliReach – Personalization engine for contextual journey recommendations and banking offers.

IntelliMarketing – AI-driven campaign engine optimizing personalization and targeting.

IntelliFinCrime – Financial crime detection tool for monitoring fraud, AML, and anomalies.

IntelliRegTech – Compliance assistant automating regulatory tracking and audit documentation.

IntelliResilience – AI-powered system for predictive recovery, failover, and SLA assurance.

IntelliSustain – ESG monitoring platform tracking emissions, waste, and regulatory compliance.

Operations & Workflow Automation
IntelliWorkflow – AI-driven orchestration of complex cross-functional business processes.

Smart Exchange – Platform for reconciling and exchanging logistics and trade financial data.

Smart Verify – Identity verification engine combining authentication, fraud detection, and user context.

Industry Platforms & Infrastructure
Freight Forwarding System (FFS) – End-to-end logistics platform managing multimodal shipments, CRM, billing, and documentation.

Warehouse Management System – Streamlines inbound/outbound processes, inventory, and reporting.

Dynamic Workforce Tracking – Real-time tracking of vehicles and workforce across the supply chain.

Smart Building Management – Centralized control and optimization of building systems and assets.

Data & Collaboration
Data Collaboration Platform – Consent-based data mesh enabling secure internal and external data sharing.

AionOS delivers industry-specific Agentic AI solutions that autonomously address complex business challenges while enhancing human collaboration—driving innovation across travel, logistics, hospitality, telecom, and transport."""
)

# Add Windows-specific event loop policy
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def search_serper_async(client: httpx.AsyncClient, company_name: str, query: str, max_results: int = 3) -> List[str]:
    """Search using Serper API with a single optimized query."""
    try:
        async with API_SEMAPHORE:
            # Single comprehensive query that includes the year to get recent results
            search_query = f"{company_name} recent {query}"
            
            url = "https://google.serper.dev/search"
            headers = {"X-API-KEY": SERPER_API_KEY}
            payload = {
                "q": search_query,
                "gl": "us",
                "hl": "en",
                "num": max_results
            }
            
            response = await client.post(url, json=payload, headers=headers, timeout=3.0)
            response.raise_for_status()
            results = response.json()
            
            all_urls = []
            if results.get("organic"):
                for result in results["organic"]:
                    if result.get("link") and not any(x in result["link"].lower() for x in ["youtube.com", "facebook.com", "twitter.com", "linkedin.com"]):
                        if result["link"] not in all_urls:
                            all_urls.append(result["link"])
                            if len(all_urls) >= max_results:
                                break
            
            return all_urls
    except Exception:
        return []

async def extract_content_async(client: httpx.AsyncClient, url: str) -> Optional[str]:
    """Extract and clean content from a URL using Trafilatura."""
    try:
        async with API_SEMAPHORE:
            # Follow redirects and handle mobile URLs
            if 'm.economictimes.com' in url:
                url = url.replace('m.economictimes.com', 'economictimes.indiatimes.com')
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = await client.get(url, headers=headers, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
            
            # Use Trafilatura to extract content
            downloaded = trafilatura.extract(response.text, include_metadata=True, include_comments=False, include_tables=False)
            
            if downloaded:
                # Clean and limit the text
                cleaned_text = clean_text(downloaded)
                if len(cleaned_text) < 100:
                    return None
                return cleaned_text[:5000]
            return None
    except httpx.HTTPStatusError as e:
        print(f"HTTP error for {url}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return None

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    text = " ".join(text.split())
    text = ''.join(char for char in text if char.isprintable())
    return text.strip()

async def extract_pdf_content(url: str) -> Optional[str]:
    """Extract text content from a PDF URL."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    pdf_data = await response.read()
                    pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
                    text = ""
                    for page in pdf_document:
                        text += page.get_text()
                    pdf_document.close()
                    return text[:10000]  # Limit text length
    except Exception as e:
        print(f"Error extracting PDF content: {str(e)}")
    return None

async def get_challenges_and_solutions(client: httpx.AsyncClient, company_name: str) -> Dict:
    """Get challenges and AIonOS solutions section."""
    final_output = []
    tried_urls = []
    
    try:
        # Search for challenges from web with more specific queries
        web_urls = await search_serper_async(client, company_name, "business challenges technology operations", max_results=3)
        
        # Process web content
        for url in web_urls:
            content = await extract_content_async(client, url)
            if content:
                tried_urls.append(url)
                prompt = f"""Based on this content about {company_name}'s business and operational challenges:
{content}

And considering AIonOS's capabilities:
{AIonOS_CAPABILITIES}

Extract up to two distinct business or operational challenges that {company_name} is facing. Focus only on:
Technology and digital transformation challenges
Operational efficiency challenges
Customer experience challenges
Business process challenges
Data and analytics challenges

For each challenge:
1. Clearly state a specific Challenge the company is facing in one short sentence
2. Immediately follow it with a corresponding AIonOS Solution that directly addresses this challenge

Exclude:
Any mention of stock market or financial performance
General industry challenges not specific to {company_name}
Market competition challenges

Important formatting instructions:
Your response must contain no introduction, summary, or explanation. Only output the two entries in the exact following format:

Format each entry exactly as:
• Challenge: [One short sentence about a specific business/operational challenge]
• AIonOS Solution: [One short sentence about how AIonOS can solve this specific challenge]
• URL: {url}

• Challenge: [One short sentence about a specific business/operational challenge]
• AIonOS Solution: [One short sentence about how AIonOS can solve this specific challenge]
• URL: {url}"""

                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        thread_pool, 
                        lambda: model.generate_content(prompt)
                    )
                    challenges = response.text.strip()
                    if challenges:
                        final_output.append(challenges)
                except Exception as e:
                    print(f"Error generating content for URL {url}: {str(e)}")
                    continue
        
        # Search for annual report PDFs with more specific query
        pdf_urls = await search_serper_async(client, company_name, "latest annual report filetype:pdf", max_results=1)
        
        # Process PDF content
        for url in pdf_urls:
            content = await extract_pdf_content(url)
            if content:
                tried_urls.append(url)
                prompt = f"""Based on this content about {company_name}'s business and operational challenges:
{content}

And considering AIonOS's capabilities:
{AIonOS_CAPABILITIES}

Extract up to two distinct business or operational challenges that {company_name} is facing. Focus only on:
Technology and digital transformation challenges
Operational efficiency challenges
Customer experience challenges
Business process challenges
Data and analytics challenges

For each challenge:
1. Clearly state a specific Challenge the company is facing in one short sentence
2. Immediately follow it with a corresponding AIonOS Solution that directly addresses this challenge

Exclude:
Any mention of stock market or financial performance
General industry challenges not specific to {company_name}
Market competition challenges

Important formatting instructions:
Your response must contain no introduction, summary, or explanation. Only output the two entries in the exact following format:

Format each entry exactly as:
• Challenge: [One short sentence about a specific business/operational challenge]
• AIonOS Solution: [One short sentence about how AIonOS can solve this specific challenge]
• URL: {url}

• Challenge: [One short sentence about a specific business/operational challenge]
• AIonOS Solution: [One short sentence about how AIonOS can solve this specific challenge]
• URL: {url}"""

                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        thread_pool, 
                        lambda: model.generate_content(prompt)
                    )
                    challenges = response.text.strip()
                    if challenges:
                        final_output.append(challenges)
                except Exception as e:
                    print(f"Error generating content for PDF URL {url}: {str(e)}")
                    continue
        
        if not final_output:
            # Fallback to general knowledge if no content found
            prompt = f"""Based on your knowledge about {company_name}'s business and operational challenges, provide two distinct entries. Focus only on:
Technology and digital transformation challenges
Operational efficiency challenges
Customer experience challenges
Business process challenges
Data and analytics challenges

For each challenge:
1. Clearly state a specific Challenge the company is facing in one short sentence
2. Immediately follow it with a corresponding AIonOS Solution that directly addresses this challenge

Exclude:
Any mention of stock market or financial performance
General industry challenges not specific to {company_name}
Market competition challenges

Important formatting instructions:
Your response must contain no introduction, summary, or explanation. Only output the two entries in the exact following format:

Format the output exactly as:
• Challenge: [One short sentence about a specific business/operational challenge]
• AIonOS Solution: [One short sentence about how AIonOS can solve this specific challenge]

• Challenge: [One short sentence about a specific business/operational challenge]
• AIonOS Solution: [One short sentence about how AIonOS can solve this specific challenge]"""
            
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    thread_pool, 
                    lambda: model.generate_content(prompt)
                )
                final_output.append(response.text.strip())
            except Exception as e:
                print(f"Error generating fallback content: {str(e)}")
        
        return {
            "summary": "\n\n".join(final_output)
        }
    except Exception as e:
        print(f"Error in get_challenges_and_solutions: {str(e)}")
        return {
            "summary": "Error occurred while generating challenges and solutions."
        }

async def main():
    """Main function to run the challenges analysis for a company."""
    print("\n=== Company Challenges & AIonOS Solutions Generator ===\n")
    company_name = input("Enter company name: ").strip()
    
    if not company_name:
        print("Error: Please provide a valid company name.")
        return
    
    print(f"\n🔍 Analyzing challenges for {company_name}...\n")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            result = await get_challenges_and_solutions(client, company_name)
            
            print("\n🎯 CHALLENGES & AIonOS SOLUTIONS")
            print("=" * 50)
            print(result["summary"])
            
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
    finally:
        # Clean up resources
        thread_pool.shutdown(wait=False)

def run_async_main():
    """Wrapper function to handle async main execution."""
    try:
        if platform.system() == 'Windows':
            # Use WindowsSelectorEventLoopPolicy for Windows
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        # Ensure thread pool is properly shut down
        thread_pool.shutdown(wait=True)

def battle_challenges(challenges_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a concise battle-ready summary of challenges.
    
    Args:
        challenges_summary (Dict[str, Any]): The original challenges summary
        
    Returns:
        Dict[str, Any]: A concise battle-ready summary
    """
    try:
        # Extract the challenges text from summary
        challenges_text = challenges_summary.get("summary", "")
        if not challenges_text:
            return {"error": "No challenges data available"}
            
        # Create a prompt for concise summarization
        prompt = f"""Based on these challenges and solutions, create a battle-ready summary with these exact sections:

Important formatting instructions:
Your response must contain no introduction, summary, or explanation. Only output the format:

• [Challenge: Most critical challenge 1]
• [Solution: Key solution strategy 1]
• [Challenge: Most critical challenge 2]
• [Solution: Key solution strategy 2]
• [Challenge: Most critical challenge 3]
• [Solution: Key solution strategy 3]
• [Challenge: Most critical challenge 4]
• [Solution: Key solution strategy 4]
• [Challenge: Most critical challenge 5]
• [Solution: Key solution strategy 5]

Content:
{challenges_text}"""

        # Get the concise summary using Gemini
        response = model.generate_content(prompt)
        battle_summary = response.text.strip()
        
        return {
            "original_challenges": challenges_summary,
            "battle_summary": battle_summary
        }
        
    except Exception as e:
        return {"error": f"Error creating battle summary: {str(e)}"}

if __name__ == "__main__":
    run_async_main()
