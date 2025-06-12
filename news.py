import requests
import trafilatura
import google.generativeai as genai
import asyncio
import httpx
from typing import List, Dict, Any, Optional

# API Keys
SERPER_API_KEY = "d38fff68cf3c2e994f15273fb1f8dc5743535d2b"
GEMINI_API_KEY = "AIzaSyAt_c0xgaXGg9H4oFX0YUqsQuhnV4gi7BY"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-lite")

async def search_news_async(client: httpx.AsyncClient, query: str) -> List[str]:
    url = "https://google.serper.dev/news"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"q": query}
    
    try:
        response = await client.post(url, headers=headers, json=payload)
        data = response.json()
        
        if "news" in data:
            # Filter out irrelevant results
            company_words = query.split()[0].lower()  # Get company name
            relevant_links = []
            seen_titles = set()
            
            for item in data["news"]:
                title = item.get("title", "").lower()
                snippet = item.get("snippet", "").lower()
                
                # Skip if we've seen a similar title
                if any(title in seen_title or seen_title in title for seen_title in seen_titles):
                    continue
                    
                # Check if the content is relevant
                if (company_words in title or company_words in snippet) and \
                   not any(word in title.lower() for word in ["stock", "share", "price", "trading", "market"]):
                    relevant_links.append(item["link"])
                    seen_titles.add(title)
            
            return relevant_links
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []
    return []

async def search_news(company: str, company_url: str = None, geography: str = None) -> list:
    # Construct base search query with site and geography filters
    base_query = company.strip()
    # if company_url:
    #     try:
    #         domain = company_url.replace("https://", "").replace("http://", "").split("/")[0]
    #         base_query += f" site:{domain}"
    #     except:
    #         pass
    # if geography:
    #     base_query += f" {geography.strip()}"

    # More focused search queries using the enhanced base query
    queries = [
        f"{base_query} recent news",
        f"{base_query} partnership",
        f"{base_query} technology innovation",
        f"{base_query} business expansion news",
        f"{base_query} major acquisition",
        f"{base_query} new product launch",
        f"{base_query} digital transformation initiative",
        f"{base_query} new office opening",
        f"{base_query} collaboration announcement",
        f"{base_query} new service offering",
        f"{base_query} industry award recognition"
    ]
    
    async with httpx.AsyncClient() as client:
        tasks = [search_news_async(client, query) for query in queries]
        results = await asyncio.gather(*tasks)
    # Flatten results and remove duplicates
    all_links = list(set([link for sublist in results for link in sublist]))
    return all_links

async def scrape_article_async(url: str) -> Dict[str, str]:
    """Scrape article content and metadata asynchronously."""
    try:
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
        
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            # Extract content using trafilatura
            downloaded = trafilatura.extract(response.text, include_metadata=True, include_comments=False, include_tables=False)
            if not downloaded:
                return {"text": "", "date": ""}
            
            # Extract metadata
            metadata = trafilatura.metadata.extract_metadata(response.text)
            date = metadata.date if metadata and hasattr(metadata, 'date') else ""
            
            return {
                "text": downloaded,
                "date": date
            }
    except httpx.HTTPStatusError as e:
        print(f"HTTP error for {url}: {str(e)}")
        return {"text": "", "date": ""}
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return {"text": "", "date": ""}

async def scrape_articles(urls: List[str]) -> List[Dict[str, str]]:
    """Scrape multiple articles in parallel."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [scrape_article_async(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed scrapes and empty content
        valid_results = []
        for url, result in zip(urls, results):
            if isinstance(result, dict) and result.get("text"):
                valid_results.append({
                    "url": url,
                    "text": result["text"],
                    "date": result["date"]
                })
        
        return valid_results

def summarize_sync(text: str, company: str) -> str:
    try:
        # Check if the text contains enough relevant content
        company_words = company.lower().split()
        text_lower = text.lower()
        
        # Count occurrences of company name and related terms
        relevance_score = sum(text_lower.count(word) for word in company_words)
        
        if relevance_score < 3:  # Minimum threshold for relevance
            return ""
            
        prompt = f"""Summarize the following article into 4 to 5 bullet points, with each point written as one concise sentence.
Focus only on concrete news and developments specifically about {company}.
Do not include any introduction, conclusion, subheadings, or labels like "point 1".
Exclude all stock prices, market analysis, and generic background information.
Return only the bullet points in plain text, one per line:

{text[:5000]}"""
        response = model.generate_content(prompt)
        summary = response.text.strip()
        
        # Validate summary quality
        if len(summary) < 50 or "no information" in summary.lower() or "doesn't contain" in summary.lower():
            return ""
            
        return ' '.join(summary.split())  # Clean up whitespace
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        return ""

async def summarize_articles(articles: List[Dict[str, str]], company: str) -> List[Dict[str, str]]:
    # Process articles in batches to avoid overwhelming the API
    batch_size = 5
    all_summaries = []
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        # Run summarization in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, summarize_sync, article["text"], company)
            for article in batch
        ]
        summaries = await asyncio.gather(*tasks)
        
        for article, summary in zip(batch, summaries):
            if summary:
                all_summaries.append({
                    "url": article["url"],
                    "summary": summary,
                    "date": article["date"]
                })
                
        # If we have enough summaries, stop processing
        if len(all_summaries) >= 10:
            break
            
    return all_summaries[:10]  # Ensure we return at most 10 summaries

def generate_themes_sync(article_summaries: List[str]) -> Dict[str, str]:
    try:
        prompt = (
            "Given the following news article summaries, organize the key points under these themes: "
            "ensure content inside each theme is relevant to the theme and the company."
            "For each theme, provide 2 or 3 concise points (comma-separated, or as a short paragraph). "
            "If there is no news for a theme, write 'No major news'. "
            "Format the output as follows (do not use markdown or bullet points):\n"
            "News\n"
            "1) Partnerships: ...\n"
            "2) AI/Tech: ...\n"
            "3) Market Strategy: ...\n"
            "4) Expansion: ...\n"
            "5) Product/Fleet: ...\n"
            "6) Infra/Invest: ...\n\n"
            "Here are the summaries:\n\n" + "\n\n".join(article_summaries)
        )
        response = model.generate_content(prompt)
        theme_text = response.text.strip()
        
        # Parse the theme text into a structured format
        themes = {}
        current_theme = None
        
        for line in theme_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(('1)', '2)', '3)', '4)', '5)', '6)')):
                theme_name = line.split(':', 1)[0].split(')')[1].strip()
                content = line.split(':', 1)[1].strip() if ':' in line else ''
                themes[theme_name] = content
                
        return themes
    except Exception as e:
        print(f"Theme generation error: {str(e)}")
        return {
            "Partnerships": "No major news",
            "AI/Tech": "No major news",
            "Market Strategy": "No major news",
            "Expansion": "No major news",
            "Product/Fleet": "No major news",
            "Infra/Invest": "No major news"
        }

async def get_news_snapshot(themes: Dict[str, str]) -> Dict[str, Any]:
    """Create a concise snapshot of news themes in bullet points."""
    try:
        # Combine all themes into a single text for summarization
        combined_themes = []
        for theme, content in themes.items():
            if content and content.strip() != "No major news":
                combined_themes.append(f"{theme}: {content}")
        
        if not combined_themes:
            return {
                "snapshot": "No significant news themes available to summarize.",
                "themes": themes
            }
        
        combined_text = "\n".join(combined_themes)
        
        prompt = f"""Based on the following news themes, create a concise bullet-point summary. Each point should be a single, clear sentence that captures the most important news. Focus on concrete developments and avoid generic statements.

News Themes:
{combined_text}

Important formatting instructions:
Your response must contain no introduction, summary, or explanation. Only output in the exact following format:
• [point 1]
• [point 2]
• [point 3]
• [point 4]
• [point 5]"""

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, model.generate_content, prompt)
        snapshot = response.text.strip()
        
        return {
            "snapshot": snapshot,
            "themes": themes
        }
    except Exception as e:
        print(f"Error creating news snapshot: {str(e)}")
        return {
            "snapshot": "Error creating news snapshot",
            "themes": themes
        }

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
        
        # Generate news snapshot
        news_snapshot = await get_news_snapshot(themes)
        
        return {
            "themes": themes,
            "articles": summaries,
            "news_snapshot": news_snapshot["snapshot"]
        }
    except Exception as e:
        print(f"News analysis error: {str(e)}")
        return {"error": f"News analysis error: {str(e)}"}

async def main():
    """Main function to run the news analysis for a company."""
    company_name = input("Enter company name: ").strip()
    
    if not company_name:
        print("Error: Please provide a valid company name.")
        return
    
    print(f"\n🔍 Analyzing news for {company_name}...\n")
    
    try:
        # Get news analysis
        analysis = await get_news_analysis(company_name)
        
        if "error" in analysis:
            print(analysis["error"])
            return
        
        # Print results
        print("\n🎯 THEMES")
        print("=" * 50)
        for theme, content in analysis["themes"].items():
            print(f"\n{theme}:")
            print("-" * len(theme))
            print(content)
        
        print("\n📝 SUMMARIES")
        print("=" * 50)
        for summary in analysis["articles"]:
            print(f"\nURL: {summary['url']}")
            print("-" * 50)
            print(summary['summary'])
            print(f"Date: {summary['date']}")
            print()
        
        print("\n📸 NEWS SNAPSHOT")
        print("=" * 50)
        print(analysis["news_snapshot"])
        
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
