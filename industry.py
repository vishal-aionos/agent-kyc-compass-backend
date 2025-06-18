import os
import asyncio
import httpx
import google.generativeai as genai
from typing import Dict, Any

# Configure Gemini
GEMINI_API_KEY = "AIzaSyAt_c0xgaXGg9H4oFX0YUqsQuhnV4gi7BY"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Prompt template for industry analysis
PROMPT_TEMPLATE = """You are an expert market analyst. For the company "{company}", provide a detailed and structured industry overview based on its country of origin and primary operating geography.

Automatically identify the company’s geographic focus based on publicly available information, such as its country of registration, headquarters location, or dominant market.

Organize the analysis into the following three major sections, tailored to the company’s core industry and geographic context. For each subsection, present exactly 4 concise, bullet-pointed insights, using current, specific, and relevant data or trends. Do not include vague or speculative statements, and avoid generic qualifiers like “as per my knowledge”.

Where applicable, cite real-world market dynamics, public sources (e.g., Statista, IMF, Gartner, IDC), or specific examples to ground the analysis.

1. Market Structure & Dynamics
Industry Definition & Scope
Clearly define the industry, its key sub-sectors, and value chain structure.

Provide 4 bullet points, each describing a distinct feature or component of the industry's scope or segmentation.

Market Size & Growth
Quantify the current market size, past growth trends, and forecasted growth rates.

Provide 4 bullet points, including figures like TAM, CAGR, and primary growth drivers.

Geographic & Segment Breakdown
Analyze how the market divides by geography, customer segments, or verticals.

Provide 4 bullet points on regional dynamics, segment-level performance, and demand patterns within the specified geography.

2. Competitive Landscape
Competitive Landscape
Identify top incumbents, challengers, and market structure (e.g., fragmented vs. consolidated).
Provide 4 bullet points with each point in one short and valid sentence, comparing player positions, market share, strategic moves, or barriers to entry.

Value Chain & Ecosystem
Map the supply chain , distribution networks, and key partnerships.

Provide 4 bullet points with each point in one short and valid sentence, in value chain basic format describing where value is generated or leaked across the chain, and ecosystem dynamics.

Technology & Innovation
Highlight emerging technologies, digital transformations, and R&D investment trends.

Provide 4 bullet points with each point in one short and valid sentence, on innovation themes, adoption rates, or competitive technology positioning.

3. Macro Environment & Forward Outlook
PESTLE Analysis
Examine macro-level forces—Political, Economic, Social, Technological, Legal, Environmental—that affect the industry.

Provide 6 bullet points, each covering one macro factor with specific, recent examples or trends.

Risks & Challenges
Describe operational, structural, regulatory, or competitive risks relevant to the company and region.

Provide 4 bullet points with each point in one short and valid sentence, on key risks or frictions hindering market efficiency or growth.

Opportunities & Outlook
Present future-looking insights, whitespace opportunities, and analyst projections.

Provide 4 bullet points with each point in one short and valid sentence, outlining specific growth levers, adjacent markets, or transformative trends forecasted over the next 1–5 years."""

# Prompt for geography detection
GEOGRAPHY_PROMPT = """For the company "{company}", identify the country or region where it was originally founded/established.
Return ONLY the location name (e.g., "United States", "Japan", "Germany") without any additional text or explanation.
If you cannot determine the origin location, return "Global"."""

async def detect_company_geography(company_name: str) -> str:
    """Detect the primary geography of a company using LLM."""
    try:
        prompt = GEOGRAPHY_PROMPT.format(company=company_name)
        response = model.generate_content(prompt)
        geography = response.text.strip()
        return geography if geography else "Global"
    except Exception as e:
        print(f"Error detecting company geography: {str(e)}")
        return "Global"

async def get_industry_analysis(company_name: str) -> Dict[str, Any]:
    """Get industry overview for a company."""
    try:
        # Detect company geography
        geography = await detect_company_geography(company_name)
        
        # Generate industry overview using Gemini
        prompt = PROMPT_TEMPLATE.format(
            company=company_name,
            geography=geography
        )
        
        response = model.generate_content(prompt)
        analysis = response.text.strip()
        
        return {
            "company_name": company_name,
            "geography": geography,
            "analysis": analysis
        }
            
    except Exception as e:
        print(f"Error in get_industry_overview: {str(e)}")
        return {
            "error": f"Error generating industry overview: {str(e)}"
        }

async def main():
    """Main function to run the industry analysis."""
    print("\n=== Industry Analysis Tool ===\n")
    company_name = input("Enter company name: ").strip()
    
    if not company_name:
        print("Error: Please provide a valid company name.")
        return
    
    print(f"\n🔍 Analyzing industry for {company_name}...\n")
    
    try:
        result = await get_industry_analysis(company_name)
        
        print("\n📊 INDUSTRY OVERVIEW")
        print("=" * 50)
        print(f"Origin Location: {result.get('geography', 'Global')}")
        print("\nAnalysis:")
        print(result["analysis"])
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

def run_async_main():
    """Run the async main function."""
    asyncio.run(main())

if __name__ == "__main__":
    run_async_main()
