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
PROMPT_TEMPLATE = """You are an expert market analyst. Analyze the company "{company}" by identifying its country of origin and primary operating geography using publicly available data (e.g., headquarters, country of incorporation, or dominant market presence).

Your task is to deliver a structured, insight-rich industry overview organized into three main sections:
Market Structure & Dynamics, Competitive Landscape, and Macro Environment & Forward Outlook.

Each subsection must contain exactly 4 concise bullet points, except PESTLE, which must have 6 bullets (one for each macro factor).

Important output formatting instructions:

Start each section with a clear header (e.g., Market Structure & Dynamics).

For each subsection, use a bold italic header (e.g., Industry Definition & Scope).

List insights as plain bullet points using this exact structure:

css
Copy
Edit
***[Subsection Name]***
• [Insight 1]
• [Insight 2]
• [Insight 3]
• [Insight 4]
Each bullet must be:

1 short, complete sentence

Data-rich, containing a stat, figure, name, or verifiable insight

Industry-specific, geography-specific, and timely

Grounded in real-world market intelligence (e.g., Statista, Gartner, IDC, World Bank, IMF)

No vague phrases like “it is believed that” or “may be”

Sections and Subsections Required:

Market Structure & Dynamics

Industry Definition & Scope: Define the company’s industry, sub-sectors, and value chain.

Market Size & Growth: Provide total market size (TAM), past growth, future CAGR, and drivers.

Geographic & Segment Breakdown: Segment market by region, customer type, or vertical.

Competitive Landscape

Competitive Landscape: Identify key players, market shares, challengers, and structure.

Value Chain & Ecosystem: Cover supply chains, distribution, partnerships, and value dynamics.

Technology & Innovation: Highlight emerging tech, R&D investment, and digital themes.

Macro Environment & Forward Outlook

PESTLE Analysis: One bullet per macro factor — Political, Economic, Social, Technological, Legal, Environmental (6 total).

Risks & Challenges: Outline 4 critical operational, regulatory, or market risks.

Opportunities & Outlook: Highlight 4 forward-looking opportunities, trends, or growth areas.

Final Output Must Be Formatted Exactly Like This:

css
Copy
Edit
**[Section Title]**

***[Subsection Title]***
• [Insight 1]
• [Insight 2]
• [Insight 3]
• [Insight 4]

***[Next Subsection Title]***
• [Insight 1]
• [Insight 2]
• [Insight 3]
• [Insight 4]
Your response must contain no introduction, summary, or explanation. Only output the format exactly as shown."""




async def get_industry_analysis(company_name: str) -> Dict[str, Any]:
    """Get industry overview for a company."""
    try:
        
        # Generate industry overview using Gemini
        prompt = PROMPT_TEMPLATE.format(
            company=company_name
        )
        
        response = model.generate_content(prompt)
        analysis = response.text.strip()
        
        return {
            "company_name": company_name,
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
