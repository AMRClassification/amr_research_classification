from utils.utils import get_categories, get_additional_info, get_keywords

# Get the categories and additional info once at module level
sector_options = get_categories("Sector")
research_area_options = get_categories("Research Area") 
research_area_keywords = get_keywords("Research Area")

def get_screening_prompt(title: str, abstract: str) -> str:
    """Generate the prompt for human therapeutics screening."""
    
    return f"""
You are an AI specialized in determining if research is undoubtably within the human therapeutics field. Follow the instructions and specifications below carefully.

**Instructions:**

1. **Classification Rules:**
   - Only classify as human therapeutics if there is clear, explicit evidence
   - Focus on the main objective and intended application of the research
   - Ignore references to other works or background information
   - Animal testing for human applications should be considered human therapeutics
   - Research must ONLY be classifiable as Human (sector) and one of the Therapeutics areas (research area)
   - Pay close attention to keywords - if keywords from non-therapeutics research areas appear frequently or are mentioned as the research focus, classify as False

2. **Available Classifications:**

Sector Classifications:
{sector_options}

Research Area Classifications:
{research_area_options}

Research Area Keywords:
{research_area_keywords}

3. **Analysis Task:**
Please analyze if the following title and abstract are undoubtably ONLY about human therapeutics research.
Return True ONLY if:
- The research is clearly and exclusively about human applications (Human sector classification)
- The research fits into one of the Therapeutics research area classifications
- No other sector or research area classifications would apply
- Keywords from non-therapeutics research areas do not appear frequently or as main focus

Return False for:
- Anything uncertain
- Research that could be classified into multiple sectors or research areas
- Research not clearly focused on human therapeutics
- Research containing many keywords from non-therapeutics areas
- Research where non-therapeutics keywords indicate the main research focus

Title: {title}
Abstract: {abstract}

**Output Format:**
Return a JSON object with the following structure:
{{
    "is_human_therapeutics": boolean,
    "explanation": "Brief explanation of your decision, referencing relevant classifications and keywords found in the text"
}}
""" 