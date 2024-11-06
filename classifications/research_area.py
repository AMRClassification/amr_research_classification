import json

from utils.utils import (
    get_categories,
    parse_non_json_response,
    get_additional_info,
    extract_json,
)
from utils.llm_call import classify_research_json


def generate_prompt(title, abstract, include_examples=True):
    research_area_options = get_categories("Research Area")
    research_area_additional_info = get_additional_info("Research Area")

    base_prompt = f"""
        You are an AI specialized in classifying research papers on antimicrobial resistance into relevant research areas based on their title and abstract. Follow the instructions and specifications below to determine the appropriate research area(s).

        **Instructions:**

        1. **Input:**
            - **Title:** {title}
            - **Abstract:** {abstract}
        2. **Classification Rules:**
            
            a. **Direct Mention or Inference:**
                - Only classify research areas that are directly mentioned or can be directly inferred from the title and abstract. Choose the one that fits the context of the research the best, taking into account the research area definition and TRL.
                
            b. **Multiple Classifications:**
                - Multiple research area classifications are permitted **only** if multiple areas are explicitly mentioned or can be directly inferred from the title and abstract.
                - Within one research area, there should not be multiple subclassifications. For example, if classifying under "Therapeutics", choose only one of "Discovery" or "Development", not both.
                
            c. **Exclude External References:**
                - Ignore any parts of the text that contain references to other resources, such as related work sections or citations to other research. Only consider the topics that are directly treated in the current research.

        3. **Classification Choices:**
            {research_area_options}

        4. **Additional Information for 'research_area' Category:**
            {research_area_additional_info}

        5. **Output Format:**
            - The output should be a JSON object with the following structure:
            {{
                "research_area": [list of research areas],
                "explanation": "explanation for the classification",
                "confidence": "float representing the confidence in the classification",
                "confidence_explanation": "explanation for the confidence"
            }}

        **Now, perform the classification for the following research paper given only these classification choices:**

        **Output:**
    """

    examples = """

    """

    options = """
    """

    if include_examples:
        return base_prompt + examples + options
    else:
        return base_prompt + options


def classify_research_area(title, abstract, model="gpt-4o-mini", include_examples=True):
    max_tries = 3
    tries = 0
    while tries < max_tries:
        try:
            prompt = generate_prompt(
                title=title, abstract=abstract, include_examples=include_examples
            )
            result = classify_research_json(prompt, model)
            if result is None:
                return None

            # Convert the result directly to the expected format
            parsed_result = {
                "research_area": result.get("research_area", []),
                "explanation": result.get("explanation", ""),
                "confidence": result.get("confidence", ""),
                "confidence_explanation": result.get("confidence_explanation", ""),
            }

            # Assert that all research area entries are in the valid categories
            research_area_categories = get_categories("Research Area")
            invalid_areas = [
                area
                for area in parsed_result["research_area"]
                if area not in research_area_categories
            ]
            assert not invalid_areas, f"The following research_area entries are not in the valid categories: {', '.join(invalid_areas)}"

            return parsed_result
        except Exception as e:
            tries += 1
            print(f"Error occurred: {str(e)}. Attempt {tries} of {max_tries}")
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None
