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

### 1. Classification Choices:
{research_area_options}

### 2. Additional Information about the Classification:
{research_area_additional_info}

### 3. Classification Rules:

#### a. Direct Mention or Inference:
- **Explicit Classification:** Assign research areas that are explicitly mentioned in the title or abstract.
- **Inferred Classification:** Assign research areas that can be directly inferred from the context.
- **Considerations:** Use research area definitions and Technology Readiness Levels (TRL) to determine the best fit.

#### b. Multiple Classifications:
- **Allowed:** Assign multiple research areas if they are explicitly mentioned or can be directly inferred.
- **Single Subcategory per Category:** Within a single main category (e.g., Therapeutics), assign only one subcategory (e.g., Discovery or Development).

#### c. Exclude External References:
- **Ignore:** References to other works, related studies, citations, or mentions of earlier work.
- **Focus:** Only on topics directly addressed in the current research.

#### d. Therapeutics Specifics:
- **Therapeutics / Development:**
  - **When to Assign:** If therapeutic products are actively being developed.
  - **Indicators:** Terms like "clinical trials", "Phase 1-3" 
- **Therapeutics / Discovery:**
  - **When to Assign:** If researching the feasibility of a product or idea is still in the early stages.
  - **Indicators:** Terms like "target identification", "lead optimization", "preclinical testing"
- **Specific Phases:** Use specific subclassifications (e.g., Phase 1, Phase 2, Phase 3) if explicitly mentioned.
- **Avoid Multiple Subclassifications:** Do not assign multiple subclassifications within the same main category.
  - **Valid:**
    - 3200 Research Area / Therapeutics / Development
    - 6100 Research Area / Operational / Operational
  - **Not Valid:**
    - 3200 Research Area / Therapeutics / Development
    - 3201 Research Area / Therapeutics / Development / Phase 1

    
### 4. Output Format:
Provide a JSON object with the following structure:
```json
{{
    "research_area": [list of research areas],
    "explanation": "Explanation for the classification",
    "confidence": float,  # Confidence score between 0 and 1
    "confidence_explanation": "Explanation for the confidence score"
}}


Now here is the research that needs to be classified:

- **Title:** {title}
- **Abstract:** {abstract}

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
