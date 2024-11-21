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
- **Primary Goal:** Focus on identifying the major goal or objective of the research.
- **Classification Assignment:** Assign research areas that are explicitly mentioned in the title/abstract or can be directly inferred from the context.

#### b. Preference for Single Classification:
- **Default to Single Classification:** Assign only one research area that best represents the main focus of the research.
- **When to Assign Multiple Classifications:**
  - **Rare Exceptions:** Assign multiple research areas only if the research equally and explicitly addresses multiple areas as main objectives.
  - **Equal Focus Required:** Multiple classifications are permitted only when multiple research areas are of equal significance and centrality in the research.
  - **Main Focus Rule:** If one research area is clearly the primary focus (e.g., constitutes 80% or more of the content), do not assign additional classifications even if other areas are mentioned.
  - **Avoid Overclassification:** Do not assign multiple classifications simply because multiple topics are mentioned; focus on the single main goal of the research.

#### c. Exclude External References:
- **Ignore:** References to other works, related studies, citations, or mentions of earlier work.
- **Focus:** Only on topics directly addressed in the current research.

#### d. Category-Specific Guidelines:
- **Translational Research:** Represents transitions between phases and should be classified based on the target phase, not as a separate category.
- **Diagnostics:** Refers specifically to the detection and identification of infectious agents to determine which agent is present.
- **Therapeutics Discovery:** Focuses on the detection and validation of therapeutic products in early stages.
- **Capacity Building:** Specifically refers to efforts aimed at refurbishing or increasing laboratory infrastructure and capabilities.

#### e. Therapeutics Specifics:


- **Therapeutics / Discovery:**
  - **When to Assign:** If researching the feasibility of a product in the lab or are in target assessment & validation.
  - **Indicators:** Terms like "target identification," "lead optimization," "preclinical trials/testing."
  - Don't be muisguided by the word "development" in the title/abstract. If the research is prior to clinical trials, classify as "Therapeutics / Discovery".
  - A transition to Development is only the case if the abstract mentions that it will move forward to clinical trials.
  
- **Therapeutics / Development:**
  - **When to Assign:** If therapeutic products are actively being tested in clinical trials.
  - **Indicators:** Terms like "clinical trials," "Phase 1-3."

- **Specific Phases:** Use specific subclassifications (e.g., Phase 1, Phase 2, Phase 3) only if explicitly mentioned.
- **Avoid Multiple Subclassifications Within the Same Category:** Do not assign multiple subclassifications within the same main category.
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
    "explanation": "Explanation for the classification. Include the words from the original text that proof the classification. If the explanation is telling that a certain research area is not explicitly mentioned, leave it out.",
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
