import json

from utils.utils import (
    get_categories,
    parse_non_json_response,
    get_additional_info,
    extract_json,
)
from utils.llm_call import classify_research_json


def generate_prompt(title, abstract, include_examples=True):
    sector_options = get_categories("Sector")
    sector_additional_info = get_additional_info("Sector")

    base_prompt = f"""
        You are an AI specialized in classifying research papers on antimicrobial resistance into relevant sectors based on their title and abstract. Follow the instructions and specifications below to determine the appropriate sector(s).

        **Instructions:**

        1. **Input:**
            - **Title:** {title}
            - **Abstract:** {abstract}

        2. **Classification Rules:**
            
            a. **Direct Mention or Inference:**
                - Only classify sectors that are directly mentioned or can be directly inferred from the title and abstract.
                
            b. **Sector Selection When Multiple Apply:**
                - If the research involves infectious agents applicable to multiple sectors, prioritize and select the sector that is explicitly mentioned in the title or abstract.
                
            c. **Default Classification:**
                - If no sector is explicitly mentioned, default the classification to the **"Human"** sector.
                
            d. **Multiple Classifications:**
                - Multiple sector classifications are permitted **only** if multiple sectors are explicitly mentioned or can be directly inferred from the title and abstract.
                
            e. **Exclude External References:**
                - Ignore any parts of the text that contain references to other resources, such as related work sections or citations to other research. Only consider the topics that are directly treated in the current research.
                
            f. **Animal Testing for Human Purposes:**
                - If the research involves **animal tests** to study the effects or treatments of AMR **for humans**, classify the sector as **"Human"** instead of **"Animal"**.
                - Indicators may include phrases like "animal models to assess human health impacts," "testing on animals for human applications," "from vegetable oils," "studying ecologic-based structures," etc.
                
            g. **Animal-Derived Derivatives for Human Use:**
                - If the research mentions that any **derivatives** (instead of "ingredients") are **derived from animals** but are intended for **human** applications, classify the sector as **"Human"**.
                - Indicators may include phrases like "animal-derived compounds for human therapy," "using animal-sourced materials in human medicine," etc.

        3. **Classification Choices:**
            {sector_options}

        4. **Additional Information for 'sector' Category:**
            {sector_additional_info}

        5. **Output Format:**
            - The output should be a JSON object with the following structure:
            {{
                "sector": [list of sectors],
                "explanation": "explanation for the classification. Include the words from the original text that proof the classification. If the explanation is telling that a certain sector is not explicitly mentioned, leave it out.",
                "confidence": "float representing the confidence in the classification",
                "confidence_explanation": "explanation for the confidence"
            }}

        **Now, perform the classification for the following research paper given only these classification choices:**

        **Output:**
    """

    return base_prompt


def classify_sector(title, abstract, model="gpt-4o-mini", include_examples=True):
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

            parsed_result = {
                "sector": result.get("sector", []),
                "explanation": result.get("explanation", ""),
                "confidence": result.get("confidence", ""),
                "confidence_explanation": result.get("confidence_explanation", ""),
            }

            sector_categories = get_categories("Sector")
            invalid_sectors = [
                sector
                for sector in parsed_result["sector"]
                if sector not in sector_categories
            ]
            assert not invalid_sectors, f"The following sector entries are not in the valid categories: {', '.join(invalid_sectors)}"

            return parsed_result
        except Exception as e:
            tries += 1
            print(f"Error occurred: {str(e)}. Attempt {tries} of {max_tries}")
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None
