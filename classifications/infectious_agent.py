import json
import re

from utils.utils import (
    get_categories,
    parse_non_json_response,
    get_additional_info,
    extract_json,
)
from utils.llm_call import classify_research_json


def generate_prompt(title, abstract, include_examples=True):
    infectious_agent_options = get_categories("Infectious Agent")
    infectious_agent_additional_info = get_additional_info("Infectious Agent")

    base_prompt = f"""You are an AI specialized in classifying research papers on antimicrobial resistance into relevant infectious agents based on their title and abstract. Follow the instructions and specifications below to determine the appropriate infectious agent(s).

    **Instructions:**

    1. **Input:**
        - **Title:** {title}
        - **Abstract:** {abstract}

    2. **Classification Choices:**
        {infectious_agent_options}

    3. **Classification Rules:**

    a. **Direct Mention Only:**
        - Only classify infectious agents that are directly and concretely mentioned in the title and abstract; do not infer agents.
        - If examples of infectious agents are mentioned using phrases like “such as”, “including”, or “for example” but they arent the concrete focus of this research, do not consider them.
        - If infectious agents are introduced as examples in the beginning but are not the main focus or are not further discussed for concrete treatment, don't consider them.

        - Classification Based on Specificity:
            1. For bacteria without specified type:
                - If bacteria are mentioned without indicating Gram status, classify as:
                    - "1500 Infectious Agent / Bacteria / Bacteria / Not Specified_Bacteria"
                - Similarly use corresponding "Not Specified" classifications for other unspecified categories
            2. For specified Gram status:
                - If Gram negative bacteria are mentioned but not specified, or if specific unlisted, classify as:
                    - "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative"
                - Apply same rule for Gram positive/variable or other infectious agents (parasites, fungi, etc.) using appropriate "Other" category
            3. For unspecified infectious agents:
                - If research relates to infectious agents but the category is not mentioned, classify as:
                    - "1901 Infectious Agent / Not Specified / Not Specified_InfectiousAgent"
            4. For non-infectious agent research:
                - If research has no relation to infectious agents, classify as:
                    - "1902 Infectious Agent / Not Applicable / Not Applicable"
        - Specific vs. General Mentions:
            - For specific agents not in classification list, use appropriate "Other" category:
                - For unlisted Gram negative bacteria: "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative"
                - For unlisted Gram positive bacteria: "1513 Infectious Agent / Bacteria / Gram positive / Other Gram positive"

    b. **Multiple Classifications:**
        - Assign multiple classifications when:
            - Multiple infectious agents are explicitly mentioned as main topics
            - Research describes multiple efforts targeting different agents/categories
        - Use single classification when:
            - One agent/category is primary focus
            - Other mentions are only examples or peripheral references

    c. **Exclude External References:**
        - Ignore:
            - References to other resources or studies
            - Infectious agents mentioned only in context of other research
        - Focus only on infectious agents directly studied in current research


    4. **Output Format:**
        - The output should be a JSON object with the following structure:
        {{
            "infectious_agent": [list of infectious agents],
            "explanation": "explanation for the classification. Include the words from the original text that proof the classification. If the explanation is telling that a certain infectious agent is not explicitly mentioned, leave it out.",
            "confidence": "float representing the confidence in the classification",
            "confidence_explanation": "explanation for the confidence"
        }}

    **Now, perform the classification for the following research paper given only these classification choices:**

    **Output:**
    """
    return base_prompt


def classify_infectious_agent(
    title, abstract, model="gpt-4o-mini", include_examples=True
):
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
                "infectious_agent": result.get("infectious_agent", []),
                "explanation": result.get("explanation", ""),
                "confidence": result.get("confidence", ""),
                "confidence_explanation": result.get("confidence_explanation", ""),
            }

            agent_categories = get_categories("Infectious Agent")
            invalid_agents = [
                agent
                for agent in parsed_result["infectious_agent"]
                if agent not in agent_categories
            ]
            try:
                assert not invalid_agents, f"The following infectious_agent entries are not in the valid categories: {', '.join(invalid_agents)}"
                return parsed_result
            except AssertionError:
                # Add warning about invalid categories to prompt and retry
                invalid_warning = f"\n\nNote: The class(es) {', '.join(invalid_agents)} are not valid infectious agent categories (potentially the number is wrong)."
                prompt += invalid_warning
                tries += 1
                continue

        except Exception as e:
            tries += 1
            print(f"Error occurred: {str(e)}. Attempt {tries} of {max_tries}")
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None
