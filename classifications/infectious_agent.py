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

    base_prompt = f"""
        You are an AI specialized in classifying research papers on antimicrobial resistance into relevant infectious agents based on their title and abstract. Follow the instructions and specifications below to determine the appropriate infectious agent(s).

        **Instructions:**

        1. **Input:**
            - **Title:** {title}
            - **Abstract:** {abstract}

        2. **Classification Rules:**
            
            a. **Direct Mention or Inference:**
                - Only classify infectious agents that are directly mentioned or can be directly inferred from the title and abstract.
                - If no specific infectious agents are mentioned or inferred, use the appropriate category:
                    - "1900 Infectious Agent / Other / Other_Other" if the research is related to infectious agents but no specific agent is mentioned.
                    - "1901 Infectious Agent / Not Specified / Not Specified_InfectiousAgent" if it's unclear whether infectious agents are involved.
                    - "1902 Infectious Agent / Not Applicable / Not Applicable" if the research clearly does not involve or is not related to any infectious agents.
                
            b. **Multiple Classifications:**
                - Multiple infectious agent classifications are permitted **only** if multiple agents are explicitly mentioned or can be directly inferred from the title and abstract.
                
            c. **Exclude External References:**
                - Ignore any parts of the text that contain references to other resources, such as related work sections or citations to other research. Only consider the topics that are directly treated in the current research.
                
        3. **Classification Choices:**
            {infectious_agent_options}

        4. **Output Format:**
            - The output should be a JSON object with the following structure:
            {{
                "infectious_agent": [list of infectious agents],
                "explanation": "explanation for the classification",
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
            assert not invalid_agents, f"The following infectious_agent entries are not in the valid categories: {', '.join(invalid_agents)}"

            return parsed_result
        except Exception as e:
            tries += 1
            print(f"Error occurred: {str(e)}. Attempt {tries} of {max_tries}")
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None
