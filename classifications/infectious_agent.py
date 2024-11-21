import json
import re

from utils.utils import (
    get_categories,
    parse_non_json_response,
    get_additional_info,
    extract_json,
)
from utils.llm_call import classify_research_json


def generate_prompt(title, abstract, include_examples=True, use_response_format=False):
    infectious_agent_options = get_categories("Infectious Agent")
    infectious_agent_additional_info = get_additional_info("Infectious Agent")

    if use_response_format:
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
                - Multiple infectious agent classifications are permitted **only** if multiple agents are explicitly mentioned or can be directly inferred as being the main topic.
                
            c. **Exclude External References:**
                - Ignore any parts of the text that contain references to other resources, such as related work sections or citations to other research. Only consider the topics that are the direct topic of this current research at hand.
                
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
"""

    else:
        base_prompt = f"""
            You are an AI specialized in classifying research papers on antimicrobial resistance into relevant infectious agents based on their title and abstract. Follow the instructions and specifications below to determine the appropriate infectious agent(s).

        **Instructions:**

        1. **Input:**
            - **Title:** {title}
            - **Abstract:** {abstract}

        2. **Classification Choices:**
            {infectious_agent_options}

        3 **Classification Rules:**
            
        a. **Direct Mention or Inference:**
            - Only classify infectious agents that are directly mentioned or can be directly inferred from the title and abstract.
            - For research mentioning a specific category without details:
                - Use "1500 Infectious Agent / Bacteria / Bacteria / Not Specified_Bacteria" if bacteria are mentioned without specifying which ones
                - Use "1801 Infectious Agent / Virus / Virus / Not Specified_Virus" if viruses are mentioned without specifying which ones
                - Use "1700 Infectious Agent / Parasite / Parasite / Not Specified_Parasite" if parasites are mentioned without specifying which ones
                - Use "1600 Infectious Agent / Fungus / Fungus / Not Specified_Fungus" if fungi are mentioned without specifying which ones
            - For research mentioning specific agents not in our categories:
                - Use "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative" or similar bacterial categories for uncategorized bacteria
                - Use "1802 Infectious Agent / Virus / Virus / Other_Virus" for uncategorized viruses
                - Use "1702 Infectious Agent / Parasite / Other_Parasite" for uncategorized parasites
                - Use "1602 Infectious Agent / Fungus / Other_Fungus" for uncategorized fungi
            - For other cases:
                - Use "1901 Infectious Agent / Not Specified / Not Specified_InfectiousAgent" if no infectious agent is specified at all
                - Use "1900 Infectious Agent / Other / Other_Other" if the infectious agent is specified but doesn't fit into bacteria, virus, parasite, or fungus categories
                - Use "1902 Infectious Agent / Not Applicable / Not Applicable" if the research is not related to any infectious agents
        b. **Multiple Classifications:**
            - Multiple infectious agent classifications are permitted **only** if multiple agents are explicitly mentioned or can be directly inferred as being the main topic.
            
        c. **Exclude External References:**
            - Ignore any parts of the text that contain references to other resources, such as related work sections, citations to other research or earlier work. Only consider the topics that are the direct topic of this current research at hand.
                
            
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
