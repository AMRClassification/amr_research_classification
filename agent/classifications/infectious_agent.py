import json
import re

from utils.utils import (
    get_categories,
    find_closest_category,
    handle_invalid_entry,
)
from utils.llm_call import call_llm
from .prompts.infectious_agent_prompts import (
    get_classification_prompt,
    get_infectious_agent_validation_prompt,
)



def classify_infectious_agent(title, abstract, model="gpt-4o-mini"):
    max_tries = 3
    tries = 0
    while tries < max_tries:
        try:
            prompt = get_classification_prompt(title=title, abstract=abstract)
            result = call_llm(prompt, model)

            # Add validation for result
            if (
                result is None
                or not isinstance(result, dict)
                or "infectious_agents" not in result
                or not isinstance(result["infectious_agents"], list)
            ):
                handle_invalid_entry("Invalid result format", f"Received: {result}")
                tries += 1
                continue

            # Parse the new format result
            parsed_result = {"infectious_agent": [], "explanation": ""}

            agents = result["infectious_agents"]

            # Validate each agent against valid categories
            valid_categories = get_categories("Infectious Agent")
            corrected_agents = []

            for item in agents:
                agent = item["infectious_agent"]
                
                if agent not in valid_categories:
                    closest_matches = find_closest_category(agent, "Infectious Agent", model=model)
                    if closest_matches:
                        corrected_agents.extend(closest_matches)
                    else:
                        tries += 1
                        continue
                else:
                    corrected_agents.append(agent)

            parsed_result["infectious_agent"] = corrected_agents
            parsed_result["explanation"] = "\n\n".join(
                [
                    f"Infectious Agent: {item['infectious_agent']}\n\n"
                    f"Evidence:\n- {'\n- '.join(item['relevant_input_snippet'])}\n"
                    f"Explanation:\n{item['explanation']}\n\n"
                    for item in agents
                ]
            )

            return parsed_result

        except Exception as e:
            handle_invalid_entry(
                "Unexpected error in classification", f"Error: {str(e)}"
            )
            tries += 1
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None


def validate_infectious_agent_classification(title, abstract, prediction, model="gpt-4o-mini"):
    max_tries = 3
    tries = 0

    while tries < max_tries:
        try:

            prompt = get_infectious_agent_validation_prompt(title, abstract, str(prediction))
            result = call_llm(prompt, model)

            if (
                result is None
                or not isinstance(result, dict)
                or "validation_result" not in result
            ):
                handle_invalid_entry(
                    "Invalid validation result format", f"Received: {result}"
                )
                tries += 1
                continue

            validation = result["validation_result"]

            # Validate suggested classification against valid categories
            if not validation["is_correct"] and validation.get(
                "correct_classification"
            ):
                valid_categories = get_categories("Infectious Agent")
                suggested_classes = validation["correct_classification"]

                # Handle suggested classifications
                corrected_classes = []

                for suggested_class in suggested_classes:
                    if suggested_class not in valid_categories:
                        # Try to find a close match
                        closest_match = find_closest_category(suggested_class, "Infectious Agent", model=model)
                        if closest_match:
                            print(
                                f"Correcting suggested classification from '{suggested_class}' to '{', '.join(closest_match)}'"
                            )
                            corrected_classes.extend(closest_match)
                        else:
                            print(f"Invalid suggested classification: {suggested_class}")
                            tries += 1
                            continue
                    else:
                        corrected_classes.append(suggested_class)

                if corrected_classes:
                    validation["correct_classification"] = corrected_classes
                    

            validation_response = {
                "is_correct": validation["is_correct"],
                "suggested_classification": validation.get("correct_classification"),
                "explanation": (
                    "\n\n".join([
                        f"Validation Result: {'CORRECT' if validation['is_correct'] else 'INCORRECT'}",
                        f"Infectious Agent: {validation['correct_classification']}\n" if not validation['is_correct'] and validation.get('correct_classification') else "",
                        f"Revision Explanation:{validation['explanation']}"
                    ])
                ),
            }

            return validation_response

        except Exception as e:
            print(f"Error in infectious agent validation: {str(e)}")
            tries += 1
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None

def map_less_relevant_infectious_agents_to_stain(agents):
    """Map specific infectious agents to their broader categories.
    
    Args:
        agents: String or list of strings representing infectious agent categories
        
    Returns:
        String or list of strings with mapped categories
    """
    # Convert single string to list for consistent processing
    input_was_string = isinstance(agents, str)
    agents_list = [agents] if input_was_string else agents.copy()
    
    # Mapping rules
    gram_negative_mappings = {
        "1506 Infectious Agent / Bacteria / Gram negative / Burkholderia spp.": "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative",
        "1505 Infectious Agent / Bacteria / Gram negative / Chlamydia": "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative", 
        "1504 Infectious Agent / Bacteria / Gram negative / Helicobacter spp.": "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative",
        "1505 Infectious Agent / Bacteria / Gram negative / Vibrio spp.": "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative"
    }
    
    gram_positive_mappings = {
        "1515 Infectious Agent / Bacteria / Gram positive / Clostridium spp.": "1513 Infectious Agent / Bacteria / Gram positive / Other Gram positive",
        "1515 Infectious Agent / Bacteria / Gram positive / Corynebacterium spp.": "1513 Infectious Agent / Bacteria / Gram positive / Other Gram positive",
        "1514 Infectious Agent / Bacteria / Gram positive / Enterococcus spp.": "1513 Infectious Agent / Bacteria / Gram positive / Other Gram positive",
        "1514 Infectious Agent / Bacteria / Gram positive / Staphylococcus spp.": "1513 Infectious Agent / Bacteria / Gram positive / Other Gram positive",
        "1514 Infectious Agent / Bacteria / Gram positive / Streptococcus spp.": "1513 Infectious Agent / Bacteria / Gram positive / Other Gram positive"
    }
    
    gram_variable_mappings = {
        "1524 Infectious Agent / Bacteria / Gram variable / Mycobacterium spp": "1523 Infectious Agent / Bacteria / Gram variable / Other Gram variable",
        "1524 Infectious Agent / Bacteria / Gram variable / Mycoplasma spp.": "1523 Infectious Agent / Bacteria / Gram variable / Other Gram variable"
    }

    mapped_agents = []
    for agent in agents_list:
        mapped_agent = agent
        for mappings in [gram_negative_mappings, gram_positive_mappings, gram_variable_mappings]:
            if agent in mappings:
                mapped_agent = mappings[agent]
                break
        mapped_agents.append(mapped_agent)

    # Return string if input was string, otherwise return list
    return mapped_agents[0] if input_was_string else mapped_agents