import json
import re

from utils.utils import (
    get_categories,
    find_closest_category,
    handle_invalid_entry,
)
from utils.llm_call import classify_research_json
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
            result = classify_research_json(prompt, model)

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
                    closest_match = find_closest_category(agent, "Infectious Agent")
                    if closest_match:
                        corrected_agents.append(closest_match)
                    else:
                        tries += 1
                        continue
                else:
                    corrected_agents.append(agent)

            parsed_result["infectious_agent"] = corrected_agents
            parsed_result["explanation"] = "\n\n".join(
                [
                    f"Classification: {item['infectious_agent']}\n"
                    f"Evidence:\n- {'\n- '.join(item['relevant_input_snippet'])}\n"
                    f"Explanation:\n{item['explanation']}\n"
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


def validate_infectious_agent_classification(title, abstract, prediction):
    max_tries = 3
    tries = 0

    while tries < max_tries:
        try:
            prompt = get_infectious_agent_validation_prompt(title, abstract, prediction)
            result = classify_research_json(prompt, "gpt-4o-mini")

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
                suggested_class = validation["correct_classification"]

                if suggested_class not in valid_categories:
                    # Try to find a close match
                    closest_match = find_closest_category(suggested_class, "Infectious Agent")
                    if closest_match:
                        print(
                            f"Correcting suggested classification from '{suggested_class}' to '{closest_match}'"
                        )
                        validation["correct_classification"] = closest_match
                    else:
                        print(f"Invalid suggested classification: {suggested_class}")
                        tries += 1
                        continue

            validation_response = {
                "is_correct": validation["is_correct"],
                "suggested_classification": validation.get("correct_classification"),
                "explanation": (
                    f"Validation Result: {'CORRECT' if validation['is_correct'] else 'INCORRECT'}\n\n"
                    f"Evidence:\n- {'\n- '.join(validation['evidence'])}\n\n"
                    f"Explanation:\n{validation['explanation']}\n\n"
                ),
            }

            return validation_response

        except Exception as e:
            print(f"Error in infectious agent validation: {str(e)}")
            tries += 1
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None

