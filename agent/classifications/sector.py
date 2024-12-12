from utils.utils import get_categories, find_closest_category, handle_invalid_entry
from utils.llm_call import call_llm
from .prompts.sector_prompts import (
    get_classification_prompt,
    get_sector_validation_prompt,
)


def classify_sector(title, abstract, model="gpt-4o-mini"):
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
                or "sectors" not in result
                or not isinstance(result["sectors"], list)
            ):
                print(f"Invalid result format received: {result}")
                tries += 1
                continue

            # Parse the new format result
            parsed_result = {"sector": [], "explanation": ""}

            sectors = result["sectors"]
            valid_categories = get_categories("Sector")
            corrected_sectors = []

            for item in sectors:
                sector = item["sector"]
                # Handle case where sector might be a string
                if isinstance(sector, str):
                    sector = [sector]
                
                for s in sector:
                    if s not in valid_categories:
                        closest_matches = find_closest_category(s, "Sector", model=model)
                        if closest_matches:
                            corrected_sectors.extend(closest_matches)
                        else:
                            tries += 1
                            continue
                    else:
                        corrected_sectors.append(s)

            parsed_result["sector"] = corrected_sectors
            parsed_result["explanation"] = "\n\n".join(
                [
                    f"Sector: {item['sector']}\n\n"
                    f"Evidence:\n- {'\n- '.join(item['relevant_input_snippet'])}\n\n"
                    f"Explanation:\n{item['explanation']}"
                    for item in sectors
                ]
            )

            return parsed_result

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            tries += 1
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None


def validate_sector_classification(title, abstract, prediction, model="gpt-4o-mini"):
    max_tries = 3
    tries = 0


    while tries < max_tries:
        try:
            prompt = get_sector_validation_prompt(title, abstract, prediction)
            result = call_llm(prompt, model)

            # print(f"Validation prompt: {prompt}")
            # print(f"Validation result: {result}")

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
                valid_categories = get_categories("Sector")
                suggested_class = validation["correct_classification"]

                if suggested_class not in valid_categories:
                    # Try to find a close match
                    closest_match = find_closest_category(suggested_class, "Sector", model=model)
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
                    "\n\n".join([
                        f"Validation Result: {'CORRECT' if validation['is_correct'] else 'INCORRECT'}",
                        f"Sector: {validation['correct_classification']}\n" if not validation['is_correct'] and validation.get('correct_classification') else "",
                        f"Revision Explanation:{validation['explanation']}"
                    ])
                ),
            }

            return validation_response

        except Exception as e:
            print(f"Error in sector validation: {str(e)}")
            tries += 1
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None

