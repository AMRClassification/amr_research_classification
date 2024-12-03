import json

from utils.utils import get_categories, find_closest_category, handle_invalid_entry
from utils.llm_call import classify_research_json
from .prompts.sector_prompts import (
    get_classification_prompt,
    get_sector_validation_prompt,
)


def validate_sector_classification(title, abstract, prediction):
    max_tries = 3
    tries = 0

    while tries < max_tries:
        try:
            prompt = get_sector_validation_prompt(title, abstract, prediction)
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
                valid_categories = get_categories("Sector")
                suggested_class = validation["correct_classification"]

                if suggested_class not in valid_categories:
                    # Try to find a close match
                    closest_match = find_closest_category(suggested_class, "Sector")
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
            print(f"Error in sector validation: {str(e)}")
            tries += 1
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None


def classify_sector(title, abstract, model="gpt-4o-mini"):
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
                or "sectors" not in result
                or not isinstance(result["sectors"], list)
            ):
                print(f"Invalid result format received: {result}")
                tries += 1
                continue

            # Parse the new format result
            parsed_result = {"sector": [], "explanation": ""}

            sectors = result["sectors"]
            parsed_result["sector"] = [item["sector"] for item in sectors]
            parsed_result["explanation"] = "\n\n".join(
                [
                    f"Classification: {item['sector']}\n"
                    f"Evidence:\n- {'\n- '.join(item['relevant_input_snippet'])}\n"
                    f"Explanation:\n{item['explanation']}\n"
                    for item in sectors
                ]
            )

            # Validate sectors
            sector_categories = get_categories("Sector")
            invalid_sectors = [
                sector
                for sector in parsed_result["sector"]
                if sector not in sector_categories
            ]
            if invalid_sectors:
                print(f"Invalid sectors found: {invalid_sectors}")
                tries += 1
                continue

            # After parsing the result, add validation
            validation_result = validate_sector_classification(
                title, abstract, str(result)
            )

            if validation_result:
                if not validation_result["is_correct"]:
                    print("\nSector Classification Validation Failed!")
                    print(validation_result["explanation"])

                    if validation_result["suggested_classification"]:
                        print(
                            f"\nUpdating classification to: {validation_result['suggested_classification']}"
                        )
                        parsed_result["sector"] = [
                            validation_result["suggested_classification"]
                        ]
                        parsed_result["explanation"] = (
                            "Original Classification:\n"
                            + parsed_result["explanation"]
                            + "\n\nValidation Result:\n"
                            + validation_result["explanation"]
                        )
                else:
                    print("\nSector Classification Validated Successfully!")
                    parsed_result["explanation"] = (
                        parsed_result["explanation"]
                        + "\n\nValidation Result:\n"
                        + validation_result["explanation"]
                    )

            return parsed_result

        except (KeyError, TypeError) as e:
            print(f"Error parsing result: {str(e)}")
            print(f"Received result: {result}")
            tries += 1
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            tries += 1
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None
