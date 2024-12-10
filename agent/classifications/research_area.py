from utils.utils import (
    get_categories,
    get_additional_info,
    find_closest_category,
    handle_invalid_entry,
)
from utils.llm_call import call_llm
from .prompts.research_area_prompts import (
    get_classification_prompt,
    get_therapeutics_validation_prompt,
)


def classify_research_area(title, abstract, model="gpt-4o-mini"):
    """Classify research area using relevant information."""
    max_tries = 3
    tries = 0
    while tries < max_tries:
        try:

            # Use the original classification prompt with added relevant info
            prompt = get_classification_prompt(
                title=title, 
                abstract=abstract, 

            )
            result = call_llm(prompt, model)

            # Rest of the validation and parsing logic
            if (
                result is None
                or not isinstance(result, dict)
                or "research_areas" not in result
                or not isinstance(result["research_areas"], list)
            ):
                print(f"Invalid result format received: {result}")
                tries += 1
                continue

            parsed_result = {"research_area": [], "explanation": ""}
            areas = result["research_areas"]
            valid_categories = get_categories("Research Area")
            corrected_areas = []

            for item in areas:
                area = item["research_area"]
                if area not in valid_categories:
                    closest_match = find_closest_category(area, "Research Area")
                    if closest_match:
                        corrected_areas.append(closest_match)
                    else:
                        tries += 1
                        continue
                else:
                    corrected_areas.append(area)

            parsed_result["research_area"] = corrected_areas

            parsed_result["explanation"] = "\n\n".join(
                [
                    f"Research Area: {item['research_area']}\n\n"
                    f"Evidence:\n" + "\n".join([
                        f"- {snippet['text_snippet']} (Keyword: {snippet['corresponding_keyword']}, from {snippet['keyword_paragraph']})"
                        for snippet in item['relevant_input_snippet']
                    ]) + "\n\n"
                    f"Explanation:\n{item['explanation']}"
                    for item in areas
                ]
            )


            return parsed_result

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            tries += 1
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None
            



def validate_therapeutics_classification(title, abstract, prediction):
    max_tries = 3
    tries = 0

    while tries < max_tries:
        try:
            prompt = get_therapeutics_validation_prompt(title, abstract, prediction)
            result = call_llm(prompt, "gpt-4o-mini")

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
                valid_categories = get_categories("Research Area")
                suggested_class = validation["correct_classification"]

                if suggested_class not in valid_categories:
                    # Try to find a close match
                    closest_match = find_closest_category(
                        suggested_class, "Research Area"
                    )
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
                        f"Research Area: {validation['correct_classification']}\n" if not validation['is_correct'] and validation.get('correct_classification') else "",
                        f"Revision Explanation:{validation['explanation']}"
                    ])
                ),
            }

            return validation_response

        except Exception as e:
            print(f"Error in therapeutics validation: {str(e)}")
            tries += 1
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None

