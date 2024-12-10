from utils.utils import (
    get_categories,
    find_closest_category,
    handle_invalid_entry,
)
from utils.llm_call import classify_research_json
from .prompts.research_area_prompts import (
    get_preselection_prompt,
    get_classification_prompt,
    get_therapeutics_validation_prompt,
    get_relevant_info_prompt,
    get_additional_info,
)


def get_potential_research_areas(title, abstract, model="gpt-4o-mini"):
    """First step: Identify potential research areas based on additional information."""
    max_tries = 3
    tries = 0
    
    while tries < max_tries:
        try:
            prompt = get_preselection_prompt(title, abstract)
            result = classify_research_json(prompt, model)

            if (
                result is None
                or not isinstance(result, dict)
                or "potential_areas" not in result
                or not isinstance(result["potential_areas"], list)
            ):
                handle_invalid_entry("Invalid preselection result format", f"Received: {result}")
                tries += 1
                continue
                
            # Validate potential areas against valid categories
            valid_categories = get_categories("Research Area")
            validated_areas = []
            
            for item in result["potential_areas"]:
                area = item["research_area"]
                if area not in valid_categories:
                    closest_match = find_closest_category(area, "Research Area")
                    if closest_match:
                        validated_areas.append({
                            "research_area": closest_match,
                            "reasoning": item["reasoning"]
                        })
                else:
                    validated_areas.append(item)
            
            return validated_areas
            
        except Exception as e:
            print(f"Error in research area preselection: {str(e)}")
            tries += 1
            if tries == max_tries:
                print("Max retries reached in preselection. Returning None.")
                return None


def get_relevant_info(title, abstract, potential_areas, model="gpt-4o-mini"):
    """Extract relevant information from additional info based on the context."""
    try:
        prompt = get_relevant_info_prompt(title, abstract, potential_areas)
        result = classify_research_json(prompt, model)
        return result.get("relevant_info", "No relevant information extracted")
        
    except Exception as e:
        print(f"Error extracting relevant information: {e}")
        return get_additional_info("Research Area")


def classify_research_area(title, abstract, model="gpt-4o-mini", potential_areas=None):
    """Classify research area using relevant information."""
    max_tries = 3
    tries = 0
    while tries < max_tries:
        try:
            # First get relevant information based on potential areas
            relevant_info = get_relevant_info(title, abstract, potential_areas, model)
            
            # Use the original classification prompt with added relevant info
            prompt = get_classification_prompt(
                title=title, 
                abstract=abstract, 
                potential_areas=potential_areas,
                relevant_info=relevant_info
            )
            result = classify_research_json(prompt, model)

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

            explanation_parts = []
            for item in areas:
                explanation_parts.append(
                    f"Research Area: {item['research_area']}\n"
                    f"Evidence: {', '.join(item['relevant_input_snippet'])}\n"
                    f"Explanation: {item['explanation']}\n"
                )
            parsed_result["explanation"] = "\n".join(explanation_parts)

            return parsed_result

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            tries += 1
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None


def validate_therapeutics_classification(title, abstract, prediction):
    """Validate therapeutics classifications."""
    max_tries = 3
    tries = 0

    while tries < max_tries:
        try:
            prompt = get_therapeutics_validation_prompt(title, abstract, prediction)
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
                valid_categories = get_categories("Research Area")
                suggested_class = validation["correct_classification"]

                if suggested_class not in valid_categories:
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
                    f"Validation Result: {'CORRECT' if validation['is_correct'] else 'INCORRECT'}\n\n"
                    f"Evidence:\n- {'\n- '.join(validation['evidence'])}\n\n"
                    f"Explanation:\n{validation['explanation']}\n\n"
                ),
            }

            return validation_response

        except Exception as e:
            print(f"Error in therapeutics validation: {str(e)}")
            tries += 1
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None
