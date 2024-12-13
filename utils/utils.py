import pandas as pd
import json
import re
import os
from difflib import get_close_matches, SequenceMatcher
from collections import Counter

file_path = "assets/Dashboard Categories to be used.xlsx"
docs_path = "assets/docs"


def get_dataframe():
    return pd.read_excel(file_path)


def filter_entries_by_category(category):
    df = get_dataframe()
    filtered_df = df[df["Text"].str.contains(category, case=False, na=False)]
    return filtered_df.shape[0], filtered_df["Text"].tolist()


def get_categories(domain):
    count, entries = filter_entries_by_category(domain)
    return entries


def parse_non_json_response(response, category_key):
    category_match = re.search(f'"{category_key}":\\s*(\\[.*?\\])', response)
    explanation_match = re.search(r'"explanation":\s*"(.*?)"', response)
    confidence_match = re.search(r'"confidence":\s*(\[.*?\])', response)
    confidence_explanations_match = re.search(
        r'"confidence_explanations?":\s*(\[.*?\])', response
    )

    category = json.loads(category_match.group(1)) if category_match else []
    explanation = explanation_match.group(1) if explanation_match else ""
    confidence = json.loads(confidence_match.group(1)) if confidence_match else []
    confidence_explanations = (
        json.loads(confidence_explanations_match.group(1))
        if confidence_explanations_match
        else []
    )

    return {
        category_key: category,
        "explanation": explanation,
        "confidence": confidence,
        "confidence_explanations": confidence_explanations,
    }


def get_additional_info(category):
    """
    Read additional information from text files in the assets/docs directory.

    :param category: The category name (e.g., 'Sector', 'Research Area', 'Infectious Agent')
    :return: A string containing the additional information
    """
    filename = f"{category.lower().replace(' ', '_')}.txt"
    file_path = os.path.join(docs_path, filename)

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        return f"No additional information found for {category}."


def get_keywords(category):
    filename = f"{category.lower().replace(' ', '_')}_keywords.txt"
    file_path = os.path.join(docs_path, filename)
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        return f"No keywords found for {category}."


def extract_json(text):
    json_match = re.search(r"\{[\s\S]*\}", text)
    return json_match.group(0) if json_match else None


def handle_invalid_entry(message, details=None):
    """
    Handle invalid entries by pausing execution and showing error details.

    Args:
        message (str): Main error message
        details (any): Additional error details
    """
    print("\n" + "=" * 50)
    print("INVALID ENTRY DETECTED!")
    print("=" * 50)
    print(f"Reason: {message}")

    if details:
        print("\nDetails:")
        print(details)

    # print("\nPress Enter to continue or type 'exit' to stop...")
    # response = input()

    # if response.lower() == "exit":
    #     raise SystemExit("Program terminated by user")


def find_closest_category(invalid_categories: str, domain: str, model="gpt-4o-mini") -> list:
    """Find the closest matching valid categories using LLM.
    
    Args:
        invalid_categories (str): String potentially containing multiple invalid categories
        domain (str): The domain to check against ("Sector", "Research Area", or "Infectious Agent")
        model (str): The LLM model to use
        
    Returns:
        list: List of closest matching valid categories, or empty list if no matches found
    """
    valid_categories = get_categories(domain)
    
    # Split input string into potential multiple categories
    categories_to_check = [cat.strip() for cat in invalid_categories.split(',')]
    
    prompt = f"""
You are a classification expert. Given a potentially invalid category or categories and a list of valid categories, determine which valid category or categories were likely meant by the invalid one(s).

Invalid Categories:
{categories_to_check}

Valid Categories:
{valid_categories}

Based on naming patterns, domain knowledge, and semantic similarity, identify which valid categories from the list above were most likely intended.

Output Format:
```json
{{
    "matches": [
        {{
            "invalid_category": "str -> original invalid category",
            "closest_match": "str -> exact match from valid categories list or null if no clear match",
            "explanation": "str -> brief explanation of the match"
        }}
    ]
}}
```
"""

    try:
        from utils.llm_call import call_llm
        result = call_llm(prompt, model)
        
        if result and isinstance(result, dict) and "matches" in result:
            closest_matches = []
            
            for match in result["matches"]:
                closest_match = match.get("closest_match")
                if closest_match in valid_categories:
                    closest_matches.append(closest_match)
                else:
                    print(f"\nNo confident match found for '{match['invalid_category']}'")
                    print(f"Closest match: {closest_match}")
            
            return closest_matches if closest_matches else None
                
    except Exception as e:
        print(f"Error in category matching: {e}")
        return None

    return None


def print_validation_result(original_classification, validation_result, classification_type):
    """Format validation result with a concise one-line status message."""
    is_correct = validation_result["is_correct"]
    suggested = validation_result.get("suggested_classification")

    # Ensure original_classification is a list
    if isinstance(original_classification, str):
        original_classification = [original_classification]
    
    # Ensure suggested is a list
    if suggested and isinstance(suggested, str):
        suggested = [suggested]

    if is_correct:
        print(f"[✓] {classification_type}: CORRECT ({', '.join(original_classification)})")
    else:
        original = ', '.join(original_classification)
        updated = ', '.join(suggested) if suggested else 'No valid suggestion'
        print(f"[✗] {classification_type}: INCORRECT ({original} -> {updated})")


def compute_average(results, classification_type, threshold=0.9):
    if not results:
        return [], ""
    
    # Get all classifications and their counts
    classifications = [r.get(classification_type, []) for r in results]
    flat_classifications = [item for sublist in classifications for item in sublist]
    classification_counts = Counter(flat_classifications)
    
    # Get explanations
    explanations = [r.get("explanation", "") for r in results if r.get("explanation")]
    combined_explanation = "\n".join(explanations)
    
    # Check if any classification meets the threshold
    total_runs = len(results)
    most_common = []
    uncertainty_info = []
    
    for classification, count in classification_counts.items():
        percentage = count / total_runs
        if percentage >= threshold:
            most_common.append(classification)
        uncertainty_info.append(f"{classification}: {percentage:.2%}")
    
    # If no classification meets threshold, mark as uncertain
    if not most_common:
        domain = classification_type.replace("_", " ").title()
        most_common = [f"0000 {domain} / Uncertain ({', '.join(uncertainty_info)})"]
    
    return most_common, combined_explanation


def print_review_result(review_result, classification_type):
    """Format review result with a concise one-line status message.
    
    Args:
        review_result (dict): Review result containing status, reason, and analysis
        classification_type (str): Type of classification (Sector, Research Area, or Infectious Agent)
    """
    if review_result["status"] == "uncertain":
        print(f"[?] {classification_type} Review: UNCERTAIN")
        print(f"    Reason: {review_result['reason']}")
    else:
        print(f"[✓] {classification_type} Review: CERTAIN")


def print_classification_results(original_classification, validation_result, review_result, classification_type):
    """Print both validation and review results for a classification type."""
    print_validation_result(original_classification, validation_result, classification_type)
    print_review_result(review_result, classification_type)


if __name__ == "__main__":
    print(f"Sector Categories: (length: {len(get_categories('Sector'))})")
    print(get_categories("Sector"))
    # print("\nAdditional Sector Information:")
    # print(get_additional_info("Sector"))

    print(
        f"\nResearch Area Categories: (length: {len(get_categories('Research Area'))})"
    )
    print(get_categories("Research Area"))
    # print("\nAdditional Research Area Information:")
    # print(get_additional_info("Research Area"))

    print(
        f"\nInfectious Agent Categories: (length: {len(get_categories('Infectious Agent'))})"
    )
    print(get_categories("Infectious Agent"))
    # print("\nAdditional Infectious Agent Information:")
    # print(get_additional_info("Infectious Agent"))