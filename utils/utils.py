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

    print("\nPress Enter to continue or type 'exit' to stop...")
    response = input()

    if response.lower() == "exit":
        raise SystemExit("Program terminated by user")


def find_closest_category(invalid_category, domain, threshold=0.92):
    """
    Find the closest matching valid category using fuzzy matching.
    """
    valid_categories = get_categories(domain)
    matches = get_close_matches(
        invalid_category, valid_categories, n=3, cutoff=threshold
    )

    if matches:
        print(
            f"\nFound close matches for invalid category '{invalid_category}' in {domain}:"
        )
        for i, match in enumerate(matches, 1):
            similarity = SequenceMatcher(None, invalid_category, match).ratio()
            print(f"{i}. '{match}' (similarity: {similarity:.2%})")
        return matches[0]  # Return the closest match

    print(
        f"\nNo close match found for invalid category '{invalid_category}' in {domain}"
    )
    return None


def format_validation_result(original_classification, validation_result, classification_type):
    """Format validation result with a concise one-line status message.
    
    Args:
        original_classification (list): List of original classifications
        validation_result (dict): Validation result containing is_correct and suggested_classification
        classification_type (str): Type of classification (Sector, Research Area, or Infectious Agent)
        
    Returns:
        str: Formatted validation message
    """
    is_correct = validation_result["is_correct"]
    suggested = validation_result.get("suggested_classification")
    
    if is_correct:
        return f"[✓] {classification_type}: CORRECT ({', '.join(original_classification)})"
    else:
        original = ', '.join(original_classification)
        updated = suggested if suggested else 'No valid suggestion'
        return f"[✗] {classification_type}: INCORRECT ({original} -> {updated})"


def compute_average(results, classification_type, threshold=0.9):
    if not results:
        return [], ""
    classifications = [r.get(classification_type, []) for r in results]
    flat_classifications = [
        item for sublist in classifications for item in sublist
    ]
    classification_counts = Counter(flat_classifications)
    most_common = []
    for c, count in classification_counts.items():
        percentage = count / len(results)
        if percentage >= threshold:
            most_common.append(c)
    if not most_common:
        most_common = [
            f"0000 {classification_type.replace('_', ' ').title()} / Uncertain ({', '.join([f'{c}: {count/len(results):.2%}' for c, count in classification_counts.items()])})"
        ]
    explanation = results[0].get("explanation", "") if results else ""
    return most_common, explanation
        

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
