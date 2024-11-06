import pandas as pd
import json
import re
import os

file_path = "assets/3. Dashboard Categories_04.04.24.xlsx"
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


def extract_json(text):
    json_match = re.search(r"\{[\s\S]*\}", text)
    return json_match.group(0) if json_match else None


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
