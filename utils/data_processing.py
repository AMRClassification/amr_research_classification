# utils/data_processing.py
import pandas as pd


def parse_classification_categories(category_string):
    """
    Parses the classification string into a dict mapping domain to list of category levels.
    Returns a dict of the form {domain: [tuple of levels]}
    """
    categories = category_string.strip().split("\n")
    domain_categories = {}

    for category in categories:
        levels = category.strip().split(" / ")
        if levels:
            # Extract domain from first level
            first_level = levels[0].strip()
            if " " in first_level:
                code, domain = first_level.split(" ", 1)
                domain = domain.strip()
            else:
                domain = first_level.strip()
            # Append the levels (excluding the code) to the domain's list as tuples
            levels_without_code = levels.copy()
            # Remove code from the first level
            if " " in levels_without_code[0]:
                levels_without_code[0] = levels_without_code[0].split(" ", 1)[1].strip()
            # Convert list of levels to tuple for immutability and hashability
            levels_tuple = tuple(levels_without_code)
            domain_categories.setdefault(domain, []).append(levels_tuple)
    return domain_categories


def process_excel_data(df):
    """
    Processes Excel data and extracts ground truth and prediction pairs.

    Args:
        df (pandas.DataFrame): DataFrame containing 'Ground Truth' and 'Prediction' columns

    Returns:
        tuple: (ground_truths, predictions, skipped_entries)
    """
    ground_truths = []
    predictions = []
    skipped_entries = 0

    for index, row in df.iterrows():
        try:
            ground_truth = row["Ground Truth"]
            prediction = row["Prediction"]

            # Skip if either value is missing or invalid
            if pd.isna(ground_truth) or pd.isna(prediction):
                skipped_entries += 1
                continue

            # Convert to string if necessary
            ground_truth = (
                str(ground_truth) if not isinstance(ground_truth, str) else ground_truth
            )
            prediction = (
                str(prediction) if not isinstance(prediction, str) else prediction
            )

            # Skip if either string is empty after conversion
            if not ground_truth.strip() or not prediction.strip():
                skipped_entries += 1
                continue

            ground_truths.append(ground_truth)
            predictions.append(prediction)

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            skipped_entries += 1
            continue

    return ground_truths, predictions, skipped_entries
