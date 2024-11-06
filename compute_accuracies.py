import pandas as pd
from utils.metrics import (
    calculate_hierarchical_accuracy,
    get_misclassifications,
)


def compute_excel_accuracies(file_path, show_misclassifications=False):
    """
    Reads an Excel file and computes hierarchical accuracies for Ground Truth vs Prediction.

    Args:
        file_path (str): Path to the Excel file containing 'Ground Truth' and 'Prediction' columns
        show_misclassifications (bool): Whether to show detailed misclassification analysis

    Returns:
        dict: Overall accuracy statistics for each level
        dict: Detailed domain-level statistics
    """
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Initialize counters for overall statistics (now only 3 levels)
    total_matches = {1: 0, 2: 0, 3: 0}
    total_counts = {1: 0, 2: 0, 3: 0}
    domain_stats = {}
    skipped_entries = 0

    # Initialize misclassifications with known domains
    if show_misclassifications:
        misclassifications = {"Sector": [], "Research Area": [], "Infectious Agent": []}

    # Process each row
    print(f"Processing {len(df)} entries...")
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

            # Calculate accuracy for this entry
            matches_totals = calculate_hierarchical_accuracy(ground_truth, prediction)

            # Track misclassifications if requested
            if show_misclassifications:
                for domain in ["Sector", "Research Area", "Infectious Agent"]:
                    domain_misclassifications = get_misclassifications(
                        ground_truth, prediction, domain
                    )
                    if domain_misclassifications:
                        misclassifications[domain].append(
                            {
                                "index": index,
                                "title": row.get("Title", "N/A"),
                                "errors": domain_misclassifications,
                            }
                        )

            # Update domain-specific statistics
            for domain, data in matches_totals.items():
                if domain not in domain_stats:
                    domain_stats[domain] = {
                        "matches": {1: 0, 2: 0, 3: 0},
                        "totals": {1: 0, 2: 0, 3: 0},
                    }

                level_matches = data["matches"]
                level_totals = data["totals"]

                # Update domain stats
                for level in range(1, 4):  # Now only 3 levels
                    domain_stats[domain]["matches"][level] += level_matches.get(
                        level, 0
                    )
                    domain_stats[domain]["totals"][level] += level_totals.get(level, 0)

                    # Update overall stats
                    total_matches[level] += level_matches.get(level, 0)
                    total_counts[level] += level_totals.get(level, 0)

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            skipped_entries += 1
            continue

    if skipped_entries > 0:
        print(f"\nSkipped {skipped_entries} entries due to missing or invalid data")

    # Calculate overall accuracies
    overall_accuracies = {}
    print("\nOverall Accuracies:")
    for level in range(1, 4):  # Now only 3 levels
        if total_counts[level] > 0:
            accuracy = total_matches[level] / total_counts[level]
            overall_accuracies[level] = accuracy
            print(
                f"Level {level}: {accuracy:.2%} ({total_matches[level]}/{total_counts[level]})"
            )
        else:
            overall_accuracies[level] = 0
            print(f"Level {level}: N/A (no entries)")

    # Print domain-specific accuracies
    print("\nAccuracies by Domain:")
    for domain, stats in domain_stats.items():
        print(f"\nDomain: {domain}")
        for level in range(1, 4):  # Now only 3 levels
            matches = stats["matches"][level]
            totals = stats["totals"][level]
            if totals > 0:
                accuracy = matches / totals
                print(f"Level {level}: {accuracy:.2%} ({matches}/{totals})")
            else:
                print(f"Level {level}: N/A (no entries)")

    # Print misclassifications if requested
    if show_misclassifications:
        print("\nMisclassification Analysis:")
        for domain, errors in misclassifications.items():
            if errors:
                print(f"\n{domain} Domain Misclassifications:")
                print("-" * 80)

                # Group by error type
                error_types = {"incorrect": [], "additional": [], "missing": []}
                for error in errors:
                    for err in error["errors"]:
                        error_types[err["type"]].append(
                            (error["index"], error["title"], err)
                        )

                # Print each error type
                for error_type, type_errors in error_types.items():
                    if type_errors:
                        print(f"\n{error_type.upper()} PREDICTIONS:")
                        print("-" * 40)
                        for index, title, err in type_errors:
                            print(f"\nEntry {index}: {title}")
                            print(f"Details: {err['details']}")
                            if error_type in ["additional", "missing"]:
                                print(f"Correct Predictions: {err['correct_hits']}")
                            print(f"Ground Truth: {err['ground_truth']}")
                            print(f"Prediction: {err['prediction']}")
                            print("-" * 40)

    return overall_accuracies, domain_stats


if __name__ == "__main__":
    # Example usage
    file_path = "results/classification_results.xlsx"
    try:
        overall_accuracies, domain_stats = compute_excel_accuracies(file_path)
    except Exception as e:
        print(f"Error processing file: {e}")
