import pandas as pd
from utils.metrics import (
    calculate_hierarchical_accuracy,
    get_misclassifications,
    summarize_predictions,
    analyze_error_patterns,
    analyze_class_errors,
    analyze_detailed_class_errors,
    analyze_classification_constellations_errors,
)
from utils.visualizations import plot_error_constellations
import os


def compute_excel_accuracies(
    file_path, show_misclassifications=False, save_plots=False
):
    """
    Reads an Excel file and computes hierarchical accuracies for Ground Truth vs Prediction.

    Args:
        file_path (str): Path to the Excel file containing 'Ground Truth' and 'Prediction' columns
        show_misclassifications (bool): Whether to show detailed misclassification analysis
        save_plots (bool): Whether to save plots of error constellations

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

    # Initialize summary counters
    total_summary = {"correct": 0, "additional": 0, "missing": 0, "incorrect": 0}

    # Initialize error pattern tracking
    total_error_patterns = {
        domain: {"incorrect": {}, "additional": {}, "missing": {}}
        for domain in ["Sector", "Research Area", "Infectious Agent"]
    }

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

            # Update summary statistics
            summary = summarize_predictions(
                ground_truth,
                prediction,
                title=row.get("Title", "N/A"),
                print_details=show_misclassifications,
            )
            for key in total_summary:
                total_summary[key] += summary[key]

            # Analyze error patterns
            error_patterns = analyze_error_patterns(
                ground_truth, prediction, print_details=False
            )
            for domain in total_error_patterns:
                for error_type in ["incorrect", "additional", "missing"]:
                    for category, count in error_patterns[domain][error_type].items():
                        total_error_patterns[domain][error_type][category] = (
                            total_error_patterns[domain][error_type].get(category, 0)
                            + count
                        )

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

    # Print overall error patterns
    # print("\nOverall Error Patterns:")
    # print("=" * 80)
    # for domain in total_error_patterns:
    #     print(f"\n{domain}:")
    #     print("-" * 80)
    #     for error_type in ["incorrect", "additional", "missing"]:
    #         errors = total_error_patterns[domain][error_type]
    #         if errors:
    #             print(f"\n{error_type.capitalize()} Predictions (Top 10):")
    #             sorted_errors = sorted(
    #                 errors.items(), key=lambda x: x[1], reverse=True
    #             )[:10]
    #             for category, count in sorted_errors:
    #                 print(f"  {category}: {count} errors")

    # Add class-specific error analysis
    if show_misclassifications:
        # Initialize class-specific error tracking
        total_class_stats = {"added_wrongly": {}, "missing_wrongly": {}}

        # Analyze each entry
        for index, row in df.iterrows():
            ground_truth = row["Ground Truth"]
            prediction = row["Prediction"]

            if pd.isna(ground_truth) or pd.isna(prediction):
                continue

            class_stats = analyze_class_errors(
                ground_truth, prediction, print_details=False
            )

            # Aggregate the statistics
            for error_type in ["added_wrongly", "missing_wrongly"]:
                for category, count in class_stats[error_type].items():
                    total_class_stats[error_type][category] = (
                        total_class_stats[error_type].get(category, 0) + count
                    )

        # Print class-specific error analysis
        print("\nOverall Class-specific Error Analysis:")
        print("=" * 80)

        print("\nTop 10 Most Frequently Added Wrong Classes:")
        print("-" * 50)
        sorted_added = sorted(
            total_class_stats["added_wrongly"].items(), key=lambda x: x[1], reverse=True
        )[:10]
        for category, count in sorted_added:
            print(f"  {category}: {count} times")

        print("\nTop 10 Most Frequently Missing Classes:")
        print("-" * 50)
        sorted_missing = sorted(
            total_class_stats["missing_wrongly"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        for category, count in sorted_missing:
            print(f"  {category}: {count} times")

    # Add detailed class-specific error analysis
    if show_misclassifications:
        print("\nDetailed Class Error Analysis:")
        print("=" * 100)

        # Initialize detailed error tracking
        total_error_patterns = {"substitutions": {}, "additional": {}, "missing": {}}

        # Analyze each entry
        for index, row in df.iterrows():
            ground_truth = row["Ground Truth"]
            prediction = row["Prediction"]

            if pd.isna(ground_truth) or pd.isna(prediction):
                continue

            error_patterns = analyze_detailed_class_errors(ground_truth, prediction)

            # Aggregate the statistics
            for error_type in ["substitutions", "additional", "missing"]:
                for pattern, count in error_patterns[error_type].items():
                    total_error_patterns[error_type][pattern] = (
                        total_error_patterns[error_type].get(pattern, 0) + count
                    )

        # Print top 10 errors for each category
        for error_type in ["substitutions", "additional", "missing"]:
            print(f"\nTop 10 Most Frequent {error_type.title()}:")
            print("-" * 80)
            sorted_errors = sorted(
                total_error_patterns[error_type].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            if sorted_errors:
                for pattern, count in sorted_errors:
                    print(f"  {count:3d}x  {pattern}")
            else:
                print("  No errors of this type found")
            print()

    # Print final summary
    print("\nOverall Prediction Summary:")
    print("-" * 40)
    print(f"Correct Predictions:   {total_summary['correct']}")
    print(f"Additional Predictions:  {total_summary['additional']}")
    print(f"Missing Predictions:    {total_summary['missing']}")
    print(f"Incorrect Predictions: {total_summary['incorrect']}")
    print("-" * 40)
    total_predictions = sum(total_summary.values())
    if total_predictions > 0:
        accuracy = total_summary["correct"] / total_predictions
        print(f"Overall Accuracy: {accuracy:.2%}")

    # Add constellation analysis with visualization
    if show_misclassifications:
        print("\nMost Common Error Constellations:")
        print("=" * 100)

        # Initialize constellation tracking
        total_constellation_patterns = {
            domain: {} for domain in ["Sector", "Research Area", "Infectious Agent"]
        }

        # Analyze each entry
        for index, row in df.iterrows():
            ground_truth = row["Ground Truth"]
            prediction = row["Prediction"]

            if pd.isna(ground_truth) or pd.isna(prediction):
                continue

            constellation_patterns = analyze_classification_constellations_errors(
                ground_truth, prediction
            )

            # Aggregate the statistics
            for domain in constellation_patterns:
                for pattern, count in constellation_patterns[domain].items():
                    total_constellation_patterns[domain][pattern] = (
                        total_constellation_patterns[domain].get(pattern, 0) + count
                    )

        # Print and plot error constellations
        for domain in total_constellation_patterns:
            print(f"\n{domain} - Top 5 Error Constellations:")
            print("-" * 80)

            sorted_constellations = sorted(
                total_constellation_patterns[domain].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            if sorted_constellations:
                for pattern, count in sorted_constellations:
                    print(f"  {count:3d}x  {pattern}")
            else:
                print("  No error constellations found")
            print()

        filename = os.path.splitext(os.path.basename(file_path))[0]
        if save_plots:
            plot_save_path = f"results/plots/{filename}/"
            # Create directory if it doesn't exist
            os.makedirs(plot_save_path, exist_ok=True)
        else:
            plot_save_path = None

        # Create visualizations
        plot_error_constellations(
            total_constellation_patterns,
            save_path=plot_save_path,
        )

    return overall_accuracies, domain_stats


if __name__ == "__main__":
    # Example usage
    file_path = "classification_results.xlsx"
    try:
        overall_accuracies, domain_stats = compute_excel_accuracies(
            file_path, show_misclassifications=False, save_plots=False
        )
    except Exception as e:
        print(f"Error processing file: {e}")
