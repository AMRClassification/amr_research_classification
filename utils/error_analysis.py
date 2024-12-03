# utils/error_analysis.py

from .data_processing import parse_classification_categories


def identify_misclassifications(ground_truth, prediction, domain):
    """
    Identifies misclassifications for a specific domain.
    """
    gt_domains = parse_classification_categories(ground_truth)
    pred_domains = parse_classification_categories(prediction)

    misclassifications = []

    gt_categories = gt_domains.get(domain, [])
    pred_categories = pred_domains.get(domain, [])

    # If both gt and pred categories are empty, no misclassification
    if not gt_categories and not pred_categories:
        return misclassifications

    # Copy for manipulation
    gt_categories_copy = gt_categories.copy()
    pred_categories_copy = pred_categories.copy()

    # Find correct predictions
    correct_predictions = []
    for pred_cat in pred_categories:
        if pred_cat in gt_categories_copy:
            correct_predictions.append(pred_cat)
            gt_categories_copy.remove(pred_cat)

    # Remaining predictions are incorrect
    incorrect_predictions = [
        pred_cat for pred_cat in pred_categories if pred_cat not in correct_predictions
    ]

    # Remaining gt categories are missing
    missing_categories = gt_categories_copy

    # Record misclassifications
    if incorrect_predictions and not correct_predictions:
        misclassifications.append(
            {
                "type": "incorrect",
                "ground_truth": "\n".join(
                    " / ".join(gt_cat) for gt_cat in gt_categories
                ),
                "prediction": "\n".join(
                    " / ".join(pred_cat) for pred_cat in incorrect_predictions
                ),
                "details": "Incorrect predictions made with no correct predictions",
            }
        )
    else:
        if missing_categories:
            misclassifications.append(
                {
                    "type": "missing",
                    "ground_truth": "\n".join(
                        " / ".join(gt_cat) for gt_cat in missing_categories
                    ),
                    "prediction": "None",
                    "correct_hits": "\n".join(
                        " / ".join(pred_cat) for pred_cat in correct_predictions
                    ),
                    "details": f"Missed {len(missing_categories)} categories",
                }
            )
        if incorrect_predictions:
            misclassifications.append(
                {
                    "type": "additional",
                    "ground_truth": "\n".join(
                        " / ".join(gt_cat) for gt_cat in gt_categories
                    ),
                    "prediction": "\n".join(
                        " / ".join(pred_cat) for pred_cat in incorrect_predictions
                    ),
                    "correct_hits": "\n".join(
                        " / ".join(pred_cat) for pred_cat in correct_predictions
                    ),
                    "details": f"Added {len(incorrect_predictions)} incorrect categories",
                }
            )

    return misclassifications


def prediction_accuracy(ground_truths, predictions, verbose=True):
    """
    Summarizes prediction results for each domain and overall.

    Args:
        ground_truths (list): List of ground truth strings
        predictions (list): List of prediction strings
        verbose (bool): Whether to print the analysis results
    """
    domains = ["Sector", "Research Area", "Infectious Agent"]
    summary = {"overall": {"correct": 0, "incorrect": 0}, "by_domain": {}}

    # Initialize domain summaries
    for domain in domains:
        summary["by_domain"][domain] = {"correct": 0, "incorrect": 0}

    # Process each prediction
    for ground_truth, prediction in zip(ground_truths, predictions):
        gt_domains = parse_classification_categories(ground_truth)
        pred_domains = parse_classification_categories(prediction)

        for domain in domains:
            gt_categories = gt_domains.get(domain, [])
            pred_categories = pred_domains.get(domain, [])

            correct_predictions = [
                pred_cat for pred_cat in pred_categories if pred_cat in gt_categories
            ]
            incorrect_predictions = [
                pred_cat
                for pred_cat in pred_categories
                if pred_cat not in gt_categories
            ]

            # Count missing predictions
            missing_predictions = [
                gt_cat for gt_cat in gt_categories if gt_cat not in pred_categories
            ]

            # Update domain-specific counts
            summary["by_domain"][domain]["correct"] += len(correct_predictions)

            # If there are both missing and incorrect predictions for the same category,
            # count it as a single error rather than two
            if incorrect_predictions and missing_predictions:
                # Count as one error per pair of missing/incorrect
                num_errors = max(len(incorrect_predictions), len(missing_predictions))
                summary["by_domain"][domain]["incorrect"] += num_errors
            else:
                # Count individual errors when only one type exists
                summary["by_domain"][domain]["incorrect"] += len(
                    incorrect_predictions
                ) + len(missing_predictions)

            # Update overall totals the same way
            summary["overall"]["correct"] += len(correct_predictions)
            if incorrect_predictions and missing_predictions:
                summary["overall"]["incorrect"] += max(
                    len(incorrect_predictions), len(missing_predictions)
                )
            else:
                summary["overall"]["incorrect"] += len(incorrect_predictions) + len(
                    missing_predictions
                )

    if verbose:
        # Print domain summaries
        print("\nPrediction Summary by Domain:")
        print("-" * 40)
        for domain in domains:
            domain_stats = summary["by_domain"][domain]
            domain_total = domain_stats["correct"] + domain_stats["incorrect"]
            if domain_total > 0:
                accuracy = domain_stats["correct"] / domain_total
                print(f"\n{domain}:")
                print(f"Correct Predictions:    {domain_stats['correct']}")
                print(f"Incorrect Predictions:  {domain_stats['incorrect']}")
                print(f"Accuracy:              {accuracy:.2%}")

        # Print overall summary
        print("\nOverall Summary:")
        print("-" * 40)
        print(f"Total Correct:         {summary['overall']['correct']}")
        print(f"Total Incorrect:       {summary['overall']['incorrect']}")
        total_predictions = sum(summary["overall"].values())
        if total_predictions > 0:
            accuracy = summary["overall"]["correct"] / total_predictions
            print(f"Overall Accuracy:      {accuracy:.2%}")

    return summary


def analyze_error_constellations(
    ground_truths, predictions, save_plots=False, verbose=False
):
    """
    Analyzes and prints error constellations across all entries.

    Args:
        ground_truths (list): List of ground truth strings
        predictions (list): List of prediction strings
        save_plots (bool): Whether to save constellation plots
        verbose (bool): Whether to print detailed analysis

    Returns:
        dict: Dictionary of constellation patterns by domain
    """
    domains = ["Sector", "Research Area", "Infectious Agent"]
    constellation_patterns = {domain: {} for domain in domains}

    # Collect constellation patterns
    for ground_truth, prediction in zip(ground_truths, predictions):
        gt_domains = parse_classification_categories(ground_truth)
        pred_domains = parse_classification_categories(prediction)

        for domain in domains:
            gt_categories = gt_domains.get(domain, [])
            pred_categories = pred_domains.get(domain, [])

            # Skip if both are empty
            if not gt_categories and not pred_categories:
                continue

            # Sort categories to ensure consistent ordering
            gt_sorted = sorted([" / ".join(cat) for cat in gt_categories])
            pred_sorted = sorted([" / ".join(cat) for cat in pred_categories])

            # Only track if prediction is different from ground truth
            if set(gt_sorted) != set(pred_sorted):
                constellation_key = (
                    f"Ground Truth: [{' | '.join(gt_sorted) if gt_sorted else 'NONE'}] "
                    f"→ "
                    f"Prediction: [{' | '.join(pred_sorted) if pred_sorted else 'NONE'}]"
                )
                constellation_patterns[domain][constellation_key] = (
                    constellation_patterns[domain].get(constellation_key, 0) + 1
                )

    if verbose:
        print("\nMost Common Error Constellations:")
        print("=" * 100)

        # Print top 5 constellations per domain
        for domain, patterns in constellation_patterns.items():
            print(f"\n{domain} - Top 5 Error Constellations:")
            print("-" * 80)

            sorted_constellations = sorted(
                patterns.items(), key=lambda x: x[1], reverse=True
            )[:5]

            if sorted_constellations:
                for pattern, count in sorted_constellations:
                    print(f"  {count:3d}x  {pattern}")
            else:
                print("  No error constellations found")
            print()

    return constellation_patterns


def analyze_misclassifications(
    ground_truths, predictions, domains=None, verbose=False, output_file=None
):
    """
    Analyzes misclassifications across all entries and optionally writes indices to a file.

    Args:
        ground_truths (list): List of ground truth strings
        predictions (list): List of prediction strings
        domains (list): List of domains to analyze
        verbose (bool): Whether to print the analysis results
        output_file (str): Path to output file for writing misclassification indices
    """
    if domains is None:
        domains = ["Sector", "Research Area", "Infectious Agent"]

    misclassifications = {domain: [] for domain in domains}

    # Dictionary to store indices by domain and error type
    error_indices = {
        domain: {"incorrect": [], "additional": [], "missing": []} for domain in domains
    }

    # Collect misclassifications
    for index, (ground_truth, prediction) in enumerate(zip(ground_truths, predictions)):
        for domain in domains:
            domain_misclassifications = identify_misclassifications(
                ground_truth, prediction, domain
            )
            if domain_misclassifications:
                misclassifications[domain].append(
                    {
                        "index": index,
                        "errors": domain_misclassifications,
                    }
                )
                # Store indices by error type
                for error in domain_misclassifications:
                    error_indices[domain][error["type"]].append(index)

    # Write indices to file if output_file is specified
    if output_file:
        with open(output_file, "w") as f:
            f.write("Misclassification Indices by Domain and Error Type\n")
            f.write("=" * 50 + "\n\n")

            for domain in domains:
                f.write(f"\n{domain} Domain:\n")
                f.write("-" * 30 + "\n")

                for error_type in ["incorrect", "additional", "missing"]:
                    indices = error_indices[domain][error_type]
                    if indices:
                        f.write(f"\n{error_type.upper()} PREDICTIONS:\n")
                        f.write(f"Count: {len(indices)}\n")
                        f.write(
                            f"Indices: {', '.join(map(str, sorted([i+2 for i in indices])))}\n"
                        )
                f.write("\n")

    if verbose:
        print("\nMisclassification Analysis:")
        for domain, errors in misclassifications.items():
            if errors:
                print(f"\n{domain} Domain Misclassifications:")
                print("-" * 80)

                # Group by error type
                error_types = {"incorrect": [], "additional": [], "missing": []}
                for error in errors:
                    for err in error["errors"]:
                        error_types[err["type"]].append((error["index"], err))

                # Print each error type
                for error_type, type_errors in error_types.items():
                    if type_errors:
                        print(f"\n{error_type.upper()} PREDICTIONS:")
                        print("-" * 40)
                        for index, err in type_errors:
                            print(f"\nEntry {index}")
                            print(f"Details: {err['details']}")
                            if error_type in ["additional", "missing"]:
                                print(
                                    f"Correct Predictions: {err.get('correct_hits', 'N/A')}"
                                )
                            print(f"Ground Truth: {err['ground_truth']}")
                            print(f"Prediction: {err['prediction']}")
                            print("-" * 40)

    return misclassifications
