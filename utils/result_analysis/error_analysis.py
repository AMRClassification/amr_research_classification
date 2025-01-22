# utils/error_analysis.py

from ..data_processing import parse_classification_categories


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
    Now includes complete match accuracy per domain.
    """
    domains = ["Sector", "Research Area", "Infectious Agent"]
    summary = {
        "overall": {"correct": 0, "incorrect": 0}, 
        "by_domain": {},
        "complete_matches": {}  # New section for complete matches
    }

    # Initialize domain summaries
    for domain in domains:
        summary["by_domain"][domain] = {"correct": 0, "incorrect": 0}
        summary["complete_matches"][domain] = {
            "exact_matches": 0,
            "total": len(ground_truths)
        }

    # Process each prediction
    for ground_truth, prediction in zip(ground_truths, predictions):
        gt_domains = parse_classification_categories(ground_truth)
        pred_domains = parse_classification_categories(prediction)

        # Check complete matches per domain
        for domain in domains:
            if gt_domains.get(domain, []) == pred_domains.get(domain, []):
                summary["complete_matches"][domain]["exact_matches"] += 1

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
    ground_truths, predictions, id_column, start_index=1, domains=None, verbose=False, output_file=None
):
    """
    Analyzes misclassifications across all entries and optionally writes indices to a file.

    Args:
        ground_truths (list): List of ground truth strings
        predictions (list): List of prediction strings
        id_column (list): List of IDs corresponding to each entry
        start_index (int): Starting row index in the Excel file (default is 1 for header row)
        domains (list): List of domains to analyze
        verbose (bool): Whether to print the analysis results
        output_file (str): Path to output file for writing misclassification indices
    """
    if domains is None:
        domains = ["Sector", "Research Area", "Infectious Agent"]

    misclassifications = {domain: [] for domain in domains}
    error_indices = {
        domain: {
            "incorrect": [], 
            "additional": [], 
            "missing": [],
            "uncertain": []
        } for domain in domains
    }

    # Collect misclassifications and uncertain predictions
    for idx, (ground_truth, prediction, entry_id) in enumerate(
        zip(ground_truths, predictions, id_column)
    ):
        # Calculate actual Excel row number (add start_index to get real Excel row)
        excel_row = idx + start_index
        
        for domain in domains:
            # Check for uncertain predictions
            if "0000" in prediction and f"{domain} / Uncertain" in prediction:
                error_indices[domain]["uncertain"].append((excel_row, entry_id))
                continue
                
            domain_misclassifications = identify_misclassifications(
                ground_truth, prediction, domain
            )
            if domain_misclassifications:
                misclassifications[domain].append(
                    {
                        "row_idx": excel_row,
                        "id": entry_id,
                        "errors": domain_misclassifications,
                    }
                )
                # Store both row index and ID by error type
                for error in domain_misclassifications:
                    error_indices[domain][error["type"]].append((excel_row, entry_id))

    # Write indices to file if output_file is specified
    if output_file:
        with open(output_file, "w") as f:
            f.write("Misclassification Indices by Domain and Error Type\n")
            f.write("=" * 50 + "\n\n")

            for domain in domains:
                f.write(f"\n{domain} Domain:\n")
                f.write("-" * 30 + "\n")

                # First write uncertain predictions
                uncertain = error_indices[domain]["uncertain"]
                if uncertain:
                    f.write(f"\nUNCERTAIN PREDICTIONS:\n")
                    f.write(f"Count: {len(uncertain)}\n")
                    f.write("IDs: ")
                    f.write(", ".join([f"(Row: {row}, ID: {id})" for row, id in sorted(uncertain)]))
                    f.write("\n")

                # Then write other error types
                for error_type in ["incorrect", "additional", "missing"]:
                    indices = error_indices[domain][error_type]
                    if indices:
                        f.write(f"\n{error_type.upper()} PREDICTIONS:\n")
                        f.write(f"Count: {len(indices)}\n")
                        f.write("IDs: ")
                        f.write(", ".join([f"(Row: {row}, ID: {id})" for row, id in sorted(indices)]))
                        f.write("\n")
                f.write("\n")

    if verbose:
        print("\nMisclassification Analysis:")
        for domain, errors in misclassifications.items():
            if errors or error_indices[domain]["uncertain"]:
                print(f"\n{domain} Domain Misclassifications:")
                print("-" * 80)

                # Print uncertain predictions first
                uncertain = error_indices[domain]["uncertain"]
                if uncertain:
                    print("\nUNCERTAIN PREDICTIONS:")
                    print("-" * 40)
                    for excel_row, entry_id in sorted(uncertain):
                        print(f"Excel Row {excel_row}, ID {entry_id}")

                # Group by error type
                error_types = {"incorrect": [], "additional": [], "missing": []}
                for error in errors:
                    for err in error["errors"]:
                        error_types[err["type"]].append((error["row_idx"], error["id"], err))

                # Print each error type
                for error_type, type_errors in error_types.items():
                    if type_errors:
                        print(f"\n{error_type.upper()} PREDICTIONS:")
                        print("-" * 40)
                        for excel_row, entry_id, err in type_errors:
                            print(f"\nExcel Row {excel_row}, ID {entry_id}")
                            print(f"Details: {err['details']}")
                            if error_type in ["additional", "missing"]:
                                print(f"Correct Predictions: {err.get('correct_hits', 'N/A')}")
                            print(f"Ground Truth: {err['ground_truth']}")
                            print(f"Prediction: {err['prediction']}")
                            print("-" * 40)

    return misclassifications


def calculate_complete_matches(ground_truths, predictions, verbose=True):
    """
    Calculates the percentage of entries where all classifications across all domains match exactly.
    
    Args:
        ground_truths (list): List of ground truth strings
        predictions (list): List of prediction strings
        verbose (bool): Whether to print the results
        
    Returns:
        dict: Statistics about complete matches
    """
    total_entries = len(ground_truths)
    complete_matches = 0
    
    for ground_truth, prediction in zip(ground_truths, predictions):
        gt_domains = parse_classification_categories(ground_truth)
        pred_domains = parse_classification_categories(prediction)
        
        # Check if all domains match exactly (order independent)
        is_complete_match = True
        for domain in ["Sector", "Research Area", "Infectious Agent"]:
            gt_categories = set(gt_domains.get(domain, []))
            pred_categories = set(pred_domains.get(domain, []))
            if gt_categories != pred_categories:
                is_complete_match = False
                break
        
        if is_complete_match:
            complete_matches += 1
    
    match_rate = complete_matches / total_entries if total_entries > 0 else 0
    
    results = {
        "complete_matches": complete_matches,
        "total_entries": total_entries,
        "match_rate": match_rate
    }
    
    if verbose:
        print("\nComplete Classification Matches:")
        print("-" * 40)
        print(f"Complete Matches:      {complete_matches}")
        print(f"Total Entries:        {total_entries}")
        print(f"Complete Match Rate:  {match_rate:.2%}")
    
    return results



def compute_hierarchical_accuracy_per_domain(
    gt_domain_categories, pred_domain_categories, max_level=3
):
    """
    Computes hierarchical accuracy for a specific domain.
    """
    level_matches = {level: 0 for level in range(1, max_level + 1)}
    level_totals = {level: 0 for level in range(1, max_level + 1)}

    # Count total levels in ground truth categories
    for gt_category_levels in gt_domain_categories:
        gt_levels = gt_category_levels[1:]  # Exclude domain name
        num_levels = len(gt_levels)
        for level in range(1, num_levels + 1):
            level_totals[level] += 1

    # Keep track of matched predictions
    matched_pred_indices = set()

    # For each ground truth category, try to find an exact matching prediction
    for gt_category_levels in gt_domain_categories:
        gt_levels = gt_category_levels[1:]  # Exclude domain name
        num_levels = len(gt_levels)
        match_found = False

        for pred_idx, pred_category_levels in enumerate(pred_domain_categories):
            if pred_idx in matched_pred_indices:
                continue  # Prediction already matched

            pred_levels = pred_category_levels[1:]  # Exclude domain name

            # Only consider exact matches (both levels and content)
            if gt_levels == pred_levels:
                match_found = True
                matched_pred_indices.add(pred_idx)
                # Update level matches
                for level in range(1, num_levels + 1):
                    level_matches[level] += 1
                break  # Stop searching after finding a match

        if not match_found:
            # No match found for this ground truth category
            pass  # level_matches remain unchanged

    return level_matches, level_totals


def compute_hierarchical_accuracy(
    ground_truths, predictions, excluded_domain="Disease", verbose=False
):
    """
    Computes hierarchical accuracy across all relevant domains.

    Args:
        ground_truths (list): List of ground truth classification strings
        predictions (list): List of predicted classification strings
        excluded_domain (str): Domain to exclude from analysis
        verbose (bool): Whether to print detailed accuracy statistics
    """
    if not ground_truths or not predictions:
        return {}

    # Track overall statistics
    total_matches = {1: 0, 2: 0, 3: 0}
    total_counts = {1: 0, 2: 0, 3: 0}
    domain_accuracy = {}

    for ground_truth, prediction in zip(ground_truths, predictions):
        if not ground_truth or not prediction:
            continue

        gt_domains = parse_classification_categories(ground_truth)
        pred_domains = parse_classification_categories(prediction)

        all_domains = (set(gt_domains.keys()) | set(pred_domains.keys())) - {
            excluded_domain
        }

        for domain in all_domains:
            if domain not in domain_accuracy:
                domain_accuracy[domain] = {
                    "matches": {1: 0, 2: 0, 3: 0},
                    "totals": {1: 0, 2: 0, 3: 0},
                }

            gt_domain_categories = gt_domains.get(domain, [])
            pred_domain_categories = pred_domains.get(domain, [])
            level_matches, level_totals = compute_hierarchical_accuracy_per_domain(
                gt_domain_categories, pred_domain_categories
            )

            # Update domain stats
            for level in range(1, 4):
                domain_accuracy[domain]["matches"][level] += level_matches.get(level, 0)
                domain_accuracy[domain]["totals"][level] += level_totals.get(level, 0)

                # Update overall totals
                total_matches[level] += level_matches.get(level, 0)
                total_counts[level] += level_totals.get(level, 0)

    if verbose:
        # Print overall accuracies
        print("\nOverall Accuracies:")
        for level in range(1, 4):
            if total_counts[level] > 0:
                accuracy = total_matches[level] / total_counts[level]
                print(
                    f"Level {level}: {accuracy:.2%} ({total_matches[level]}/{total_counts[level]})"
                )
            else:
                print(f"Level {level}: N/A (no entries)")

        # Print domain-specific accuracies
        print("\nAccuracies by Domain:")
        for domain, stats in domain_accuracy.items():
            print(f"\nDomain: {domain}")
            for level in range(1, 4):
                matches = stats["matches"][level]
                totals = stats["totals"][level]
                if totals > 0:
                    accuracy = matches / totals
                    print(f"Level {level}: {accuracy:.2%} ({matches}/{totals})")
                else:
                    print(f"Level {level}: N/A (no entries)")

    return domain_accuracy
