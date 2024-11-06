def parse_categories(category_string):
    """
    Parses the category string into a dict mapping domain to list of category levels.
    Returns a dict of the form {domain: [list of levels]}
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
            # Append the levels (excluding the code) to the domain's list
            levels_without_code = levels
            # Remove code from the first level
            if " " in levels_without_code[0]:
                levels_without_code[0] = levels_without_code[0].split(" ", 1)[1].strip()
            domain_categories.setdefault(domain, []).append(levels_without_code)
    return domain_categories


def calculate_hierarchical_accuracy_per_domain(
    gt_domain_categories, pred_domain_categories
):
    """
    Calculates hierarchical accuracy for a specific domain.
    gt_domain_categories and pred_domain_categories are lists of lists of levels.
    Returns level_matches and level_totals dictionaries.
    """
    max_level = 3
    level_matches = {level: 0 for level in range(1, max_level + 1)}
    level_totals = {level: 0 for level in range(1, max_level + 1)}

    # For each ground truth category in the domain
    for gt_category_levels in gt_domain_categories:
        gt_levels = gt_category_levels[1:]
        num_levels = len(gt_levels)
        # Update level_totals
        for i in range(1, num_levels + 1):
            level_totals[i] += 1

        # Find the best matching prediction category
        best_match_level = 0
        for pred_category_levels in pred_domain_categories:
            pred_levels = pred_category_levels[1:]
            max_possible_level = min(len(gt_levels), len(pred_levels))
            current_match_level = 0
            for level in range(max_possible_level):
                if gt_levels[level] == pred_levels[level]:
                    current_match_level += 1
                else:
                    break
            if current_match_level > best_match_level:
                best_match_level = current_match_level
        # Update level_matches
        for level in range(1, best_match_level + 1):
            level_matches[level] += 1

    return level_matches, level_totals


def calculate_hierarchical_accuracy(ground_truth, prediction):
    """
    Calculate matches and totals at each hierarchical level per domain.
    Returns a dict with matches and totals for each level and domain.
    """
    if not ground_truth or not prediction:
        return {}

    # Parse ground truth and prediction into domains
    gt_domains = parse_categories(ground_truth)
    pred_domains = parse_categories(prediction)

    # Get all domains except "Disease"
    all_domains = (set(gt_domains.keys()) | set(pred_domains.keys())) - {"Disease"}

    domain_matches_totals = {}

    for domain in all_domains:
        gt_domain_categories = gt_domains.get(domain, [])
        pred_domain_categories = pred_domains.get(domain, [])
        level_matches, level_totals = calculate_hierarchical_accuracy_per_domain(
            gt_domain_categories, pred_domain_categories
        )
        domain_matches_totals[domain] = {
            "matches": level_matches,
            "totals": level_totals,
        }

    return domain_matches_totals


def get_misclassifications(ground_truth, prediction, domain):
    """
    Identifies misclassifications for a specific domain.

    Types of errors:
    - additional: correctly predicted one or more categories but added extra incorrect ones
    - missing: correctly predicted one or more categories but missed some ground truth ones
    - incorrect: didn't predict any correct categories

    Args:
        ground_truth (str): Ground truth classification string
        prediction (str): Predicted classification string
        domain (str): Domain to analyze

    Returns:
        list: List of dictionaries containing misclassification details
    """
    gt_domains = parse_categories(ground_truth)
    pred_domains = parse_categories(prediction)

    misclassifications = []

    gt_categories = gt_domains.get(domain, [])
    pred_categories = pred_domains.get(domain, [])

    # Check if any predictions match ground truth
    any_correct = False
    correct_predictions = []
    incorrect_predictions = []

    for pred_cat in pred_categories:
        found_match = False
        for gt_cat in gt_categories:
            if pred_cat == gt_cat:
                found_match = True
                any_correct = True
                correct_predictions.append(pred_cat)
                break
        if not found_match:
            incorrect_predictions.append(pred_cat)

    # Case 1: No correct predictions at all
    if not any_correct and pred_categories:
        misclassifications.append(
            {
                "type": "incorrect",
                "ground_truth": "\n".join(
                    " / ".join(gt_cat) for gt_cat in gt_categories
                )
                if gt_categories
                else "None",
                "prediction": "\n".join(
                    " / ".join(pred_cat) for pred_cat in pred_categories
                ),
                "details": "No correct predictions made",
            }
        )

    # Case 2: Additional incorrect predictions
    if incorrect_predictions and any_correct:
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
                "details": f"Made {len(correct_predictions)} correct predictions but added {len(incorrect_predictions)} incorrect ones",
            }
        )

    # Case 3: Missing ground truth categories
    missing_categories = []
    for gt_cat in gt_categories:
        if gt_cat not in correct_predictions:
            missing_categories.append(gt_cat)

    if missing_categories and any_correct:
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
                "details": f"Made {len(correct_predictions)} correct predictions but missed {len(missing_categories)} categories",
            }
        )

    return misclassifications


def summarize_predictions(ground_truth, prediction, title="", print_details=True):
    """
    Summarizes the prediction results across all domains and prints detailed analysis.
    Only prints errors, with a simple format showing type, prediction, and ground truth.
    """
    domains = ["Sector", "Research Area", "Infectious Agent"]
    summary = {"correct": 0, "additional": 0, "missing": 0, "incorrect": 0}

    if print_details and (title):
        print(f"\nPaper: {title}")
        print("-" * 80)

    for domain in domains:
        gt_domains = parse_categories(ground_truth)
        pred_domains = parse_categories(prediction)

        gt_categories = gt_domains.get(domain, [])
        pred_categories = pred_domains.get(domain, [])

        # Count correct predictions
        correct_predictions = []
        incorrect_predictions = []

        for pred_cat in pred_categories:
            found_match = False
            for gt_cat in gt_categories:
                if pred_cat == gt_cat:
                    found_match = True
                    correct_predictions.append(pred_cat)
                    break
            if not found_match:
                incorrect_predictions.append(pred_cat)

        if print_details and (
            incorrect_predictions or len(correct_predictions) < len(gt_categories)
        ):
            print(f"\n{domain}:")

            # Case 1: No correct predictions at all
            if not correct_predictions and pred_categories:
                print("  Type: Incorrect")
                print("  Ground Truth:")
                for gt in gt_categories:
                    print(f"    - {' / '.join(gt)}")
                print("  Prediction:")
                for pred in pred_categories:
                    print(f"    - {' / '.join(pred)}")
                summary["incorrect"] += len(pred_categories)

            else:
                # Case 2: Additional incorrect predictions
                if incorrect_predictions:
                    print("  Type: Additional")
                    print("  Ground Truth:")
                    for gt in gt_categories:
                        print(f"    - {' / '.join(gt)}")
                    print("  Prediction (incorrect ones only):")
                    for pred in incorrect_predictions:
                        print(f"    - {' / '.join(pred)}")
                    summary["additional"] += len(incorrect_predictions)

                # Case 3: Missing categories
                missing_categories = [
                    cat for cat in gt_categories if cat not in correct_predictions
                ]
                if missing_categories:
                    print("  Type: Missing")
                    print("  Ground Truth (missing ones only):")
                    for cat in missing_categories:
                        print(f"    - {' / '.join(cat)}")
                    print("  Prediction: None")
                    summary["missing"] += len(missing_categories)

        # Update summary for correct predictions
        if correct_predictions:
            summary["correct"] += len(correct_predictions)

    return summary


def analyze_error_patterns(ground_truth, prediction, print_details=True):
    """
    Analyzes patterns in prediction errors by class level.
    Returns statistics about which classes are most problematic.
    Shows full class paths and limits to top 10 most frequent errors.
    """
    domains = ["Sector", "Research Area", "Infectious Agent"]
    error_stats = {
        domain: {
            "incorrect": {},  # Classes that were completely wrong
            "additional": {},  # Classes that were incorrectly added
            "missing": {},  # Classes that were missed
        }
        for domain in domains
    }

    for domain in domains:
        gt_domains = parse_categories(ground_truth)
        pred_domains = parse_categories(prediction)

        gt_categories = gt_domains.get(domain, [])
        pred_categories = pred_domains.get(domain, [])

        # Count correct predictions
        correct_predictions = []
        incorrect_predictions = []

        for pred_cat in pred_categories:
            found_match = False
            for gt_cat in gt_categories:
                if pred_cat == gt_cat:
                    found_match = True
                    correct_predictions.append(pred_cat)
                    break
            if not found_match:
                incorrect_predictions.append(pred_cat)
                # Track incorrect predictions with full path
                if not correct_predictions:  # Completely incorrect prediction
                    full_path = " / ".join(pred_cat)
                    error_stats[domain]["incorrect"][full_path] = (
                        error_stats[domain]["incorrect"].get(full_path, 0) + 1
                    )
                else:  # Additional incorrect prediction
                    full_path = " / ".join(pred_cat)
                    error_stats[domain]["additional"][full_path] = (
                        error_stats[domain]["additional"].get(full_path, 0) + 1
                    )

        # Track missing categories with full path
        missing_categories = [
            cat for cat in gt_categories if cat not in correct_predictions
        ]
        for cat in missing_categories:
            full_path = " / ".join(cat)
            error_stats[domain]["missing"][full_path] = (
                error_stats[domain]["missing"].get(full_path, 0) + 1
            )

    if print_details:
        print("\nError Pattern Analysis:")
        print("=" * 80)

        for domain in domains:
            print(f"\n{domain}:")
            print("-" * 80)

            for error_type in ["incorrect", "additional", "missing"]:
                errors = error_stats[domain][error_type]
                if errors:
                    print(f"\n{error_type.capitalize()} Predictions (Top 10):")
                    # Sort by frequency and take top 10
                    sorted_errors = sorted(
                        errors.items(), key=lambda x: x[1], reverse=True
                    )[:10]
                    for category, count in sorted_errors:
                        print(f"  {category}: {count} errors")

    return error_stats
