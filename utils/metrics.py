# utils/metrics.py

from .data_processing import parse_classification_categories


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
