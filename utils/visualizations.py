# utils/visualizations.py

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_accuracy_metrics(
    domain_accuracies, prediction_summary, save_path=None, show_plot=True
):
    """
    Creates visualization of accuracy percentages using seaborn.
    """
    # Set the style
    sns.set_style("whitegrid")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Level-wise accuracies (top)
    level_data = []
    domains = []
    for domain, stats in domain_accuracies.items():
        domains.append(domain)
        for level in range(1, 4):
            matches = stats["matches"][level]
            totals = stats["totals"][level]
            accuracy = matches / totals if totals > 0 else 0
            level_data.append(
                {
                    "Domain": domain,
                    "Level": f"Level {level}",
                    "Accuracy": accuracy * 100,
                }
            )

    df_levels = pd.DataFrame(level_data)
    sns.barplot(data=df_levels, x="Domain", y="Accuracy", hue="Level", ax=ax1)
    ax1.set_title("Level-wise Accuracies by Domain")
    ax1.set_ylabel("Accuracy (%)")

    # Fix x-axis labels
    ax1.set_xticks(range(len(domains)))
    ax1.set_xticklabels(domains, rotation=45, ha="right")

    # Add value labels
    for container in ax1.containers:
        ax1.bar_label(container, fmt="%.1f%%", padding=3)

    # Prediction-wise accuracies (bottom)
    pred_data = []
    categories = []
    for domain, stats in prediction_summary["by_domain"].items():
        categories.append(domain)
        total = stats["correct"] + stats["incorrect"]
        accuracy = stats["correct"] / total if total > 0 else 0
        pred_data.append(
            {
                "Category": domain,
                "Accuracy": accuracy * 100,
            }
        )

    # Add overall accuracy
    categories.append("Overall")
    total_overall = (
        prediction_summary["overall"]["correct"]
        + prediction_summary["overall"]["incorrect"]
    )
    overall_accuracy = (
        prediction_summary["overall"]["correct"] / total_overall
        if total_overall > 0
        else 0
    )
    pred_data.append({"Category": "Overall", "Accuracy": overall_accuracy * 100})

    df_pred = pd.DataFrame(pred_data)
    sns.barplot(
        data=df_pred, x="Category", y="Accuracy", ax=ax2, color=sns.color_palette()[0]
    )
    ax2.set_title("Prediction-wise Accuracies")
    ax2.set_ylabel("Accuracy (%)")

    # Fix x-axis labels
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha="right")

    # Add value labels
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%.1f%%", padding=3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_prediction_counts(prediction_summary, save_path=None, show_plot=True):
    """
    Creates visualization of prediction counts using seaborn.
    """
    # Set the style
    sns.set_style("whitegrid")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Prediction counts by domain (top)
    count_data = []
    categories = []
    for domain, stats in prediction_summary["by_domain"].items():
        categories.append(domain)
        count_data.extend(
            [
                {"Category": domain, "Type": "Correct", "Count": stats["correct"]},
                {"Category": domain, "Type": "Incorrect", "Count": stats["incorrect"]},
            ]
        )

    df_counts = pd.DataFrame(count_data)
    sns.barplot(
        data=df_counts,
        x="Category",
        y="Count",
        hue="Type",
        ax=ax1,
        palette=["#2ecc71", "#e74c3c"],
    )
    ax1.set_title("Prediction Counts by Domain")
    ax1.set_ylabel("Number of Predictions")

    # Fix x-axis labels
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha="right")

    # Add value labels
    for container in ax1.containers:
        ax1.bar_label(container, padding=3)

    # Overall prediction counts (bottom)
    overall_data = [
        {"Type": "Correct", "Count": prediction_summary["overall"]["correct"]},
        {"Type": "Incorrect", "Count": prediction_summary["overall"]["incorrect"]},
    ]
    df_overall = pd.DataFrame(overall_data)

    # Updated barplot call to use hue instead of palette directly
    sns.barplot(
        data=df_overall,
        x="Type",
        y="Count",
        hue="Type",  # Use Type for both x and hue
        ax=ax2,
        palette=["#2ecc71", "#e74c3c"],
        legend=False,  # Hide the redundant legend
    )
    ax2.set_title("Overall Prediction Counts")
    ax2.set_ylabel("Number of Predictions")

    # Add value labels
    for container in ax2.containers:
        ax2.bar_label(container, padding=3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_error_constellations(constellation_patterns, save_path=None, show_plot=True):
    """
    Creates visualization of error constellations using seaborn.
    """
    # Set the style
    sns.set_style("whitegrid")

    # Prepare data
    constellation_data = []
    for domain, patterns in constellation_patterns.items():
        # Sort patterns by frequency and take top 5
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]

        for pattern, count in sorted_patterns:
            # Split and format pattern
            gt, pred = pattern.split(" → ")
            gt = gt.replace("Ground Truth: [", "GT: [")
            pred = pred.replace("Prediction: [", "P: [")
            formatted_pattern = f"{gt}\n{pred}"

            constellation_data.append(
                {"Domain": domain, "Pattern": formatted_pattern, "Count": count}
            )

    df_const = pd.DataFrame(constellation_data)

    # Create figure
    plt.figure(figsize=(15, 10))

    # Create the plot
    g = sns.barplot(
        data=df_const, y="Pattern", x="Count", hue="Domain", dodge=False, palette="deep"
    )

    # Customize the plot
    plt.title("Top Error Constellations by Domain")
    plt.xlabel("Count")
    plt.ylabel("")

    # Add value labels
    for i in g.containers:
        g.bar_label(i, padding=5)

    # Adjust layout for long labels
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()
