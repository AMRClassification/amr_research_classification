# utils/visualizations.py

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import textwrap


def plot_accuracy_metrics(
    domain_accuracies, prediction_summary, save_path=None, show_plot=True
):
    """Creates visualization of accuracy metrics using seaborn."""
    # Check if we have any data to plot
    if not domain_accuracies or not prediction_summary:
        print("No data available for visualization")
        return

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])  # Level-wise accuracies
    ax2 = fig.add_subplot(gs[1, 0])  # Uncertain predictions
    ax3 = fig.add_subplot(gs[:, 1])  # Prediction-wise accuracies

    # Level-wise accuracies (top-left)
    level_data = []
    domains = []
    available_levels = set()

    for domain, stats in domain_accuracies.items():
        domains.append(domain)
        for level in range(1, 4):
            totals = stats["totals"][level]
            if totals > 0:
                available_levels.add(level)
                matches = stats["matches"][level]
                accuracy = matches / totals
                level_data.append(
                    {
                        "Domain": domain,
                        "Level": f"Level {level}",
                        "Accuracy": accuracy * 100,
                    }
                )

    if level_data:  # Only plot if we have data
        df_levels = pd.DataFrame(level_data)
        df_levels["Level"] = pd.Categorical(
            df_levels["Level"],
            categories=[f"Level {l}" for l in sorted(available_levels)],
            ordered=True,
        )
        sns.barplot(data=df_levels, x="Domain", y="Accuracy", hue="Level", ax=ax1)
        ax1.set_title("Level-wise Accuracies by Domain")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        for container in ax1.containers:
            ax1.bar_label(container, fmt="%.1f%%", padding=3)
    else:
        ax1.text(
            0.5, 0.5, "No level-wise accuracy data available", ha="center", va="center"
        )
        ax1.set_title("Level-wise Accuracies by Domain")

    # Uncertain predictions (bottom-left)
    uncertain_data = []
    for domain, stats in domain_accuracies.items():
        uncertain_count = stats.get("uncertain", 0)
        uncertain_data.append({"Domain": domain, "Uncertain Count": uncertain_count})

    if uncertain_data:  # Only plot if we have data
        df_uncertain = pd.DataFrame(uncertain_data)
        sns.barplot(
            data=df_uncertain,
            x="Domain",
            y="Uncertain Count",
            ax=ax2,
            color=sns.color_palette()[2],
        )
        ax2.set_title("Uncertain Predictions by Domain")
        ax2.set_ylabel("Number of Uncertain Predictions")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        for container in ax2.containers:
            ax2.bar_label(container, padding=3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No uncertain predictions data available",
            ha="center",
            va="center",
        )
        ax2.set_title("Uncertain Predictions by Domain")

    # Complete match accuracy (right side)
    pred_data = []
    if prediction_summary.get("complete_matches"):
        for domain, stats in prediction_summary["complete_matches"].items():
            exact_matches = stats["exact_matches"]
            total = stats["total"]
            accuracy = (exact_matches / total * 100) if total > 0 else 0
            
            pred_data.append({
                "Category": domain,
                "Accuracy": accuracy,
                "Exact Matches": exact_matches,
                "Total": total
            })

    if pred_data:  # Only plot if we have data
        df_pred = pd.DataFrame(pred_data)
        sns.barplot(
            data=df_pred,
            x="Category",
            y="Accuracy",
            ax=ax3,
            color=sns.color_palette()[0]
        )
        ax3.set_title("Complete Match Accuracy by Domain")
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")

        # Add value labels showing both percentage and fraction
        for i, row in df_pred.iterrows():
            ax3.text(
                i,
                row["Accuracy"],
                f'{row["Accuracy"]:.1f}%\n({row["Exact Matches"]}/{row["Total"]})',
                ha='center',
                va='bottom',
                fontsize=10
            )
    else:
        ax3.text(
            0.5,
            0.5,
            "No prediction data available",
            ha="center",
            va="center",
        )
        ax3.set_title("Complete Match Accuracy by Domain")

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
    """Creates visualization of error constellations using seaborn."""
    # Check if we have any data to plot
    if not constellation_patterns:
        print("No error constellation data available for visualization")
        plt.figure(figsize=(15, 10))
        plt.text(
            0.5, 0.5, "No error constellation data available", ha="center", va="center"
        )
        plt.title("Error Constellations by Domain")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        if show_plot:
            plt.show()
        else:
            plt.close()
        return

    # Set the style
    sns.set_style("whitegrid")

    # Prepare data
    constellation_data = []
    for domain, patterns in constellation_patterns.items():
        if not patterns:  # Skip if no patterns for this domain
            continue

        # Sort patterns by frequency and take top 5
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]

        for pattern, count in sorted_patterns:
            # Split and format pattern
            try:
                gt, pred = pattern.split(" → ")
                gt = gt.replace("Ground Truth: [", "GT: [")
                pred = pred.replace("Prediction: [", "P: [")
                formatted_pattern = f"{gt}\n{pred}"

                constellation_data.append(
                    {"Domain": domain, "Pattern": formatted_pattern, "Count": count}
                )
            except Exception as e:
                print(f"Error processing pattern {pattern}: {str(e)}")
                continue

    # Check if we have any processed data
    if not constellation_data:
        plt.figure(figsize=(16, 10))
        plt.text(
            0.5,
            0.5,
            "No valid error constellation patterns found",
            ha="center",
            va="center"
        )
        plt.title("Error Constellations by Domain")
    else:
        df_const = pd.DataFrame(constellation_data)

        # Wrap long labels with smaller width
        df_const['Domain'] = df_const['Domain'].apply(lambda x: '\n'.join(textwrap.wrap(x, width=40)))
        df_const['Pattern'] = df_const['Pattern'].apply(lambda x: '\n'.join(textwrap.wrap(x, width=50)))

        plt.figure(figsize=(16, 10))

        g = sns.barplot(
            data=df_const,
            y="Pattern",
            x="Count",
            hue="Domain",
            dodge=False,
            palette="deep"
        )

        # Customize the plot with smaller font sizes
        plt.title("Top Error Constellations by Domain", fontsize=12)
        plt.xlabel("Count", fontsize=10)
        plt.ylabel("Labels", fontsize=10)

        # Add value labels with smaller font size
        for container in g.containers:
            g.bar_label(container, padding=5, fontsize=8)

        # Adjust tick labels
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=12)

        # Adjust legend
        plt.legend(fontsize=8)

        # Adjust margins to fit labels
        plt.subplots_adjust(left=0.8)

        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()
