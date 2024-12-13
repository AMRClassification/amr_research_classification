# utils/processing.py

import pandas as pd
from .metrics import compute_hierarchical_accuracy
from .error_analysis import (
    identify_misclassifications,
    prediction_accuracy,
    analyze_error_constellations,
    analyze_misclassifications,
)
from .visualizations import (
    plot_accuracy_metrics,
    plot_error_constellations,
    plot_prediction_counts,
)
from .data_processing import process_excel_data
import os


def compute_excel_accuracies(
    file_path,
    print_options=None,
    viz_options=None,
):
    """
    Reads an Excel file and computes hierarchical accuracies for Ground Truth vs Prediction.

    Args:
        file_path (str): Path to the Excel file containing 'Ground Truth' and 'Prediction' columns
        print_options (dict): Options controlling what analysis to print
            - level_wise (bool): Print level-wise hierarchical accuracies
            - prediction_wise (bool): Print prediction-wise accuracies
            - misclassifications (bool): Print detailed misclassification analysis
            - constellations (bool): Print error constellation analysis
        viz_options (dict): Options controlling visualization outputs
            - visualize_analysis (bool): Whether to display analysis plots
            - save_plots (bool): Whether to save plots to files
            - plot_save_dir (str): Directory for saving plots

    Returns:
        tuple: (domain_accuracies, prediction_summary, misclassifications, constellation_patterns)
    """
    # Set default options if none provided
    if print_options is None:
        print_options = {
            "level_wise": True,
            "prediction_wise": True,
            "misclassifications": False,
            "constellations": False,
        }

    if viz_options is None:
        viz_options = {
            "visualize_analysis": True,
            "save_plots": False,
            "plot_save_dir": "results/plots/",
        }

    # Read the Excel file
    df = pd.read_excel(file_path)

    # Process the data
    ground_truths, predictions, skipped_entries = process_excel_data(df)

    # Add tracking of uncertain predictions by domain
    uncertain_by_domain = {"Sector": [], "Research Area": [], "Infectious Agent": []}

    certain_ground_truths = []
    certain_predictions = []

    for gt, pred in zip(ground_truths, predictions):
        has_uncertain = False
        if pred:
            pred_lines = pred.split("\n")
            for line in pred_lines:
                if "0000" in line and "Uncertain" in line:
                    domain = line.split("/")[0].strip().split(" ", 1)[1]
                    uncertain_by_domain[domain].append(line)
                    has_uncertain = True

        if not has_uncertain:
            certain_ground_truths.append(gt)
            certain_predictions.append(pred)

    print(f"Processing {len(df)} entries...")
    print(f"Found uncertain predictions:")
    for domain, uncertains in uncertain_by_domain.items():
        print(f"  {domain}: {len(uncertains)}")
    if skipped_entries > 0:
        print(f"Skipped {skipped_entries} entries due to missing or invalid data")

    # Run analyses with only certain predictions
    domain_accuracies = compute_hierarchical_accuracy(
        ground_truths=certain_ground_truths,
        predictions=certain_predictions,
        verbose=print_options["level_wise"],
    )

    # Add uncertain counts to domain_accuracies
    for domain in domain_accuracies:
        domain_accuracies[domain]["uncertain"] = len(uncertain_by_domain[domain])

    prediction_summary = prediction_accuracy(
        ground_truths=certain_ground_truths,
        predictions=certain_predictions,
        verbose=print_options["prediction_wise"],
    )

    # Generate output file path based on input file
    filename = os.path.splitext(os.path.basename(file_path))[0]
    misclass_output = os.path.join(os.path.dirname(file_path), "misclassifications.txt")

    # Analyze misclassifications with output file and ID column
    misclassifications = analyze_misclassifications(
        ground_truths=certain_ground_truths,
        predictions=certain_predictions,
        id_column=df["Id"].iloc[
            len(df) - len(certain_ground_truths) :
        ],  # Get IDs for non-uncertain entries
        verbose=print_options.get("misclassifications", False),
        output_file=misclass_output,
    )

    # Set up plot save path if needed
    plot_save_path = None
    if viz_options["save_plots"]:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        plot_save_path = os.path.join(viz_options["plot_save_dir"], filename)
        os.makedirs(plot_save_path, exist_ok=True)

    constellation_patterns = analyze_error_constellations(
        ground_truths=certain_ground_truths,
        predictions=certain_predictions,
        save_plots=viz_options["save_plots"],
        verbose=print_options["constellations"],
    )

    if viz_options["save_plots"]:
        plot_error_constellations(
            constellation_patterns,
            save_path=plot_save_path,
        )

    # Handle visualization and saving of analysis plots
    if viz_options["visualize_analysis"] or viz_options["save_plots"]:
        if viz_options["save_plots"]:
            filename = os.path.splitext(os.path.basename(file_path))[0]
            metrics_save_path = os.path.join(
                viz_options["plot_save_dir"], f"{filename}_metrics.png"
            )
            counts_save_path = os.path.join(
                viz_options["plot_save_dir"], f"{filename}_counts.png"
            )
            const_save_path = os.path.join(
                viz_options["plot_save_dir"], f"{filename}_constellations.png"
            )
            os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
        else:
            metrics_save_path = None
            counts_save_path = None
            const_save_path = None

        # Plot accuracy metrics
        plot_accuracy_metrics(
            domain_accuracies,
            prediction_summary,
            save_path=metrics_save_path,
            show_plot=viz_options["visualize_analysis"],
        )

        # Plot prediction counts
        plot_prediction_counts(
            prediction_summary,
            save_path=counts_save_path,
            show_plot=viz_options["visualize_analysis"],
        )

        # Plot error constellations
        plot_error_constellations(
            constellation_patterns,
            save_path=const_save_path,
            show_plot=viz_options["visualize_analysis"],
        )

    return (
        domain_accuracies,
        prediction_summary,
        misclassifications,
        constellation_patterns,
    )
