# main.py

from utils.processing import compute_excel_accuracies


def main():
    # Define the path to your Excel file
    file_name = "results/all_5_runs 13500_500"
    file_path = f"{file_name}.xlsx"

    # Define print options for different analysis views
    print_options = {
        "level_wise": True,  # Print level-wise hierarchical accuracies
        "prediction_wise": True,  # Print prediction-wise accuracies
        "misclassifications": True,  # Print detailed misclassification analysis
        "constellations": True,  # Print error constellation analysis
    }

    # Define visualization options
    viz_options = {
        "visualize_analysis": True,  # Show the analysis plots
        "save_plots": True,  # Save plots to files
        "plot_save_dir": "results/plots/",  # Directory for saving plots
    }

    try:
        # Call the processing function with the specified options
        compute_excel_accuracies(
            file_path=file_path, print_options=print_options, viz_options=viz_options
        )
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
