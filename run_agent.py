import pandas as pd
from agent.agent_classifier import perform_classification
import json
from datetime import datetime
from utils.processing import compute_excel_accuracies


def main():
    # Load your data into DataFrame
    file_name = "Human_Therapeutics_1060"
    file_path = f"assets/{file_name}.xlsx"

    try:
        categorised_df = pd.read_excel(file_path)
        print(f"Loaded {len(categorised_df)} entries from dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Parameters
    start_index = 800
    num_entries = 1
    num_runs = 1

    # Model selection
    model_choices = {
        "1": "o1-mini",  # OpenAI
        "2": "gpt-4o-mini",  # OpenAI
        "3": "gemini-1.5-flash",  # Google
        "4": "gemini-1.5-pro",  # Google
    }

    print("\nAvailable Models:")
    for key, value in model_choices.items():
        print(f"{key}: {value}")

    model_choice = input("\nSelect model (1-4): ")
    model = model_choices.get(model_choice, "gpt-4o-mini")

    # Set abbreviation for output file
    model_abbreviation = {
        "o1-mini": "o1",
        "gpt-4o-mini": "4o",
        "gemini-1.5-flash": "flash",
        "gemini-1.5-pro": "pro",
    }.get(model, "4o")

    output_file = f"{model_abbreviation}_{file_name}_{start_index}_{num_entries}.xlsx"

    print(
        f"\nStarting classification of {num_entries} entries from index {start_index}"
    )
    print(f"Using model: {model}")
    print("=" * 50)

    # Run the classification
    results_df = perform_classification(
        df=categorised_df,
        start_index=start_index,
        num_entries=num_entries,
        model=model,
    )

    # Save results
    results_df.to_excel(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Compute accuracies if we have results
    if not results_df.empty:
        print("\nComputing accuracies from results file...")

        # Define print options for different analysis views
        print_options = {
            "level_wise": True,
            "prediction_wise": True,
            "misclassifications": True,
            "constellations": True,
        }

        # Define visualization options
        viz_options = {
            "visualize_analysis": True,
            "save_plots": False,
            "plot_save_dir": "results/plots/",
        }

        compute_excel_accuracies(
            file_path=output_file,
            print_options=print_options,
            viz_options=viz_options,
        )


if __name__ == "__main__":
    main()
