import os
import sys

sys.path.insert(1, os.getcwd())

import pandas as pd
from datetime import datetime
from utils.data_processing import process_excel_data
from utils.processing import compute_excel_accuracies
from agent_classifier import Agent


def main():
    # Load your data
    file_name = "Human_Therapeutics_1060"
    file_path = f"assets/{file_name}.xlsx"

    try:
        df = pd.read_excel(file_path)
        print(f"Loaded {len(df)} entries from dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Parameters
    start_index = 800
    num_entries = 2
    num_runs = 5
    threshold = 0.8

    # # Model selection
    # model_choices = {
    #     "1": "o1-mini",  # OpenAI
    #     "2": "gpt-4o-mini",  # OpenAI
    #     "3": "gemini-1.5-flash",  # Google
    #     "4": "gemini-1.5-pro",  # Google
    # }

    # print("\nAvailable Models:")
    # for key, value in model_choices.items():
    #     print(f"{key}: {value}")

    # model_choice = input("\nSelect model (1-4): ")
    # model = model_choices.get(model_choice, "gpt-4o-mini")

    model = "gpt-4o-mini"

    print(
        f"\nStarting classification of {num_entries} entries from index {start_index}"
    )
    print(f"Using model: {model}")
    print("=" * 50)

    # Initialize agent with all parameters
    agent = Agent(model=model, num_runs=num_runs, threshold=threshold)

    # Process entries
    try:
        for current_index in range(
            start_index, min(start_index + num_entries, len(df))
        ):
            print(f"\nProcessing entry {current_index} of {len(df)}")

            # Get the row data
            row = df.iloc[current_index]

            # Process the entry
            agent.perform_classification(
                index=current_index,
                title=str(row["Title"]),
                abstract=str(row["Abstract"]),
                original_id=str(row["Id"]),
                ground_truth=str(row["Categories"]),
            )

        results_df = agent.get_results()

        # Set output file name after classification
        model_abbreviation = {
            "o1-mini": "o1",
            "gpt-4o-mini": "4o",
            "gemini-1.5-flash": "flash",
            "gemini-1.5-pro": "pro",
        }.get(model, "4o")

        output_file = (
            f"{model_abbreviation}_{file_name}_{start_index}_{num_entries}.xlsx"
        )

        # Save final results
        if not results_df.empty:
            results_df.to_excel(output_file, index=False)
            print(f"\nFinal results saved to: {output_file}")

            print("\nComputing accuracies from results file...")
            compute_excel_accuracies(
                file_path=output_file,
                print_options={
                    "level_wise": True,
                    "prediction_wise": True,
                    "misclassifications": True,
                    "constellations": True,
                },
                viz_options={
                    "visualize_analysis": True,
                    "save_plots": False,
                    "plot_save_dir": "results/plots/",
                },
            )

    except Exception as e:
        print(f"Error in classification process: {e}")
        raise


if __name__ == "__main__":
    main()
