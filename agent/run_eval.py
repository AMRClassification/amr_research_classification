# TODO Rule: For Humas only bacterial and fungal pathogens are tagged in the classifcation,  HOWEVER: budget relevance is also decided among virus, parasites (Post processing)


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
    file_name = "4. Data_Dynamic Dashboard_test_19032024"
    file_path = f"assets/{file_name}.xlsx"

    try:
        df = pd.read_excel(file_path)
        print(f"Loaded {len(df)} entries from dataset") 
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Parameters
    start_index = 13500
    num_entries = 100
    num_runs = 1
    required_consistent = 4  # Number of required consistent results
    
    required_consistent = min(required_consistent, num_runs)
    threshold = required_consistent / num_runs  # Calculate threshold

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

    # Set output file name
    model_abbreviation = {
        "o1-mini": "o1",
        "gpt-4o-mini": "4o",
        "gemini-1.5-flash": "1.5-flash",
        "gemini-1.5-pro": "1.5-pro",
        "gemini-2.0-flash-exp": "2.0-flash",
    }.get(model, "4o")

    output_file = f"{model_abbreviation}_{file_name}_{start_index}_{num_entries}.xlsx"

    # Initialize agent in eval mode
    agent = Agent(
        model=model, 
        num_runs=num_runs, 
        threshold=threshold,
        output_file=output_file,
        eval_mode=True
    )

    # Process entries
    try:
        for current_index in range(start_index, min(start_index + num_entries, len(df))):
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

            print("=" * 50)

        print("\nComputing accuracies from results file...")
        # Update the file path to match where Agent saved it
        results_file_path = os.path.join("results", output_file)
        compute_excel_accuracies(
            file_path=results_file_path,  # Use the correct path
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