import os
import sys
import traceback

sys.path.insert(1, os.getcwd())

import pandas as pd
from datetime import datetime
from typing import Dict, List
import json


from agent.classifications.infectious_agent_tree import InfectiousAgentTreeClassifier
from utils.processing import compute_excel_accuracies

def run_infectious_agent_evaluation(
    input_file: str,
    model: str = "gpt-4",
    start_index: int = 0,
    num_entries: int = None
) -> None:
    """
    Run evaluation of the infectious agent tree classifier.
    
    Args:
        input_file: Path to input Excel file with ground truth
        model: Model to use for classification
        start_index: Starting index for processing
        num_entries: Number of entries to process
    """
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Load data
    df = pd.read_excel(input_file)
    if num_entries is None:
        num_entries = len(df)
    end_index = min(start_index + num_entries, len(df))
    
    # Initialize classifier
    classifier = InfectiousAgentTreeClassifier(model=model)
    
    # Initialize results storage
    results = []
    
    # Process entries
    for idx in range(start_index, end_index):
        print(f"\nProcessing entry {idx} of {end_index-1}")
        
        try:
            # Get entry
            entry = df.iloc[idx]

            print(f"Title: {entry['Title']}")

            # Run classification
            classification_result = classifier.classify(
                title=str(entry["Title"]),
                abstract=str(entry["Abstract"])
            )
            
            # Store results
            result = {
                "Index": idx,
                "Id": entry["Id"],
                "Title": entry["Title"],
                "Abstract": entry["Abstract"],
                "Ground Truth": entry["Categories"],
                "Prediction": classification_result["classification"],
                "Explanation": classification_result["explanation"]
            }
            results.append(result)
            
            # Save intermediate results
            results_df = pd.DataFrame(results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/infectious_agent_tree_results_{timestamp}.xlsx"
            results_df.to_excel(output_file, index=False)
            
            # Print current result
            print(f"Ground Truth: {entry['Categories']}")
            print("=" * 80)
        except Exception:
            print(f"Error processing entry {idx}:")
            print(traceback.format_exc())
            continue
    
    
if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "assets/4. Data_Dynamic Dashboard_test_19032024.xlsx"  # Update with your input file
    MODEL = "gpt-4o-mini"
    START_INDEX = 14000
    NUM_ENTRIES = 10  # Adjust as needed
    
    # Run evaluation
    run_infectious_agent_evaluation(
        input_file=INPUT_FILE,
        model=MODEL,
        start_index=START_INDEX,
        num_entries=NUM_ENTRIES
    )