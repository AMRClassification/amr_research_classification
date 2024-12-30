import time
import pandas as pd
import random
from statistics import mean
from collections import Counter

from classifications.sector import classify_sector
from classifications.infectious_agent import classify_infectious_agent
from classifications.research_area import classify_research_area
from stats import compute_excel_accuracies

from utils.utils import compute_average

def construct_classification_string(data):
    return "\n".join(filter(None, data))


def perform_classification(
    df,
    start_index,
    num_entries,
    model="gpt-4o-mini",
    num_runs=10,
    threshold=0.8,
    output_file=None,
):
    # Validate indices
    if start_index >= len(df) or start_index < 0:
        raise ValueError("Start index is beyond the dataframe length")

    # Initialize tracking variables
    successful_entries = 0
    current_index = start_index
    total_time = 0
    category_counter = Counter()

    # Initialize empty results DataFrame with columns
    columns = [
        "Index",
        "Id",
        "Title",
        "Abstract",
        "Ground Truth",
        "Prediction",
        "Sector Overall Explanation",
        "Research Area Overall Explanation",
        "Infectious Agent Overall Explanation",
        "Categorisation Time",
    ]
    results_df = pd.DataFrame(columns=columns)

    while successful_entries < num_entries and current_index < len(df):
        # Ask for continuation every 50 samples
        # if successful_entries > 0 and successful_entries % 50 == 0:
        #     user_input = input(
        #         f"\nProcessed {successful_entries} entries. Continue? (y/n): "
        #     )
        #     if user_input.lower() != "y":
        #         print("Stopping classification process...")
        #         break

        row = df.iloc[current_index]
        title = row["Title"]
        abstract = row["Abstract"]
        ground_truth = row["Categories"]
        original_id = row["Id"]

        if not isinstance(abstract, str) or not isinstance(title, str):
            current_index += 1
            continue
        if not len(str(title) + str(abstract)) > 500:
            current_index += 1
            continue

        start_time = time.time()
        print(f"{current_index} - Title: {title}")

        sector_results = []
        research_area_results = []
        infectious_agent_results = []

        try:
            # Run classifications `num_runs` times
            for _ in range(num_runs):
                sector = classify_sector(title, abstract, model=model)
                if not sector:
                    continue

                research_area = classify_research_area(title, abstract, model=model)
                if not research_area:
                    continue

                infectious_agent = classify_infectious_agent(
                    title, abstract, model=model
                )
                if not infectious_agent:
                    continue

                sector_results.append(sector)
                research_area_results.append(research_area)
                infectious_agent_results.append(infectious_agent)

                # Update category_counter for each run
                for category in sector.get("sector", []):
                    category_counter[f"Sector: {category}"] += 1
                for category in research_area.get("research_area", []):
                    category_counter[f"Research Area: {category}"] += 1
                for category in infectious_agent.get("infectious_agent", []):
                    category_counter[f"Infectious Agent: {category}"] += 1

            # Compute average results


            sector_avg = compute_average(sector_results, "sector", threshold)
            research_area_avg = compute_average(research_area_results, "research_area", threshold)
            infectious_agent_avg = compute_average(
                infectious_agent_results, "infectious_agent", threshold
            )

            # Only add to results_df if classification was successful
            if sector_results and research_area_results and infectious_agent_results:
                new_row = {
                    "Index": current_index,
                    "Id": original_id,
                    "Title": title,
                    "Abstract": abstract,
                    "Ground Truth": ground_truth,
                    "Sector Overall Explanation": sector_avg[1],
                    "Research Area Overall Explanation": research_area_avg[1],
                    "Infectious Agent Overall Explanation": infectious_agent_avg[1],
                }

                # Construct overall prediction
                classification = construct_classification_string(
                    [
                        "\n".join(sector_avg[0]),
                        "\n".join(research_area_avg[0]),
                        "\n".join(infectious_agent_avg[0]),
                    ]
                )
                new_row["Prediction"] = classification

                # Add timing information
                end_time = time.time()
                classification_time = end_time - start_time
                new_row["Categorisation Time"] = classification_time
                total_time += classification_time

                # Add the new row to results_df
                results_df.loc[successful_entries] = new_row
                successful_entries += 1

                # Print results
                print("Overall Classification:")
                print(classification)
                print("Ground Truth:")
                print(ground_truth)
                print(f"Successful entries: {successful_entries}/{num_entries}")

                # Save after each successful classification
                results_df.to_excel(output_file, index=False)

        except Exception as e:
            print(
                f"Exception occurred during classification of index {current_index}: {e}"
            )
            print("Skipping this entry.")

        current_index += 1

    if successful_entries == 0:
        print("No valid classifications were made. Exiting.")
        return pd.DataFrame(), {}

    # Calculate and print statistics
    average_time = total_time / successful_entries
    print(f"\nAverage classification time: {average_time:.2f} seconds")

    print("\nCategory occurrences:")
    for category, count in category_counter.items():
        percentage = (count / (num_runs * successful_entries)) * 100
        print(f"{category}: {count} ({percentage:.2f}%)")

    return results_df


if __name__ == "__main__":
    # Load your data into DataFrame
    file_name = "Human_Therapeutics_1060"
    file_path = f"assets/{file_name}.xlsx"
    categorised_df = pd.read_excel(file_path)

    start_index = 800
    num_entries = 200

    num_runs = 5

    # Allow selection of model
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

    # Perform classification
    results = perform_classification(
        categorised_df,
        start_index,
        num_entries,
        model=model,
        num_runs=num_runs,
        threshold=0.8,
        output_file=output_file,
    )

    if isinstance(results, pd.DataFrame) and not results.empty:
        # Compute accuracies using the saved results
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
            file_path=output_file, print_options=print_options, viz_options=viz_options
        )
