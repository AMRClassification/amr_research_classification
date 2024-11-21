import time
import pandas as pd
import random
from statistics import mean
from collections import Counter

from classifications.sector import classify_sector
from classifications.infectious_agent import classify_infectious_agent
from classifications.research_area import classify_research_area
from stats import compute_excel_accuracies


def construct_classification_string(data):
    return "\n".join(filter(None, data))


def perform_classification(
    df,
    start_index,
    num_entries,
    model="gpt-4o-mini",
    include_examples=False,
    num_runs=10,
    threshold=0.8,
    output_file=None,
):
    # Validate indices
    if start_index >= len(df):
        raise ValueError("Start index is beyond the dataframe length")

    # Initialize tracking variables
    successful_entries = 0
    current_index = start_index
    total_time = 0
    category_counter = Counter()

    # Initialize empty results DataFrame with columns
    columns = [
        "Title",
        "Abstract",
        "Ground Truth",
        "Prediction",
        "Sector",
        "Sector Overall Explanation",
        "Sector Confidence",
        "Sector Confidence Explanation",
        "Research Area",
        "Research Area Overall Explanation",
        "Research Area Confidence",
        "Research Area Confidence Explanation",
        "Infectious Agent",
        "Infectious Agent Overall Explanation",
        "Infectious Agent Confidence",
        "Infectious Agent Confidence Explanation",
        "Categorisation Time",
    ]
    results_df = pd.DataFrame(columns=columns)

    while successful_entries < num_entries and current_index < len(df):
        row = df.iloc[current_index]
        title = row["Title"]
        abstract = row["Abstract"]
        ground_truth = row["Categories"]

        if not isinstance(abstract, str) or not isinstance(title, str):
            current_index += 1
            continue
        if not len(str(title) + str(abstract)) > 500:
            current_index += 1
            continue

        start_time = time.time()
        print(f"Processing index {current_index} - Title: {title}")

        sector_results = []
        research_area_results = []
        infectious_agent_results = []

        try:
            # Run classifications `num_runs` times
            for _ in range(num_runs):
                sector = classify_sector(
                    title, abstract, model=model, include_examples=include_examples
                )
                if not sector:
                    continue

                research_area = classify_research_area(
                    title, abstract, model=model, include_examples=include_examples
                )
                if not research_area:
                    continue

                infectious_agent = classify_infectious_agent(
                    title, abstract, model=model, include_examples=include_examples
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
            def compute_average(results, classification_type):
                if not results:
                    return [], "", 0, ""
                classifications = [r.get(classification_type, []) for r in results]
                flat_classifications = [
                    item for sublist in classifications for item in sublist
                ]
                classification_counts = Counter(flat_classifications)
                most_common = []
                for c, count in classification_counts.items():
                    percentage = count / len(results)
                    if percentage >= threshold:
                        most_common.append(c)
                if not most_common:
                    most_common = [
                        f"0000 {classification_type.replace('_', ' ').title()} / Uncertain ({', '.join([f'{c}: {count/len(results):.2%}' for c, count in classification_counts.items()])})"
                    ]
                explanation = results[0].get("explanation", "") if results else ""
                confidence = (
                    mean([float(r.get("confidence", 0)) for r in results])
                    if results
                    else 0
                )
                confidence_explanation = (
                    results[0].get("confidence_explanation", "") if results else ""
                )
                return most_common, explanation, confidence, confidence_explanation

            sector_avg = compute_average(sector_results, "sector")
            research_area_avg = compute_average(research_area_results, "research_area")
            infectious_agent_avg = compute_average(
                infectious_agent_results, "infectious_agent"
            )

            # Only add to results_df if classification was successful
            if sector_results and research_area_results and infectious_agent_results:
                new_row = {
                    "Title": title,
                    "Abstract": abstract,
                    "Ground Truth": ground_truth,
                    "Sector": "\n".join(sector_avg[0]),
                    "Sector Overall Explanation": sector_avg[1],
                    "Sector Confidence": sector_avg[2],
                    "Sector Confidence Explanation": sector_avg[3],
                    "Research Area": "\n".join(research_area_avg[0]),
                    "Research Area Overall Explanation": research_area_avg[1],
                    "Research Area Confidence": research_area_avg[2],
                    "Research Area Confidence Explanation": research_area_avg[3],
                    "Infectious Agent": "\n".join(infectious_agent_avg[0]),
                    "Infectious Agent Overall Explanation": infectious_agent_avg[1],
                    "Infectious Agent Confidence": infectious_agent_avg[2],
                    "Infectious Agent Confidence Explanation": infectious_agent_avg[3],
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
    file_name = "4. Data_Dynamic Dashboard_test_19032024"
    file_path = f"assets/{file_name}.xlsx"
    categorised_df = pd.read_excel(file_path)

    start_index = 2000
    num_entries = 200  # Specify desired number of entries

    model = "gpt-4o-mini"
    model_abbreviation = "o1" if model == "o1-mini" else "4o"
    output_file = (
        f"{model_abbreviation}_{file_name}_{start_index}_{num_entries}_entries.xlsx"
    )

    # Perform classification
    results = perform_classification(
        categorised_df,
        start_index,
        num_entries,
        model=model,
        num_runs=1,
        threshold=0.8,
        output_file=output_file,
    )

    if not results.empty:
        # Compute accuracies using the saved results
        print("\nComputing accuracies from results file...")
        overall_accuracies, domain_stats = compute_excel_accuracies(
            output_file,
            print_misclassifications=True,
        )
