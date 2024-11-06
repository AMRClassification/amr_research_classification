import time
import pandas as pd
import random
from statistics import mean
from collections import Counter

from classifications.sector import classify_sector
from classifications.infectious_agent import classify_infectious_agent
from classifications.research_area import classify_research_area
from compute_accuracies import compute_excel_accuracies


def construct_classification_string(data):
    return "\n".join(filter(None, data))


def perform_classification(
    df,
    start_index,
    end_index,
    model="gpt-4o-mini",
    include_examples=False,
    num_runs=10,
    threshold=0.8,
):
    results_df = df.iloc[start_index:end_index][["Title", "Abstract", "Categories"]]
    results_df = results_df.rename(columns={"Categories": "Ground Truth"})
    results_df[
        [
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
    ] = ""
    total_time = 0
    category_counter = Counter()

    # Predictions
    for index, row in df.iloc[start_index:end_index].iterrows():
        title = row["Title"]
        abstract = row["Abstract"]
        ground_truth = row["Categories"]

        start_time = time.time()

        print(f"Title: {title}")

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

            # Update results_df
            results_df.at[index, "Sector"] = "\n".join(sector_avg[0])
            results_df.at[index, "Sector Overall Explanation"] = sector_avg[1]
            results_df.at[index, "Sector Confidence"] = sector_avg[2]
            results_df.at[index, "Sector Confidence Explanation"] = sector_avg[3]

            results_df.at[index, "Research Area"] = "\n".join(research_area_avg[0])
            results_df.at[index, "Research Area Overall Explanation"] = (
                research_area_avg[1]
            )
            results_df.at[index, "Research Area Confidence"] = research_area_avg[2]
            results_df.at[index, "Research Area Confidence Explanation"] = (
                research_area_avg[3]
            )

            results_df.at[index, "Infectious Agent"] = "\n".join(
                infectious_agent_avg[0]
            )
            results_df.at[index, "Infectious Agent Overall Explanation"] = (
                infectious_agent_avg[1]
            )
            results_df.at[index, "Infectious Agent Confidence"] = infectious_agent_avg[
                2
            ]
            results_df.at[index, "Infectious Agent Confidence Explanation"] = (
                infectious_agent_avg[3]
            )

            classification = construct_classification_string(
                [
                    "\n".join(sector_avg[0]),
                    "\n".join(research_area_avg[0]),
                    "\n".join(infectious_agent_avg[0]),
                ]
            )

            results_df.at[index, "Prediction"] = classification
            print("Overall Classification:")
            print(classification)
            print("Ground Truth:")
            print(ground_truth)

            end_time = time.time()
            classification_time = end_time - start_time
            results_df.at[index, "Categorisation Time"] = classification_time
            total_time += classification_time

        except Exception as e:
            print(f"Exception occurred during classification of index {index}: {e}")
            print("Skipping this entry.")
            results_df = results_df.drop(index)
            continue

    if results_df.empty:
        print("No valid classifications were made. Exiting.")
        return results_df, {}

    average_time = total_time / len(results_df["Title"])
    print(f"Average classification time: {average_time} seconds")

    # Print category occurrences
    print("\nCategory occurrences:")
    for category, count in category_counter.items():
        percentage = (count / (num_runs * (end_index - start_index))) * 100
        print(f"{category}: {count} ({percentage:.2f}%)")

    return results_df


if __name__ == "__main__":
    # Load your data into DataFrame
    file_path = "assets/4. Data_Dynamic Dashboard_test_19032024.xlsx"

    categorised_df = pd.read_excel(file_path)

    # Generate a random index between 1 and the total number of rows in the DataFrame
    random_index = random.randint(1, len(categorised_df))

    start_index = 1000
    end_index = 1200

    # Perform classification
    results = perform_classification(
        categorised_df,
        start_index,
        end_index,
        model="gpt-4o-mini",
        num_runs=5,
        threshold=0.8,
    )

    if not results.empty:
        # Save results to a new Excel file
        output_file = "classification_results.xlsx"
        results.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")

        # Compute accuracies using the saved results
        print("\nComputing accuracies from results file...")
        overall_accuracies, domain_stats = compute_excel_accuracies(
            output_file,
            show_misclassifications=True,  # Enable misclassification analysis
        )
