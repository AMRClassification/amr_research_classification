import os
import sys

sys.path.insert(1, os.getcwd())

from typing import Any, Dict
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv
import os
import pandas as pd
from collections import Counter
from utils.utils import compute_average, print_validation_result, print_review_result

from agent.classifications.prompts.sector_prompts import get_sector_validation_review_prompt
from agent.classifications.prompts.research_area_prompts import get_research_area_validation_review_prompt
from agent.classifications.prompts.infectious_agent_prompts import get_infectious_agent_validation_review_prompt
from utils.llm_call import call_llm

from agent.schema import ClassificationState
from agent.classifications.infectious_agent import map_less_relevant_infectious_agents_to_stain


# Load environment variables
load_dotenv()


def replace_reducer(existing_value, new_value):
    """Reducer that replaces the existing value with the new value."""
    return new_value



class Agent:
    def __init__(self, model: str, num_runs: int = 5, threshold: float = 0.8, output_file: str = None):
        """Initialize the agent with model and run parameters."""
        self.model = model
        self.num_runs = num_runs
        self.threshold = threshold
        self.classification_stats = Counter()
        self.output_file = output_file
        
        # Initialize results DataFrame
        self.results_df = pd.DataFrame(
            columns=[
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
        )

        # Initialize workflow
        self._setup_workflow()

    def classify_sector(self, state: ClassificationState) -> Dict[str, Any]:
        """Classify the sector."""
        try:
            from classifications.sector import classify_sector
            
            result = classify_sector(
                title=state["input"]["title"],
                abstract=state["input"]["abstract"],
                model=self.model
            )
            
            
            if result:
                return {
                    "sector_result": {
                        "sector": result["sector"],
                        "explanation": result["explanation"]
                    }
                }
            return state
        except Exception as e:
            print(f"Error in sector classification: {e}")
            return state

    def validate_sector(self, state: ClassificationState) -> Dict[str, Any]:
        """Validate the sector classification."""
        try:
            from classifications.sector import validate_sector_classification
            
            if not state.get("sector_result"):
                return state

            validation = validate_sector_classification(
                title=state["input"]["title"],
                abstract=state["input"]["abstract"],
                prediction=str(state["sector_result"]["sector"]),
                model=self.model
            )

            if validation:
                print_validation_result(
                    original_classification=state["sector_result"]["sector"],
                    validation_result=validation,
                    classification_type="Sector"
                )

                if validation["is_correct"]:
                    return {
                        "sector_result": {
                            "sector": state["sector_result"]["sector"],
                            "explanation": (
                                state["sector_result"]["explanation"]
                                + "\n\nValidation Result:\n"
                                + validation["explanation"]
                            )
                        },
                        "current_step": "validate_sector"
                    }
                else:
                    return {
                        "sector_result": {
                            "sector": validation["suggested_classification"],
                            "explanation": (
                                state["sector_result"]["explanation"]
                                + "\n\nValidation Result:\n"
                                + validation["explanation"]
                            )
                        },
                        "current_step": "validate_sector"
                    }
            return state
        except Exception as e:
            print(f"Error in sector validation node: {e}")
            return state

    def review_sector_validation(self, state: dict) -> dict:
        """Reviews the sector validation explanation and determines next steps."""
        prompt = get_sector_validation_review_prompt(
            title=state["input"]["title"],
            abstract=state["input"]["abstract"],
            validation_result=state["sector_result"]["explanation"]
        )
        
        result = call_llm(prompt, self.model)

        if not result or "review_result" not in result:
            print("Invalid review result format")
            return {"sector_next": ["combined_validation"]}
        
        review = result["review_result"]
        print_review_result(review, "Sector")

        if review["status"] == "uncertain":
            domain_name = "Sector"
            uncertain_classification = f"0000 {domain_name} / Uncertain"
            
            return {
                "sector_result": {
                    "sector": [uncertain_classification],
                    "explanation": (
                        state["sector_result"]["explanation"] +
                        f"\n\n{review['status'].upper()}\n\nUncertainty Analysis:\n{review['reason']}\n{review['analysis']}"
                    )
                },
                "sector_next": ["combined_validation"]
            }
        else:
            return {
                "sector_result": {
                    "sector": state["sector_result"]["sector"],
                    "explanation": (
                        state["sector_result"]["explanation"] +
                        f"\n\n{review['status'].upper()}\n\nUncertainty Analysis:\n{review['reason']}\n{review['analysis']}"
                    )
                },
                "sector_next": ["combined_validation"]
            }

    def classify_research_area(self, state: ClassificationState) -> Dict[str, Any]:
        """Second step: Detailed classification based on preselected areas."""
        try:
            from classifications.research_area import classify_research_area
            
            result = classify_research_area(
                title=state["input"]["title"],
                abstract=state["input"]["abstract"],
                model=self.model,
            )
            
            if result:
                return {
                    "research_area_result": {
                        "research_area": result["research_area"],
                        "explanation": result["explanation"]
                    },
                }
            return state
            
        except Exception as e:
            print(f"Error in research area classification: {e}")
            return state

    def validate_research_area(self, state: ClassificationState) -> Dict[str, Any]:
        """Validate research area classification."""
        try:
            if "research_area_result" not in state:
                return state
            
            result = state["research_area_result"]
            
            # Check if any therapeutics classifications are present
            therapeutics_codes = ["3100", "3200", "3201", "3202", "3203", "3204", "3205", "3400"]
            has_therapeutics = any(
                any(code in area for code in therapeutics_codes)
                for area in result.get("research_area", [])
            )
            
            if has_therapeutics:
                from classifications.research_area import validate_therapeutics_classification
                
                validation = validate_therapeutics_classification(
                    title=state["input"]["title"],
                    abstract=state["input"]["abstract"],
                    prediction=str(result["research_area"]),
                    model=self.model
                )

                if validation:
                    print_validation_result(
                        original_classification=result["research_area"],
                        validation_result=validation,
                        classification_type="Research Area"
                    )
                    if not validation["is_correct"]:
                        if validation["suggested_classification"]:
                            return {
                                "research_area_result": {
                                    "research_area": validation["suggested_classification"],
                                    "explanation": (
                                        result["explanation"]
                                        + "\n\nValidation Result:\n"
                                        + validation["explanation"]
                                    )
                                },
                                "current_step": "validate_research_area"
                            }
                    else:
                        return {
                            "research_area_result": {
                                "research_area": result["research_area"],
                                "explanation": (
                                    result["explanation"]
                                    + "\n\nValidation Result:\n"
                                    + validation["explanation"]
                                )
                            },
                            "current_step": "validate_research_area"
                        }
            else:
                # For non-therapeutics classifications, add explicit validation skip message
                print("[✓] Research Area: VALIDATION SKIPPED (no therapeutics classification)")
                return {
                    "research_area_result": {
                        "research_area": result["research_area"],
                        "explanation": (
                            result["explanation"]
                            + "\n\nValidation: Not required (no therapeutics classification)"
                        )
                    },
                    "current_step": "validate_research_area"
                }
            
            return state
            
        except Exception as e:
            print(f"Error in research area validation node: {e}")
            return state

    def review_research_area_validation(self, state: dict) -> dict:
        """Reviews the research area validation explanation and determines next steps."""
        prompt = get_research_area_validation_review_prompt(
            title=state["input"]["title"],
            abstract=state["input"]["abstract"],
            validation_result=state["research_area_result"]["explanation"]
        )
        
        result = call_llm(prompt, self.model)

        if not result or "review_result" not in result:
            print("Invalid review result format")
            return {"research_area_next": ["combined_validation"]}
        
        review = result["review_result"]
        print_review_result(review, "Research Area")

        if review["status"] == "uncertain":
            domain_name = "Research Area"
            uncertain_classification = f"0000 {domain_name} / Uncertain"
            
            print(f"\n\nUncertainty Analysis:\n{review['reason']}\n{review['analysis']}")

            return {
                "research_area_result": {
                    "research_area": [uncertain_classification],
                    "explanation": (
                    state["research_area_result"]["explanation"] +
                    f"\n\n{review['status'].upper()}\n\nUncertainty Analysis:\n{review['reason']}\n{review['analysis']}"
                )
            },
                "research_area_next": ["combined_validation"]
            }
        else:
            return {
                "research_area_result": {
                    "research_area": state["research_area_result"]["research_area"],
                    "explanation": (
                        state["research_area_result"]["explanation"] +
                        f"\n\n{review['status'].upper()}\n\nUncertainty Analysis:\n{review['reason']}\n{review['analysis']}"
                    )
                },
                "research_area_next": ["combined_validation"]
            }



    def classify_infectious_agent(self, state: ClassificationState) -> Dict[str, Any]:
        """Classify the infectious agent."""
        try:
            from classifications.infectious_agent import classify_infectious_agent
            
            result = classify_infectious_agent(
                title=state["input"]["title"],
                abstract=state["input"]["abstract"],
                model=self.model
            )
            
            if result:
                return {
                    "infectious_agent_result": {
                        "infectious_agent": result["infectious_agent"],
                        "explanation": result["explanation"]
                    }
                }
            return state
        except Exception as e:
            print(f"Error in infectious agent classification: {e}")
            return state

    def validate_infectious_agent(self, state: ClassificationState) -> Dict[str, Any]:
        """Validate infectious agent classification."""

        from classifications.infectious_agent import validate_infectious_agent_classification
        
        if not state.get("infectious_agent_result"):
            return state

        validation = validate_infectious_agent_classification(
            title=state["input"]["title"],
            abstract=state["input"]["abstract"],
            prediction=str(state["infectious_agent_result"]["infectious_agent"]),
            model=self.model
        )

        if validation:
            print_validation_result(
                original_classification=state["infectious_agent_result"]["infectious_agent"],
                validation_result=validation,
                classification_type="Infectious Agent"
            )

            if validation["is_correct"]:
                return {
                    "infectious_agent_result": {
                        "infectious_agent": state["infectious_agent_result"]["infectious_agent"],
                        "explanation": (
                            state["infectious_agent_result"]["explanation"]
                            + "\n\nValidation Result:\n"
                            + validation["explanation"]
                        )
                    },
                    "current_step": "validate_infectious_agent"
                }
            else:
                return {
                    "infectious_agent_result": {
                        "infectious_agent": validation["suggested_classification"],
                        "explanation": (
                            state["infectious_agent_result"]["explanation"]
                            + "\n\nValidation Result:\n"
                            + validation["explanation"]
                        )
                    },
                    "current_step": "validate_infectious_agent"
                }
        return state
        # except Exception as e:
        #     print(f"Error in infectious agent validation node: {e}")
        #     return state

    def review_infectious_agent_validation(self, state: dict) -> dict:
        """Reviews the infectious agent validation explanation and determines next steps."""
        prompt = get_infectious_agent_validation_review_prompt(
            state["input"]["title"],
            state["input"]["abstract"],
            state["infectious_agent_result"]["explanation"]
        )
        
        result = call_llm(prompt, self.model)

        if not result or "review_result" not in result:
            print("Invalid review result format")
            return {"infectious_agent_next": ["combined_validation"]}
        
        review = result["review_result"]
        print_review_result(review, "Infectious Agent")

        if review["status"] == "uncertain":
            domain_name = "Infectious Agent"
            uncertain_classification = f"0000 {domain_name} / Uncertain"
            
            print(f"\n\nUncertainty Analysis:\n{review['reason']}\n{review['analysis']}")

            return {
                "infectious_agent_result": {
                    "infectious_agent": [uncertain_classification],
                    "explanation": (
                        state["infectious_agent_result"]["explanation"] +
                        f"\n\n{review['status'].upper()}\n\nUncertainty Analysis:\n{review['reason']}\n{review['analysis']}"
                    )
                },
                "infectious_agent_next": ["combined_validation"]
            }
        else:
            # Map infectious agents to their broader categories
            agent = state["infectious_agent_result"]["infectious_agent"]
            mapped_agent = map_less_relevant_infectious_agents_to_stain(agent)

            return {
                "infectious_agent_result": {
                    "infectious_agent": mapped_agent,
                    "explanation": (
                        state["infectious_agent_result"]["explanation"] +
                        f"\n\n{review['status'].upper()}\n\nUncertainty Analysis:\n{review['reason']}\n{review['analysis']}"
                    )
                },
                "infectious_agent_next": ["combined_validation"]
            }


    def _setup_workflow(self):
        """Set up the classification workflow."""
        workflow = StateGraph(ClassificationState)

        # Add nodes
        workflow.add_node("classify_sector", self.classify_sector)
        workflow.add_node("validate_sector", self.validate_sector)
        workflow.add_node("review_sector_validation", self.review_sector_validation)
        
        workflow.add_node("classify_research_area", self.classify_research_area)
        workflow.add_node("validate_research_area", self.validate_research_area)
        workflow.add_node("review_research_area_validation", self.review_research_area_validation)
        
        workflow.add_node("classify_infectious_agent", self.classify_infectious_agent)
        workflow.add_node("validate_infectious_agent", self.validate_infectious_agent)
        workflow.add_node("review_infectious_agent_validation", self.review_infectious_agent_validation)

        workflow.add_node("combined_validation", self.combined_validation)

        # Parallel sector classification path
        workflow.add_edge(START, "classify_sector")
        workflow.add_edge("classify_sector", "validate_sector")
        workflow.add_edge("validate_sector", "review_sector_validation")
        workflow.add_conditional_edges(
            "review_sector_validation",
            lambda x: x["sector_next"],
            ["classify_sector", "combined_validation"]
        )

        # Parallel research area path
        workflow.add_edge(START, "classify_research_area")
        workflow.add_edge("classify_research_area", "validate_research_area")
        workflow.add_edge("validate_research_area", "review_research_area_validation")
        workflow.add_conditional_edges(
            "review_research_area_validation",
            lambda x: x["research_area_next"],
            ["classify_research_area", "combined_validation"]
        )

        # Parallel infectious agent path
        workflow.add_edge(START, "classify_infectious_agent")
        workflow.add_edge("classify_infectious_agent", "validate_infectious_agent")
        workflow.add_edge("validate_infectious_agent", "review_infectious_agent_validation")
        workflow.add_conditional_edges(
            "review_infectious_agent_validation",
            lambda x: x["infectious_agent_next"],
            ["classify_infectious_agent", "combined_validation"]
        )

        # Final steps
        workflow.add_edge("combined_validation", END)

        self.app = workflow.compile()

    def print_classification_stats(self, successful_runs):
        """Print statistics for all classifications."""
        print("\nClassification Statistics:")
        print("-" * 50)
        
        # Group stats by classification type
        stats_by_type = {
            "Sector": {},
            "Research Area": {},
            "Infectious Agent": {}
        }
        
        # Process and clean statistics
        for category, count in self.classification_stats.items():
            # Skip invalid single characters and empty strings
            if len(category) <= 1 or category.isspace():
                continue
            
            if category.startswith("Sector:"):
                clean_category = category[7:].strip()
                if clean_category:  # Skip empty categories
                    stats_by_type["Sector"][clean_category] = count
                
            elif category.startswith("Research Area:"):
                clean_category = category[14:].strip()
                if clean_category:
                    stats_by_type["Research Area"][clean_category] = count
                
            elif category.startswith("Infectious Agent:"):
                clean_category = category[17:].strip()
                if clean_category:
                    stats_by_type["Infectious Agent"][clean_category] = count
        
        # Print stats for each type
        for classification_type, stats in stats_by_type.items():
            if stats:
                print(f"\n{classification_type}:")
                for category, count in stats.items():
                    percentage = (count / successful_runs) * 100
                    print(f"   {category}: {count} ({percentage:.2f}%)")

    def save_results(self):
        """Save current results to Excel file."""
        if self.output_file and not self.results_df.empty:
            try:
                self.results_df.to_excel(self.output_file, index=False)
                print(f"\nResults saved to: {self.output_file}")
            except Exception as e:
                print(f"Error saving results to file: {e}")

    def perform_classification(self, index: int, title: str, abstract: str, original_id: str, ground_truth: str) -> pd.DataFrame:
        """Run multiple classification attempts and aggregate results."""
        # Check minimum content length
        if not len(str(title) + str(abstract)) > 500:
            print(f"Entry {index} skipped: Content length below minimum threshold")
            return self.results_df
        
        sector_results = []
        research_area_results = []
        infectious_agent_results = []
        
        print(f"\nPerforming {self.num_runs} classification runs for entry {index}")
        successful_runs = 0
        
        for run in range(self.num_runs):
            try:
                # Create a fresh input state for each run
                input_state = {
                    "title": str(title),
                    "abstract": str(abstract),
                    "original_id": str(original_id),
                    "ground_truth": str(ground_truth)
                }

                # Create a fresh initial state for each run
                initial_state = {
                    "index": index,
                    "input": input_state,
                    "classifications": [],
                    "results_df": self.results_df,
                    "successful_entries": len(self.results_df),
                    # Initialize empty results for each domain
                    "sector_result": {},
                    "research_area_result": {},
                    "infectious_agent_result": {},
                    # Initialize empty next states
                    "sector_next": [],
                    "research_area_next": [],
                    "infectious_agent_next": []
                }

                final_state = self.app.invoke(initial_state)
                
                # Collect results from this run, ensuring list format
                if "sector_result" in final_state:
                    result = final_state["sector_result"]
                    if isinstance(result.get("sector"), str):
                        result["sector"] = [result["sector"]]
                    sector_results.append(result)
                    
                if "research_area_result" in final_state:
                    result = final_state["research_area_result"]
                    if isinstance(result.get("research_area"), str):
                        result["research_area"] = [result["research_area"]]
                    research_area_results.append(result)
                    
                if "infectious_agent_result" in final_state:
                    result = final_state["infectious_agent_result"]
                    if isinstance(result.get("infectious_agent"), str):
                        result["infectious_agent"] = [result["infectious_agent"]]
                    infectious_agent_results.append(result)
                
                successful_runs += 1

                print("-" * 50)
                
            except Exception as e:
                print(f"Error in run {run + 1}: {e}")
                continue

        # Compute averages with proper formatting
        sector_avg = compute_average(sector_results, "sector", self.threshold)
        research_area_avg = compute_average(research_area_results, "research_area", self.threshold)
        infectious_agent_avg = compute_average(infectious_agent_results, "infectious_agent", self.threshold)

        # Format predictions consistently
        prediction_parts = []
        if sector_avg[0]:
            prediction_parts.extend(sector_avg[0])
        if research_area_avg[0]:
            prediction_parts.extend(research_area_avg[0])
        if infectious_agent_avg[0]:
            prediction_parts.extend(infectious_agent_avg[0])

        # Create new row with properly formatted data
        new_data = {
            "Index": index,
            "Id": str(original_id),
            "Title": str(title),
            "Abstract": str(abstract),
            "Ground Truth": str(ground_truth),
            "Prediction": "\n".join(filter(None, prediction_parts)),
            "Sector Overall Explanation": sector_avg[1],
            "Research Area Overall Explanation": research_area_avg[1],
            "Infectious Agent Overall Explanation": infectious_agent_avg[1],
            "Categorisation Time": 0
        }

        # Update results DataFrame
        new_row = pd.DataFrame([new_data])
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)

        # Save results after each successful classification
        self.save_results()

        # # Print statistics for this entry
        # self.print_classification_stats(successful_runs)

        
        return self.results_df

    def get_results(self) -> pd.DataFrame:
        """Get the current results DataFrame."""
        return self.results_df

    def clear_results(self):
        """Clear the results DataFrame and statistics."""
        self.results_df = pd.DataFrame(
            columns=[
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
        )
        self.classification_stats.clear()

    def combined_validation(self, state: ClassificationState) -> ClassificationState:
        """Perform final validation of all classifications."""
        # print("\n[Node: combined_validation] Performing combined validation...")
        try:
            # Here you can add logic to validate the consistency between different classifications
            # For now, we'll just pass through the state
            return state
        except Exception as e:
            print(f"Error in combined validation: {e}")
            return state

