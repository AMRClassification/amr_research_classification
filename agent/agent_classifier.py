import os
import sys

sys.path.insert(1, os.getcwd())

from typing import TypedDict, Annotated, Any, List, Optional, Dict, Union
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime
from collections import Counter
import operator
from functools import partial
from utils.utils import compute_average, format_validation_result


# Load environment variables
load_dotenv()
MODEL = "gpt-4o-mini"


def replace_reducer(existing_value, new_value):
    """Reducer that replaces the existing value with the new value."""
    return new_value


class InputState(TypedDict):
    title: str
    abstract: str
    original_id: str
    ground_truth: str


class SectorClassification(TypedDict):
    """Result of sector classification."""
    sector: str
    explanation: str


class ResearchAreaClassification(TypedDict):
    """Result of research area classification."""
    research_area: str
    explanation: str


class InfectiousAgentClassification(TypedDict):
    """Result of infectious agent classification."""
    infectious_agent: str
    explanation: str


class ClassificationResults(TypedDict):
    type: str
    classification: Dict


class OutputState(TypedDict):
    index: int
    classification_time: float
    prediction: str
    explanations: dict


class ClassificationState(TypedDict):
    """Combined state for the classification workflow."""
    input: InputState
    index: int
    sector_result: Optional[SectorClassification]
    potential_research_areas: Optional[List[Dict[str, str]]]
    research_area_result: Annotated[ResearchAreaClassification, replace_reducer]
    infectious_agent_result: Annotated[InfectiousAgentClassification, replace_reducer]
    classifications: List[Dict]
    results_df: pd.DataFrame
    successful_entries: int
    next: List[str]


class Agent:
    def __init__(self, model: str, num_runs: int = 5, threshold: float = 0.8):
        """Initialize the agent with model and run parameters."""
        self.model = model
        self.num_runs = num_runs
        self.threshold = threshold
        self.classification_stats = Counter()
        
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
                state["input"]["title"],
                state["input"]["abstract"],
                model=self.model
            )
            
            if result:
                return {
                    "sector_result": {
                        "sector": result["sector"],
                        "relevant_input_snippet": result.get("relevant_input_snippet", []),
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
                state["input"]["title"],
                state["input"]["abstract"],
                state["sector_result"]["sector"]
            )

            if validation:
                validation_message = format_validation_result(
                    original_classification=state["sector_result"]["sector"],
                    validation_result=validation,
                    classification_type="Sector"
                )
                print(validation_message)

                if validation["is_correct"]:
                    return {
                        "sector_result": state["sector_result"],
                        "current_step": "validate_sector"
                    }
                else:
                    return {
                        "sector_result": {
                            "sector": validation["suggested_classification"],
                            "relevant_input_snippet": validation.get("evidence", []),
                            "explanation": validation["explanation"]
                        },
                        "current_step": "validate_sector"
                    }
            return state
        except Exception as e:
            print(f"Error in sector validation: {e}")
            return state
        

    def preselect_research_areas(self, state: ClassificationState) -> Dict[str, Any]:
        """First step: Identify potential research areas."""
        try:
            from classifications.research_area import get_potential_research_areas
            
            result = get_potential_research_areas(
                state["input"]["title"],
                state["input"]["abstract"],
                model=self.model
            )
            
            if result:
                return {
                    "potential_research_areas": result,
                    "next": ["classify_research_area"]
                }
            return state
            
        except Exception as e:
            print(f"Error in research area preselection: {e}")
            return state


    def classify_research_area(self, state: ClassificationState) -> Dict[str, Any]:
        """Second step: Detailed classification based on preselected areas."""
        try:
            from classifications.research_area import classify_research_area
            
            # Get the potential areas from state
            potential_areas = state.get("potential_research_areas", [])
            
            if not potential_areas:
                print("No potential research areas available in state")
                return state
            
            result = classify_research_area(
                state["input"]["title"],
                state["input"]["abstract"],
                model=self.model,
                potential_areas=potential_areas
            )
            
            if result:
                return {
                    "research_area_result": {
                        "research_area": result["research_area"],
                        "relevant_input_snippet": result.get("relevant_input_snippet", []),
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
                    state["input"]["title"],
                    state["input"]["abstract"],
                    str(result["research_area"])
                )
                
                if validation:
                    validation_message = format_validation_result(
                        original_classification=result["research_area"],
                        validation_result=validation,
                        classification_type="Research Area"
                    )
                    print(validation_message)

                    if not validation["is_correct"]:
                        if validation["suggested_classification"]:
                            return {
                                "research_area_result": {
                                    "research_area": [validation["suggested_classification"]],
                                    "explanation": (
                                        "Original Classification:\n"
                                        + result["explanation"]
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
                validation_message = "[✓] Research Area: VALIDATION SKIPPED (no therapeutics classification)"
                print(validation_message)
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
            print(f"Error in research area validation: {e}")
            return state

    def classify_infectious_agent(self, state: ClassificationState) -> Dict[str, Any]:
        """Classify the infectious agent."""
        try:
            from classifications.infectious_agent import classify_infectious_agent
            
            result = classify_infectious_agent(
                state["input"]["title"],
                state["input"]["abstract"],
                model=self.model
            )
            
            if result:
                return {
                    "infectious_agent_result": {
                        "infectious_agent": result["infectious_agent"],
                        "relevant_input_snippet": result.get("relevant_input_snippet", []),
                        "explanation": result["explanation"]
                    }
                }
            return state
        except Exception as e:
            print(f"Error in infectious agent classification: {e}")
            return state

    def validate_infectious_agent(self, state: ClassificationState) -> Dict[str, Any]:
        """Validate infectious agent classification."""
        try:
            from classifications.infectious_agent import validate_infectious_agent_classification
            
            if not state.get("infectious_agent_result"):
                return state

            validation = validate_infectious_agent_classification(
                state["input"]["title"],
                state["input"]["abstract"],
                state["infectious_agent_result"]["infectious_agent"]
            )

            if validation:
                validation_message = format_validation_result(
                    original_classification=state["infectious_agent_result"]["infectious_agent"],
                    validation_result=validation,
                    classification_type="Infectious Agent"
                )
                print(validation_message)

                if validation["is_correct"]:
                    return {
                        "infectious_agent_result": state["infectious_agent_result"],
                        "current_step": "validate_infectious_agent"
                    }
                else:
                    return {
                        "infectious_agent_result": {
                            "infectious_agent": validation["suggested_classification"],
                            "relevant_input_snippet": validation.get("evidence", []),
                            "explanation": validation["explanation"]
                        },
                        "current_step": "validate_infectious_agent"
                    }
            return state
        except Exception as e:
            print(f"Error in infectious agent validation: {e}")
            return state

    def store_results(self, state: ClassificationState) -> ClassificationState:
        """Store classification results in DataFrame."""
        try:
            # Format prediction string - handle lists by joining them
            sector = state.get("sector_result", {}).get("sector", [])
            research_area = state.get("research_area_result", {}).get("research_area", [])
            infectious_agent = state.get("infectious_agent_result", {}).get("infectious_agent", [])

            # Convert lists to strings if necessary
            sector_str = "\n".join(sector) if isinstance(sector, list) else str(sector)
            research_area_str = "\n".join(research_area) if isinstance(research_area, list) else str(research_area)
            infectious_agent_str = "\n".join(infectious_agent) if isinstance(infectious_agent, list) else str(infectious_agent)

            prediction = "\n".join(filter(None, [
                sector_str,
                research_area_str,
                infectious_agent_str
            ]))

            # Create new row data
            new_data = {
                "Index": state["index"],
                "Id": str(state["input"]["original_id"]),
                "Title": str(state["input"]["title"]),
                "Abstract": str(state["input"]["abstract"]),
                "Ground Truth": str(state["input"]["ground_truth"]),
                "Prediction": prediction,
                "Sector Overall Explanation": state.get("sector_result", {}).get("explanation", ""),
                "Research Area Overall Explanation": state.get("research_area_result", {}).get("explanation", ""),
                "Infectious Agent Overall Explanation": state.get("infectious_agent_result", {}).get("explanation", "")
            }

            new_row = pd.DataFrame([new_data])
            updated_df = pd.concat([state["results_df"], new_row], ignore_index=True)
            
        except Exception as e:
            print(f"Error storing results: {e}")

    def _setup_workflow(self):
        """Set up the classification workflow."""
        workflow = StateGraph(ClassificationState)

        def start_classifications(state: ClassificationState) -> Dict[str, Any]:
            """Entry point that initiates parallel classification tasks."""
            return {
                "next": [
                    "classify_sector",
                    "preselect_research_areas",
                    "classify_infectious_agent"
                ]
            }

        # Add nodes
        workflow.add_node("start_classifications", start_classifications)
        workflow.add_node("classify_sector", self.classify_sector)
        workflow.add_node("validate_sector", self.validate_sector)
        
        # Research area nodes
        workflow.add_node("preselect_research_areas", self.preselect_research_areas)
        workflow.add_node("classify_research_area", self.classify_research_area)
        workflow.add_node("validate_research_area", self.validate_research_area)
        
        workflow.add_node("classify_infectious_agent", self.classify_infectious_agent)
        workflow.add_node("validate_infectious_agent", self.validate_infectious_agent)
        workflow.add_node("combined_validation", self.combined_validation)
        workflow.add_node("store_results", self.store_results)

        # Connect START to entry point
        workflow.add_edge(START, "start_classifications")

        # Parallel sector classification path
        workflow.add_edge("start_classifications", "classify_sector")
        workflow.add_edge("classify_sector", "validate_sector")
        workflow.add_edge("validate_sector", "combined_validation")

        # Parallel research area path (modified)
        workflow.add_edge("start_classifications", "preselect_research_areas")
        workflow.add_edge("preselect_research_areas", "classify_research_area")
        workflow.add_edge("classify_research_area", "validate_research_area")
        workflow.add_edge("validate_research_area", "combined_validation")

        # Parallel infectious agent path
        workflow.add_edge("start_classifications", "classify_infectious_agent")
        workflow.add_edge("classify_infectious_agent", "validate_infectious_agent")
        workflow.add_edge("validate_infectious_agent", "combined_validation")

        # Final steps
        workflow.add_edge("combined_validation", "store_results")
        workflow.add_edge("store_results", END)

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
                input_state = {
                    "title": str(title),
                    "abstract": str(abstract),
                    "original_id": str(original_id),
                    "ground_truth": str(ground_truth)
                }

                initial_state = {
                    "index": index,
                    "input": input_state,
                    "classifications": [],
                    "results_df": self.results_df,
                    "successful_entries": len(self.results_df)
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

        # Print statistics for this entry
        self.print_classification_stats(successful_runs)

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

