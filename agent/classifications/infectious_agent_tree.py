from typing import List, Dict, Optional, TypedDict
from uuid import UUID
from datetime import datetime


import os
import sys

sys.path.insert(1, os.getcwd())

import json
from langgraph.graph import Graph, StateGraph, START, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage

from utils.utils import get_categories, find_closest_category
from utils.llm_call import call_llm
from agent.classifications.prompts.infectious_agent_prompts import (
    get_mentions_group_prompt,
    get_mentions_infectious_agent_prompt,
    get_example_check_prompt,
    get_unlisted_example_check_prompt,
    get_agent_classification_prompt,
    )

import traceback

class InfectiousAgentState(TypedDict):
    """State object for infectious agent classification."""
    
    # Workflow
    workflow_finished: bool

    # Input
    title: str
    abstract: str

    # Group check
    found_groups: List[str]

    # Mention check
    has_infectious_agent: bool
    found_agents: List[str]

    # Example check
    example_agents: List[str]
    research_agents: List[str]
    
    # Target check
    target_agents: List[str]
    uncertain_agents: List[str]
    
    # Explanation tracking
    classification: List[str]
    explanation: str



def create_infectious_agent_graph(model: str = "gpt-4o-mini") -> Graph:
    """Creates the classification graph for infectious agents."""
    
    # Initialize workflow graph
    workflow = StateGraph(InfectiousAgentState)


    def check_mentions_group(state: InfectiousAgentState) -> Dict:
        """Check if the abstract mentions any broad infectious agent groups."""
        try:
            result = call_llm(
                get_mentions_group_prompt(
                    title=state["title"],
                    abstract=state["abstract"]
                ),
                model
            )

            if not result or "has_groups" not in result:
                print("Invalid group check result format")
                return {
                    "has_groups": False,
                    "found_groups": [],
                    "explanation": "Error: Invalid response format from group check"
                }
            
            explanation = (
                "1. Groups Mentioned:\n"
                f"Groups identified: {', '.join(result['found_groups']) if result['found_groups'] else 'None'}\n"
                f"\nReasoning: {result['explanation']}\n"
                f"\nMentions: {', '.join(result['mentions']) if result.get('mentions') else 'None'}\n"
            )
            
            if result["has_groups"]:
                return {
                    "workflow_finished": False,
                    "has_groups": True,
                    "found_groups": result.get("found_groups", []),
                    "explanation": explanation 
                }
            else:
                if result["not_applicable_or_not_specified"] == "not_applicable":   
                    classification = ["1902 Infectious Agent / Not Applicable / Not Applicable"]
                elif result["not_applicable_or_not_specified"] == "not_specified":
                    classification = ["1901 Infectious Agent / Not Specified / Not Specified_InfectiousAgent"]

                return {
                    "workflow_finished": True,
                    "has_groups": False,
                    "found_groups": [],
                    "explanation": explanation,
                    "classification": classification
                }
                
        except Exception:
            print(f"Error in group check:\n{traceback.format_exc()}")
            return {
                "workflow_finished": True,
                "classification": "",
                "explanation": "Error: Failed to perform group check"
            }

    def check_mentions_infectious_agent(state: InfectiousAgentState) -> Dict:
        """Extract all infectious agents mentioned in the text."""
        try:
            result = call_llm(
                get_mentions_infectious_agent_prompt(
                    title=state["title"],
                    abstract=state["abstract"]
                ),
                model
            )

            if not result or "found_agents" not in result:
                print("Invalid agent detection result format")
                return {
                    "workflow_finished": True,
                    "found_agents": [],
                    "explanation": "Error: Invalid response format from agent detection"
                }
            
            # Build explanation
            mentions_explanation = (
                "\n-------------------------------------------------------\n"
                "2. Detected Infectious Agents:\n\n"
            )
            
            mentions_explanation += (
                f"Found agents: {', '.join(result['found_agents'])}\n\n"
                f"Mentions:\n{result.get('mentions', [])}\n\n"
                f"Explanation: {result.get('explanation', '')}"
            )

            return {
                "workflow_finished": False,
                "found_agents": result["found_agents"], 
                "explanation": state["explanation"] + mentions_explanation
            }
            
        except Exception:
            print(f"Error in infectious agent detection:\n{traceback.format_exc()}")
            return {
                "workflow_finished": True,
                "classification": [],
                "explanation": state["explanation"] + "\n\nError: Failed to detect infectious agents"
            }

    def categorize_agents(state: InfectiousAgentState) -> Dict:
        """Categorize all found agents together into in_list, not_in_list, or not_targeted."""
        try:
            found_agents = state.get("found_agents", [])
            if not found_agents:
                return {
                    "workflow_finished": True,
                    "classification": state.get("found_groups", []),
                    "explanation": state["explanation"] + "\nNo specific agents to classify."
                }

            # Get classifications for all agents at once
            result = call_llm(
                get_agent_classification_prompt(
                    title=state["title"],
                    abstract=state["abstract"],
                    agents=found_agents
                ),
                model
            )

            if not result or "agent_classifications" not in result:
                print("Invalid agent classification result format")
                return {
                    "workflow_finished": True,
                    "classification": [],
                    "explanation": state["explanation"] + "\n\nError: Invalid response format from agent classification"
                }

            # Build explanation
            classification_explanation = (
                "\n-------------------------------------------------------\n"
                "3. Agent Classifications:\n\n"
                f"{result.get('analysis_summary', '')}\n\n"
                "Detailed Classifications:\n"
            )

            # Extract final classifications and build explanation
            final_classifications = []
            for agent_class in result["agent_classifications"]:
                classification_explanation += (
                    f"\nAgent: {agent_class['agent']}\n"
                    f"Category: {agent_class['category']}\n"
                    f"Class: {agent_class['class']}\n"
                    f"Explanation: {agent_class['explanation']}\n"
                    f"Evidence:\n" + "\n".join(f"  - {e}" for e in agent_class['evidence']) + "\n"
                )

                if agent_class["category"] in ["in_list", "not_in_list"] and agent_class["class"]:
                    final_classifications.append(agent_class["class"])

            # If no targeted agents found, use the broad group classifications
            if not final_classifications:
                final_classifications = state.get("found_groups", [])

            print("[✓] Infectious Agent:", ", ".join(final_classifications))
            return {
                "workflow_finished": True,
                "classification": final_classifications,
                "explanation": state["explanation"] + classification_explanation
            }

        except Exception:
            print(f"Error in agent categorization:\n{traceback.format_exc()}")
            return {
                "workflow_finished": True,
                "classification": [],
                "explanation": state["explanation"] + "\n\nError: Failed to categorize agents"
            }

    def clean_classification(state: InfectiousAgentState) -> Dict:
        """Clean up the classification by finding nearest matches for each entry."""
        try:
            classification = state.get("classification", [])
            if not classification:
                return {
                    "classification": "",
                    "workflow_finished": True
                }
            
            # Get valid categories
            valid_categories = get_categories("Infectious Agent")
            
            # Split entries by newline and process each
            cleaned_entries = []
            
            for entry in classification:
                if not entry.strip():
                    continue
                
                # Check if entry is already valid
                if entry in valid_categories:
                    cleaned_entries.append(entry)
                    continue
                
                # Try to find closest match
                closest_matches = find_closest_category(entry, "Infectious Agent", model=model)
                if closest_matches:
                    print(f"Correcting classification from '{entry}' to '{', '.join(closest_matches)}'")
                    cleaned_entries.extend(closest_matches)
                else:
                    print(f"Could not find match for: {entry}")
                    # If no match found, keep original to maintain information
                    cleaned_entries.append(entry)
            
            # Join back with newlines and remove duplicates while maintaining order
            seen = set()
            unique_entries = []
            for entry in cleaned_entries:
                if entry not in seen:
                    seen.add(entry)
                    unique_entries.append(entry)
            
            return {
                "classification": '\n'.join(unique_entries),
                "workflow_finished": True
            }
            
        except Exception:
            print(f"Error in classification cleanup:\n{traceback.format_exc()}")
            return {
                "classification": '\n'.join(classification) if isinstance(classification, list) else classification,
                "workflow_finished": True
            }

    # Add nodes to graph
    workflow.add_node("mentions_group", check_mentions_group)
    workflow.add_node("mentions_infectious_agent", check_mentions_infectious_agent)
    workflow.add_node("categorize_agents", categorize_agents)
    workflow.add_node("clean_classification", clean_classification)

    # Add conditional edges based on the decision tree
    workflow.add_edge(START, "mentions_group")
    
    workflow.add_conditional_edges(
        "mentions_group",
        lambda x: {
            True: "mentions_infectious_agent",  # Found groups
            False: "clean_classification"  # No groups found and workflow finished
        }[not x["workflow_finished"]]
    )

    workflow.add_conditional_edges(
        "mentions_infectious_agent",
        lambda x: {
            True: "categorize_agents",  # Found infectious agents
            False: "clean_classification"   # No infectious agents found
        }[not x["workflow_finished"]]
    )
    
    workflow.add_edge("categorize_agents", "clean_classification")
    workflow.add_edge("clean_classification", END)

    # Compile graph
    app = workflow.compile()
    return app

class InfectiousAgentTreeClassifier:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.graph = create_infectious_agent_graph(model)
        
    def classify(self, title: str, abstract: str) -> Dict:
        """Run the classification graph for the given title and abstract."""
        # Initialize state
        state = InfectiousAgentState()
        state["title"] = title
        state["abstract"] = abstract
        
        # Run graph
        final_state = self.graph.invoke(state)

        # Get classification and convert list to string with linebreaks if needed
        classification = final_state.get("classification", [])
        if isinstance(classification, list):
            classification = "\n".join(classification)
        
        # Process results
        results = {
            "classification": classification,
            "explanation": final_state.get("explanation", ""),
        }
        
        return results