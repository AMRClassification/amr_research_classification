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
    get_mentions_infectious_agent_prompt,
    get_example_check_prompt,
    get_mentions_group_prompt
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
    listed_agents: List[str]
    unlisted_agents: List[str]

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
                    classification = "1902 Infectious Agent / Not Applicable / Not Applicable"
                elif result["not_applicable_or_not_specified"] == "not_specified":
                    classification = "1901 Infectious Agent / Not Specified / Not Specified_InfectiousAgent"

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
        """Check if the abstract mentions any infectious agents from the list."""
        try:
            result = call_llm(
                get_mentions_infectious_agent_prompt(
                    title=state["title"],
                    abstract=state["abstract"]
                ),
                model
            )

            if not result or "has_infectious_agent" not in result:
                print("Invalid initial check result format")
                return {
                    "has_infectious_agent": False,
                    "listed_agents": [],
                    "unlisted_agents": [],
                    "explanation": state["explanation"] + "\n\nError: Invalid response format from initial check",
                }
            
            # Build explanation for listed agents
            listed_explanation = (
                "\n-------------------------------------------------------\n2. Listed Infectious Agents:\n"
                f"{'\n'.join(result['listed_agents']) if result['listed_agents'] else '-'}\n"
            )

            # Build explanation for unlisted agents 
            unlisted_explanation = (
                f"\n-------------------------------------------------------\n3. Unlisted Infectious Agents:\n"
                f"{'\n'.join(result['unlisted_agents']) if result['unlisted_agents'] else '-'}\n"
            )

            # Combine explanations
            combined_explanation = (
                state["explanation"] +  # Previous explanation from group check
                listed_explanation +
                unlisted_explanation +
                f"\nAnalysis: {result['explanation']}\n\nListed Mentions:\n{result.get('listed_mentions', '-')}\n\nUnlisted Mentions:\n{result.get('unlisted_mentions', '-')}"
            )
            
            if result["has_infectious_agent"]:
                return {
                    "workflow_finished": False,
                    "has_infectious_agent": True,
                    "listed_agents": result.get("listed_agents", []),
                    "unlisted_agents": result.get("unlisted_agents", []),
                    "explanation": combined_explanation,
                }
            else:
                classification = state["found_groups"]
                return {
                    "workflow_finished": True,
                    "has_infectious_agent": False,
                    "listed_agents": [],
                    "unlisted_agents": [],
                    "explanation": combined_explanation,
                    "classification": classification,
                }
                
        except Exception:
            print(f"Error in initial infectious agent check:\n{traceback.format_exc()}")
            return {
                "workflow_finished": True,
                "classification": "",
                "explanation": state["explanation"] + "\n\nError: Failed to perform initial infectious agent check"
            }

    def check_example_and_recurring(state: InfectiousAgentState) -> Dict:
        """
        Categorize found agents into examples/related research vs. main research focus.
        Also handles found groups and ensures unique entries.
        """
        try:
            listed_agents = state.get("listed_agents", [])
            example_agents = []
            research_agents = []
            found_group = []  # Use set to ensure uniqueness
            agent_explanations = []

            for agent in listed_agents:
                # Try up to 3 times to get a valid response
                max_tries = 3
                tries = 0
                valid_result = False
                
                while tries < max_tries and not valid_result:
                    try:
                        result = call_llm(
                            get_example_check_prompt(
                                title=state["title"],
                                abstract=state["abstract"],
                                agent=agent
                            ),
                            model
                        )
                        # print(f"\nAgent: {agent}")
                        # print(f"Classification: {result['agent_classification']}")
                        # print(f"Explanation: {result['explanation']}")
                        # print("Mentions:", 
                        #       "\n  - " + "\n  - ".join(result['mentions'])
                        #       if result['mentions'] else "None")
                        # print("-" * 80)

                        if not result or "is_target" not in result:
                            print(f"Invalid example check result format for agent {agent}, attempt {tries + 1}")
                            tries += 1
                            continue

                        valid_result = True

                        # Categorize agent based on result
                        if result["is_target"]:
                            research_agents.append(agent)
                            mentions = "\n".join(f"  - {mention}" for mention in result['mentions'])
                            agent_explanations.append(f"'{agent}': Research focus\n{result['explanation']}\nMentions:\n{mentions}\n\n")
                        else:
                            example_agents.append(agent)
                            mentions = "\n".join(f"  - {mention}" for mention in result['mentions'])
                            agent_explanations.append(f"'{agent}': Example/related research\n{result['explanation']}\nMentions:\n{mentions}\n\n")
                    
                    except Exception:
                        print(f"Error processing agent {agent}, attempt {tries + 1}:\n{traceback.format_exc()}")
                        tries += 1

                # If we couldn't get a valid result after 3 tries, default to research agent
                if not valid_result:
                    print(f"Failed to get valid classification for {agent} after {max_tries} attempts")
                    research_agents.append(agent)
                    agent_explanations.append(f"'{agent}': Defaulted to research focus after failed attempts")
                    continue

            # # Add found groups to research agents (ensuring uniqueness)
            # if not research_agents:
            #     research_agents.extend(found_group)
            #     research_agents = list(set(research_agents))  # Remove duplicates


            # Process unlisted agents
            unlisted_agents = state.get("unlisted_agents", [])
            if unlisted_agents:
                for unlisted_agent in unlisted_agents:
                    max_tries = 3
                    tries = 0
                    valid_result = False
                    while tries < max_tries and not valid_result:
                        try:
                            result = call_llm(
                                get_example_check_prompt(
                                    title=state["title"],
                                    abstract=state["abstract"],
                                    agent=unlisted_agent
                                ),
                                model
                            )

                            print(result)

                            if not result or "is_target" not in result:
                                print(f"Invalid example check result format for agent {unlisted_agent}, attempt {tries + 1}")
                                tries += 1
                                continue

                            valid_result = True

                            # If it's a research target, add the "Other" classification for its type
                            if result["is_target"]:
                                search_string = unlisted_agent.lower() + result["found_group"].lower()

                                # Bacteria
                                if "bacteria" in search_string and "gram negative" in search_string:
                                    research_agents.append("1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative")
                                elif "bacteria" in search_string and "gram positive" in search_string:
                                    research_agents.append("1513 Infectious Agent / Bacteria / Gram positive / Other Gram positive")
                                elif "bacteria" in search_string and "gram variable" in search_string:
                                    research_agents.append("1523 Infectious Agent / Bacteria / Gram variable / Other Gram variable")
                                
                                # Fungus
                                elif "fungus" in search_string or "fungi" in search_string:
                                    research_agents.append("1602 Infectious Agent / Fungus / Fungus / Other_Fungus")

                                # Parasite
                                elif "parasite" in search_string and "protozoa" in search_string:
                                    research_agents.append("1713 Infectious Agent / Parasite / Protozoa / Other_Protozoa")
                                elif "parasite" in search_string and "helminth" in search_string:
                                    research_agents.append("1723 Infectious Agent / Parasite / Helminth / Other_Helminth")
                                elif "parasite" in search_string:
                                    research_agents.append("1702 Infectious Agent / Parasite / Other_Parasite")

                                # Virus
                                elif "virus" in search_string:
                                    research_agents.append("1802 Infectious Agent / Virus / Virus / Other_Virus")

                                # Other
                                else:
                                    research_agents.append("1902 Infectious Agent / Not Specified / Not Specified_InfectiousAgent")
                                    agent_explanations.append(f"Warning: Unlisted agent '{unlisted_agent}' could not be assigned to a specific group\n")
                                    raise Exception(f"Unlisted agent couldnt be assigned any group: {unlisted_agent}")
                                
                                agent_explanations.append(f"Added Other category for unlisted agent '{unlisted_agent}'")
                                
                            elif result["is_example"] or result["is_from_related_research"]:
                                example_agents.append(unlisted_agent)

                        except Exception:
                            print(f"Error processing unlisted agent {unlisted_agent}, attempt {tries + 1}:\n{traceback.format_exc()}")
                            tries += 1

                    # If we couldn't get a valid result, use the default Other category
                    if not valid_result:
                        research_agents.append("1900 Infectious Agent / Not Specified / Not Specified_InfectiousAgent")

            if not research_agents:
                research_agents = state.get("found_groups", [])


            # Update the explanation format
            explanation = (
                f"-------------------------------------------------------\n4. Agent Categorization:\n"
                # First, add the overview
                f"Overview:\n" + 
                "\n".join([f"'{agent}': {'Research focus' if agent in research_agents else 'Example/related research'}" 
                          for agent in (listed_agents + unlisted_agents)]) +
                "\n\nDetails:\n" + 
                "\n".join(agent_explanations)
            )

            return {
                "workflow_finished": True,
                "classification": research_agents,
                "explanation": state["explanation"] + "\n\n" + explanation,
            }
                
        except Exception:
            print(f"Error in example and recurring check:\n{traceback.format_exc()}")
            return {
                "workflow_finished": True,
                "classification": "",
                "explanation": state["explanation"] + "\n\nError: Failed to categorize agents"
            }
        

    def clean_classification(classification: str) -> str:
        """Clean up the classification by finding nearest matches for each entry."""
        try:
            if not classification:
                return ""
            
            # Get valid categories
            valid_categories = get_categories("Infectious Agent")
            
            # Split entries by newline and process each
            entries = classification.split('\n')
            cleaned_entries = []
            
            for entry in entries:
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
            
            return '\n'.join(unique_entries)
            
        except Exception:
            print(f"Error in classification cleanup:\n{traceback.format_exc()}")
            return classification  # Return original if cleanup fails

    

    # Add nodes to graph
    workflow.add_node("mentions_group", check_mentions_group)
    workflow.add_node("mentions_infectious_agent", check_mentions_infectious_agent)
    workflow.add_node("check_example_and_recurring", check_example_and_recurring)
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
            True: "check_example_and_recurring",  # Found infectious agents
            False: "clean_classification"   # No infectious agents found
        }[not x["workflow_finished"]]
    )
    workflow.add_edge("check_example_and_recurring", "clean_classification")
    workflow.add_edge("clean_classification", END)
    
    # # Branch based on whether infectious agents are found
    # workflow.add_conditional_edges(
    #     "mentions_infectious_agent",
    #     lambda x: {
    #         True: "mentions_which_infectious_agent",  # Found infectious agents from list
    #         False: "respective_category_other"  # No infectious agents from list
    #     }[x["has_infectious_agent"]]
    # )
    
    # # For each found agent, check if it's an example and not recurring
    # workflow.add_edge("mentions_which_infectious_agent", "check_example_and_recurring")
    
    # # Branch based on example and recurring check
    # workflow.add_conditional_edges(
    #     "check_example_and_recurring",
    #     lambda x: {
    #         True: "check_is_target",  # Is example and not recurring
    #         False: "respective_infectious_agent"  # Not example or is recurring
    #     }[x["is_example_and_not_recurring"]]
    # )
    
    # # Final branch for target check
    # workflow.add_conditional_edges(
    #     "check_is_target",
    #     lambda x: {
    #         True: "respective_infectious_agent",  # Is target
    #         False: "uncertain"  # Not target
    #     }[x["is_target"]]
    # )

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
        
        print(classification)
        print(final_state.get("explanation", ""))

        # Process results
        results = {
            "classification": classification,
            "explanation": final_state.get("explanation", ""),
        }
        
        return results