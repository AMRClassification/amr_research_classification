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
    get_mention_check_prompt,
    get_example_check_prompt,
    get_target_check_prompt
)

import traceback

class InfectiousAgentState(TypedDict):
    """State object for infectious agent classification."""
    title: str
    abstract: str
    has_infectious_agent: bool
    found_agents: List[str]
    classifications: List[str]
    current_agent: Optional[str]
    is_explicitly_mentioned: bool
    is_target: bool
    keep_recurring: bool    

    # Example check
    example_agents: List[str]
    research_agents: List[str]
    
    # Target check
    target_agents: List[str]
    uncertain_agents: List[str]
    
    # Explanation tracking
    explanation: str

def create_infectious_agent_graph(model: str = "gpt-4o-mini") -> Graph:
    """Creates the classification graph for infectious agents."""
    
    # Initialize workflow graph
    workflow = StateGraph(InfectiousAgentState)

    def check_mentions_infectious_agent(state: InfectiousAgentState) -> Dict:
        """Check if the abstract mentions any infectious agents from the list."""
        try:
            result = call_llm(
                get_mention_check_prompt(
                    title=state["title"],
                    abstract=state["abstract"]
                ),
                model
            )

            if not result or "has_infectious_agent" not in result:
                print("Invalid initial check result format")
                return {
                    "has_infectious_agent": False,
                    "found_agents": [],
                    "explanation": "Error: Invalid response format from initial check"
                }
            # Print agents and their mentions
            print("\nFound Agents and Mentions:")
            if result["has_infectious_agent"] and result["found_agents"]:
                print(f"\nAgent: {result['found_agents'][0]}")
                print(f"Mention: {result['mentions'][0]}")
            else:
                print("No agents found")
            print("-" * 80)
            
            explanation = (
                f"Initial Check:\n"
                f"{'Found' if result['has_infectious_agent'] else 'No'} infectious agents in text.\n"
                f"Agents found: {', '.join(result['found_agents']) if result['found_agents'] else 'None'}\n"
                f"Analysis: {result['explanation']}"
            )
            
            return {
                "has_infectious_agent": result["has_infectious_agent"],
                "found_agents": result["found_agents"],
                "explanation": explanation
            }
                
        except Exception:
            print(f"Error in initial infectious agent check:\n{traceback.format_exc()}")
            return {
                "has_infectious_agent": False,
                "found_agents": [],
                "explanation": "Error: Failed to perform initial infectious agent check"
            }

    def check_example_and_recurring(state: InfectiousAgentState) -> Dict:
        """
        Categorize found agents into examples/related research vs. main research focus.
        Also handles found groups and ensures unique entries.
        """
        try:
            found_agents = state.get("found_agents", [])
            example_agents = []
            research_agents = []
            found_group = []  # Use set to ensure uniqueness
            agent_explanations = []

            for agent in found_agents:
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
                        print(f"\nAgent: {agent}")
                        print(f"Classification: {result['agent_classification']}")
                        print(f"Explanation: {result['explanation']}")
                        print("Mentions:", 
                              "\n  - " + "\n  - ".join(result['mentions'])
                              if result['mentions'] else "None")
                        print("-" * 80)

                        if not result or "agent_classification" not in result:
                            print(f"Invalid example check result format for agent {agent}, attempt {tries + 1}")
                            tries += 1
                            continue

                        # Assert that an agent cannot be both recurring and an example/related research
                        is_example_or_related = result["is_example"] or result["is_from_related_research"]
                        assert not (result["is_recurringly_mention"] and is_example_or_related), \
                            f"Agent {agent} cannot be both recurring and an example/related research"

                        valid_result = True

                    except AssertionError:
                        print(f"Assertion failed for agent {agent}, attempt {tries + 1}")
                        tries += 1
                    except Exception:
                        print(f"Error processing agent {agent}, attempt {tries + 1}:\n{traceback.format_exc()}")
                        tries += 1

                # If we couldn't get a valid result after 3 tries, default to research agent
                if not valid_result:
                    print(f"Failed to get valid classification for {agent} after {max_tries} attempts")
                    research_agents.append(agent)
                    agent_explanations.append(f"'{agent}': Defaulted to research focus after failed attempts")
                    continue

                # Handle found groups if present and agent is an example
                if result["is_example"] and "found_group" in result and result["found_group"]:
                    found_group.append(result["found_group"])
                    agent_explanations.append(
                        f"Found additional groups through example {agent}: {', '.join(result['found_group'])}"
                    )

                print(f"FOUND GROUPS: {found_group}")

                # Categorize agent based on result
                if result["is_recurringly_mention"]:
                    research_agents.append(agent)
                    agent_explanations.append(f"'{agent}': Research focus (recurring mentions)")
                elif is_example_or_related:
                    example_agents.append(agent)
                    agent_explanations.append(f"'{agent}': Example/related research")
                else:
                    research_agents.append(agent)
                    agent_explanations.append(f"'{agent}': Research focus")

            # Add found groups to research agents (ensuring uniqueness)
            if not research_agents:
                research_agents.extend(found_group)
                research_agents = list(set(research_agents))  # Remove duplicates

            explanation = (
                f"Agent Categorization:\n"
                f"Research focus: {', '.join(research_agents) if research_agents else 'None'}\n"
                f"Examples/related: {', '.join(example_agents) if example_agents else 'None'}\n"
                f"Additional groups found: {', '.join(found_group) if found_group else 'None'}\n"
                f"Details:\n" + "\n".join(agent_explanations)
            )

            print(f"EXAMPLE: {example_agents}")
            print(f"RESEARCH: {research_agents}")

            return {
                "example_agents": example_agents,
                "research_agents": research_agents,
                "explanation": state["explanation"] + "\n\n" + explanation
            }
                
        except Exception:
            print(f"Error in example and recurring check:\n{traceback.format_exc()}")
            return {
                "example_agents": [],
                "research_agents": [],
                "explanation": state["explanation"] + "\n\nError: Failed to categorize agents"
            }

    def check_is_target(state: dict) -> dict:
        """
        Check if research agents are explicitly mentioned as treatment targets.
        Analyzes each research agent to determine if it's a target of the treatment/discovery.
        """
        try:
            research_agents = state.get("research_agents", [])
            target_agents = []
            uncertain_agents = []
            explanations = {}

            for agent in research_agents:
                result = call_llm(
                    get_target_check_prompt(
                        title=state["title"],
                        abstract=state["abstract"],
                        agent=agent
                    ),
                    model
                )

                if not result or "is_target" not in result:
                    print(f"Invalid target check result format for agent {agent}")
                    uncertain_agents.append(agent)
                    continue

                # Store the explanation
                explanations[f"target_check_{agent}"] = result["explanation"]

                if result["is_target"]:
                    target_agents.append(agent)
                else:
                    uncertain_agents.append(agent)
        
            print(f"TARGET: {target_agents}")
            print(f"UNCERTAIN: {uncertain_agents}")

            return {
                "target_agents": target_agents,
                "uncertain_agents": uncertain_agents,
            }
            
        except Exception:
            print(f"Error in target check:\n{traceback.format_exc()}")
            return {
                "target_agents": [],
                "uncertain_agents": research_agents,  # Move all to uncertain on error
            }

    # Add nodes to graph
    workflow.add_node("mentions_infectious_agent", check_mentions_infectious_agent)
    workflow.add_node("check_example_and_recurring", check_example_and_recurring)
    workflow.add_node("check_is_target", check_is_target)

    # Add conditional edges based on the decision tree
    workflow.add_edge(START, "mentions_infectious_agent")

    workflow.add_edge("mentions_infectious_agent", "check_example_and_recurring")
    workflow.add_edge("check_example_and_recurring",END)
    
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
        
        # Process results
        results = {
            "classifications": final_state.get("classifications", []),
            "explanations": {
                "has_infectious_agent": final_state.get("has_infectious_agent", False),
                "found_agents": final_state.get("found_agents", []),
                "is_explicitly_mentioned": final_state.get("is_explicitly_mentioned", False),
                "is_target": final_state.get("is_target", False),
                "keep_recurring": final_state.get("keep_recurring", False)
            }
        }
        
        return results