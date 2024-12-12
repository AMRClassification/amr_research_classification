from typing import TypedDict, List, Dict
import pandas as pd

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
    sector_result: SectorClassification
    research_area_result: ResearchAreaClassification
    infectious_agent_result: InfectiousAgentClassification
    classifications: List[Dict]
    results_df: pd.DataFrame
    successful_entries: int
    sector_next: List[str]  # Separate next state for sector path
    research_area_next: List[str]  # Separate next state for research area path
    infectious_agent_next: List[str]  # Separate next state for infectious agent path
