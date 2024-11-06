import os
import time
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Union
from utils.pydantic.sector import SectorClassificationResult
from utils.pydantic.research_area import ResearchAreaClassificationResult
from utils.pydantic.infectious_agent import InfectiousAgentClassificationResult
from utils.utils import get_categories, extract_json

# Load environment variables from .env file
load_dotenv()

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")


def classify_research(
    prompt: str, model: str, classification_type: str
) -> Union[
    SectorClassificationResult,
    ResearchAreaClassificationResult,
    InfectiousAgentClassificationResult,
]:
    # Step 1: Call LLM with prompt and JSON response format
    json_result = classify_research_json(prompt, model)

    # Step 2: Validate and convert JSON to Pydantic format
    validated = validate_and_convert_json(json_result, classification_type)
    return validated


def classify_research_json(prompt: str, model: str) -> dict:
    try:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": model,
            "messages": messages,
        }
        if model != "o1-mini":
            kwargs["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": "Classify the research based on the given information. Respond in JSON format.",
                },
            )
            kwargs["response_format"] = {"type": "json_object"}

        completion = client.beta.chat.completions.parse(**kwargs)

        if model == "o1-mini":
            # Parse the o1-mini format
            content = completion.choices[0].message.content
            json_str = extract_json(content)
            if json_str:
                parsed_result = json.loads(json_str)
                if isinstance(parsed_result, str):
                    parsed_result = json.loads(parsed_result)
                if not isinstance(parsed_result, dict):
                    parsed_result = {
                        "classification": parsed_result,
                        "explanation": "",
                        "confidence": "",
                        "confidence_explanation": "",
                    }
            else:
                raise json.JSONDecodeError("No JSON object found", content, 0)
            return parsed_result
        else:
            return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"Error in classify_research_json: {str(e)}")
        return None


def validate_and_convert_json(
    json_dict: dict, classification_type: str
) -> Union[
    SectorClassificationResult,
    ResearchAreaClassificationResult,
    InfectiousAgentClassificationResult,
]:
    categories = get_categories(classification_type.replace("_", " ").title())
    validation_prompt = f"""
    Validate and convert the following JSON to the correct Pydantic model format for the {classification_type} classification:

    {json.dumps(json_dict)}

    Ensure all required fields are present and in the correct format.
    The valid categories for this classification are:
    {json.dumps(categories)}

    Make sure the classification(s) in the JSON match these categories exactly.
    """

    if classification_type == "sector":
        response_format = SectorClassificationResult
    elif classification_type == "research_area":
        response_format = ResearchAreaClassificationResult
    elif classification_type == "infectious_agent":
        response_format = InfectiousAgentClassificationResult
    else:
        raise ValueError(f"Invalid classification type: {classification_type}")

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can parse the JSON into a Pydantic model.",
            },
            {"role": "user", "content": validation_prompt},
        ],
        response_format=response_format,
    )
    response = completion.choices[0].message.parsed

    return response
