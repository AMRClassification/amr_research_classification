import os
import time
import json
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Union
from utils.pydantic.sector import SectorClassificationResult
from utils.research_area import ResearchAreaClassificationResult
from utils.pydantic.infectious_agent import InfectiousAgentClassificationResult
from utils.utils import get_categories, extract_json

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI()
openai_client.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Google client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-pro")


def classify_research(
    prompt: str, model: str, classification_type: str
) -> Union[
    SectorClassificationResult,
    ResearchAreaClassificationResult,
    InfectiousAgentClassificationResult,
]:
    # Step 1: Call LLM with prompt and JSON response format
    json_result = call_llm(prompt, model)

    # Step 2: Validate and convert JSON to Pydantic format
    validated = validate_and_convert_json(json_result, classification_type)
    return validated


def call_llm(prompt: str, model: str) -> dict:
    try:
        if model.startswith("gemini"):
            return call_gemini(prompt)
        else:
            return call_openai(prompt, model)
    except Exception as e:
        print(f"Error in call_llm: {str(e)}")
        return None


def call_openai(prompt: str, model: str) -> dict:
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

    completion = openai_client.chat.completions.create(**kwargs)

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


def call_gemini(prompt: str) -> dict:
    try:
        # Add explicit instruction for JSON format
        formatted_prompt = f"""
        {prompt}
        
        IMPORTANT: Your response must be valid JSON. Wrap your entire response in a JSON object.
        """

        response = gemini_model.generate_content(formatted_prompt)

        # Extract JSON from the response
        content = response.text
        json_str = extract_json(content)

        if not json_str:
            print("No JSON found in Gemini response")
            print(f"Raw response: {content}")
            return None

        try:
            parsed_result = json.loads(json_str)
            if isinstance(parsed_result, str):
                parsed_result = json.loads(parsed_result)
            return parsed_result
        except json.JSONDecodeError as e:
            print(f"Error parsing Gemini JSON: {str(e)}")
            print(f"Attempted to parse: {json_str}")
            return None

    except Exception as e:
        print(f"Error in Gemini call: {str(e)}")
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

    completion = openai_client.chat.completions.create(
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
