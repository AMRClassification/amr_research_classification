import os
import json
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from utils.utils import extract_json

# Load environment variables from .env file
load_dotenv()

def get_openai_client():
    """Initialize and return OpenAI client with API key."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client

def get_google_client(model: str):
    """Initialize and return Google client with API key."""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel(model)
    return gemini_model


def call_llm(prompt: str, model: str) -> str:
    """Call the appropriate LLM based on the model name."""
    try:
        if model.startswith("gemini"):
            return call_gemini(prompt, model)
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

    client = get_openai_client()
    completion = client.chat.completions.create(**kwargs)
    
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


def call_gemini(prompt: str, model: str) -> dict:
    try:
        # Add explicit instruction for JSON format
        formatted_prompt = f"""
        {prompt}
        
        IMPORTANT: Your response must be valid JSON. Wrap your entire response in a JSON object.
        """

        client = get_google_client(model)
        response = client.generate_content(formatted_prompt)

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
