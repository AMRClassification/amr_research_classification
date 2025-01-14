from utils.utils import get_categories, get_additional_info
from utils.llm_call import call_llm
from .prompts.human_therapeutics_prompts import get_screening_prompt

def check_human_therapeutics(title: str, abstract: str, model: str = "gpt-4-turbo") -> dict:
    """Check if the content is undoubtably about human therapeutics."""
    max_tries = 3
    tries = 0
    
    while tries < max_tries:
        try:
            # Get the screening prompt
            prompt = get_screening_prompt(
                title=title,
                abstract=abstract
            )
            
            result = call_llm(prompt, model)
            
            if (
                result is None
                or not isinstance(result, dict)
                or "is_human_therapeutics" not in result
                or "explanation" not in result
                or not isinstance(result["is_human_therapeutics"], bool)
            ):
                print(f"Invalid screening result format: {result}")
                tries += 1
                continue
                
            return result

        except Exception as e:
            print(f"Error in human therapeutics screening: {str(e)}")
            tries += 1
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None