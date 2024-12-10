from utils.utils import get_categories, get_additional_info

sector_options = get_categories("Sector")
sector_additional_info = get_additional_info("Sector")


def get_classification_prompt(
    title,
    abstract,
):
    return f"""
        You are an AI specialized in classifying research papers on antimicrobial resistance into relevant sectors based on their title and abstract. Follow the instructions and specifications below to determine the appropriate sector(s).

        **Instructions:**

        1. **Input:**
            - **Title:** {title}
            - **Abstract:** {abstract}

        2. **Classification Rules:**
            
            a. **Direct Mention:**
                - Only classify sectors that are directly mentioned or can be directly inferred from the title and abstract.
                
            b. **Sector Selection When Multiple Apply:**
                - If the research involves infectious agents applicable to multiple sectors, prioritize and select the sector that is explicitly mentioned in the title or abstract.
                
            c. **Default Classification:**
                - If no sector is explicitly mentioned, default the classification to the **"Human"** sector.
                
            d. **Multiple Classifications:**
                - Multiple sector classifications are permitted **only** if multiple sectors are explicitly mentioned or can be directly inferred from the title and abstract.
                
            e. **Exclude External References:**
                - Ignore any parts of the text that contain references to other resources, such as related work sections or citations to other research. Only consider the topics that are directly treated in the current research.
                
            f. **Animal Testing for Human Purposes:**
                - If the research involves **animal tests** to study the effects or treatments of AMR **for humans**, classify the sector as **"Human"** instead of **"Animal"**.
                - Indicators may include phrases like "animal models to assess human health impacts," "testing on animals for human applications," "from vegetable oils," "studying ecologic-based structures," etc.
                
            g. **Animal-Derived Derivatives for Human Use:**
                - If the research mentions that any **derivatives** (instead of "ingredients") are **derived from animals** but are intended for **human** applications, classify the sector as **"Human"**.
                - Indicators may include phrases like "animal-derived compounds for human therapy," "using animal-sourced materials in human medicine," etc.
            
            h. **Animal Testing Exclusion:**
                - If animal tests are performed but the intended use is specifically for human applications, do NOT classify as "Animal" sector.
                - Examples include:
                    - Testing antibiotics on mice to develop human treatments
                    - Using animal models to study human drug resistance
                    - Animal trials for human therapeutics development
                - The focus should be on the intended end use (human) rather than the testing method (animal)

        3. **Classification Choices:**
            {sector_options}

        4. **Additional Information for 'sector' Category:**
            {sector_additional_info}

        5. **Output Format:**

        - The output should be a JSON object with the following structure:
        ```json
        {{
            "sectors": [
                {{
                    "sector": "str -> 1 sector",
                    "relevant_input_snippet": ["List[str] -> the actual quote(s) from the input text where it takes the information from"],
                    "explanation": "str -> Explanation how this explains the addition of this sector to the classification",
                }}
            ]
        }}
        ```
    """


def get_sector_validation_prompt(title, abstract, prediction):
    return f"""
You are an AI specialized in validating the classification of research papers into sectors. Your task is to verify if the current given classification is correct for the paper's title and abstract.

**Input:**
- **Title:** {title}
- **Abstract:** {abstract}

- **Current Classification:** {prediction}

**Validation Rules:**

1. **Human vs Animal Classification:**
   - Research using animal models for human applications should be classified as "Human"
   - Animal testing for human therapeutics should be classified as "Human"
   - Only classify as "Animal" if the research is specifically focused on animal health/treatment

2. **Default Classification:**
   - If no sector is explicitly mentioned, validate that "Human" is the assigned sector
   - Challenge any non-Human classification that lacks explicit evidence

3. **Multiple Classifications:**
   - Multiple sectors should only be present if explicitly mentioned
   - Validate that each sector has direct evidence in the text

**Classification Choices:**
{sector_options}

**Output Format:**
```json
{{
    "validation_result": {{
        "is_correct": true/false -> indicating if the current classification is correct,
        "correct_classification": "str -> the correct classification if current is wrong, otherwise null",
        "evidence": ["List[str] -> relevant quotes from input"],
        "explanation": "str -> detailed explanation of the validation decision",
    }}
}}
```
"""
