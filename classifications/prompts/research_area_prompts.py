from utils.utils import get_categories, get_additional_info, get_keywords


research_area_options = get_categories("Research Area")
research_area_additional_info = get_additional_info("Research Area")
research_area_keywords = get_keywords("Research Area")


def get_classification_prompt(title, abstract):
    return f"""
You are an AI specialized in classifying research papers on antimicrobial resistance into relevant research areas based on their title and abstract. Follow the instructions and specifications below to determine the appropriate research area(s).

**Instructions:**

### 1. Classification Choices:
{research_area_options}

### 3. Classification Rules:

#### a. Direct Mention:
- **Primary Goal:** Focus on identifying the major goal or objective of the research.
- **Classification Assignment:** Assign research areas that are explicitly mentioned in the title/abstract as defined by the keywords.

#### b. Preference for Single Classification:
- **Default to Single Classification:** Assign only one research area that best represents the main focus of the research.
- **When to Assign Multiple Classifications:**
  - **Rare Exceptions:** Assign multiple research areas only if the research equally and explicitly addresses multiple areas as main objectives.
  - **Equal Focus Required:** Multiple classifications are permitted only when multiple research areas are of equal significance and centrality in the research.
  - **Main Focus Rule:** If one research area is clearly the primary focus (e.g., constitutes 80% or more of the content), do not assign additional classifications even if other areas are mentioned.
  - **Avoid Overclassification:** Do not assign multiple classifications simply because multiple topics are mentioned; focus on the single main goal of the research.

#### c. Exclude External References:
- **Ignore:** References to other works, related studies, citations, or mentions of earlier work.
- **Focus:** Only on topics directly addressed in the current research.

#### d. Category-Specific Guidelines:
- **Translational Research:** Represents transitions between phases and should be classified based on the target phase, not as a separate category.
- **Diagnostics:** Refers specifically to the detection and identification of infectious agents to determine which agent is present.
- **Therapeutics Discovery:** Focuses on the detection and validation of therapeutic products.
- **Capacity Building:** Specifically refers to efforts aimed at refurbishing or increasing laboratory infrastructure and capabilities.

#### 4. Keywords:
Focus on the following keywords to determine the research area.
{research_area_keywords}

Only assign classes for which these according keywords or variants are mentioned to be actively performed within the current research.
Use the keywords to think about what stage of research they are currently in.

### 5. Output Format:
The output should be a JSON object with the following structure:
```json
{{
    "research_areas": [
        {{
            "research_area": "str -> 1 research area",
            "relevant_input_snippet": ["List[str] -> the actual quote(s) from the input text where it takes the information from"],
            "explanation": "str -> Explanation how this explains the addition of this research area to the classification",
        }}
    ]
}}
```

Now here is the research that needs to be classified:

- **Title:** {title}
- **Abstract:** {abstract}
"""


def get_therapeutics_validation_prompt(title, abstract, prediction):
    return f"""
You are an AI specialized in validating the classification of research papers in the Therapeutics area. Your task is to verify if the given classification is correct based on the paper's title and abstract.

**Input:**
- **Title:** {title}
- **Abstract:** {abstract}
- **Current Classification:** {prediction}

**Validation Rules for Therapeutics:**

1. **Discovery vs Development:**
   
   a) **Therapeutics / Discovery** (3100):
   - Research is in early stages (lab/preclinical)
   - Focuses on target identification, validation
   - Involves lead optimization
   - Preclinical testing/trials
   - No mention of clinical trials
   - The word "development" alone doesn't indicate Development phase
   
   b) **Therapeutics / Development** (3200):
   - Research involves clinical trials
   - Explicitly mentions Phase 1, 2, or 3
   - Testing in human subjects
   - Moving beyond preclinical stage
   - Must explicitly state clinical trial involvement
   - Does the title or abstract explicitly mention that there are clinical trials performed? Otherwise, classify only as "Therapeutics / Discovery".


Here are some more general guidelines:
#### a. Direct Mention:
- **Primary Goal:** Focus on identifying the major goal or objective of the research.
- **Classification Assignment:** Assign research areas that are explicitly mentioned in the title/abstract.

#### b. Preference for Single Classification:
- **Default to Single Classification:** Assign only one research area that best represents the main focus of the research.
- **When to Assign Multiple Classifications:**
  - **Rare Exceptions:** Assign multiple research areas only if the research equally and explicitly addresses multiple areas as main objectives.
  - **Equal Focus Required:** Multiple classifications are permitted only when multiple research areas are of equal significance and centrality in the research.
  - **Main Focus Rule:** If one research area is clearly the primary focus (e.g., constitutes 80% or more of the content), do not assign additional classifications even if other areas are mentioned.
  - **Avoid Overclassification:** Do not assign multiple classifications simply because multiple topics are mentioned; focus on the single main goal of the research.

#### c. Exclude External References:
- **Ignore:** References to other works, related studies, citations, or mentions of earlier work.
- **Focus:** Only on topics directly addressed in the current research.

### Classification Choices:
{research_area_options}

**Output Format:**
```json
{{
    "validation_result": {{
        "is_correct": true/false,
        "correct_classification": "str -> the correct classification if current is wrong, otherwise null",
        "evidence": ["List[str] -> relevant quotes from input"],
        "explanation": "str -> detailed explanation of the validation decision",
    }}
}}
```
"""


# 2. **Phase-Specific Classifications:**
#    - Only assign Phase 1, 2, or 3 if explicitly mentioned
#    - Default to general Development if phase isn't specified
#    - Don't assign multiple phases simultaneously
