from utils.utils import get_categories, get_additional_info, get_keywords


research_area_options = get_categories("Research Area")
research_area_additional_info = get_additional_info("Research Area")
research_area_keywords = get_keywords("Research Area")


def get_preselection_prompt(title, abstract):
    """Prompt for the initial preselection of potential research areas."""
    prompt = f"""
You are an AI specialized in identifying potential research areas for papers on antimicrobial resistance. Your task is to analyze the title and abstract and identify which research areas could potentially apply based on the general descriptions below.

**Available Research Areas:**
{research_area_options}

**Research Area Descriptions:**
{research_area_additional_info}

**Instructions:**
1. Read the title and abstract carefully
2. Compare the content with the research area descriptions
3. Identify potential research areas from the available options above that could apply based on the general descriptions
4. For each potential area, provide reasoning based on the content matching the description
5. Only suggest areas from the provided list of available research areas
6. Use the exact spelling of the classes including the numbers and uppercase/lowercase letters

**Input:**
- **Title:** {title}
- **Abstract:** {abstract}

**Output Format:**
```json
{{
    "potential_areas": [
        {{
            "research_area": "str -> exact complete name from available options",
            "reasoning": "str -> explanation why this area could apply based on the description"
        }}
    ]
}}
```

Note: This is an initial screening step in a multi-stage pipeline. Please be comprehensive and include all potentially relevant areas, even if you're not completely certain. Over-inclusion at this stage is preferable to missing potential matches, as subsequent steps will refine these candidates.
"""
    return prompt

def get_relevant_info_prompt(title, abstract, potential_areas):
    prompt = f"""
Given a research paper's title and abstract, extract the relevant paragraphs from the additional information that match the potential research areas identified.

Title: {title}
Abstract: {abstract}

Potential Research Areas identified:
{potential_areas}

Additional Information Available:
{research_area_additional_info}

Instructions:
1. For each potential research area identified above, find the matching paragraphs from the additional information
2. Return the complete, unmodified paragraphs that describe those research areas
3. Only include paragraphs for the potential research areas identified
4. Do not modify, summarize or rewrite the paragraphs - return them exactly as they appear in the additional information
5. Make sure to include the relevant headings of the paragraphs
6. If applicable for any of the potential areas include the Product Development Stages and the TLR definitions

Output Format:```json
{{
    "relevant_info": "str -> The complete, unmodified paragraphs from the additional information that describe the potential research areas, separated by newlines"
}}
```
"""
    return prompt


# ### 2. Relevant Additional Information:
# {relevant_info if relevant_info else "No additional information provided"}

def get_classification_prompt(title, abstract, potential_areas=None, relevant_info=None):
    prompt = f"""
You are an AI specialized in classifying research papers on antimicrobial resistance into relevant research areas based on their title and abstract. Follow the instructions and specifications below to determine the appropriate research area(s).

**Instructions:**       

### 1. Classification Choices:
{potential_areas if potential_areas else "No potential areas provided"}

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
    return prompt   


def get_therapeutics_validation_prompt(title, abstract, prediction):
    prompt = f"""
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
    return prompt   


# 2. **Phase-Specific Classifications:**
#    - Only assign Phase 1, 2, or 3 if explicitly mentioned
#    - Default to general Development if phase isn't specified
#    - Don't assign multiple phases simultaneously
