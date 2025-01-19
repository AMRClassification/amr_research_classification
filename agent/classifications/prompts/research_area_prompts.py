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

### 2. Additional Information:
{research_area_additional_info} 

### 3. Classification Rules:

#### a. Direct Mention:
- **Primary Goal:** Focus on identifying the major goal or objective of the research.
- **Classification Assignment:** Assign research areas that are explicitly mentioned in the title/abstract as defined by the keywords.

#### b. Preference for Single Classification:
- **Default to Single Classification:** Assign only one research area that best represents the main focus of the research.
- **When to Assign Multiple Classifications:**
  - **Rare Exceptions:** Assign multiple research areas only if the research equally and explicitly addresses multiple areas as main objectives.
  - **Equal Focus Required:** Multiple classifications are permitted only when multiple research areas are mentioned as separate efforts with clear significance and focus in the research.
  - **Main Focus Rule:** If one research area is clearly the primary focus (e.g., constitutes 80% or more of the content), do not assign additional classifications even if other areas are mentioned.
  - **Discovery to Clinical Testing Exception:** If the research explicitly covers the complete process from hit discovery (including target assessment, validation, hit identification, hit to lead, lead identification, lead optimization) through to testing in the lab, classify as BOTH Discovery AND Clinical Testing.
  - **Avoid Overclassification:** Do not assign multiple classifications simply because multiple topics are mentioned; focus on the single main goal of the research unless multiple efforts are clearly mentioned separately.

#### c. Exclude External References:
- **Ignore:** References to other works, related studies, citations, or mentions of earlier work.
- **Focus:** Only on topics directly addressed in the current paper.


#### 4. Keywords:
Focus on the following keywords to determine the research area.
{research_area_keywords}

Only assign classes for which these according keywords or variants are mentioned to be actively performed within the current research.
Use the keywords to determine what stage of research they are currently in.

### 5. Output Format:
The output should be a JSON object with the following structure:
```json
{{
    "research_areas": [
        {{
            "research_area": "str -> 1 research area; make sure you return the exact glassification strings as in the classification choices",
            "relevant_input_snippet": [
                {{
                    "text_snippet": "str -> the actual quote(s) from the input text where it takes the information from",
                    "corresponding_keyword": "str -> the keyword from the Kewords section that this connects to",
                    "keyword_paragraph": "str -> the paragraph in the Keywords section, where the keyword is drawn in the following format: [Vaccines / Clinical Testing]"
                }}
            ],
            "explanation": "str -> Explanation how this explains the addition of this research area to the classification",
        }}
    ]
}}
```

Now, perform the classification for the following research paper given only these classification choices:
- **Title:** {title}
- **Abstract:** {abstract}
"""


def get_therapeutics_validation_prompt(title, abstract, prediction):
    prompt = f"""
You are an AI specialized in validating the classification of research papers in the Therapeutics area. Your task is to verify if the current given classification is correct for the paper's title and abstract.

**Input:**
- **Title:** {title}
- **Abstract:** {abstract}

- **Current Classification:** {prediction}

**Validation Rules for Therapeutics:**

1. **Discovery vs Clinical Testing:**
   
   a) **Therapeutics / Discovery** (3100):
   - Research is in early stages (lab/preclinical)
   - Focuses on target identification, validation
   - Involves lead optimization
   - Preclinical testing/trials
   - No mention of clinical trials
   
   b) **Therapeutics / Clinical Testing** (3200):
   - Research involves clinical trials
   - Explicitly mentions Phase 1, 2, or 3
   - Testing in human subjects
   - Moving beyond preclinical stage
   - Must explicitly state clinical trial involvement
   - Does the title or abstract explicitly mention that there are clinical trials performed? Otherwise, classify only as "Therapeutics / Discovery"

   c) **Both Discovery and Clinical Testing:**
   - Research explicitly covers the complete process from hit discovery through clinical phases
   - Must include target assessment, validation, hit identification, hit to lead, lead identification, lead optimization
   - AND explicitly mention actual involvement in clinical trials
   - Both aspects must be active parts of the current research, not just future plans


Here are some more general guidelines:
#### a. Direct Mention:
- **Primary Goal:** Focus on identifying the major goal or objective of the research.
- **Classification Assignment:** Assign research areas that are explicitly mentioned in the title/abstract.

#### b. Preference for Single Classification:
- **Default to Single Classification:** Assign only one research area that best represents the main focus of the research.
- **When to Assign Multiple Classifications:**
  - **Rare Exceptions:** Assign multiple research areas only if the research equally and explicitly addresses multiple areas as main objectives.
  - **Equal Focus Required:** Multiple classifications are permitted only when multiple research areas are mentioned as separate efforts with clear significance and focus in the research.
  - **Main Focus Rule:** If one research area is clearly the primary focus (e.g., constitutes 80% or more of the content), do not assign additional classifications even if other areas are mentioned.
  - **Avoid Overclassification:** Do not assign multiple classifications simply because multiple topics are mentioned; focus on the single main goal of the research unless multiple efforts are clearly mentioned separately.

#### c. Exclude External References:
- **Ignore:** References to other works, related studies, citations, or mentions of earlier work.
- **Focus:** Only on topics directly addressed in the current research.

### Classification Choices:
{research_area_options}

**Output Format:**
```json
{{
    "validation_result": {{
        "is_correct": true/false -> indicating if the original classifications are correct,
        "correct_classification": ["List[str] -> the correct classifications"],
        "evidence": ["List[str] -> relevant quotes from input"],
        "explanation": "str -> brief explanation of the validation decision",
    }}
}}
```
"""
    return prompt



def get_research_area_validation_prompt(title, abstract, prediction):
    return f"""
You are an AI specialized in validating the classification of research papers into research areas. Your task is to verify if the current given classification is correct for the paper's title and abstract.

**Input:**
- **Title:** {title}
- **Abstract:** {abstract}
- **Current Classification:** {prediction}

**Classification Choices:**
{research_area_options}

### 1. Additional Information Categories and Subcategories:
{research_area_additional_info}

**Validation Rules:**

### 2. Classification Rules:

#### a. Direct Mention:
- **Primary Goal:** Focus on identifying the major goal or objective of the research.
- **Classification Assignment:** Assign research areas that are explicitly mentioned in the title/abstract as defined by the keywords.

#### b. Preference for Single Classification:
- **Default to Single Classification:** Assign only one research area that best represents the main focus of the research.
- **When to Assign Multiple Classifications:**
  - **Rare Exceptions:** Assign multiple research areas only if the research equally and explicitly addresses multiple areas as main objectives.
  - **Equal Focus Required:** Multiple classifications are permitted only when multiple research areas are mentioned as separate efforts with clear significance and focus in the research.
  - **Main Focus Rule:** If one research area is clearly the primary focus (e.g., constitutes 80% or more of the content), do not assign additional classifications even if other areas are mentioned.
  - **Discovery to Clinical Testing Exception:** If the research explicitly covers the complete process from hit discovery (including target assessment, validation, hit identification, hit to lead, lead identification, lead optimization) through to testing in the lab, classify as BOTH Discovery AND Clinical Testing.
  - **Avoid Overclassification:** Do not assign multiple classifications simply because multiple topics are mentioned; focus on the single main goal of the research unless multiple efforts are clearly mentioned separately.
  - **Clinical Testing Requirement:** Never classify as Clinical Testing if there is no explicit mention that clinical trials (one of the phases 1-3) are being performed currently.
  - **Preclinical Testing:** In case preclinical testing is explicitly said to be performed, classify as Discovery.
  
#### c. Exclude External References:
- **Ignore:** References to other works, related studies, citations, or mentions of earlier work.
- **Focus:** Only on topics directly addressed in the current research.

#### 3. Keywords:
Focus on the following keywords and variants you find in the input text to determine the research area.
{research_area_keywords}

**Output Format:**
```json
{{
    "validation_result": {{
        "is_correct": true/false -> indicating if the original classifications are correct,
        "correct_classification": ["List[str] -> the correct classifications"],
        "evidence": ["List[str] -> relevant quotes from input"],
        "explanation": "str -> Brief explanation of the validation decision. Never use words like "indication", but instead infer from the text and don't guess whether a phase could be present,
    }}
}}
```
"""


def get_research_area_validation_review_prompt(title, abstract, validation_result):
    return f"""
You are an AI specialized in determining if a research area classification should be marked as "uncertain". Your task is to evaluate if the research stage is clearly identifiable or if there is significant ambiguity.

**Input to Review:**
Title: {title}
Abstract: {abstract}

Validation Result: {validation_result}

**Classification Choices:**
{research_area_options}

**Keywords:**
{research_area_keywords}

**Review Instructions:**
1. Evaluate if the research stage is unclear due to:
   - Multiple stages being mentioned without a clear main focus
   - A transition between stages where neither is clearly dominant
   - Ambiguity about which stage is currently being researched

2. Focus specifically on:
   - What research activities are currently happening
   - What is explicitly planned for the immediate future
   - Whether one stage clearly dominates the research focus, in this case it should only classify this stage, and if this happened, it shall be reviewed as "certain"

3. Consider marking as Uncertain if:
   - Multiple stages are discussed with similar emphasis
   - The research appears to be between stages
   - Cannot clearly determine which stage is the primary focus
   - The timeline or progression between stages is unclear
   - Note that multiple stages are only allowed when they are mentioned as separate efforts with clear significance and focus in the research.
   - The argumentation in the validation is not clear or logical
   - You don't agree with the classification result based on the reasoning laid out in the validation

**Note:**
- In the case of distinction between Discovery and Clinical Testing, if the title/abstract doesn't explicitly mention clinical trials, this is a clear indicator for being classified as Discovery and should still be considered as "certain". However, sometimes from the wording it is hard to tell how far the research already progressed (e.g. "We will demonstrate the preclinical safety, pharmacokinetics and efficacy of APC247, and advance this candidate into the first FIH clinical trial to provide human dose and safety data."), and in these cases it should be reviewed as "uncertain". But if there clearly isn't any clinical trials mentioned and the validation result is it shall be classified as Discovery, it should be classified as "certain".
- If the research covers the complete discovery process (including target assessment & validation, hit identification, hit to lead, lead identification, lead optimization) AND continues into clinical phases, it should be classified as BOTH Discovery and Clinical Testing
- If there is a clear focus that can be identified and the respective classification was made, it should be classified as "certain"
- A validation result of INCORRECT doesn't mean the review should be uncertain, as long as the explanation of the correction during validation is logical and correct
- If the research expresses hope or intention to reach testing stage in the future (e.g. "we hope to advance to clinical trials"), this should be marked as "uncertain" since the testing stage is not yet confirmed

**Output Format:**
Return a JSON object with the following structure:
{{
    "review_result": {{
        "status": str,  # One of: "certain", "uncertain"
        "reason": str,  # Brief explanation of why the stage is clear or unclear
        "analysis": str,  # Brief analysis of the current and planned research activities (do not include the input text from the former validation results)
        "possible_areas": [str]  # List of research areas that could potentially apply
    }}
}}

Where:
- "certain" means one or more research stages are clearly dominant
- "uncertain" means the research stages cannot be confidently determined

Only classify as uncertain if the argumentation of the validation is unclear or wrong, or if the title/abstract are ambigious in terms of the correct answer
"""
