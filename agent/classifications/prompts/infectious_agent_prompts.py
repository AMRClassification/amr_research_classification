from utils.utils import get_categories

infectious_agent_options = get_categories("Infectious Agent")


def get_classification_prompt(title, abstract):
    return f"""You are an AI specialized in classifying research papers on antimicrobial resistance into relevant infectious agents based on their title and abstract. Follow the instructions and specifications below to determine the appropriate infectious agent(s).

**Instructions:**

1. **Classification Choices:**
    {infectious_agent_options}

2. **Classification Rules:**

a. **Direct Mention Only:**
    - Only classify infectious agents that are directly and concretely mentioned in the title and abstract; do not infer agents.
    - If examples of infectious agents are mentioned using phrases like “such as”, “including”, or “for example” but they arent the concrete focus of this research, do not consider them.
    - If infectious agents are introduced as examples in the beginning but are not the main focus or are not further discussed for concrete treatment, don't consider them.

    - Classification Based on Specificity:
        1. For bacteria without specified type:
            - If bacteria are mentioned without indicating Gram status, classify as:
                - "1500 Infectious Agent / Bacteria / Bacteria / Not Specified_Bacteria"
            - Similarly use corresponding "Not Specified" classifications for other unspecified categories
        2. For specified Gram status:
            - If Gram negative bacteria are mentioned but not specified, or if specific unlisted, classify as:
                - "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative"
            - Apply same rule for Gram positive/variable or other infectious agents (parasites, fungi, etc.) using appropriate "Other" category
        3. For unspecified infectious agents:
            - If research relates to infectious agents but the category is not mentioned, classify as:
                - "1901 Infectious Agent / Not Specified / Not Specified_InfectiousAgent"
        4. For non-infectious agent research:
            - If research has no relation to infectious agents, classify as:
                - "1902 Infectious Agent / Not Applicable / Not Applicable"
    - Specific vs. General Mentions:
        - For specific agents not in classification list, use appropriate "Other" category:
            - For unlisted Gram negative bacteria: "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative"
            - For unlisted Gram positive bacteria: "1513 Infectious Agent / Bacteria / Gram positive / Other Gram positive"

b. **Multiple Classifications:**
    - Assign multiple classifications when:
        - Multiple infectious agents are explicitly mentioned as main topics
        - Research describes multiple efforts targeting different agents/categories
    - Use single classification when:
        - One agent/category is primary focus
        - Other mentions are only examples or peripheral references

c. **Exclude External References:**
    - Ignore:
        - References to other resources or studies
        - Infectious agents mentioned only in context of other research
    - Focus only on infectious agents directly studied in current research

d. **Definition of ESKAPE:**
    - If the research mentions ESKAPE, this includes the following agents:
        - Enterococcus faecium (Gram positive)
        - Staphylococcus aureus (Gram positive) 
        - Klebsiella pneumoniae (Gram negative)
        - Acinetobacter baumannii (Gram negative)
        - Pseudomonas aeruginosa (Gram negative)
        - Enterobacter species (Gram negative)

3. **Output Format:**

    - The output should be a JSON array of objects. Each object should have the following structure:
    ```json
    {{
        "infectious_agents": [
            {{
                "infectious_agent": "str -> 1 agent",
                "relevant_input_snippet": ["List[str] -> the actual quote(s) from the input text where it takes the information from"],
                "explanation": "str -> Explanation how this explains the addition of this infectious agent to the classification",
            }}
        ]
    }}
    ```

Now, perform the classification for the following research paper given only these classification choices:
    - **Title:** {title}
    - **Abstract:** {abstract}

**Output:**
    """


def get_infectious_agent_validation_prompt(title, abstract, prediction):
    return f"""
You are an AI specialized in validating the classification of research papers for infectious agents. Your task is to verify if the current given classification is correct for the paper's title and abstract.

**Input:**
- **Title:** {title}
- **Abstract:** {abstract}

- **Current Classification:** {prediction}

**Validation Rules for Infectious Agents:**

1. **Direct Mention Rule:**
   - Only validate infectious agents that are directly and concretely mentioned
   - Reject classifications based on inference or indirect references
   - Verify that classified agents are the actual focus of the research, not just examples
   - Focus especially on the main topic which is also disscussed in the latter parts of the abstract if they go into more detail, and not just mentioned as an example in the beginning

2. **Specificity Rules:**
   a. For bacteria without specified type:
      - If bacteria are mentioned without indicating Gram status, classify as:
         - "1500 Infectious Agent / Bacteria / Bacteria / Not Specified_Bacteria"
      - Similarly use corresponding "Not Specified" classifications for other unspecified categories
   b. For specified Gram status:
      - If Gram negative bacteria are mentioned but not specified, or if specific unlisted, classify as:
         - "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative"
      - Apply same rule for Gram positive/variable or other infectious agents (parasites, fungi, etc.) using appropriate "Other" category
   c. For unspecified infectious agents:
      - If research relates to infectious agents but the category is not mentioned, classify as:
         - "1901 Infectious Agent / Not Specified / Not Specified_InfectiousAgent"
   d. For non-infectious agent research:
      - If research has no relation to infectious agents, classify as:
         - "1902 Infectious Agent / Not Applicable / Not Applicable"

   Note:
   - Do not specify individual test organisms if they are only used as representative examples for demonstrating effectiveness against a larger group of pathogens. This is indicated when infectious agents are mentioned by phrases such as "such as", "including" or "among others". If the infectious agents aren't then further mentioned in detail in the following texts, in this case the classification should be "Not Specified" of the respective category
   - If the research keeps mentionioning the same infectious agent this is a strong indication that this agent actually is the main focus
         
   e. **Definition of ESKAPE:**
    - If the research mentions ESKAPE, this includes the following agents:
        - Enterococcus faecium (Gram positive)
        - Staphylococcus aureus (Gram positive) 
        - Klebsiella pneumoniae (Gram negative)
        - Acinetobacter baumannii (Gram negative)
        - Pseudomonas aeruginosa (Gram negative)
        - Enterobacter species (Gram negative)


3. **Multiple Classifications:**
   - Validate that multiple classifications are only used when:
     * Multiple agents are explicitly studied as main topics
     * Research clearly describes multiple efforts targeting different agents
   - Confirm single classification when one agent is the primary focus

4. **Context Validation:**
   - Verify agents are not just mentioned in background/introduction
   - Confirm agents are actively studied in the current research
   - Check that classified agents are not just examples or peripheral references

5. **Common Errors to Check:**
   - Over-classification: Including agents mentioned only as examples for a larger category
   - Over-classification: A specific infectious agent is used as an testing example to produce a treatment for a larger group of infectious agents. The larger group must however be explicitly mentioned
   - Under-specification: Using general categories when the targeted infectious agent is specifically mentioned
   - Missing classifications: Overlooking clearly mentioned agents
   - Context errors: Including agents from referenced studies 
   - Mistakenly inference: Inferring agents because they are "often related with..." or "common for...", instead only consider agents that are explicitly mentioned

**Classification Choices:**
{infectious_agent_options}

**Output Format:**
```json
{{
    "validation_result": {{
        "is_correct": true/false -> indicating if the current classification is correct,
        "correct_classification": "str -> the correct classification if current is wrong, otherwise null",
        "evidence": ["List[str] -> relevant quotes from input"],
        "explanation": "str -> brief explanation of the validation decision",
    }}
}}
```

Validate the classification based on these rules and provide a clear explanation of your decision.
"""


def get_infectious_agent_validation_review_prompt(title, abstract, validation_result):
    return f"""
You are an AI specialized in determining if an infectious agent classification should be marked as Uncertain. Your task is to evaluate if the infectious agent is clearly identifiable or if there is significant ambiguity.

**Input to Review:**
Title: {title}
Abstract: {abstract}

Validation Result: {validation_result}

**Classification Choices:**
{infectious_agent_options}

**Review Instructions:**
1. Evaluate if the infectious agent is unclear due to:
   - Multiple agents mentioned without clear primary focus
   - Ambiguous specificity level (e.g., general bacteria vs specific strain)
   - Lack of explicit agent information

2. Consider these rules in case the title/abstract doesn't include a explicitly mentioned infectious agent:
      a. For bacteria without specified type:
      - If bacteria are mentioned without indicating Gram status, classify as:
         - "1500 Infectious Agent / Bacteria / Bacteria / Not Specified_Bacteria"
      - Similarly use corresponding "Not Specified" classifications for other unspecified categories
   b. For specified Gram status:
      - If Gram negative bacteria are mentioned but not specified, or if specific unlisted, classify as:
         - "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative"
      - Apply same rule for Gram positive/variable or other infectious agents (parasites, fungi, etc.) using appropriate "Other" category
   c. For unspecified infectious agents:
      - If research relates to infectious agents but the category is not mentioned, classify as:
         - "1901 Infectious Agent / Not Specified / Not Specified_InfectiousAgent"
   d. For non-infectious agent research:
      - If research has no relation to infectious agents, classify as:
         - "1902 Infectious Agent / Not Applicable / Not Applicable"

   - If a specific infectious agent is used for testing/experiments but the research focuses on treating a broader group of infectious agents (e.g., "tested on E. coli but effective against gram-negative bacteria"), classify according to the broader group that is explicitly mentioned as treatable
   - Do not specify individual test organisms if they are only used as representative examples for demonstrating effectiveness against a larger group of pathogens


2. Focus specifically on:
   - Which agents are explicitly mentioned as the main research target, inferring ideas that are not explicitly mentioned in the text are no valid
   - Whether one agent clearly dominates (>80%) the research focus
   - If the research targets multiple agents equally
   - If agents are only mentioned as examples, consider them as the targeted pathogen throughout

3. Consider marking as Uncertain if:
   - It's hard to tell if one or multiple specific pathogen are being targeted or the respective overarching category of them is the main focus
      - If the research only mentions examples of pathogens but doesn't mention concrete focus on them, and we correctly classify it as Not Specified, this should still be marked as certain 
      - Conversely, if the research clearly mentions a specific pathogen(s) and we correctly classify it with this or those infectious agents, this should be marked as certain
   - The argumentation in the validation is not clear or logical
   - You don't agree with the classification result based on the reasoning laid out in the validation

   Note: If the title/abstract don't include a explicitly mentioned infectious agent, that doesnt necessarily mean it's uncertain. For these cases the Other and Not Specified classifications are used. Its only uncertain if the correct answer can not be clearly determined on the title/abstract.

**Output Format:**
Return a JSON object with the following structure:
{{
    "review_result": {{
        "status": str,  # One of: "certain", "uncertain"
        "reason": str,  # Brief explanation of why the agent is clear or unclear
        "analysis": str,  # Brief analysis of the agent focus (do not include the input text from the former validation results)
        "possible_agents": [str]  # List of agents that could potentially apply
    }}
}}

Where:
- "certain" means one or more research stages are clearly dominant
- "uncertain" means the research stages cannot be confidently determined

Only classify as uncertain if the argumentation of the validation is unclear or wrong, or if the title/abstract are ambigious in terms of the correct answer
"""
