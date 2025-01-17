from utils.utils import get_categories
from typing import List

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

1. **Classification Rules:**

a. **Direct Mention Only:**
    - Only classify infectious agents that are directly and concretely mentioned in the title and abstract; do not infer agents.
    - If examples of infectious agents are mentioned using phrases like “such as”, “including”, or “for example” but they arent the concrete focus of this research, do not consider them.
    - If infectious agents are mentioned without any of these "example-like" phrases, this is a strong indication that this agent is the main focus
    - If infectious agents are introduced as examples in the beginning but are not the main focus or are not further discussed for concrete treatment, don't consider them.
    - If the research keeps mentionioning the same infectious agent(s) this is a strong indication that this agent actually is the main focus

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
    - if the research is focused on ESKAPE, include all of the above infectious agents in the classification. If it only focuses on one of the two stains, add the according infectious agent

2. **Context Validation:**
   - Verify agents are not just mentioned in background/introduction
   - Confirm agents are actively studied in the current research
   - Check that classified agents are not just examples or peripheral references from or related to former work, related research or citations

3. **Common Errors to Check:**
   - Over-classification: A specific infectious agent is only mentioned as an example of a general group of infectious agents, then the general category should be classified
   - Over-classification: If they mention that a test is made on one infectious agent but they explicitly mention that the treatment is focused towards a general group of infectious agents, then the general category should be classified
   - Under-specification: Using general categories when specific infectious agents are explicitly mentioned as targets for treatment development - in these cases, the specific infectious agent should be classified and not the general category
   - Missing classifications: Overlooking clearly mentioned agents
   - Context errors: Mistakenly including agents from referenced studies 
   - Mistakenly inference: Inferring agents because they are "often related with..." or "common for...", instead only consider agents that are explicitly mentioned
   - If there is a list of infectious agents, did you really include all of them?

**Classification Choices:**
{infectious_agent_options}

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

3. Focus specifically on:
   - Which agents are explicitly mentioned as the main research target, inferring ideas that are not explicitly mentioned in the text are no valid
   - If a specific infectious agent is used for testing/experiments but the research focuses on treating a broader group of infectious agents (e.g., "tested on E. coli but effective against gram-negative bacteria"), classify according to the broader group that is explicitly mentioned as treatable
   - Do not specify individual test organisms if they are only used as representative examples for demonstrating effectiveness against a larger group of pathogens
   - It is not about if one infectious agent has priority over another one, but finding out which all are investigated in the research

4. Consider marking as Uncertain if:
   - It's hard to tell if one or multiple specific pathogen are being targeted or the respective overarching category of them is the main focus
      - If the research only mentions examples of pathogens but doesn't mention concrete focus on them, and we correctly classify it as Not Specified, this should still be marked as certain 
      - Conversely, if the research clearly mentions a specific pathogen(s) and we correctly classify it with this or those infectious agents, this should be marked as certain
   - The argumentation in the validation is not clear or logical
   - You don't agree with the classification result based on the reasoning laid out in the validation, e.g. not all mentioned infectious agents are included in the validation result, or the wrong category was chosen

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


#---------------------------------------------------------------------------------------------------



def get_mentions_group_prompt(title: str, abstract: str) -> str:
    return f"""Given the following title and abstract, determine if it mentions any broad groups of infectious agents (bacteria, fungi, parasites, viruses).

Title: {title}
Abstract: {abstract}

If any infectious agents from one of these groups are mentioned to be target of the research add the corresponding class to the found_groups list:
{{
    "Bacteria": "1500 Infectious Agent / Bacteria / Bacteria / Not Specified_Bacteria",
    "Fungus": "1601 Infectious Agent / Fungus / Fungus / Not Specified_Fungus", 
    "Parasite": "1700 Infectious Agent / Parasite / Parasite / Not Specified_Parasite",
    "Protozoa": "1712 Infectious Agent / Parasite / Protozoa / Not Specified_Protozoa",
    "Helminth": "1722 Infectious Agent / Parasite / Helminth / Not Specified_Helminth",
    "Virus": "1801 Infectious Agent / Virus / Virus / Not Specified_Virus"
}}

Include all infectious agents groups that are mentioned as target of the research unless:
1. the infectious agents are only mentioned as examples
2. the infectious agents are only from related work but not part of this current research
3. the infectious agents will not be looked into as part of the research 


Only if has_groups is false, add the following in the "not_applicable_or_not_specified" field:
- "not_applicable" -> the research is not related to any infectious agents
- "not_specified" -> the research is related to infectious agents but it doesn't even mention the group

Respond in JSON format:
{{
    "has_groups": boolean,
    "found_groups": [list of the full classification strings for found broad groups],
    "mentions": [list of relevant quotes from the text showing the group mentions],
    "not_applicable_or_not_specified": "not_applicable" | "not_specified",
    "explanation": "Detailed explanation of which groups were found and why"
}}"""



def get_mentions_infectious_agent_prompt(title: str, abstract: str) -> str:
    return f"""Given the following title and abstract, determine if it explicitly mentions any infectious agents, which are to be researched in this paper:

Title: {title}
Abstract: {abstract}



Only add the infectious agents for which one of the following is true:
- the infectious agent is explicitly mentioned in the title/abstract
- in case the research is focused on ESKAPE, include all of the following infectious agents in the classification:
    - Enterococcus faecium (Gram positive)
    - Staphylococcus aureus (Gram positive) 
    - Klebsiella pneumoniae (Gram negative)
    - Acinetobacter baumannii (Gram negative)
    - Pseudomonas aeruginosa (Gram negative)
    - Enterobacter species (Gram negative)
  - if only the gram-negative or gram-positive category of ESKAPE is mentioned, include all of the infectious agents in the respective category
- in case the research mentions a medicine/treatment and that treatment is focused only on a specific infectious agent, include that agent in the classification
- infectious agents that are paraphrased by words like antipseudomonal (-> Pseudomonas), etc.
- "MRSA" is the Staphylococcus aureus


Don't come up with infectious agents that are not mentioned in the title/abstract.

Respond in JSON format:
{{
    "has_infectious_agent": boolean,
    "found_agents": [list of infectious agents that match exactly with the provided categories (write the full classification string as given in the Infectious Agents Class List)],
    "mentions": [list of relevant quotes from the text showing mentions of  agents],
    "explanation": "Detailed explanation of your decision and which agents were found"
}}

Example response:
{{
    "has_infectious_agent": true,
    "found_agents": [
        "Pseudomonas aeruginosa",
        "Staphylococcus aureus"
    ],
    "mentions": [
        "MRSA infections are becoming increasingly difficult to treat",
        "antipseudomonal activity was observed"
    ],
    "explanation": "The text mentions MRSA which refers to Staphylococcus aureus, and uses the term 'antipseudomonal' which indicates Pseudomonas aeruginosa as a target organism."
}}
"""

def get_example_check_prompt(title: str, abstract: str, agent: str) -> str:
    return f"""Analyze how the infectious agent '{agent}' is mentioned in the text and determine if it's a primary research target.

Title: {title}
Abstract: {abstract}

Consider it a research target (true) if atleast one of the following applies:
1. It appears in the title
2. It's explicitly mentioned as the being (one of) the target(s) of the currently performed research, which means a eventually resulting treatment would be focused on this specific infectious agent (but it is not just mentioned as an example for a larger group of infectious agents by indicating phrases like "such as", "for example", "including")
3. It keeps recurring throughout the text, especially in methodology and results
4. It's discussed in detail in the research findings

Consider it NOT a research target (false) if:
1. It's mentioned as an example (e.g., "such as", "for example", "including")
2. It's only mentioned in context of other research or background
3. It appears only in passing or comparative references
4. There's uncertainty about whether it's directly addressed 

Not Specified Categories for agent_group:
{{
    "Bacteria": "1500 Infectious Agent / Bacteria / Bacteria / Not Specified_Bacteria",
    "Fungus": "1601 Infectious Agent / Fungus / Fungus / Not Specified_Fungus", 
    "Parasite": "1700 Infectious Agent / Parasite / Parasite / Not Specified_Parasite",
    "Protozoa": "1712 Infectious Agent / Parasite / Protozoa / Not Specified_Protozoa",
    "Helminth": "1722 Infectious Agent / Parasite / Helminth / Not Specified_Helminth",
    "Virus": "1801 Infectious Agent / Virus / Virus / Not Specified_Virus",
}}

Never classify as a target if the infectious agent's name does not appear explicitly in the text (common abbreviations like MRSA, S. aureus, etc. are allowed). You must take reference to at least one text snippet in your explanation that shows where the infectious agent is directly mentioned by its own name (not inferring from other phrases).
For example the explanation "Although it is not mentioned directly by name, the focus on 'the spread of pathogens and antibiotic resistance genes' indicates a serious concern about bacterial agents, including Staphylococcus spp." is not allowed!!!

If an infectious agent is a super group of another infectious agent, classify the super group as not being target, but only classify the individual infectious agent as target.
Here is a list of all classifications that can be made:
{infectious_agent_options}

Respond in JSON format:
{{
    "is_target": boolean,

    "mentions": [list of any mentions in the text by giving the relevant sentence snippet for the agent_classificaiton],
    "agent_group": the full string of the Not Specified category of the respective larger group of the infectious agent (e.g. "1500 Infectious Agent / Bacteria / Bacteria / Not Specified_Bacteria" for any gram-negative, gram-positive or gram-variable bacterial infectious agent),
    "explanation": "Justification of the categorization decision and relation to the specific mention that you base the decision on"
}}"""



def get_unlisted_example_check_prompt(title: str, abstract: str, agent: str) -> str:
    return f"""Analyze how the infectious agent '{agent}' is mentioned in the text and determine if it's a primary research target.

Title: {title}
Abstract: {abstract}

Consider it a research target (true) if atleast one of the following applies:
1. It appears in the title
2. It's explicitly mentioned as the being (one of) the target(s) of the currently performed research, which means a eventually resulting treatment would be focused on this specific infectious agent (but it is not just mentioned as an example for a larger group of infectious agents by indicating phrases like "such as", "for example", "including")
3. It keeps recurring throughout the text, especially in methodology and results
4. It's discussed in detail in the research findings

Consider it NOT a research target (false) if:
1. It's mentioned as an example (e.g., "such as", "for example", "including")
2. It's only mentioned in context of other research or background
3. It appears only in passing or comparative references
4. There's uncertainty about whether it's directly addressed 

Other Categories for agent_group:
{{
    "Gram negative": "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative",
    "Gram positive": "1513 Infectious Agent / Bacteria / Gram positive / Other Gram positive",
    "Gram variable": "1523 Infectious Agent / Bacteria / Gram variable / Other Gram variable",
    "Fungus": "1602 Infectious Agent / Fungus / Fungus / Other_Fungus",
    "Parasite": "1702 Infectious Agent / Parasite / Other_Parasite",
    "Protozoa": "1713 Infectious Agent / Parasite / Protozoa / Other_Protozoa",
    "Helminth": "1723 Infectious Agent / Parasite / Helminth / Other_Helminth",
    "Virus": "1802 Infectious Agent / Virus / Virus / Other_Virus"
}}

Never classify as a target if the infectious agent's name does not appear explicitly in the text (common abbreviations like MRSA, S. aureus, etc. are allowed). You must take reference to at least one text snippet in your explanation that shows where the infectious agent is directly mentioned by its own name (not inferring from other phrases).
For example the explanation "Although it is not mentioned directly by name, the focus on 'the spread of pathogens and antibiotic resistance genes' indicates a serious concern about bacterial agents, including Staphylococcus spp." is not allowed!!!

First check if this agent is actually in the classification list:
{infectious_agent_options}

Respond in JSON format:
{{
    "is_target": boolean,

    "is_in_list": boolean,  # Whether the agent is found in the classification list
    "associated_entry_in_list": str,  # If is_in_list is true, provide the full classification string from the list

    "mentions": [list of any mentions in the text by giving the relevant sentence snippet for the agent_classificaiton],
    "agent_group": the full string of the respective Other category of the infectious agent (e.g. "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative" for an unlisted gram-negative bacterial infectious agent),
    "explanation": "Justification of the categorization decision and relation to the specific mention that you base the decision on"
}}"""


def get_agent_classification_prompt(title: str, abstract: str, agents: List[str]) -> str:
    return f"""Analyze how each of the following infectious agents is mentioned in the text and categorize them according to the rules below.

Title: {title}
Abstract: {abstract}

Agents to analyze: {agents}


Consider it a research target if atleast one of the following applies:
1. It appears in the title
2. It's explicitly mentioned as the being (one of) the target(s) of the currently performed research, which means a eventually resulting treatment would be focused on this specific infectious agent (but it is not just mentioned as an example for a larger group of infectious agents by indicating phrases like "such as", "for example", "including")
3. It keeps recurring throughout the text, especially in methodology and results
4. It's discussed in detail in the research findings

Consider it NOT a research target if:
1. It's mentioned as an example following words like "such as", "for example", "including", etc.
2. It's only mentioned in context of other research or background
3. It appears only in passing or comparative references
4. There's uncertainty about whether it's directly addressed 
5. The actual target is a larger group of infectious agents, and the infectious agent is only used as an research example.


If it IS a research target, check if it's in this classification list:
{infectious_agent_options}

If it's in the list, categorize as "in_list" and provide the full classification string.
If it's not in the list, categorize as "not_in_list" and provide the appropriate Other category using these mappings:
{{
    "Gram negative": "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative",
    "Gram positive": "1513 Infectious Agent / Bacteria / Gram positive / Other Gram positive",
    "Gram variable": "1523 Infectious Agent / Bacteria / Gram variable / Other Gram variable",
    "Fungus": "1602 Infectious Agent / Fungus / Fungus / Other_Fungus",
    "Parasite": "1702 Infectious Agent / Parasite / Other_Parasite",
    "Protozoa": "1713 Infectious Agent / Parasite / Protozoa / Other_Protozoa",
    "Helminth": "1723 Infectious Agent / Parasite / Helminth / Other_Helminth",
    "Virus": "1802 Infectious Agent / Virus / Virus / Other_Virus"
}}

Note that it is also possible to classify all agents as "not_targeted", for example if they are just mentioned as examples for a larger group and play no explicit larger role in the paper.

Respond in JSON format:
{{
    "agent_classifications": [
        {{
            "agent": str,  # Name of the agent
            "category": "in_list" | "not_in_list" | "not_targeted",
            "class": str | null,  # Full classification string or Other category if applicable, null if not_targeted
            "evidence": [str],  # Relevant quotes from text supporting this classification
            "explanation": str  # Brief explanation of why this categorization was chosen
        }}
    ],
    "analysis_summary": str  # Brief overview of the classification decisions
}}"""

# Consider these rules:
#     "Other": "1900 Infectious Agent / Other / Other_Other",
#     "Not Applicable": "1902 Infectious Agent / Not Applicable / Not Applicable",
#     "Not Specified": "1901 Infectious Agent / Not Specified / Not Specified_InfectiousAgent"

#    For unspecified infectious agents:
#       - If research relates to infectious agents but the category is not mentioned, classify as:
#          - "1901 Infectious Agent / Not Specified / Not Specified_InfectiousAgent"
#    For non-infectious agent research:
#       - If research has no relation to infectious agents, classify as:
#          - "1902 Infectious Agent / Not Applicable / Not Applicable"