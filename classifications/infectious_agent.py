import json
import re

from utils.utils import (
    get_categories,
    parse_non_json_response,
    get_additional_info,
    extract_json,
)
from utils.llm_call import classify_research_json


def generate_prompt(title, abstract, include_examples=True, use_response_format=False):
    infectious_agent_options = get_categories("Infectious Agent")
    infectious_agent_additional_info = get_additional_info("Infectious Agent")

    if use_response_format:
        base_prompt = f"""
            You are an AI specialized in classifying research papers on antimicrobial resistance into relevant infectious agents based on their title and abstract. Follow the instructions and specifications below to determine the appropriate infectious agent(s).

        **Instructions:**

        1. **Input:**
            - **Title:** {title}
            - **Abstract:** {abstract}

        2. **Classification Rules:**
            
            a. **Direct Mention or Inference:**
                - Only classify infectious agents that are directly mentioned or can be directly inferred from the title and abstract.
                - If no specific infectious agents are mentioned or inferred, use the appropriate category:
                    - "1900 Infectious Agent / Other / Other_Other" if the research is related to infectious agents but no specific agent is mentioned.
                    - "1901 Infectious Agent / Not Specified / Not Specified_InfectiousAgent" if it's unclear whether infectious agents are involved.
                    - "1902 Infectious Agent / Not Applicable / Not Applicable" if the research clearly does not involve or is not related to any infectious agents.
                
            b. **Multiple Classifications:**
                - Multiple infectious agent classifications are permitted **only** if multiple agents are explicitly mentioned or can be directly inferred as being the main topic.
                
            c. **Exclude External References:**
                - Ignore any parts of the text that contain references to other resources, such as related work sections or citations to other research. Only consider the topics that are the direct topic of this current research at hand.
                
        3. **Classification Choices:**
            {infectious_agent_options}

        4. **Output Format:**
            - The output should be a JSON object with the following structure:
            {{
                "infectious_agent": [list of infectious agents],
                "explanation": "explanation for the classification",
                "confidence": "float representing the confidence in the classification",
                "confidence_explanation": "explanation for the confidence"
            }}

        **Now, perform the classification for the following research paper given only these classification choices:**
"""

    else:
        base_prompt = f"""
            You are an AI specialized in classifying research papers on antimicrobial resistance into relevant infectious agents based on their title and abstract. Follow the instructions and specifications below to determine the appropriate infectious agent(s).

        **Instructions:**

        1. **Input:**
            - **Title:** {title}
            - **Abstract:** {abstract}

        2. **Classification Choices:**
            Only the terminal nodes (leaves) of the following dictionary should be considered for classification:
            "Bacteria": {
                "Bacteria": {
                    None: "1501 Infectious Agent / Bacteria / Bacteria",
                    "Not Specified": "1500 Infectious Agent / Bacteria / Bacteria / Not Specified_Bacteria",
                },
                "Gram negative": {
                    None: "1502 Infectious Agent / Bacteria / Gram negative",
                    "Other Gram negative": "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative",
                    "Acinetobacter spp.": "1504 Infectious Agent / Bacteria / Gram negative / Acinetobacter spp.",
                    "Bordetella spp.": "1504 Infectious Agent / Bacteria / Gram negative / Bordetella spp.",
                    "Campylobacter spp.": "1504 Infectious Agent / Bacteria / Gram negative / Campylobacter spp.",
                    "Enterobacter spp.": "1504 Infectious Agent / Bacteria / Gram negative / Enterobacter spp.",
                    "Enterobacteriaceae": "1504 Infectious Agent / Bacteria / Gram negative / Enterobacteriaceae",
                    "Escherichia spp.": "1504 Infectious Agent / Bacteria / Gram negative / Escherichia spp.",
                    "Haemophilus spp.": "1504 Infectious Agent / Bacteria / Gram negative / Haemophilus spp.",
                    "Helicobacter spp.": "1504 Infectious Agent / Bacteria / Gram negative / Helicobacter spp.",
                    "Klebsiella spp.": "1504 Infectious Agent / Bacteria / Gram negative / Klebsiella spp.",
                    "Morganella spp.": "1504 Infectious Agent / Bacteria / Gram negative / Morganella spp.",
                    "Neisseria spp.": "1504 Infectious Agent / Bacteria / Gram negative / Neisseria spp.",
                    "Proteus spp.": "1504 Infectious Agent / Bacteria / Gram negative / Proteus spp.",
                    "Providencia spp.": "1504 Infectious Agent / Bacteria / Gram negative / Providencia spp.",
                    "Pseudomonas spp.": "1504 Infectious Agent / Bacteria / Gram negative / Pseudomonas spp.",
                    "Salmonella spp.": "1504 Infectious Agent / Bacteria / Gram negative / Salmonella spp.",
                    "Serratia spp.": "1504 Infectious Agent / Bacteria / Gram negative / Serratia spp.",
                    "Shigella spp.": "1504 Infectious Agent / Bacteria / Gram negative / Shigella spp.",
                    "Actinobacillus spp.": "1505 Infectious Agent / Bacteria / Gram negative / Actinobacillus spp.",
                    "Aeromonas spp.": "1505 Infectious Agent / Bacteria / Gram negative / Aeromonas spp.",
                    "Anaplasma spp.": "1505 Infectious Agent / Bacteria / Gram negative / Anaplasma spp.",
                    "Bibersteinia spp.": "1505 Infectious Agent / Bacteria / Gram negative / Bibersteinia spp.",
                    "Brachyspira spp.": "1505 Infectious Agent / Bacteria / Gram negative / Brachyspira spp.",
                    "Brucella spp.": "1505 Infectious Agent / Bacteria / Gram negative / Brucella spp.",
                    "Chlamydia": "1505 Infectious Agent / Bacteria / Gram negative / Chlamydia",
                    "Dichelobacter spp.": "1505 Infectious Agent / Bacteria / Gram negative / Dichelobacter spp.",
                    "Edwardsiella spp.": "1505 Infectious Agent / Bacteria / Gram negative / Edwardsiella spp.",
                    "Ehrlichia spp.": "1505 Infectious Agent / Bacteria / Gram negative / Ehrlichia spp.",
                    "Flavobacterium spp.": "1505 Infectious Agent / Bacteria / Gram negative / Flavobacterium spp.",
                    "Fusobacterium spp.": "1505 Infectious Agent / Bacteria / Gram negative / Fusobacterium spp.",
                    "Histophilus spp.": "1505 Infectious Agent / Bacteria / Gram negative / Histophilus spp.",
                    "Lawsonia spp.": "1505 Infectious Agent / Bacteria / Gram negative / Lawsonia spp.",
                    "Leptospira spp.": "1505 Infectious Agent / Bacteria / Gram negative / Leptospira spp.",
                    "Mannheimia spp.": "1505 Infectious Agent / Bacteria / Gram negative / Mannheimia spp.",
                    "Pasteurella spp.": "1505 Infectious Agent / Bacteria / Gram negative / Pasteurella spp.",
                    "Photobacterium spp.": "1505 Infectious Agent / Bacteria / Gram negative / Photobacterium spp.",
                    "Piscirickettsia spp.": "1505 Infectious Agent / Bacteria / Gram negative / Piscirickettsia spp.",
                    "Vibrio spp.": "1505 Infectious Agent / Bacteria / Gram negative / Vibrio spp.",
                    "Yersinia spp.": "1505 Infectious Agent / Bacteria / Gram negative / Yersinia spp.",
                    "Acidovorax spp.": "1506 Infectious Agent / Bacteria / Gram negative / Acidovorax spp.",
                    "Burkholderia spp.": "1506 Infectious Agent / Bacteria / Gram negative / Burkholderia spp.",
                    "Erwinia spp.": "1506 Infectious Agent / Bacteria / Gram negative / Erwinia spp.",
                    "Ralstonia spp.": "1506 Infectious Agent / Bacteria / Gram negative / Ralstonia spp.",
                    "Xanthomonas spp.": "1506 Infectious Agent / Bacteria / Gram negative / Xanthomonas spp.",
                    "Xylella spp.": "1506 Infectious Agent / Bacteria / Gram negative / Xylella spp.",
                },
                "Gram positive": {
                    None: "1512 Infectious Agent / Bacteria / Gram positive",
                    "Other Gram positive": "1513 Infectious Agent / Bacteria / Gram positive / Other Gram positive",
                    "Clostridioides spp.": "1514 Infectious Agent / Bacteria / Gram positive / Clostridioides spp.",
                    "Enterococcus spp.": "1514 Infectious Agent / Bacteria / Gram positive / Enterococcus spp.",
                    "Staphylococcus spp.": "1514 Infectious Agent / Bacteria / Gram positive / Staphylococcus spp.",
                    "Streptococcus spp.": "1514 Infectious Agent / Bacteria / Gram positive / Streptococcus spp.",
                    "Bacillus spp.": "1515 Infectious Agent / Bacteria / Gram positive / Bacillus spp.",
                    "Clostridium spp.": "1515 Infectious Agent / Bacteria / Gram positive / Clostridium spp.",
                    "Corynebacterium spp.": "1515 Infectious Agent / Bacteria / Gram positive / Corynebacterium spp.",
                    "Dermatophilus spp.": "1515 Infectious Agent / Bacteria / Gram positive / Dermatophilus spp.",
                    "Trueperella spp.": "1515 Infectious Agent / Bacteria / Gram positive / Trueperella spp.",
                },
                "Gram variable": {
                    None: "1522 Infectious Agent / Bacteria / Gram variable",
                    "Other Gram variable": "1523 Infectious Agent / Bacteria / Gram variable / Other Gram variable",
                    "Mycobacterium spp": "1524 Infectious Agent / Bacteria / Gram variable / Mycobacterium spp",
                    "Mycoplasma spp.": "1524 Infectious Agent / Bacteria / Gram variable / Mycoplasma spp.",
                },
            },
            "Fungus": {
                "Fungus": {
                    None: "1600 Infectious Agent / Fungus / Fungus",
                    "Not Specified": "1601 Infectious Agent / Fungus / Fungus / Not Specified_Fungus",
                    "Other": "1602 Infectious Agent / Fungus / Fungus / Other_Fungus",
                    "Aspergillus": "1603 Infectious Agent / Fungus / Fungus / Aspergillus",
                    "Blastomyces": "1603 Infectious Agent / Fungus / Fungus / Blastomyces",
                    "Candida": "1603 Infectious Agent / Fungus / Fungus / Candida",
                    "Cladophialophora": "1603 Infectious Agent / Fungus / Fungus / Cladophialophora",
                    "Coccidioides": "1603 Infectious Agent / Fungus / Fungus / Coccidioides",
                    "Cryptococcus": "1603 Infectious Agent / Fungus / Fungus / Cryptococcus",
                    "Epidermophyton": "1603 Infectious Agent / Fungus / Fungus / Epidermophyton",
                    "Fonsecaea": "1603 Infectious Agent / Fungus / Fungus / Fonsecaea",
                    "Fusarium": "1603 Infectious Agent / Fungus / Fungus / Fusarium",
                    "Histoplasma": "1603 Infectious Agent / Fungus / Fungus / Histoplasma",
                    "Lichtheimia": "1603 Infectious Agent / Fungus / Fungus / Lichtheimia",
                    "Lomentospora": "1603 Infectious Agent / Fungus / Fungus / Lomentospora",
                    "Microsporum": "1603 Infectious Agent / Fungus / Fungus / Microsporum",
                    "Mucor": "1603 Infectious Agent / Fungus / Fungus / Mucor",
                    "Paracoccidioides": "1603 Infectious Agent / Fungus / Fungus / Paracoccidioides",
                    "Phialophora": "1603 Infectious Agent / Fungus / Fungus / Phialophora",
                    "Pneumocystis": "1603 Infectious Agent / Fungus / Fungus / Pneumocystis",
                    "Rhizopus": "1603 Infectious Agent / Fungus / Fungus / Rhizopus",
                    "Scedosporium": "1603 Infectious Agent / Fungus / Fungus / Scedosporium",
                    "Sporothrix": "1603 Infectious Agent / Fungus / Fungus / Sporothrix",
                    "Talaromyces": "1603 Infectious Agent / Fungus / Fungus / Talaromyces",
                    "Trichophyton": "1603 Infectious Agent / Fungus / Fungus / Trichophyton",
                    "Ascomycota Other": "1604 Infectious Agent / Fungus / Fungus / Ascomycota Other",
                    "Basidiomycota Other": "1604 Infectious Agent / Fungus / Fungus / Basidiomycota Other",
                    "Mucorales Other": "1604 Infectious Agent / Fungus / Fungus / Mucorales Other",
                    "Ascomycota Not specified": "1605 Infectious Agent / Fungus / Fungus / Ascomycota Not specified",
                    "Basidiomycota Not specified": "1605 Infectious Agent / Fungus / Fungus / Basidiomycota Not specified",
                    "Mucorales Not specified": "1605 Infectious Agent / Fungus / Fungus / Mucorales Not specified",
                    "Alternaria spp.": "1606 Infectious Agent / Fungus / Fungus / Alternaria spp.",
                    "Blumeria spp.": "1606 Infectious Agent / Fungus / Fungus / Blumeria spp.",
                    "Botrytis": "1606 Infectious Agent / Fungus / Fungus / Botrytis",
                    "Cercospora": "1606 Infectious Agent / Fungus / Fungus / Cercospora",
                    "Corynespora spp.": "1606 Infectious Agent / Fungus / Fungus / Corynespora spp.",
                    "Dydimella spp.": "1606 Infectious Agent / Fungus / Fungus / Dydimella spp.",
                    "Mycosphaerella": "1606 Infectious Agent / Fungus / Fungus / Mycosphaerella",
                    "Phakopsora": "1606 Infectious Agent / Fungus / Fungus / Phakopsora",
                    "Plasmopara spp.": "1606 Infectious Agent / Fungus / Fungus / Plasmopara spp.",
                    "Pseudocercospora": "1606 Infectious Agent / Fungus / Fungus / Pseudocercospora",
                    "Pseudoperonospora": "1606 Infectious Agent / Fungus / Fungus / Pseudoperonospora",
                    "Pyricularia": "1606 Infectious Agent / Fungus / Fungus / Pyricularia",
                    "Ramularia": "1606 Infectious Agent / Fungus / Fungus / Ramularia",
                    "Sphaerotheca": "1606 Infectious Agent / Fungus / Fungus / Sphaerotheca",
                    "Venturia": "1606 Infectious Agent / Fungus / Fungus / Venturia",
                    "Zymoseptoria": "1606 Infectious Agent / Fungus / Fungus / Zymoseptoria",
                    "Medium Priority": "1607 Infectious Agent / Fungus / Fungus / Medium Priority",
                }
            },
            "Parasite": {
                "Parasite": {
                    None: "1700 Infectious Agent / Parasite / Parasite",
                    "Not Specified": "1700 Infectious Agent / Parasite / Parasite / Not Specified_Parasite",
                    "Other": "1702 Infectious Agent / Parasite / Other_Parasite",
                    "Protozoa": {
                        None: "1710 Infectious Agent / Parasite / Protozoa",
                        "Babesia": "1711 Infectious Agent / Parasite / Protozoa / Babesia",
                        "Cryptosporidium": "1711 Infectious Agent / Parasite / Protozoa / Cryptosporidium",
                        "Eimeria": "1711 Infectious Agent / Parasite / Protozoa / Eimeria",
                        "Theileria": "1711 Infectious Agent / Parasite / Protozoa / Theileria",
                        "Trypanosoma": "1711 Infectious Agent / Parasite / Protozoa / Trypanosoma",
                        "Not Specified_Protozoa": "1712 Infectious Agent / Parasite / Protozoa / Not Specified_Protozoa",
                        "Other_Protozoa": "1713 Infectious Agent / Parasite / Protozoa / Other_Protozoa",
                    },
                    "Helminth": {
                        None: "1720 Infectious Agent / Parasite / Helminth",
                        "Nematodes": "1721 Infectious Agent / Parasite / Helminth / Nematodes",
                        "Not Specified_Helminth": "1722 Infectious Agent / Parasite / Helminth / Not Specified_Helminth",
                        "Other_Helminth": "1723 Infectious Agent / Parasite / Helminth / Other_Helminth",
                    },
                    "Ectoparasites": "1730 Infectious Agent / Parasite / Ectoparasites",
                }
            },
            "Virus": {
                "Virus": {
                    None: "1800 Infectious Agent / Virus / Virus",
                    "Not Specified_Virus": "1801 Infectious Agent / Virus / Virus / Not Specified_Virus",
                    "Other_Virus": "1802 Infectious Agent / Virus / Virus / Other_Virus",
                    "Arteriviridae": "1803 Infectious Agent / Virus / Virus / Arteriviridae",
                    "Birnaviridae": "1803 Infectious Agent / Virus / Virus / Birnaviridae",
                    "Coronaviridae": "1803 Infectious Agent / Virus / Virus / Coronaviridae",
                    "Orthomyxoviridae": "1803 Infectious Agent / Virus / Virus / Orthomyxoviridae",
                    "Paramyxoviridae": "1803 Infectious Agent / Virus / Virus / Paramyxoviridae",
                    "Pestivirus/Flaviviridae": "1803 Infectious Agent / Virus / Virus / Pestivirus/Flaviviridae",
                    "Poxviridae": "1803 Infectious Agent / Virus / Virus / Poxviridae",
                    "Reoviridae": "1803 Infectious Agent / Virus / Virus / Reoviridae",
                }
            },
            "Other": {"Other": {None: "1900 Infectious Agent / Other / Other_Other"}},
            "Not Applicable": {
                "Not Applicable": {
                    None: "1902 Infectious Agent / Not Applicable / Not Applicable"
                }
            },
            "Not Specified": {
                "Not Specified": {
                    None: "1901 Infectious Agent / Not Specified / Not Specified_InfectiousAgent"
                }
            },

        **Classification Rules:**
            
        a. **Direct Mention or Inference:**
            - Only classify infectious agents that are directly mentioned or can be directly inferred from the title and abstract.
            - For research mentioning a specific category without details:
                - Use "1500 Infectious Agent / Bacteria / Bacteria / Not Specified_Bacteria" if bacteria are mentioned without specifying which ones
                - Use "1801 Infectious Agent / Virus / Virus / Not Specified_Virus" if viruses are mentioned without specifying which ones
                - Use "1700 Infectious Agent / Parasite / Parasite / Not Specified_Parasite" if parasites are mentioned without specifying which ones
                - Use "1600 Infectious Agent / Fungus / Fungus / Not Specified_Fungus" if fungi are mentioned without specifying which ones
            - For research mentioning specific agents not in our categories:
                - Use "1503 Infectious Agent / Bacteria / Gram negative / Other Gram negative" or similar bacterial categories for uncategorized bacteria
                - Use "1802 Infectious Agent / Virus / Virus / Other_Virus" for uncategorized viruses
                - Use "1702 Infectious Agent / Parasite / Other_Parasite" for uncategorized parasites
                - Use "1602 Infectious Agent / Fungus / Other_Fungus" for uncategorized fungi
            - For other cases:
                - Use "1901 Infectious Agent / Not Specified / Not Specified_InfectiousAgent" if no infectious agent is specified at all
                - Use "1900 Infectious Agent / Other / Other_Other" if the infectious agent is specified but doesn't fit into bacteria, virus, parasite, or fungus categories
                - Use "1902 Infectious Agent / Not Applicable / Not Applicable" if the research is not related to any infectious agents
        b. **Multiple Classifications:**
            - Multiple infectious agent classifications are permitted **only** if multiple agents are explicitly mentioned or can be directly inferred as being the main topic.
            
        c. **Exclude External References:**
            - Ignore any parts of the text that contain references to other resources, such as related work sections or citations to other research. Only consider the topics that are the direct topic of this current research at hand.
                
            
        4. **Output Format:**
            - The output should be a JSON object with the following structure:
            {{
                "infectious_agent": [list of infectious agents],
                "explanation": "explanation for the classification",
                "confidence": "float representing the confidence in the classification",
                "confidence_explanation": "explanation for the confidence"
            }}

        **Now, perform the classification for the following research paper given only these classification choices:**

        **Output:**
    """
    return base_prompt


def classify_infectious_agent(
    title, abstract, model="gpt-4o-mini", include_examples=True
):
    max_tries = 3
    tries = 0
    while tries < max_tries:
        try:
            prompt = generate_prompt(
                title=title, abstract=abstract, include_examples=include_examples
            )
            result = classify_research_json(prompt, model)
            if result is None:
                return None

            parsed_result = {
                "infectious_agent": result.get("infectious_agent", []),
                "explanation": result.get("explanation", ""),
                "confidence": result.get("confidence", ""),
                "confidence_explanation": result.get("confidence_explanation", ""),
            }

            agent_categories = get_categories("Infectious Agent")
            invalid_agents = [
                agent
                for agent in parsed_result["infectious_agent"]
                if agent not in agent_categories
            ]
            assert not invalid_agents, f"The following infectious_agent entries are not in the valid categories: {', '.join(invalid_agents)}"

            return parsed_result
        except Exception as e:
            tries += 1
            print(f"Error occurred: {str(e)}. Attempt {tries} of {max_tries}")
            if tries == max_tries:
                print("Max retries reached. Returning None.")
                return None
