from pydantic import BaseModel, model_validator
from typing import Union, Literal, Optional, List
from enum import Enum

# Define the mapping with unique keys
INFECTIOUS_AGENT_MAPPING = {
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
}


# Define Enums for Categories and Subcategories


class InfectiousAgentCategory(str, Enum):
    BACTERIA = "Bacteria"
    FUNGUS = "Fungus"
    PARASITE = "Parasite"
    VIRUS = "Virus"
    NOT_APPLICABLE = "Not Applicable"
    NOT_SPECIFIED = "Not Specified"
    OTHER = "Other"


# Bacteria Subcategories
class BacteriaSubcategory(str, Enum):
    BACTERIA = "Bacteria"
    GRAM_NEGATIVE = "Gram negative"
    GRAM_POSITIVE = "Gram positive"
    GRAM_VARIABLE = "Gram variable"


class BacteriaBacteriaSpecificType(str, Enum):
    NOT_SPECIFIED_BACTERIA = "Not Specified_Bacteria"


# Bacteria Specific Types for Gram Negative
class BacteriaGramNegativeSpecificType(str, Enum):
    OTHER_GRAM_NEGATIVE = "Other Gram negative"
    ACINETOBACTER_SPP = "Acinetobacter spp."
    BORDETELLA_SPP = "Bordetella spp."
    CAMPYLOBACTER_SPP = "Campylobacter spp."
    ENTEROBACTER_SPP = "Enterobacter spp."
    ENTEROBACTERIACEAE = "Enterobacteriaceae"
    ESCHERICHIA_SPP = "Escherichia spp."
    HAEMOPHILUS_SPP = "Haemophilus spp."
    HELICOBACTER_SPP = "Helicobacter spp."
    KLEBSIELLA_SPP = "Klebsiella spp."
    MORGANELLA_SPP = "Morganella spp."
    NEISSERIA_SPP = "Neisseria spp."
    PROTEUS_SPP = "Proteus spp."
    PROVIDENCIA_SPP = "Providencia spp."
    PSEUDOMONAS_SPP = "Pseudomonas spp."
    SALMONELLA_SPP = "Salmonella spp."
    SERRATIA_SPP = "Serratia spp."
    SHIGELLA_SPP = "Shigella spp."
    ACTINOBACILLUS_SPP = "Actinobacillus spp."
    AEROMONAS_SPP = "Aeromonas spp."
    ANAPLASMA_SPP = "Anaplasma spp."
    BIBERSTEINIA_SPP = "Bibersteinia spp."
    BRACHYSPIRA_SPP = "Brachyspira spp."
    BRUCELLA_SPP = "Brucella spp."
    CHLAMYDIA = "Chlamydia"
    DICHELOBACTER_SPP = "Dichelobacter spp."
    EDWARDSIELLA_SPP = "Edwardsiella spp."
    EHRLICHIA_SPP = "Ehrlichia spp."
    FLAVOBACTERIUM_SPP = "Flavobacterium spp."
    FUSOBACTERIUM_SPP = "Fusobacterium spp."
    HISTOPHILUS_SPP = "Histophilus spp."
    LAWSONIA_SPP = "Lawsonia spp."
    LEPTOSPIRA_SPP = "Leptospira spp."
    MANNHEIMIA_SPP = "Mannheimia spp."
    PASTEURELLA_SPP = "Pasteurella spp."
    PHOTOBACTERIUM_SPP = "Photobacterium spp."
    PISCIRICKETTSIA_SPP = "Piscirickettsia spp."
    VIBRIO_SPP = "Vibrio spp."
    YERSINIA_SPP = "Yersinia spp."
    ACIDOVORAX_SPP = "Acidovorax spp."
    BURKHOLDERIA_SPP = "Burkholderia spp."
    ERWINIA_SPP = "Erwinia spp."
    RALSTONIA_SPP = "Ralstonia spp."
    XANTHOMONAS_SPP = "Xanthomonas spp."
    XYLELLA_SPP = "Xylella spp."


# Bacteria Specific Types for Gram Positive
class BacteriaGramPositiveSpecificType(str, Enum):
    OTHER_GRAM_POSITIVE = "Other Gram positive"
    CLOSTRIDIOIDES_SPP = "Clostridioides spp."
    ENTEROCOCCUS_SPP = "Enterococcus spp."
    STAPHYLOCOCCUS_SPP = "Staphylococcus spp."
    STREPTOCOCCUS_SPP = "Streptococcus spp."
    BACILLUS_SPP = "Bacillus spp."
    CLOSTRIDIUM_SPP = "Clostridium spp."
    CORYNEBACTERIUM_SPP = "Corynebacterium spp."
    DERMATOPHILUS_SPP = "Dermatophilus spp."
    TRUEPERELLA_SPP = "Trueperella spp."


# Bacteria Specific Types for Gram Variable
class BacteriaGramVariableSpecificType(str, Enum):
    OTHER_GRAM_VARIABLE = "Other Gram variable"
    MYCOBACTERIUM_SPP = "Mycobacterium spp"
    MYCOPLASMA_SPP = "Mycoplasma spp."


# Fungus Subcategories
class FungusSubcategory(str, Enum):
    FUNGUS = "Fungus"


# Fungus Specific Types
class FungusSpecificType(str, Enum):
    NOT_SPECIFIED_FUNGUS = "Not Specified_Fungus"
    OTHER_FUNGUS = "Other_Fungus"
    ASPERGILLUS = "Aspergillus"
    BLASTOMYCES = "Blastomyces"
    CANDIDA = "Candida"
    CLADOPHIALOPHORA = "Cladophialophora"
    COCCIDIOIDES = "Coccidioides"
    CRYPTOCOCCUS = "Cryptococcus"
    EPIDERMOPHYTON = "Epidermophyton"
    FONSECAEA = "Fonsecaea"
    FUSARIUM = "Fusarium"
    HISTOPLASMA = "Histoplasma"
    LICHTHEIMIA = "Lichtheimia"
    LOMENTOSPORA = "Lomentospora"
    MICROSPORUM = "Microsporum"
    MUCOR = "Mucor"
    PARACOCCIDIOIDES = "Paracoccidioides"
    PHIALOPHORA = "Phialophora"
    PNEUMOCYSTIS = "Pneumocystis"
    RHIZOPUS = "Rhizopus"
    SCEDOSPORIUM = "Scedosporium"
    SPOROTHRIX = "Sporothrix"
    TALAROMYCES = "Talaromyces"
    TRICHOPHYTON = "Trichophyton"
    ASCOMYCOTA_OTHER = "Ascomycota Other"
    BASIDIOMYCOTA_OTHER = "Basidiomycota Other"
    MUCORALES_OTHER = "Mucorales Other"
    ASCOMYCOTA_NOT_SPECIFIED = "Ascomycota Not specified"
    BASIDIOMYCOTA_NOT_SPECIFIED = "Basidiomycota Not specified"
    MUCORALES_NOT_SPECIFIED = "Mucorales Not specified"
    ALTERNARIA_SPP = "Alternaria spp."
    BLUMERIA_SPP = "Blumeria spp."
    BOTRYTIS = "Botrytis"
    CERCOSPORA = "Cercospora"
    CORYNESPORA_SPP = "Corynespora spp."
    DYDIMELLA_SPP = "Dydimella spp."
    MYCOSPHAERELLA = "Mycosphaerella"
    PHAKOPSORA = "Phakopsora"
    PLASMOPARA_SPP = "Plasmopara spp."
    PSEUDOCERCOSPORA = "Pseudocercospora"
    PSEUDOPERONOSPORA = "Pseudoperonospora"
    PYRICULARIA = "Pyricularia"
    RAMULARIA = "Ramularia"
    SPHAEROTHECA = "Sphaerotheca"
    VENTURIA = "Venturia"
    ZYMOSEPTORIA = "Zymoseptoria"
    MEDIUM_PRIORITY = "Medium Priority"


# Parasite Subcategories
class ParasiteSubcategory(str, Enum):
    PARASITE = "Parasite"
    PROTOZOA = "Protozoa"
    HELMINTH = "Helminth"
    ECTOPARASITES = "Ectoparasites"
    NOT_SPECIFIED_PARASITE = "Not Specified_Parasite"
    OTHER_PARASITE = "Other_Parasite"


# Parasite Specific Types for Protozoa
class ParasiteProtozoaSpecificType(str, Enum):
    BABESIA = "Babesia"
    CRYPTOSPORIDIUM = "Cryptosporidium"
    EIMERIA = "Eimeria"
    THEILERIA = "Theileria"
    TRYPANOSOMA = "Trypanosoma"
    NOT_SPECIFIED_PROTOZOA = "Not Specified_Protozoa"
    OTHER_PROTOZOA = "Other_Protozoa"


# Parasite Specific Types for Helminth
class ParasiteHelminthSpecificType(str, Enum):
    NEMATODES = "Nematodes"
    NOT_SPECIFIED_HELMINTH = "Not Specified_Helminth"
    OTHER_HELMINTH = "Other_Helminth"


# Virus Subcategories
class VirusSubcategory(str, Enum):
    VIRUS = "Virus"


# Virus Specific Types
class VirusSpecificType(str, Enum):
    NOT_SPECIFIED_VIRUS = "Not Specified_Virus"
    OTHER_VIRUS = "Other_Virus"
    ARTERIVIRIDAE = "Arteriviridae"
    BIRNAVIRIDAE = "Birnaviridae"
    CORONAVIRIDAE = "Coronaviridae"
    ORTHOMYXOVIRIDAE = "Orthomyxoviridae"
    PARAMYXOVIRIDAE = "Paramyxoviridae"
    PESTIVIRUS_FLAVIVIRIDAE = "Pestivirus/Flaviviridae"
    POXVIRIDAE = "Poxviridae"
    REOVIRIDAE = "Reoviridae"


# Not Applicable Subcategories
class NotApplicableSubcategory(str, Enum):
    NOT_APPLICABLE = "Not Applicable"


# Not Specified Subcategories
class NotSpecifiedSubcategory(str, Enum):
    NOT_SPECIFIED = "Not Specified"


# Other Subcategories
class OtherSubcategory(str, Enum):
    OTHER = "Other"


# Define Classification classes for Infectious Agent Categories


class BacteriaClassification(BaseModel):
    classification_type: Literal["Bacteria"] = "Bacteria"  # Unique first key
    category: Literal[InfectiousAgentCategory.BACTERIA]
    subcategory: BacteriaSubcategory
    specific_type: Optional[
        Union[
            BacteriaBacteriaSpecificType,
            BacteriaGramNegativeSpecificType,
            BacteriaGramPositiveSpecificType,
            BacteriaGramVariableSpecificType,
        ]
    ] = None

    def map_to_string(self) -> str:
        if self.specific_type:
            mapped = INFECTIOUS_AGENT_MAPPING[self.category.value][
                self.subcategory.value
            ][self.specific_type.value]
        else:
            sub_mapping = INFECTIOUS_AGENT_MAPPING[self.category.value][
                self.subcategory.value
            ]
            if isinstance(sub_mapping, dict):
                mapped = sub_mapping.get(None)
            else:
                mapped = sub_mapping
        return mapped

    @model_validator(mode="after")
    def validate_specific_type(self):
        if self.subcategory in {
            BacteriaSubcategory.BACTERIA,
            BacteriaSubcategory.GRAM_NEGATIVE,
            BacteriaSubcategory.GRAM_POSITIVE,
            BacteriaSubcategory.GRAM_VARIABLE,
        }:
            if self.subcategory == BacteriaSubcategory.BACTERIA:
                if self.specific_type not in {
                    BacteriaBacteriaSpecificType.NOT_SPECIFIED_BACTERIA
                }:
                    raise ValueError(
                        f"specific_type must be '{BacteriaBacteriaSpecificType.NOT_SPECIFIED_BACTERIA.value}' when subcategory is '{self.subcategory.value}'"
                    )
            elif self.subcategory == BacteriaSubcategory.GRAM_NEGATIVE:
                if not isinstance(self.specific_type, BacteriaGramNegativeSpecificType):
                    raise ValueError(
                        f"specific_type must be a Gram Negative specific type when subcategory is '{self.subcategory.value}'"
                    )
            elif self.subcategory == BacteriaSubcategory.GRAM_POSITIVE:
                if not isinstance(self.specific_type, BacteriaGramPositiveSpecificType):
                    raise ValueError(
                        f"specific_type must be a Gram Positive specific type when subcategory is '{self.subcategory.value}'"
                    )
            elif self.subcategory == BacteriaSubcategory.GRAM_VARIABLE:
                if not isinstance(self.specific_type, BacteriaGramVariableSpecificType):
                    raise ValueError(
                        f"specific_type must be a Gram Variable specific type when subcategory is '{self.subcategory.value}'"
                    )
        else:
            if self.specific_type is not None:
                raise ValueError(
                    f"specific_type should not be set when subcategory is '{self.subcategory.value}'"
                )
        return self


class FungusClassification(BaseModel):
    classification_type: Literal["Fungus"] = "Fungus"  # Unique first key
    category: Literal[InfectiousAgentCategory.FUNGUS]
    subcategory: FungusSubcategory
    specific_type: Optional[FungusSpecificType] = None

    def map_to_string(self) -> str:
        if self.specific_type:
            mapped = INFECTIOUS_AGENT_MAPPING[self.category.value][
                self.subcategory.value
            ][self.specific_type.value]
        else:
            sub_mapping = INFECTIOUS_AGENT_MAPPING[self.category.value][
                self.subcategory.value
            ]
            if isinstance(sub_mapping, dict):
                mapped = sub_mapping.get(None)
            else:
                mapped = sub_mapping
        return mapped

    @model_validator(mode="after")
    def validate_specific_type(self):
        if self.specific_type:
            if not isinstance(self.specific_type, FungusSpecificType):
                raise ValueError(
                    f"specific_type must be a valid Fungus specific type when subcategory is '{self.subcategory.value}'"
                )
        return self


class ParasiteClassification(BaseModel):
    classification_type: Literal["Parasite"] = "Parasite"  # Unique first key
    category: Literal[InfectiousAgentCategory.PARASITE]
    subcategory: ParasiteSubcategory
    specific_type: Optional[
        Union[
            ParasiteProtozoaSpecificType,
            ParasiteHelminthSpecificType,
        ]
    ] = None

    def map_to_string(self) -> str:
        if self.specific_type:
            mapped = INFECTIOUS_AGENT_MAPPING[self.category.value][
                self.subcategory.value
            ][self.specific_type.value]
        else:
            sub_mapping = INFECTIOUS_AGENT_MAPPING[self.category.value][
                self.subcategory.value
            ]
            if isinstance(sub_mapping, dict):
                mapped = sub_mapping.get(None)
            else:
                mapped = sub_mapping
        return mapped

    @model_validator(mode="after")
    def validate_specific_type(self):
        if self.subcategory in {
            ParasiteSubcategory.PROTOZOA,
            ParasiteSubcategory.HELIMNTH,
        }:
            if self.subcategory == ParasiteSubcategory.PROTOZOA:
                if not isinstance(self.specific_type, ParasiteProtozoaSpecificType):
                    raise ValueError(
                        f"specific_type must be a Protozoa specific type when subcategory is '{self.subcategory.value}'"
                    )
            elif self.subcategory == ParasiteSubcategory.HELIMNTH:
                if not isinstance(self.specific_type, ParasiteHelminthSpecificType):
                    raise ValueError(
                        f"specific_type must be a Helminth specific type when subcategory is '{self.subcategory.value}'"
                    )
        else:
            if self.specific_type is not None:
                raise ValueError(
                    f"specific_type should not be set when subcategory is '{self.subcategory.value}'"
                )
        return self


class VirusClassification(BaseModel):
    classification_type: Literal["Virus"] = "Virus"  # Unique first key
    category: Literal[InfectiousAgentCategory.VIRUS]
    subcategory: VirusSubcategory
    specific_type: Optional[VirusSpecificType] = None

    def map_to_string(self) -> str:
        if self.specific_type:
            mapped = INFECTIOUS_AGENT_MAPPING[self.category.value][
                self.subcategory.value
            ][self.specific_type.value]
        else:
            sub_mapping = INFECTIOUS_AGENT_MAPPING[self.category.value][
                self.subcategory.value
            ]
            if isinstance(sub_mapping, dict):
                mapped = sub_mapping.get(None)
            else:
                mapped = sub_mapping
        return mapped

    @model_validator(mode="after")
    def validate_specific_type(self):
        if self.subcategory == VirusSubcategory.VIRUS:
            if self.specific_type is None:
                raise ValueError(
                    f"specific_type must be specified when subcategory is '{self.subcategory.value}'"
                )
        else:
            if self.specific_type is not None:
                raise ValueError(
                    f"specific_type should not be set when subcategory is '{self.subcategory.value}'"
                )
        return self


class NotApplicableClassification(BaseModel):
    classification_type: Literal["Not Applicable"] = (
        "Not Applicable"  # Unique first key
    )
    category: Literal[InfectiousAgentCategory.NOT_APPLICABLE]
    subcategory: NotApplicableSubcategory

    def map_to_string(self) -> str:
        mapped = INFECTIOUS_AGENT_MAPPING[self.category.value][self.subcategory.value]
        return mapped


class NotSpecifiedClassification(BaseModel):
    classification_type: Literal["Not Specified"] = "Not Specified"  # Unique first key
    category: Literal[InfectiousAgentCategory.NOT_SPECIFIED]
    subcategory: NotSpecifiedSubcategory

    def map_to_string(self) -> str:
        mapped = INFECTIOUS_AGENT_MAPPING[self.category.value][self.subcategory.value]
        return mapped


class OtherClassification(BaseModel):
    classification_type: Literal["Other"] = "Other"  # Unique first key
    category: Literal[InfectiousAgentCategory.OTHER]
    subcategory: OtherSubcategory

    def map_to_string(self) -> str:
        mapped = INFECTIOUS_AGENT_MAPPING[self.category.value][self.subcategory.value]
        return mapped


# Union of all possible classifications
InfectiousAgentClassification = Union[
    BacteriaClassification,
    FungusClassification,
    ParasiteClassification,
    VirusClassification,
    NotApplicableClassification,
    NotSpecifiedClassification,
    OtherClassification,
]


class InfectiousAgentClassificationResult(BaseModel):
    infectious_agent: List[InfectiousAgentClassification]
    explanation: str
    confidence: float
    confidence_explanation: str


def count_infectious_agent_classifications(mapping):
    total = 0
    for category, subcategories in mapping.items():
        for subcategory, specifics in subcategories.items():
            if isinstance(specifics, dict):
                total += len(specifics)
            else:
                total += 1
    return total


if __name__ == "__main__":
    total_classifications = count_infectious_agent_classifications(
        INFECTIOUS_AGENT_MAPPING
    )
    print(f"Total Infectious Agent Classifications: {total_classifications}")

# TODO: not all classes mapped
