from pydantic import BaseModel, model_validator, ValidationError
from typing import Union, Literal, Optional, List
from enum import Enum


RESEARCH_AREA_MAPPING = {
    "Basic Research": {
        "Fundamental": "2100 Research Area / Basic Research / Fundamental",
        "Towards a Product": "2200 Research Area / Basic Research / Towards a Product",
    },
    "Therapeutics": {
        "Discovery": "3100 Research Area / Therapeutics / Discovery",
        "Development": {
            None: "3200 Research Area / Therapeutics / Development",
            "Phase 1": "3201 Research Area / Therapeutics / Development / Phase 1",
            "Phase 2": "3202 Research Area / Therapeutics / Development / Phase 2",
            "Phase 3": "3203 Research Area / Therapeutics / Development / Phase 3",
            "Approval (pipeline)": "3204 Research Area / Therapeutics / Development / Approval (pipeline)",
            "Marketed (pipeline)": "3205 Research Area / Therapeutics / Development / Marketed (pipeline)",
        },
        "Approval & Post approval": "3400 Research Area / Therapeutics / Approval & Post approval",
    },
    "Vaccines": {
        "Discovery": "4110 Research Area / Vaccines / Discovery",
        "Development": {
            None: "4120 Research Area / Vaccines / Development",
            "Phase 1": "4122 Research Area / Vaccines / Development / Phase 1",
            "Phase 2": "4123 Research Area / Vaccines / Development / Phase 2",
            "Phase 3": "4124 Research Area / Vaccines / Development / Phase 3",
        },
        "Approval & Post approval": "4140 Research Area / Vaccines / Approval & Post approval",
    },
    "Diagnostics": {
        "Discovery": "5100 Research Area / Diagnostics / Discovery",
        "Development": "5200 Research Area / Diagnostics / Development",
        "Approval & Post approval": "5400 Research Area / Diagnostics / Approval & Post approval",
    },
    "Preventives Other": {
        "Other_PreventivesOther": "4200 Research Area / Preventives Other / Other_PreventivesOther",
        "Discovery": "4210 Research Area / Preventives Other / Discovery",
        "Development": "4220 Research Area / Preventives Other / Development",
        "Approval & Post approval": "4240 Research Area / Preventives Other / Approval & Post approval",
    },
    "Operational": {"Operational": "6100 Research Area / Operational / Operational"},
    "Policy": {"Policy": "7100 Research Area / Policy / Policy"},
    "Capacity Building": {
        "Capacity Building": "7200 Research Area / Capacity Building / Capacity Building"
    },
    "Other Products": {
        "Other Products": "5500 Research Area / Other Products / Other Products"
    },
    "Promotants": {"Promotants": "5600 Research Area / Promotants / Promotants"},
}


# Define Enums for Categories and Subcategories
class ResearchAreaCategory(str, Enum):
    BASIC_RESEARCH = "Basic Research"
    THERAPEUTICS = "Therapeutics"
    PREVENTIVES_OTHER = "Preventives Other"
    DIAGNOSTICS = "Diagnostics"
    OPERATIONAL = "Operational"
    VACCINES = "Vaccines"
    POLICY = "Policy"
    CAPACITY_BUILDING = "Capacity Building"
    OTHER_PRODUCTS = "Other Products"
    PROMOTANTS = "Promotants"


class BasicResearchSubcategory(str, Enum):
    FUNDAMENTAL = "Fundamental"
    TOWARDS_A_PRODUCT = "Towards a Product"


class TherapeuticsSubcategory(str, Enum):
    DISCOVERY = "Discovery"
    DEVELOPMENT = "Development"
    APPROVAL_POST_APPROVAL = "Approval & Post approval"


class TherapeuticsDevelopmentSpecificType(str, Enum):
    PHASE_1 = "Phase 1"
    PHASE_2 = "Phase 2"
    PHASE_3 = "Phase 3"
    APPROVAL_PIPELINE = "Approval (pipeline)"
    MARKETED_PIPELINE = "Marketed (pipeline)"


class VaccinesSubcategory(str, Enum):
    DISCOVERY = "Discovery"
    DEVELOPMENT = "Development"
    APPROVAL_POST_APPROVAL = "Approval & Post approval"


class VaccinesDevelopmentSpecificType(str, Enum):
    PHASE_1 = "Phase 1"
    PHASE_2 = "Phase 2"
    PHASE_3 = "Phase 3"


class DiagnosticsSubcategory(str, Enum):
    DISCOVERY = "Discovery"
    DEVELOPMENT = "Development"
    APPROVAL_POST_APPROVAL = "Approval & Post approval"


class PreventivesOtherSubcategory(str, Enum):
    OTHER_PREVENTIVES_OTHER = "Other_PreventivesOther"
    DISCOVERY = "Discovery"
    DEVELOPMENT = "Development"
    APPROVAL_POST_APPROVAL = "Approval & Post approval"


class OperationalSubcategory(str, Enum):
    OPERATIONAL = "Operational"


class PolicySubcategory(str, Enum):
    POLICY = "Policy"


class CapacityBuildingSubcategory(str, Enum):
    CAPACITY_BUILDING = "Capacity Building"


class OtherProductsSubcategory(str, Enum):
    OTHER_PRODUCTS = "Other Products"


class PromotantsSubcategory(str, Enum):
    PROMOTANTS = "Promotants"


# Define Classification classes for each category
class BasicResearchClassification(BaseModel):
    classification_type: Literal["BasicResearch"] = "BasicResearch"  # Unique first key
    category: Literal[ResearchAreaCategory.BASIC_RESEARCH]
    subcategory: BasicResearchSubcategory

    def map_to_string(self) -> str:
        mapped = RESEARCH_AREA_MAPPING[self.category.value][self.subcategory.value]
        return mapped


class TherapeuticsClassification(BaseModel):
    classification_type: Literal["Therapeutics"] = "Therapeutics"
    category: Literal[ResearchAreaCategory.THERAPEUTICS]
    subcategory: TherapeuticsSubcategory
    specific_type: Optional[TherapeuticsDevelopmentSpecificType] = None

    def map_to_string(self) -> str:
        if self.specific_type:
            mapped = RESEARCH_AREA_MAPPING[self.category.value][self.subcategory.value][
                self.specific_type.value
            ]
        else:
            sub_mapping = RESEARCH_AREA_MAPPING[self.category.value][
                self.subcategory.value
            ]
            if isinstance(sub_mapping, dict):
                mapped = sub_mapping.get(None)
            else:
                mapped = sub_mapping
        return mapped

    @model_validator(mode="after")
    def validate_specific_type(self):
        if self.subcategory == TherapeuticsSubcategory.DEVELOPMENT:
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


class VaccinesClassification(BaseModel):
    classification_type: Literal["Vaccines"] = "Vaccines"
    category: Literal[ResearchAreaCategory.VACCINES]
    subcategory: VaccinesSubcategory
    specific_type: Optional[VaccinesDevelopmentSpecificType] = None

    def map_to_string(self) -> str:
        if self.specific_type:
            mapped = RESEARCH_AREA_MAPPING[self.category.value][self.subcategory.value][
                self.specific_type.value
            ]
        else:
            sub_mapping = RESEARCH_AREA_MAPPING[self.category.value][
                self.subcategory.value
            ]
            if isinstance(sub_mapping, dict):
                mapped = sub_mapping.get(None)
            else:
                mapped = sub_mapping
        return mapped

    @model_validator(mode="after")
    def validate_specific_type(self):
        if self.subcategory == VaccinesSubcategory.DEVELOPMENT:
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


class DiagnosticsClassification(BaseModel):
    classification_type: Literal["Diagnostics"] = "Diagnostics"  # Unique first key
    category: Literal[ResearchAreaCategory.DIAGNOSTICS]
    subcategory: DiagnosticsSubcategory

    def map_to_string(self) -> str:
        mapped = RESEARCH_AREA_MAPPING[self.category.value][self.subcategory.value]
        return mapped


class PreventivesOtherClassification(BaseModel):
    classification_type: Literal["PreventivesOther"] = (
        "PreventivesOther"  # Unique first key
    )
    category: Literal[ResearchAreaCategory.PREVENTIVES_OTHER]
    subcategory: PreventivesOtherSubcategory

    def map_to_string(self) -> str:
        mapped = RESEARCH_AREA_MAPPING[self.category.value][self.subcategory.value]
        return mapped


class OperationalClassification(BaseModel):
    classification_type: Literal["Operational"] = "Operational"  # Unique first key
    category: Literal[ResearchAreaCategory.OPERATIONAL]
    subcategory: Literal[OperationalSubcategory.OPERATIONAL]

    def map_to_string(self) -> str:
        mapped = RESEARCH_AREA_MAPPING[self.category.value][self.subcategory.value]
        return mapped


class PolicyClassification(BaseModel):
    classification_type: Literal["Policy"] = "Policy"  # Unique first key
    category: Literal[ResearchAreaCategory.POLICY]
    subcategory: Literal[PolicySubcategory.POLICY]

    def map_to_string(self) -> str:
        mapped = RESEARCH_AREA_MAPPING[self.category.value][self.subcategory.value]
        return mapped


class CapacityBuildingClassification(BaseModel):
    classification_type: Literal["CapacityBuilding"] = (
        "CapacityBuilding"  # Unique first key
    )
    category: Literal[ResearchAreaCategory.CAPACITY_BUILDING]
    subcategory: Literal[CapacityBuildingSubcategory.CAPACITY_BUILDING]

    def map_to_string(self) -> str:
        mapped = RESEARCH_AREA_MAPPING[self.category.value][self.subcategory.value]
        return mapped


class OtherProductsClassification(BaseModel):
    classification_type: Literal["OtherProducts"] = "OtherProducts"  # Unique first key
    category: Literal[ResearchAreaCategory.OTHER_PRODUCTS]
    subcategory: Literal[OtherProductsSubcategory.OTHER_PRODUCTS]

    def map_to_string(self) -> str:
        mapped = RESEARCH_AREA_MAPPING[self.category.value][self.subcategory.value]
        return mapped


class PromotantsClassification(BaseModel):
    classification_type: Literal["Promotants"] = "Promotants"  # Unique first key
    category: Literal[ResearchAreaCategory.PROMOTANTS]
    subcategory: Literal[PromotantsSubcategory.PROMOTANTS]

    def map_to_string(self) -> str:
        mapped = RESEARCH_AREA_MAPPING[self.category.value][self.subcategory.value]
        return mapped


# Union of all possible classifications
ResearchAreaClassification = Union[
    BasicResearchClassification,
    TherapeuticsClassification,
    VaccinesClassification,
    DiagnosticsClassification,
    PreventivesOtherClassification,
    OperationalClassification,
    PolicyClassification,
    CapacityBuildingClassification,
    OtherProductsClassification,
    PromotantsClassification,
]


class ResearchAreaClassificationResult(BaseModel):
    research_area: List[ResearchAreaClassification]
    explanation: str
    confidence: float
    confidence_explanation: str


# def count_research_area_classifications(mapping):
#     total = 0
#     for category, subcategories in mapping.items():
#         if isinstance(subcategories, dict):
#             for subcat, specifics in subcategories.items():
#                 if isinstance(specifics, dict):
#                     total += len(specifics)
#                 else:
#                     total += 1
#         else:
#             total += 1
#     return total


# if __name__ == "__main__":
#     total_classifications = count_research_area_classifications(RESEARCH_AREA_MAPPING)
#     print(f"Total Research Area Classifications: {total_classifications}")

#     # TODO: not all classes mapped
#     total_classifications = count_research_area_classifications(RESEARCH_AREA_MAPPING)
#     print(f"Total Research Area Classifications: {total_classifications}")

# # TODO: not all classes mapped
