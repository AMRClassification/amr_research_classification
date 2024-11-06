from pydantic import BaseModel, model_validator
from typing import Union, Literal, Optional, List
from enum import Enum

# Define the mapping based on the provided sector categories
SECTOR_MAPPING = {
    "Human": {
        "Human": "1100 Sector / Human / Human",
    },
    "Animal": {
        "Not Specified_Animal": "1200 Sector / Animal / Not Specified_Animal",
        "Other_Animal": "1201 Sector / Animal / Other_Animal",
        "Livestock": {
            None: "1210 Sector / Animal / Livestock",
            "Cattle": "1211 Sector / Animal / Livestock / Cattle",
            "Small Ruminants": "1212 Sector / Animal / Livestock / Small Ruminants",
            "Pig": "1213 Sector / Animal / Livestock / Pig",
            "Not Specified_Livestock": "1214 Sector / Animal / Livestock / Not Specified_Livestock",
            "Other_Livestock": "1215 Sector / Animal / Livestock / Other_Livestock",
        },
        "Poultry": {
            None: "1220 Sector / Animal / Poultry",
            "Chicken": "1221 Sector / Animal / Poultry / Chicken",
            "Not Specified_Poultry": "1222 Sector / Animal / Poultry / Not Specified_Poultry",
            "Other_Poultry": "1223 Sector / Animal / Poultry / Other_Poultry",
        },
        "Companion": "1230 Sector / Animal / Companion",
        "Wildlife": "1240 Sector / Animal / Wildlife",
        "Insects": "1250 Sector / Animal / Insects",
        "Aquaculture": {
            None: "1260 Sector / Animal / Aquaculture",
            "Fish": "1261 Sector / Animal / Aquaculture / Fish",
            "Not Specified_Aquaculture": "1262 Sector / Animal / Aquaculture / Not Specified_Aquaculture",
            "Other_Aquaculture": "1263 Sector / Animal / Aquaculture / Other_Aquaculture",
        },
    },
    "Plant": {
        "Not Specified_Plant": "1300 Sector / Plant / Not Specified_Plant",
        "Other_Plant": "1301 Sector / Plant / Other_Plant",
        "Cereals": "1310 Sector / Plant / Cereals",
        "Crop": {
            None: "1320 Sector / Plant / Crop",
            "Not Specified_Crop": "1321 Sector / Plant / Crop / Not Specified_Crop",
            "Other_Crop": "1322 Sector / Plant / Crop / Other_Crop",
        },
        "Fruits": {
            None: "1330 Sector / Plant / Fruits",
            "Berries": "1331 Sector / Plant / Fruits / Berries",
            "Citrus": "1332 Sector / Plant / Fruits / Citrus",
            "Stone": "1333 Sector / Plant / Fruits / Stone",
            "Not Specified_Fruits": "1334 Sector / Plant / Fruits / Not Specified_Fruits",
            "Other_Fruits": "1335 Sector / Plant / Fruits / Other_Fruits",
        },
        "Oil Seeds": "1340 Sector / Plant / Oil Seeds",
        "Pulses/Beans": "1350 Sector / Plant / Pulses/Beans",
        "Tree Nuts": "1360 Sector / Plant / Tree Nuts",
        "Vegetables": {
            None: "1370 Sector / Plant / Vegetables",
            "Fruits": "1371 Sector / Plant / Vegetables / Fruits",
            "Leafy Greens": "1372 Sector / Plant / Vegetables / Leafy Greens",
            "Roots/Tubers": "1373 Sector / Plant / Vegetables / Roots/Tubers",
            "Stems/Flowers": "1374 Sector / Plant / Vegetables / Stems/Flowers",
            "Mushrooms": "1375 Sector / Plant / Vegetables / Mushrooms",
            "Not Specified_Vegetables": "1376 Sector / Plant / Vegetables / Not Specified_Vegetables",
            "Other_Vegetables": "1377 Sector / Plant / Vegetables / Other_Vegetables",
        },
        "Wild Plants": "1380 Sector / Plant / Wild Plants",
    },
    "Environment": {
        "Not Specified_Environment": "1400 Sector / Environment / Not Specified_Environment",
        "Other_Environment": "1401 Sector / Environment / Other_Environment",
        "Wastewater": {
            None: "1410 Sector / Environment / Wastewater",
            "Health Facilities": "1411 Sector / Environment / Wastewater / Health Facilities",
            "Agriculture": "1412 Sector / Environment / Wastewater / Agriculture",
            "Aquaculture": "1413 Sector / Environment / Wastewater / Aquaculture",
            "Industrial": "1414 Sector / Environment / Wastewater / Industrial",
            "Domestic": "1415 Sector / Environment / Wastewater / Domestic",
            "Not Specified_Wastewater": "1416 Sector / Environment / Wastewater / Not Specified_Wastewater",
            "Other_Wastewater": "1417 Sector / Environment / Wastewater / Other_Wastewater",
        },
        "Water": {
            None: "1420 Sector / Environment / Water",
            "Surface Water": "1421 Sector / Environment / Water / Surface Water",
            "Water Not Specified_Water": "1422 Sector / Environment / Water / Water Not Specified_Water",
            "Other_Water": "1423 Sector / Environment / Water / Other_Water",
        },
        "Soil-Waste": {
            None: "1440 Sector / Environment / Soil-Waste",
            "Compost": "1441 Sector / Environment / Soil-Waste / Compost",
            "Manure": "1442 Sector / Environment / Soil-Waste / Manure",
            "Sludge": "1443 Sector / Environment / Soil-Waste / Sludge",
            "Not Specified_Soil-Waste": "1444 Sector / Environment / Soil-Waste / Not Specified_Soil-Waste",
            "Other_Soil-Waste": "1445 Sector / Environment / Soil-Waste / Other_Soil-Waste",
        },
        "Soil": {
            None: "1450 Sector / Environment / Soil",
            "Not Specified_Soil": "1451 Sector / Environment / Soil / Not Specified_Soil",
            "Other_Soil": "1452 Sector / Environment / Soil / Other_Soil",
        },
        "Air": "1460 Sector / Environment / Air",
    },
    "Not Specified": {
        "Not Specified_Sector": "1490 Sector / Not Specified / Not Specified_Sector",
    },
}


# Define Enums for Categories and Subcategories
class SectorCategory(str, Enum):
    HUMAN = "Human"
    ANIMAL = "Animal"
    PLANT = "Plant"
    ENVIRONMENT = "Environment"
    NOT_SPECIFIED = "Not Specified"


class HumanSubcategory(str, Enum):
    HUMAN = "Human"


class NotSpecifiedSubcategory(str, Enum):
    NOT_SPECIFIED_SECTOR = "Not Specified_Sector"


class AnimalSubcategory(str, Enum):
    NOT_SPECIFIED_ANIMAL = "Not Specified_Animal"
    LIVESTOCK = "Livestock"
    POULTRY = "Poultry"
    COMPANION = "Companion"
    WILDLIFE = "Wildlife"
    INSECTS = "Insects"
    AQUACULTURE = "Aquaculture"
    OTHER_ANIMAL = "Other_Animal"


class LivestockSpecificType(str, Enum):
    CATTLE = "Cattle"
    SMALL_RUMINANTS = "Small Ruminants"
    PIG = "Pig"
    NOT_SPECIFIED_LIVESTOCK = "Not Specified_Livestock"
    OTHER_LIVESTOCK = "Other_Livestock"


class PoultrySpecificType(str, Enum):
    CHICKEN = "Chicken"
    NOT_SPECIFIED_POULTRY = "Not Specified_Poultry"
    OTHER_POULTRY = "Other_Poultry"


class AquacultureSpecificType(str, Enum):
    FISH = "Fish"
    NOT_SPECIFIED_AQUACULTURE = "Not Specified_Aquaculture"
    OTHER_AQUACULTURE = "Other_Aquaculture"


class PlantSubcategory(str, Enum):
    NOT_SPECIFIED_PLANT = "Not Specified_Plant"
    OTHER_PLANT = "Other_Plant"
    CEREALS = "Cereals"
    CROP = "Crop"
    FRUITS = "Fruits"
    OIL_SEEDS = "Oil Seeds"
    PULSES_BEANS = "Pulses/Beans"
    TREE_NUTS = "Tree Nuts"
    VEGETABLES = "Vegetables"
    WILD_PLANTS = "Wild Plants"


class CropSpecificType(str, Enum):
    NOT_SPECIFIED_CROP = "Not Specified_Crop"
    OTHER_CROP = "Other_Crop"


class FruitsSpecificType(str, Enum):
    BERRIES = "Berries"
    CITRUS = "Citrus"
    STONE = "Stone"
    NOT_SPECIFIED_FRUITS = "Not Specified_Fruits"
    OTHER_FRUITS = "Other_Fruits"


class VegetablesSpecificType(str, Enum):
    FRUITS = "Fruits"
    LEAFY_GREENS = "Leafy Greens"
    ROOTS_TUBERS = "Roots/Tubers"
    STEMS_FLOWERS = "Stems/Flowers"
    MUSHROOMS = "Mushrooms"
    NOT_SPECIFIED_VEGETABLES = "Not Specified_Vegetables"
    OTHER_VEGETABLES = "Other_Vegetables"


class EnvironmentSubcategory(str, Enum):
    NOT_SPECIFIED_ENVIRONMENT = "Not Specified_Environment"
    OTHER_ENVIRONMENT = "Other_Environment"
    WASTEWATER = "Wastewater"
    WATER = "Water"
    SOIL_WASTE = "Soil-Waste"
    SOIL = "Soil"
    AIR = "Air"


class WastewaterSpecificType(str, Enum):
    HEALTH_FACILITIES = "Health Facilities"
    AGRICULTURE = "Agriculture"
    AQUACULTURE = "Aquaculture"
    INDUSTRIAL = "Industrial"
    DOMESTIC = "Domestic"
    NOT_SPECIFIED_WASTEWATER = "Not Specified_Wastewater"
    OTHER_WASTEWATER = "Other_Wastewater"


class WaterSpecificType(str, Enum):
    SURFACE_WATER = "Surface Water"
    WATER_NOT_SPECIFIED_WATER = "Water Not Specified_Water"
    OTHER_WATER = "Other_Water"


class SoilWasteSpecificType(str, Enum):
    COMPOST = "Compost"
    MANURE = "Manure"
    SLUDGE = "Sludge"
    NOT_SPECIFIED_SOIL_WASTE = "Not Specified_Soil-Waste"
    OTHER_SOIL_WASTE = "Other_Soil-Waste"


class SoilSpecificType(str, Enum):
    NOT_SPECIFIED_SOIL = "Not Specified_Soil"
    OTHER_SOIL = "Other_Soil"


# Define Classification classes for each category
class HumanClassification(BaseModel):
    classification_type: Literal["Human"] = "Human"  # Unique first key
    category: Literal[SectorCategory.HUMAN]
    subcategory: Literal[HumanSubcategory.HUMAN]

    def map_to_string(self) -> str:
        mapped = SECTOR_MAPPING[self.category.value][self.subcategory.value]
        return f"{mapped.split(' ')[0]} Sector / {self.category.value} / {self.subcategory.value}"


class NotSpecifiedClassification(BaseModel):
    classification_type: Literal["NotSpecified"] = "NotSpecified"  # Unique first key
    category: Literal[SectorCategory.NOT_SPECIFIED]
    subcategory: Literal[NotSpecifiedSubcategory.NOT_SPECIFIED_SECTOR]

    def map_to_string(self) -> str:
        mapped = SECTOR_MAPPING[self.category.value][self.subcategory.value]
        return f"{mapped.split(' ')[0]} Sector / {self.category.value} / {self.subcategory.value}"


class AnimalClassification(BaseModel):
    classification_type: Literal["Animal"] = "Animal"  # Unique first key
    category: Literal[SectorCategory.ANIMAL]
    subcategory: AnimalSubcategory
    specific_type: Optional[
        Union[LivestockSpecificType, PoultrySpecificType, AquacultureSpecificType, str]
    ] = None

    def map_to_string(self) -> str:
        if self.specific_type:
            mapped = SECTOR_MAPPING[self.category.value][self.subcategory.value][
                self.specific_type.value
            ]
        else:
            sub_mapping = SECTOR_MAPPING[self.category.value][self.subcategory.value]
            if isinstance(sub_mapping, dict):
                mapped = sub_mapping.get(None)
            else:
                mapped = sub_mapping
        result = f"{mapped.split(' ')[0]} Sector / {self.category.value} / {self.subcategory.value}"
        if self.specific_type:
            result += f" / {self.specific_type.value}"
        return result

    @model_validator(mode="after")
    def validate_specific_type(self):
        if self.subcategory in [
            AnimalSubcategory.LIVESTOCK,
            AnimalSubcategory.POULTRY,
            AnimalSubcategory.AQUACULTURE,
        ]:
            if self.specific_type is None:
                raise ValueError(
                    f"specific_type must be specified when subcategory is '{self.subcategory.value}'"
                )
        else:
            self.specific_type = None
        return self


class PlantClassification(BaseModel):
    classification_type: Literal["Plant"] = "Plant"  # Unique first key
    category: Literal[SectorCategory.PLANT]
    subcategory: PlantSubcategory
    specific_type: Optional[
        Union[CropSpecificType, FruitsSpecificType, VegetablesSpecificType, str]
    ] = None

    def map_to_string(self) -> str:
        if self.specific_type:
            mapped = SECTOR_MAPPING[self.category.value][self.subcategory.value][
                self.specific_type.value
            ]
        else:
            sub_mapping = SECTOR_MAPPING[self.category.value][self.subcategory.value]
            if isinstance(sub_mapping, dict):
                mapped = sub_mapping.get(None)
            else:
                mapped = sub_mapping
        result = f"{mapped.split(' ')[0]} Sector / {self.category.value} / {self.subcategory.value}"
        if self.specific_type:
            result += f" / {self.specific_type.value}"
        return result

    @model_validator(mode="after")
    def validate_specific_type(self):
        if self.subcategory in [
            PlantSubcategory.CROP,
            PlantSubcategory.FRUITS,
            PlantSubcategory.VEGETABLES,
        ]:
            if self.specific_type is None:
                raise ValueError(
                    f"specific_type must be specified when subcategory is '{self.subcategory.value}'"
                )
        else:
            self.specific_type = None
        return self


class EnvironmentClassification(BaseModel):
    classification_type: Literal["Environment"] = "Environment"  # Unique first key
    category: Literal[SectorCategory.ENVIRONMENT]
    subcategory: EnvironmentSubcategory
    specific_type: Optional[
        Union[
            WastewaterSpecificType,
            WaterSpecificType,
            SoilWasteSpecificType,
            SoilSpecificType,
            str,
        ]
    ] = None

    def map_to_string(self) -> str:
        if self.specific_type:
            mapped = SECTOR_MAPPING[self.category.value][self.subcategory.value][
                self.specific_type.value
            ]
        else:
            sub_mapping = SECTOR_MAPPING[self.category.value][self.subcategory.value]
            if isinstance(sub_mapping, dict):
                mapped = sub_mapping.get(None)
            else:
                mapped = sub_mapping
        result = f"{mapped.split(' ')[0]} Sector / {self.category.value} / {self.subcategory.value}"
        if self.specific_type:
            result += f" / {self.specific_type.value}"
        return result

    @model_validator(mode="after")
    def validate_specific_type(self):
        if self.subcategory in [
            EnvironmentSubcategory.WASTEWATER,
            EnvironmentSubcategory.WATER,
            EnvironmentSubcategory.SOIL_WASTE,
            EnvironmentSubcategory.SOIL,
        ]:
            if self.specific_type is None:
                raise ValueError(
                    f"specific_type must be specified when subcategory is '{self.subcategory.value}'"
                )
        else:
            self.specific_type = None
        return self


# Union of all possible classifications
SectorClassification = Union[
    HumanClassification,
    AnimalClassification,
    PlantClassification,
    EnvironmentClassification,
    NotSpecifiedClassification,
]


class SectorClassificationResult(BaseModel):
    sector: List[SectorClassification]
    explanation: str
    confidence: float
    confidence_explanation: str


def count_classifications(mapping):
    total = 0
    for category, subcategories in mapping.items():
        for subcategory, specifics in subcategories.items():
            if isinstance(specifics, dict):
                total += len(specifics)
            else:
                total += 1
    return total


if __name__ == "__main__":
    total_classifications = count_classifications(SECTOR_MAPPING)
    print(f"Total Classifications: {total_classifications}")
    # Output: Total Classifications: 69
