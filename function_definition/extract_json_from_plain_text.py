from function_calling_setup.function_calling import (
    Function,
    FunctionBuilder,
    Parameter,
    Property
)
from function_definition.base_function import BaseFunction


class ExtractJsonFromPlainText(BaseFunction):

    FAMILY_RELATION_TYPES = [
        "Pater",
        "Mater",
        "Frater",
        "Soror",
        "Filius",
        "Filia",
        "Avus",
        "Avia",
        "Avunculus",
        "Consanguineus",
        "Noverca",
        "Vitricus",
        "Privignus",
        "Privigna",
        "Matertera",
        "Patruus",
        "Amita",
        "Nepos",
        "Neptis",
        "Maritus",
        "Uxor",
        "Progenitor",
        "Progenies",
    ]

    LEGAL_RELATION_TYPES = [
        "Tenens",
        "Dominus Feodi",
        "Testator",
        "Heres",
        "Plegiarius",
        "Attornatus",
        "Principalis",
        "Essoniator",
        "Particeps",
        "Reus",
        "Petitor",
        "Custos",
        "Pupillus",
        "Warantus"
    ]

    @staticmethod
    def apply_default_values(properties, data):
        for key, prop in properties.items():
            if key == "id" and key in data and data[key] is not None:
                continue

            if key not in data or data[key] is None:
                data[key] = prop.default_value

            elif prop.type == "object" and prop.properties:
                ExtractJsonFromPlainText.apply_default_values(
                    prop.properties,
                    data[key],
                )

            elif (
                prop.type == "array"
                and prop.items
                and isinstance(data[key], list)
            ):
                for item in data[key]:
                    if isinstance(item, dict):
                        ExtractJsonFromPlainText.apply_default_values(
                            prop.items.properties,
                            item,
                        )

    @staticmethod
    def _relation_property(
        description: str,
        relation_types: list[str],
    ) -> Property:
        return Property(
            property_type="object",
            description=description,
            properties={
                "relation_type": Property(
                    property_type="string",
                    description="Type of relation",
                    enum=relation_types,
                    default_value="",
                ),
                "related_person": Property(
                    property_type=["integer", "null"],
                    description="ID of the related person",
                    default_value=None,
                ),
            },
            required=[
                "relation_type",
                "related_person",
            ],
            additional_properties=False,
        )

    def get_definition(self) -> FunctionBuilder:
        person_property = Property(
            property_type="object",
            description=(
                "Description of an individual person "
                "mentioned in the input plain text"
            ),
            properties={
                "id": Property(
                    property_type="integer",
                    description="Unique identifier for the person",
                ),
                "name": Property(
                    property_type="string",
                    description="Name of the person",
                ),
                "cognomen": Property(
                    property_type="string",
                    description="Addition to first name",
                    default_value="",
                ),
                "profession": Property(
                    property_type="string",
                    description="Profession of the person",
                    default_value="",
                ),
                "family_relations": Property(
                    property_type="array",
                    description="List of family relations",
                    items=self._relation_property(
                        description="Family relation object",
                        relation_types=self.FAMILY_RELATION_TYPES,
                    ),
                    default_value=[],
                ),
                "legal_relationship": Property(
                    property_type="array",
                    description="List of legal relations",
                    items=self._relation_property(
                        description="Legal relationship object",
                        relation_types=self.LEGAL_RELATION_TYPES,
                    ),
                    default_value=[],
                ),
                "place_of_origin": Property(
                    property_type="string",
                    description="Place of origin of the person",
                    default_value="",
                ),
                "title": Property(
                    property_type="string",
                    description="Title of the person",
                    default_value="",
                ),
            },
            required=[
                "id",
                "name",
                "cognomen",
                "profession",
                "family_relations",
                "legal_relationship",
                "place_of_origin",
                "title",
            ],
            additional_properties=False,
        )

        return FunctionBuilder(
            tool_type="function",
            function=Function(
                name="extract_json_from_plaintext",
                description=(
                    "Extract all information about all the persons "
                    "mentioned in the input plain text"
                ),
                parameters=Parameter(
                    parameter_type="object",
                    properties={
                        "person_list": Property(
                            property_type="array",
                            description=(
                                "List of people mentioned "
                                "in the input plain text"
                            ),
                            items=person_property,
                        ),
                    },
                    required=["person_list"],
                    additionalProperties=False,
                ),
            ),
        )

    def run(self, **kwargs):
        properties = (
            self.get_definition()
            .function
            .parameters
            .properties["person_list"]
            .items
            .properties
        )

        person_list = kwargs.get("person_list", [])

        for person in person_list:
            self.apply_default_values(properties, person)

        kwargs["person_list"] = person_list

        print(kwargs)
        return kwargs