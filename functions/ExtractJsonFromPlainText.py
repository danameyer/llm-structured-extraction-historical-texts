import json

from function_calling_components.function_calling import Property, Parameter, FunctionBuilder, Function
from functions.BaseFunction import BaseFunction


class ExtractJsonFromPlainText(BaseFunction):

    def __init__(self, gpt_model):
        super().__init__()
        self.gpt_model = gpt_model

    @staticmethod
    def apply_default_values(properties, data):
        for key, prop in properties.items():
            if key == 'id' and 'id' in data and data['id'] is not None:
                continue
            if key not in data or data[key] is None:
                data[key] = prop.default_value
            elif prop.type == "object" and prop.properties:
                ExtractJsonFromPlainText.apply_default_values(prop.properties, data[key])
            elif prop.type == "array" and prop.items and isinstance(data[key], list):
                for item in data[key]:
                    if isinstance(item, dict):
                        ExtractJsonFromPlainText.apply_default_values(prop.items.properties, item)

    def get_definition(self) -> FunctionBuilder:
        return FunctionBuilder(
            tool_type="function",
            function=Function(
                name="extract_json_from_plaintext",
                description="Extract all information about all the persons mentioned in the input plain text",
                parameters=Parameter(
                    parameter_type="object",
                    properties={
                        "person_list": Property(
                            property_type="array",
                            description="List of people mentioned in the input plain text",
                            items=Property(
                                property_type="object",
                                description="Description of individual person mentioned in the input plain text",
                                properties={
                                    "id": Property(property_type="integer",
                                                   description="Unique identifier for the person"),
                                    "name": Property(property_type="string",
                                                     description="Name of the person"),
                                    "cognomen": Property(property_type="string",
                                                         description="Addition to first name",
                                                         default_value=""),
                                    "profession": Property(property_type="string",
                                                           description="Profession of the person",
                                                           default_value=""),
                                    "family_relations": Property(
                                        property_type="array",
                                        description="List of family relations",
                                        items=Property(
                                            property_type="object",
                                            description="Family relation object",
                                            properties={
                                                "relation_type": Property(property_type="string",
                                                                          enum=["pater",
                                                                                "mater",
                                                                                "frater",
                                                                                "soror",
                                                                                "filius",
                                                                                "filia",
                                                                                "avus",
                                                                                "avia",
                                                                                "noverca",
                                                                                "vitricus",
                                                                                "privignus",
                                                                                "privigna",
                                                                                "matertera",
                                                                                "patruus",
                                                                                "amita",
                                                                                "nepos",
                                                                                "neptis"],
                                                                          description="Type of family relation (e.g., "
                                                                                      "pater, frater, filius, "
                                                                                      "filia)",
                                                                          default_value=""),
                                                "related_person": Property(property_type="integer",
                                                                           description="ID of the related person",
                                                                           default_value=None)
                                            }
                                        ),
                                        default_value=[]
                                    ),
                                    "legal_relationship": Property(
                                        property_type="array",
                                        description="List of legal relations such as custos and heres",
                                        items=Property(
                                            property_type="object",
                                            description="Legal relationship object",
                                            properties={
                                                "relation_type": Property(property_type="string",
                                                                          enum=[
                                                                              "tenens",
                                                                              "dominus feodi",
                                                                              "testator",
                                                                              "heres",
                                                                              "plegiarius",
                                                                              "attornatus",
                                                                              "reus",
                                                                              "petitor",
                                                                              "custos",
                                                                              "pupillus",
                                                                              "progenitor",
                                                                              "progenies"
                                                                          ],
                                                                          description="Type of legal relation",
                                                                          default_value=""),
                                                "related_person": Property(property_type="integer",
                                                                           description="ID of the related person",
                                                                           default_value=None)
                                            }
                                        ),
                                        default_value=[]
                                    ),
                                    "place_of_origin": Property(property_type="string",
                                                                description="Place of origin of the person",
                                                                default_value=""),
                                    "title": Property(property_type="string",
                                                      description="Title of the person",
                                                      default_value="")
                                }
                            )
                        )
                    },
                    additionalProperties=json.loads(json.dumps(False)),
                    required=["name",
                              "id",
                              "cognomen",
                              "profession",
                              "place_of_origin",
                              "title",
                              "family_relations",
                              "legal_relationship",
                              "related_person",
                              "relation_type"
                              ]
                )
            )
        )

    def get_definition_structured_outputs(self) -> FunctionBuilder:
        return FunctionBuilder(
            tool_type="function",
            strict=json.loads(json.dumps(True)),
            function=Function(
                name="extract_json_from_plaintext",
                description="Extract all information about all the persons mentioned in the input plain text",
                parameters=Parameter(
                    parameter_type="object",
                    properties={
                        "person_list": Property(
                            property_type="array",
                            description="List of people mentioned in the input plain text",
                            items=Property(
                                property_type="object",
                                description="Description of individual person mentioned in the input plain text",
                                properties={
                                    "id": Property(property_type="integer",
                                                   description="Unique identifier for the person"),
                                    "name": Property(property_type="string",
                                                     description="Name of the person"),
                                    "cognomen": Property(property_type="string",
                                                         description="Addition to first name",
                                                         default_value=""),
                                    "profession": Property(property_type="string",
                                                           description="Profession of the person",
                                                           default_value=""),
                                    "family_relations": Property(
                                        property_type="array",
                                        description="List of family relations",
                                        items=Property(
                                            property_type="object",
                                            description="Family relation object",
                                            properties={
                                                "relation_type": Property(property_type="string",
                                                                          enum=["pater",
                                                                                "mater",
                                                                                "frater",
                                                                                "soror",
                                                                                "filius",
                                                                                "filia",
                                                                                "avus",
                                                                                "avia",
                                                                                "noverca",
                                                                                "vitricus",
                                                                                "privignus",
                                                                                "privigna",
                                                                                "matertera",
                                                                                "patruus",
                                                                                "amita",
                                                                                "nepos",
                                                                                "neptis"],
                                                                          description="Type of family relation (e.g., "
                                                                                      "pater, frater, filius, "
                                                                                      "filia)",
                                                                          default_value=""),
                                                "related_person": Property(property_type="integer",
                                                                           description="ID of the related person",
                                                                           default_value=None)
                                            }
                                        ),
                                        default_value=[]
                                    ),
                                    "legal_relationship": Property(
                                        property_type="array",
                                        description="List of legal relations such as custos and heres",
                                        items=Property(
                                            property_type="object",
                                            description="Legal relationship object",
                                            properties={
                                                "relation_type": Property(property_type="string",
                                                                          enum=[
                                                                              "tenens",
                                                                              "dominus feodi",
                                                                              "testator",
                                                                              "heres",
                                                                              "plegiarius",
                                                                              "attornatus",
                                                                              "reus",
                                                                              "petitor",
                                                                              "custos",
                                                                              "pupillus",
                                                                              "progenitor",
                                                                              "progenies"
                                                                          ],
                                                                          description="Type of legal relation",
                                                                          default_value=""),
                                                "related_person": Property(property_type="integer",
                                                                           description="ID of the related person",
                                                                           # default_value=None
                                                                           )
                                            }
                                        ),
                                        default_value=[]
                                    ),
                                    "place_of_origin": Property(property_type="string",
                                                                description="Place of origin of the person",
                                                                default_value=""),
                                    "title": Property(property_type="string",
                                                      description="Title of the person",
                                                      default_value="")
                                }
                            )
                        )
                    },
                    additionalProperties=json.loads(json.dumps(False)),
                    required=["name",
                              "id",
                              "cognomen",
                              "profession",
                              "place_of_origin",
                              "title",
                              "family_relations",
                              "legal_relationship",
                              "related_person",
                              "relation_type"
                              ]
                )
            )
        )

    def run(self, **kwargs):
        if self.gpt_model == "gpt-3.5-turbo":
            properties = self.get_definition().function.parameters.properties["person_list"].items.properties
        else:
            properties = self.get_definition_structured_outputs().function.parameters.properties["person_list"].items.properties

        person_list = kwargs.get('person_list', [])
        for person in person_list:
            ExtractJsonFromPlainText.apply_default_values(properties, person)
        print(kwargs)
        return kwargs
