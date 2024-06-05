from function_calling_components.function_calling import Property, FunctionBuilder, Parameter
from functions.BaseFunction import BaseFunction


class ExtractJsonFromPlainText(BaseFunction):
    def get_definition(self) -> FunctionBuilder:
        return FunctionBuilder(
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

                                "id": Property(property_type="integer", description="Unique identifier for the person"),
                                "name": Property(property_type="string", description="Name of the person"),
                                "profession": Property(property_type="string", description="Profession of the person"),
                                "family_relations": Property(
                                    property_type="array",
                                    description="List of family relations",
                                    items=Property(
                                        property_type="object",
                                        description="Family relation object",
                                        properties={
                                            "relation_type": Property(property_type="string",
                                                                      description="Type of family relation (e.g., pater, frater, filius, filia)"),
                                            "related_person": Property(property_type="integer",
                                                                       description="ID of the related person")
                                        }
                                    )
                                ),
                                "power_relations": Property(
                                    property_type="array",
                                    description="List of power relations",
                                    items=Property(
                                        property_type="object",
                                        description="Power relation object",
                                        properties={
                                            "relation_type": Property(property_type="string",
                                                                      description="Type of power relation"),
                                            "related_person": Property(property_type="integer",
                                                                       description="ID of the related person")
                                        }
                                    )
                                ),
                                "place_of_origin": Property(property_type="string",
                                                            description="Place of origin of the person"),
                                "title": Property(property_type="string", description="Title of the person"),
                                "org_role": Property(property_type="string",
                                                     description="Organizational role of the person")

                            }
                        )
                    )

                },
                required=["name", "id"]
            )
        )

    def run(self, **kwargs):
        print(kwargs)
        return kwargs
