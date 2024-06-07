from function_calling_components.function_calling import FunctionBuilder, Parameter, Property
from functions.BaseFunction import BaseFunction


class ComparePersonsViaLlm(BaseFunction):
    def get_definition(self) -> FunctionBuilder:
        return FunctionBuilder(
            name="compare_persons_via_llm",
            description="Decide if two people with the same name are identical based on the other attributes",
            parameters=Parameter(
                parameter_type="object",
                properties={
                    "reasoning": Property(
                        property_type="object",
                        description="Reasoning if two people with the same name are identical based on the other attributes",
                        properties={
                            "name": Property(property_type="string", description="name of the persons for whom attributes should be compared"),
                            "estimate": Property(property_type="boolean", description="true or false depending on whether persons are the same based on attributes"),
                            "similarities": Property(
                                    property_type="array",
                                    description="List of similarities between persons",
                                    items=Property(
                                        property_type="object",
                                        description="attribute object",
                                        properties={
                                            "attribute_name": Property(property_type="string",
                                                                      description="name of the attribute that is compared"),
                                            "attribute_person_a": Property(property_type="string",
                                                                       description="attribute of the first person"),
                                            "attribute_person_b": Property(property_type="string",
                                                                           description="attribute of the second person"),
                                        }
                                    )
                            ),
                            "differences": Property(
                                property_type="array",
                                description="List of similarities between persons",
                                items=Property(
                                    property_type="object",
                                    description="attribute object",
                                    properties={
                                        "attribute_name": Property(property_type="string",
                                                                   description="name of the attribute that is compared"),
                                        "attribute_person_a": Property(property_type="integer",
                                                                       description="attribute of the first person"),
                                        "attribute_person_b": Property(property_type="integer",
                                                                       description="attribute of the second person"),
                                    }
                                )
                            ),
                        }
                    )
                },
                required=["name", "reasoning", "similarities", "differences"]
            )
        )

    def run(self, **kwargs):
        print(kwargs)
        return kwargs
