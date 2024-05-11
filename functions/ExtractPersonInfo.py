import json

import jsons

from function_calling_components.function_calling import FunctionBuilder, Parameter, Property


class ExtractPersonInfo:

    @classmethod
    def get_definition(cls) -> FunctionBuilder:
        return FunctionBuilder(
            name="extract_person_info",
            description="Extract all information about a person with a given name from a json file",
            parameters=Parameter(
                parameter_type="object",
                properties={
                    "name": Property(property_type="string", description="name of the person for whom info should be extracted"),
                },
                required=["name"]
            )
        )

    @classmethod
    def get_definition_dict(cls):
        return json.loads(jsons.dumps(cls.get_definition()))

    @classmethod
    def run(cls, name):
        json_data = cls._read_json("./test_data/sample_json.json")
        person_info = []
        for person in json_data:
            if person['name'] == name:
                person_info.append(person)
        return person_info

    @classmethod
    def _read_json(cls, json_file):
        with open(json_file, 'r') as file:
            json_content = file.read()
        json_data = json.loads(json_content)
        return json_data

