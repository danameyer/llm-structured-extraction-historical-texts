import json
import os
from pathlib import Path

import jsons

from function_calling_components.function_calling import FunctionBuilder, Parameter, Property
from functions.BaseFunction import BaseFunction


class ExtractPersonInfo(BaseFunction):

    def get_definition(self) -> FunctionBuilder:
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

    def run(self, name):
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        test_file_path = os.path.join(base_dir, "test_data", "sample_json.json")
        json_data = self._read_json(test_file_path)
        person_info = []
        for person in json_data:
            if person['name'] == name:
                person_info.append(person)
        return person_info

    def _read_json(self, json_file):
        with open(json_file, 'r') as file:
            json_content = file.read()
        json_data = json.loads(json_content)
        return json_data

