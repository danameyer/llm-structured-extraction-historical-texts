import json
import os
from pathlib import Path

from function_calling_components.function_calling import FunctionBuilder, Parameter, Property, Function
from functions.BaseFunction import BaseFunction


class DecideIfSamePerson(BaseFunction):
    def get_definition(self) -> FunctionBuilder:
        return FunctionBuilder(
            tool_type="function",
            function=Function(
                name="decide_if_same_person",
                description="Decide if two people with the same name are the really identical based on the other attributes",
                parameters=Parameter(
                    parameter_type="object",
                    properties={
                        "name": Property(property_type="string",
                                         description="name of the person for whom attributes should be compared")
                    },
                    additionalProperties=json.loads(json.dumps(False)),
                    required=["name"]
                )
            )
        )

    def run(self, name):
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        test_file_path = os.path.join(base_dir, "test_data", "test_gt", "test_output.json")
        try:
            with open(test_file_path, 'r') as file:
                json_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading JSON file: {e}")
            return False

        # Find people with the given name
        people_with_name = [person for person in json_data['person_list'] if person['name'] == name]

        # If no person found with the given name
        if not people_with_name:
            return "No person found with the name '{}'.".format(name)

        # Compare attributes of people with the same name
        identical_attributes = True
        different_attributes = []
        attributes = {}
        for person in people_with_name:
            for key, value in person.items():
                if key != 'id':
                    if key not in attributes:
                        attributes[key] = set()
                    attributes[key].add(str(value))

        # Check if attributes are identical
        for key, value in attributes.items():
            if len(value) > 1:
                identical_attributes = False
                different_attributes.append(key)

        # Construct response
        if identical_attributes:
            response = "Attributes of people with the name '{}' are identical. They share the following attributes:".format(
                name)
            for key in attributes.keys():
                response += "\n- {}: {}".format(key, next(iter(attributes[key])))
        else:
            response = "Attributes of people with the name '{}' are not identical. They differ in the following attributes:".format(
                name)
            for key in different_attributes:
                response += "\n- {}: {}".format(key, ", ".join(attributes[key]))

        return response

    def _read_json(self, json_file):
        with open(json_file, 'r') as file:
            json_content = file.read()
        json_data = json.loads(json_content)
        return json_data
