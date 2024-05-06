import json

import jsons

from chat_completion_blueprint import FunctionBuilder, Parameter, Property, Message, ChatCompletion


class JsonInfoExtractor:

    def __init__(self):
        pass

    def read_json(self, json_file):
        with open(json_file, 'r') as file:
            json_content = file.read()
        json_data = json.loads(json_content)
        return json_data

    def extract_person_info(self, name):
        json_data = self.read_json("./test_data/sample_json.json")
        person_info = []
        for person in json_data:
            if person['name'] == name:
                person_info.append(person)
        return person_info

    def get_info_for_person(self):
        json_content = self.read_json("./test_data/sample_json.json")
        extract_person_info_function_builder = FunctionBuilder(
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
        # print(json_content)
        extract_person_info_as_dict = json.loads(jsons.dumps(extract_person_info_function_builder))
        message_1 = Message(role="user",
                          content="Please extract all the information about every person from the JSON file. Ask for clarification if you don't know the name. The name of the person is Willelmus:" + jsons.dumps(json_content))

        message_1_as_dict = json.loads(jsons.dumps(message_1))

        # Create an instance of ChatCompletion
        chat_completion = ChatCompletion(model='gpt-3.5-turbo')

        # Generate response with a user message and the extract_entities function
        response = chat_completion.generate_response(
            messages=[message_1_as_dict],
            functions=[extract_person_info_as_dict]
        )

        print(response)
        name = json.loads(response.function_call.arguments).get("name")
        function_name = response.function_call.name

        globals()["extract_person_info"] = self.extract_person_info

        chosen_function = eval(function_name)

        person_info = chosen_function(name)

        print(name)
        print(function_name)
        print(chosen_function)
        print(person_info)


