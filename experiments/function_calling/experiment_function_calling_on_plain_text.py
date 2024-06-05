import os
from pathlib import Path

from function_calling_components.chat_completion_blueprint import DialogueCompletion
from functions.ExtractJsonFromPlainText import ExtractJsonFromPlainText


class ExperimentFunctionCallingOnPlainText:

    def __init__(self):
        self.dialogue = DialogueCompletion(model='gpt-3.5-turbo')

    def read_input_text(self, filename):
        with open(filename, 'r') as file:
            content = file.read()
            return content

    def generate_prompt(self):
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        file_path = os.path.join(base_dir, "test_data", "test_txt", "sample_text.txt")
        input_text = self.read_input_text(file_path)
        prompt = "Please extract the following information about all of the persons mentioned in the text from the given text and return it as a JSON object: name, profession, family_relations, power_relations, place_of_origin, title, org_role. Assign an id to each person. This is the text to extract the information from:" + input_text
        prompt_name = "extract_json_from_plaintext"
        return prompt, prompt_name

    def run(self):
        prompt, filename = self.generate_prompt()
        extract_json_from_plaintext = ExtractJsonFromPlainText()
        function_list = [extract_json_from_plaintext]
        self.dialogue.prompt_assistant_response(prompt, filename, function_list)


if __name__ == '__main__':
    experiment = ExperimentFunctionCallingOnPlainText()
    experiment.run()
