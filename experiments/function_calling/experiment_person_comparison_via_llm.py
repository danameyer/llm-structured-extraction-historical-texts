import json
import os
from pathlib import Path

from function_calling_components.chat_completion_blueprint import DialogueCompletion
from functions.PersonComparisonViaLlm import ComparePersonsViaLlm


class ExperimentPersonComparisonViaLlm:
    def __init__(self, json_data, experiment_dir):
        self.experiment_dir = experiment_dir
        self.dialogue = DialogueCompletion(model='gpt-3.5-turbo', experiment_dir=experiment_dir)
        self.json_data = json_data

    def generate_prompt(self):
        prompt = "Decide if two people of the same name are the same. The name of the person is Willelmus." + json.dumps(self.json_data)
        prompt_name = "person_comparison_via_llm"
        return prompt, prompt_name

    def _read_json(self, json_file):
        with open(json_file, 'r') as file:
            json_content = file.read()
        json_data = json.loads(json_content)
        return json_data

    def run(self):
        prompt, filename = self.generate_prompt()
        compare_persons_via_llm = ComparePersonsViaLlm()
        function_list = [compare_persons_via_llm]
        self.dialogue.prompt_assistant_response(prompt, filename, function_list, validate=False)


if __name__ == '__main__':
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    test_file_path = os.path.join(base_dir, "test_data", "test_gt", "test_output.json")
    try:
        with open(test_file_path, 'r') as file:
            json_data = json.load(file)
            experiment_1 = ExperimentPersonComparisonViaLlm(json_data, "/tmp/experiments/experiment_person_comparison_via_llm")
            experiment_1.run()
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
