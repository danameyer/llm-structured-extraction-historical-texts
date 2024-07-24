import json
import os
import uuid
from pathlib import Path

from experiments.prompting.experiment_split_up_prompt import ExperimentSplitUpPrompt
from functions.ExtractJsonFromPlainText import ExtractJsonFromPlainText
from prompting.prompting_strategies import PromptBuilder


class OpenAiFineTuningHistory:
    def __init__(self):
        self.messages = []
        self.tools = []
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        self.test_files = os.path.join(base_dir, "test_data", "test_txt", "sample_text.txt")
        self.demonstrations = os.path.join(base_dir, "test_data", "demonstrations", "demonstration_1.txt")

    def add_user_message(self, content):
        self.messages.append({
            "role": "user",
            "content": content
        })

    def add_system_message(self, content):
        self.messages.append({
            "role": "system",
            "content": content
        })

    def add_assistant_tool_call(self, function_name, arguments):
        call_id = str(uuid.uuid4())
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(arguments)
            }
        }
        self.messages.append({
            "role": "assistant",
            "tool_calls": [tool_call]
        })

    def add_tool(self, function_name, description, parameters):
        parameters_dict = self._serialize_parameters(parameters)
        tool = {
            "type": "function",
            "function": {
                "name": function_name,
                "description": description,
                "parameters": parameters_dict
            }
        }
        self.tools.append(tool)

    def build_prompt(self, file_path):
        prompt_builder = PromptBuilder()
        prompt = (prompt_builder.add_input_text([file_path])
                  .add_demonstrations([self.demonstrations])
                  .add_schema_information()
                  .add_constraints()
                  .add_emotional_prompting()
                  .add_task_2()
                  .build_prompt())
        return prompt

    def build(self):
        return {
            "messages": self.messages,
            "tools": self.tools
        }

    def _serialize_parameters(self, parameters):
        # Convert parameters to a serializable format
        if isinstance(parameters, dict):
            return {k: self._serialize_parameters(v) for k, v in parameters.items()}
        elif isinstance(parameters, list):
            return [self._serialize_parameters(i) for i in parameters]
        elif hasattr(parameters, "__dict__"):
            return self._serialize_parameters(parameters.__dict__)
        else:
            return parameters


def create_fine_tuning_for_single_prompt(file_path, gt_function_arguments):
    function_object = ExtractJsonFromPlainText()
    open_ai_fine_tuning_history = OpenAiFineTuningHistory()
    prompt = open_ai_fine_tuning_history.build_prompt(file_path)
    open_ai_fine_tuning_history.add_user_message(prompt)

    with open(gt_function_arguments, 'r') as f:
        gt_content = f.read()

    open_ai_fine_tuning_history.add_assistant_tool_call(function_object.get_definition().name,
                                                        gt_content)
    open_ai_fine_tuning_history.add_tool(function_object.get_definition().name,
                                         function_object.get_definition().description,
                                         function_object.get_definition().parameters)

    return open_ai_fine_tuning_history


def run():
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    text_files_dir = os.path.join(base_dir, "test_data", "test_txt")
    pred_files_dir = os.path.join(base_dir, "test_data", "test_json_diff", "ground_truth")

    fine_tuning_objects = []

    for text_file in os.listdir(text_files_dir):
        if text_file.endswith(".txt"):
            base_name = os.path.splitext(text_file)[0]
            gt_file_name = f"{base_name}.json"
            gt_file_path = os.path.join(pred_files_dir, gt_file_name)

            if os.path.exists(gt_file_path):
                fine_tuning_object = create_fine_tuning_for_single_prompt(
                    os.path.join(text_files_dir, text_file),
                    gt_file_path
                )
                fine_tuning_objects.append(fine_tuning_object.build())
            else:
                print(f"Ground truth file {gt_file_name} does not exist for text file {text_file}")

    fine_tuning_file_path = os.path.join(base_dir, "test_data", "fine_tuning")
    os.makedirs(fine_tuning_file_path, exist_ok=True)
    output_file_path = os.path.join(fine_tuning_file_path, "open_ai_fine_tuning.jsonl")
    with open(output_file_path, 'w') as output_file:
        for obj in fine_tuning_objects:
            output_file.write(json.dumps(obj) + '\n')


if __name__ == "__main__":
    run()
