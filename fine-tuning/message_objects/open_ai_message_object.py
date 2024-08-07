import json
import os
import uuid
from pathlib import Path
from functions.ExtractJsonFromPlainText import ExtractJsonFromPlainText
from prompting.prompting_strategies import PromptBuilder


class OpenAiMessageObject:
    def __init__(self):
        self.messages = []
        self.tools = []
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
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

    def add_assistant_tool_call(self, function_name, arguments: str):
        call_id = str(uuid.uuid4())
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": arguments
            }
        }
        self.messages.append({
            "role": "assistant",
            "tool_calls": [tool_call]
        })

    def add_tool(self, function):
        tool = {
            "type": "function",
            "function": function
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
        if isinstance(parameters, dict):
            return {k: self._serialize_parameters(v) for k, v in parameters.items()}
        elif isinstance(parameters, list):
            return [self._serialize_parameters(i) for i in parameters]
        elif hasattr(parameters, "__dict__"):
            return self._serialize_parameters(parameters.__dict__)
        else:
            return parameters


def create_fine_tuning_for_single_file(file_path, gt_function_arguments):
    function_object = ExtractJsonFromPlainText()
    open_ai_message_object = OpenAiMessageObject()
    prompt = open_ai_message_object.build_prompt(file_path)
    open_ai_message_object.add_user_message(prompt)

    with open(gt_function_arguments, 'r') as f:
        gt_content = f.read()

    open_ai_message_object.add_assistant_tool_call(function_object.get_definition().name,
                                                   gt_content)
    open_ai_message_object.add_tool(function=function_object.get_definition_dict())

    return open_ai_message_object


def run():
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    text_files_dir = os.path.join(base_dir, "evaluation_results", "text_files")
    gt_files_dir = os.path.join(base_dir, "evaluation_results", "ground_truth")

    fine_tuning_objects = []

    for text_file in os.listdir(text_files_dir):
        if text_file.endswith(".txt"):
            base_name = os.path.splitext(text_file)[0]
            gt_file_name = f"{base_name}.json"
            gt_file_path = os.path.join(gt_files_dir, gt_file_name)

            if os.path.exists(gt_file_path):
                fine_tuning_object = create_fine_tuning_for_single_file(
                    os.path.join(text_files_dir, text_file),
                    gt_file_path
                )
                fine_tuning_objects.append(fine_tuning_object.build())
            else:
                print(f"Ground truth file {gt_file_name} does not exist for text file {text_file}")

    fine_tuning_file_path = os.path.join(base_dir, "test_data", "fine_tuning")
    os.makedirs(fine_tuning_file_path, exist_ok=True)
    output_file_path = os.path.join(fine_tuning_file_path, "file-finetune-openai.jsonl")
    with open(output_file_path, 'w') as output_file:
        for obj in fine_tuning_objects:
            json_string = json.dumps(obj)
            cleaned_json_string = (
                json_string
                .replace("\\\\n", "")  # newline in json
                .replace("\\n", " ")  # newline in text
                .replace("\n", " ")
                .replace("\\t", " ")  # tab in text
                .replace("\\u00ad", "")
                .replace("\\u2022", "")
            )
            output_file.write(cleaned_json_string + '\n')


if __name__ == "__main__":
    run()
