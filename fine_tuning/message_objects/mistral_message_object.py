import json
import os
import string
import uuid
from pathlib import Path
import random

from dotenv import load_dotenv

from functions.ExtractJsonFromPlainText import ExtractJsonFromPlainText
from prompting.prompting_strategies import PromptBuilder


class MistralMessageObject:
    def __init__(self):
        self.messages = []
        self.tools = []
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        self.demonstrations = os.path.join(base_dir, "test_data", "demonstrations", "demonstration_1.txt")

    def add_message(self, role, content, tool_call_id=None):
        message = {
            "role": role,
            "content": content
        }
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        self.messages.append(message)

    def add_tool(self, function):
        tool = {
            "type": "function",
            "function": function
        }
        self.tools.append(tool)

    @staticmethod
    def generate_tool_call_id(length=9):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    def generate_tool_call(self, function_name, arguments):
        tool_call_id = self.generate_tool_call_id()
        tool = {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": arguments
            }
        }

        self.messages.append({
            "role": "assistant",
            "tool_calls": [tool]
        })

        return tool_call_id

    def build(self):
        return {
            "messages": self.messages,
            "tools": self.tools
        }

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

    @staticmethod
    def build_system_message():
        prompt_builder = PromptBuilder()
        prompt = (prompt_builder.add_persona_modelling()
                  .add_context()
                  .add_iterative_approach()
                  .add_q_and_a_prompting()
                  .build_prompt())
        return prompt


def create_fine_tuning_for_single_prompt(file_path, gt_function_arguments):
    function_object = ExtractJsonFromPlainText()
    mistral_message_object = MistralMessageObject()

    system_message = mistral_message_object.build_system_message()
    mistral_message_object.add_message('system', system_message)

    prompt = mistral_message_object.build_prompt(file_path)
    mistral_message_object.add_message('user', prompt)

    with open(gt_function_arguments, 'r') as f:
        gt_content = f.read()

    tool_call_id = mistral_message_object.generate_tool_call(function_object.get_definition().name,
                                                             gt_content)

    mistral_message_object.add_message('tool', gt_content, tool_call_id)

    mistral_message_object.add_tool(function=function_object.get_definition_dict())

    mistral_message_object.add_message('assistant', "This is the structured JSON output: " + gt_content)

    return mistral_message_object


def run():
    load_dotenv()
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
                fine_tuning_object = create_fine_tuning_for_single_prompt(
                    os.path.join(text_files_dir, text_file),
                    gt_file_path
                )
                fine_tuning_objects.append(fine_tuning_object.build())
            else:
                print(f"Ground truth file {gt_file_name} does not exist for text file {text_file}")

    fine_tuning_file_path = os.path.join(base_dir, "test_data", "fine_tuning")
    os.makedirs(fine_tuning_file_path, exist_ok=True)
    output_file_path = os.path.join(fine_tuning_file_path, "file-finetune-mistral.jsonl")
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
