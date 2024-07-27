import json
import os
from pathlib import Path

from experiments.function_calling.super_function_calling_experiment import BaseExperimentFunctionCalling
from function_calling_components.chat_completion_blueprint import DialogueCompletion
from function_calling_components.chat_file_writer import ChatFileWriter
from functions.ExtractJsonFromPlainText import ExtractJsonFromPlainText
from prompting.prompting_strategies import PromptBuilder


class ExperimentFunctionCallingWithOptimisedPrompt(BaseExperimentFunctionCalling):
    def __init__(self, test_files, demonstrations, gpt_model):
        super().__init__(test_files, demonstrations)
        self.dialogue = DialogueCompletion(model=gpt_model)


    def generate_system_message(self):
        prompt_builder = PromptBuilder()
        prompt = (prompt_builder.add_persona_modelling()
                  .add_context()
                  .add_iterative_approach()
                  .add_q_and_a_prompting()
                  .build_prompt())
        return prompt

    def add_input(self):
        prompt_builder = PromptBuilder()
        prompt = (prompt_builder.add_input_text(self.test_files)
                  .add_demonstrations(self.demonstrations)
                  .add_schema_information()
                  .add_constraints()
                  .add_emotional_prompting()
                  .build_prompt())
        return prompt

    def add_task_1(self):
        prompt_builder = PromptBuilder()
        prompt = (prompt_builder.add_task_1()
                  .build_prompt())
        return prompt

    def add_task_2(self):
        prompt_builder = PromptBuilder()
        prompt = (prompt_builder.add_task_2()
                  .build_prompt())
        return prompt

    def run(self):
        filename = "function_calling_with_optimised_prompt"

        prompt = self.generate_system_message()
        self.dialogue.add_system_prompt(prompt, filename, print_conversation=False)

        prompt = self.add_input()
        self.dialogue.prompt_assistant_response(prompt, filename, print_conversation=False)

        prompt = self.add_task_1()
        self.dialogue.prompt_assistant_response(prompt, filename, print_conversation=False)

        prompt = self.add_task_2()
        extract_json_from_plaintext = ExtractJsonFromPlainText()
        function_list = [extract_json_from_plaintext]
        self.dialogue.prompt_assistant_response(prompt, filename, function_list, print_conversation=False)
        function_call_result = self.dialogue.function_call_result

        return function_call_result


if __name__ == '__main__':
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    file_path_test_data = os.path.join(base_dir, "test_data", "test_txt", "sample_text.txt")
    file_path_demonstrations = os.path.join(base_dir, "test_data", "demonstrations", "demonstration_1.txt")
    experiment = ExperimentFunctionCallingWithOptimisedPrompt(file_path_test_data,
                                                              file_path_demonstrations,
                                                              'gpt-3.5-turbo')
    response_json = experiment.run()
    response_json_str = json.dumps(response_json, indent=4)
    base_name = os.path.splitext("sample_text.json")[0]
    pred_filename = f'pred_{base_name}.json'
    output_path_response = os.path.join(base_dir, "test_data", "test_json_diff", "predictions", pred_filename)
    chat_file_writer = ChatFileWriter()
    chat_file_writer.save_response(response_json_str, output_path_response, timestamp=False)
    print(response_json)
