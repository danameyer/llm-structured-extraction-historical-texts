import json
import os
from pathlib import Path

from experiments.function_calling.super_function_calling_experiment import BaseExperimentFunctionCalling
from function_calling_components.chat_completion_blueprint import DialogueCompletion
from function_calling_components.chat_file_writer import ChatFileWriter
from functions.ExtractJsonFromPlainText import ExtractJsonFromPlainText
from prompting.prompting_strategies import PromptBuilder


class ExperimentFunctionCallingNoPrinciplesBasePrompt(BaseExperimentFunctionCalling):
    def __init__(self, test_files, demonstrations, gpt_model, experiment_dir):
        super().__init__(test_files, demonstrations, experiment_dir)
        self.dialogue = DialogueCompletion(model=gpt_model, experiment_dir=experiment_dir)

    def experiment_base_prompt(self):
        prompt_builder = PromptBuilder()
        prompt = prompt_builder.add_base_prompt(self.test_files).build_prompt()
        return prompt

    def run(self):
        filename = "function_calling_no_principles_base_prompt"

        prompt = self.experiment_base_prompt()
        extract_json_from_plaintext = ExtractJsonFromPlainText()
        function_list = [extract_json_from_plaintext]
        self.dialogue.prompt_assistant_response(prompt, filename, function_list, print_conversation=False)
        function_call_result = self.dialogue.function_call_result

        return function_call_result

