from experiments.super_function_calling_experiment import BaseExperimentFunctionCalling
from function_calling_setup.providers.base_provider import LLMProvider
from function_definition.extract_json_from_plain_text import ExtractJsonFromPlainText
from prompting.prompting_strategies import PromptBuilder


class ExperimentFunctionCallingNoPrinciplesBasePrompt(BaseExperimentFunctionCalling):
    def __init__(
            self,
            test_files,
            demonstrations,
            provider: LLMProvider,
            experiment_dir,
            pred_file_name,
    ):
        super().__init__(
            test_files,
            demonstrations,
            experiment_dir,
            pred_file_name,
            provider,
        )

    def experiment_base_prompt(self):
        prompt_builder = PromptBuilder()
        prompt = prompt_builder.add_base_prompt(self.test_files).build_prompt()
        return prompt

    def run(self):
        prompt = self.experiment_base_prompt()
        extract_json_from_plaintext = ExtractJsonFromPlainText()
        function_list = [extract_json_from_plaintext]
        self.dialogue.prompt_assistant_response(prompt,
                                                self.pred_file_name,
                                                function_list,
                                                print_conversation=False)
        function_call_result = self.dialogue.function_call_result

        return function_call_result

