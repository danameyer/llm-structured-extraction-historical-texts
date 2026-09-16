from experiments.super_function_calling_experiment import BaseExperimentFunctionCalling
from function_calling_setup.providers.base_provider import LLMProvider
from function_definition.extract_json_from_plain_text import ExtractJsonFromPlainText
from prompting.prompting_strategies import PromptBuilder


class ExperimentFunctionCallingWithOptimisedPrompt(BaseExperimentFunctionCalling):
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

    @staticmethod
    def generate_system_message():
        prompt_builder = PromptBuilder()
        prompt = (prompt_builder.add_persona_modelling()
                  .add_context()
                  .add_iterative_approach()
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

    @staticmethod
    def add_task():
        prompt_builder = PromptBuilder()
        prompt = prompt_builder.add_task().build_prompt()
        return prompt

    def run(self):
        prompt = self.generate_system_message()
        self.dialogue.add_system_prompt(prompt, self.pred_file_name, print_conversation=False)
        prompt = self.add_input() + "\n\n" + self.add_task()
        extract_json_from_plaintext = ExtractJsonFromPlainText()
        function_list = [extract_json_from_plaintext]

        self.dialogue.prompt_assistant_response(
            prompt,
            self.pred_file_name,
            function_list,
            print_conversation=False
        )

        return self.dialogue.function_call_result
