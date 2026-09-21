from experiments.super_function_calling_experiment import BaseExperimentFunctionCalling
from function_calling_setup.providers.base_provider import LLMProvider
from prompting.prompting_strategies import PromptBuilder
from function_calling_setup.dialogue_completion_no_function_calling import DialogueCompletionNoFunctionCalling, ResponseMode


class ExperimentOptimisedPromptNoFunctionCalling(BaseExperimentFunctionCalling):
    def __init__(
            self,
            test_files,
            demonstrations,
            provider: LLMProvider,
            experiment_dir,
            pred_file_name,
            final_response_mode: ResponseMode,
    ):
        super().__init__(
            test_files,
            demonstrations,
            experiment_dir,
            pred_file_name,
            provider,
            dialogue_cls=DialogueCompletionNoFunctionCalling,
        )

        self.final_response_mode = final_response_mode

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
        prompt = (prompt_builder.add_base_prompt_without_tool_calling(self.test_files)
                  .add_demonstrations(self.demonstrations)
                  .add_schema_information()
                  .add_constraints()
                  .add_emotional_prompting()
                  .build_prompt())
        return prompt

    def run(self):
        prompt = self.generate_system_message()
        self.dialogue.add_system_prompt(prompt, self.pred_file_name, print_conversation=False)
        prompt = self.add_input()

        self.dialogue.prompt_assistant_response(
            prompt,
            self.pred_file_name,
            print_conversation=False,
            response_mode=self.final_response_mode,
        )

        return self.dialogue.json_result
