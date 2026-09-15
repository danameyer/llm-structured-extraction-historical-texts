from experiments.super_function_calling_experiment import BaseExperimentFunctionCalling
from prompting.prompting_strategies import PromptBuilder
from function_calling_setup.chat_completion_no_function_calling_new import DialogueCompletionNoFunctionCallingNew, ResponseMode


class ExperimentOptimisedPromptNoFunctionCalling(BaseExperimentFunctionCalling):
    def __init__(
            self,
            test_files,
            demonstrations,
            gpt_model,
            experiment_dir,
            pred_file_name,
            final_response_mode: ResponseMode
    ):
        super().__init__(
            test_files,
            demonstrations,
            experiment_dir,
            pred_file_name,
            gpt_model
        )

        self.gpt_model = gpt_model

        self.dialogue = (
            DialogueCompletionNoFunctionCallingNew(model=gpt_model, experiment_dir=experiment_dir)
        )

        self.pred_file_name = pred_file_name
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
        prompt = (prompt_builder.add_task_without_tool_calling()
                  .build_prompt())
        return prompt

    def run(self):
        prompt = self.generate_system_message()
        self.dialogue.add_system_prompt(prompt, self.pred_file_name, print_conversation=False)

        prompt = (
                self.add_input()
                + "\n\n"
                + self.add_task()
        )

        self.dialogue.prompt_assistant_response(
            prompt,
            self.pred_file_name,
            print_conversation=False,
            response_mode=self.final_response_mode,
        )

        return self.dialogue.json_result
