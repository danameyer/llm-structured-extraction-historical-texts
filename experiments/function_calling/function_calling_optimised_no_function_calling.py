from experiments.function_calling.super_function_calling_experiment import BaseExperimentFunctionCalling
from function_calling_components.chat_completion_no_function_calling_new import DialogueCompletionNoFunctionCallingNew
from prompting.prompting_strategies import PromptBuilder


class ExperimentOptimisedPromptNoFunctionCalling(BaseExperimentFunctionCalling):
    def __init__(
            self,
            test_files,
            demonstrations,
            gpt_model,
            experiment_dir,
            pred_file_name,
            final_response_mode="json_schema",
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
        prompt = (prompt_builder.add_task_2_without_tool_calling()
                  .build_prompt())
        return prompt

    def run(self):
        # filename = "function_calling_with_optimised_prompt"

        prompt = self.generate_system_message()
        self.dialogue.add_system_prompt(prompt, self.pred_file_name, print_conversation=False)

        prompt = self.add_input()
        self.dialogue.prompt_assistant_response(prompt, self.pred_file_name, print_conversation=False)

        prompt = self.add_task_1()
        self.dialogue.prompt_assistant_response(prompt, self.pred_file_name, print_conversation=False)

        prompt = self.add_task_2()
        self.dialogue.prompt_assistant_response(
            prompt,
            self.pred_file_name,
            print_conversation=False,
            response_mode=self.final_response_mode,
        )

        json_result = self.dialogue.json_result

        return json_result
