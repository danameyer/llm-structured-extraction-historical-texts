from experiments.function_calling.super_function_calling_experiment import BaseExperimentFunctionCalling
from function_calling_components.chat_completion_blueprint import DialogueCompletion
from functions.ExtractJsonFromPlainText import ExtractJsonFromPlainText
from prompting.prompting_strategies import PromptBuilder


class ExperimentAllPrinciplesZeroShotPrompt(BaseExperimentFunctionCalling):
    def __init__(self, test_files, demonstrations, gpt_model, experiment_dir, pred_file_name):
        super().__init__(test_files, demonstrations, experiment_dir, pred_file_name, gpt_model)
        self.dialogue = DialogueCompletion(model=gpt_model, experiment_dir=experiment_dir)
        self.pred_file_name = pred_file_name
        self.gpt_model = gpt_model

    @staticmethod
    def generate_system_message():
        prompt_builder = PromptBuilder()
        prompt = (prompt_builder.add_persona_modelling()
                  .add_context()
                  .add_iterative_approach()
                  .add_q_and_a_prompting()
                  .build_prompt())
        return prompt

    def experiment_all_principles_zero_shot(self):
        prompt_builder = PromptBuilder()
        prompt = (prompt_builder
                  .add_base_prompt(self.test_files)
                  .add_schema_information()
                  .add_constraints()
                  .add_emotional_prompting()
                  .build_prompt())
        return prompt

    @staticmethod
    def add_task_2():
        prompt_builder = PromptBuilder()
        prompt = (prompt_builder.add_task_2()
                  .build_prompt())
        return prompt

    def run(self):
        # filename = "function_calling_all_principles_zero_shot"

        prompt = self.generate_system_message()
        self.dialogue.add_system_prompt(prompt, self.pred_file_name, print_conversation=False)

        prompt = self.experiment_all_principles_zero_shot()

        extract_json_from_plaintext = ExtractJsonFromPlainText()
        function_list = [extract_json_from_plaintext]
        self.dialogue.prompt_assistant_response(prompt,
                                                self.pred_file_name,
                                                function_list,
                                                print_conversation=False)
        function_call_result = self.dialogue.function_call_result

        return function_call_result

