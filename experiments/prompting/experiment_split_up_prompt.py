from experiments.prompting.super_prompting_experiment import BaseExperimentPrompting
from prompting.prompting_strategies import PromptBuilder


class ExperimentSplitUpPrompt(BaseExperimentPrompting):
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
        prompt = (prompt_builder.add_input_text([self.test_files])
                  .add_demonstrations([self.demonstrations])
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

    def add_task_3(self):
        prompt_builder = PromptBuilder()
        prompt = (prompt_builder.add_task_3()
                  .build_prompt())
        return prompt

    def run(self):
        filename = "split_up_prompt"

        prompt = self.generate_system_message()
        self.dialogue.add_system_prompt(prompt, filename, print_conversation=False)

        prompt = self.add_input()
        self.dialogue.prompt_assistant_response(prompt, filename, print_conversation=False)

        prompt = self.add_task_1()
        self.dialogue.prompt_assistant_response(prompt, filename, print_conversation=False)

        prompt = self.add_task_2()
        self.dialogue.prompt_assistant_response(prompt, filename, print_conversation=False)

        prompt = self.add_task_3()
        self.dialogue.prompt_assistant_response(prompt, filename)

        while True:
            self.dialogue.add_dynamic_prompting(filename)


if __name__ == '__main__':
    experiment_split_up_prompt = ExperimentSplitUpPrompt()
    experiment_split_up_prompt.run()
