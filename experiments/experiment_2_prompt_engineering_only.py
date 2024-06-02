import os
from pathlib import Path

from dotenv import load_dotenv

from function_calling_components.chat_completion_blueprint import DialogueCompletion
from prompting.prompting_strategies import PromptBuilder


class Experiment2:

    def __init__(self):
        load_dotenv()
        self.prompt_builder = PromptBuilder()
        self.dialogue = DialogueCompletion(model='gpt-3.5-turbo')
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        self.test_files = os.path.join(base_dir, "test_data", "test_txt", "sample_text.txt")
        self.demonstrations = os.path.join(base_dir, "test_data", "demonstrations", "demonstration_1.txt")

    def experiment_base_prompt(self):
        prompt = self.prompt_builder.add_base_prompt([self.test_files]).build_prompt()
        return prompt, "base_prompt"

    def experiment_all_principles_zero_shot(self):
        prompt = (self.prompt_builder.add_persona_modelling()
                  .add_context()
                  .add_iterative_approach()
                  .add_q_and_a_prompting()
                  .add_base_prompt([self.test_files])
                  .add_schema_information()
                  .add_constraints()
                  .add_emotional_prompting()
                  .build_prompt())
        return prompt, "all_principles_zero_shot"

    def experiment_all_principles_few_shot(self):
        prompt = (self.prompt_builder.add_persona_modelling()
                  .add_context()
                  .add_iterative_approach()
                  .add_q_and_a_prompting()
                  .add_base_prompt([self.test_files])
                  .add_demonstrations([self.demonstrations])
                  .add_schema_information()
                  .add_constraints()
                  .add_emotional_prompting()
                  .build_prompt())
        return prompt, "all_principles_few_shot"

    def generate_system_message(self):
        prompt = (self.prompt_builder.add_persona_modelling()
                  .add_context()
                  .add_iterative_approach()
                  .add_q_and_a_prompting()
                  .build_prompt())
        return prompt

    # def run_experiment(self, prompt_choice_list):
    #     self.dialogue.prompt_assistant_response(prompt_choice_list)

    def run(self):
        # self.run_experiment(self.experiment_base_prompt)
        # self.run_experiment(self.experiment_all_principles_zero_shot)
        # self.run_experiment(self.experiment_all_principles_few_shot)

        prompt, filename = self.experiment_all_principles_few_shot()
        self.dialogue.prompt_assistant_response(prompt, filename)
        #
        # prompt, filename = self.experiment_all_principles_zero_shot()
        # self.dialogue.prompt_assistant_response(prompt, filename)

        # prompt, filename = self.experiment_base_prompt()
        # self.dialogue.prompt_assistant_response(prompt, filename)

        while True:
            self.dialogue.add_dynamic_prompting(filename)


if __name__ == '__main__':
    experiment_2 = Experiment2()
    experiment_2.run()
