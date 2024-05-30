import os
from pathlib import Path

from dotenv import load_dotenv

from function_calling_components.chat_completion_blueprint import DialogueCompletion
from function_calling_components.function_calling import Message
from prompting.prompting_strategies import PromptBuilder
from datetime import datetime


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

    def run_experiment(self, prompt_choice):
        self.dialogue.perform_initial_prompting(prompt_choice)

    def run(self):
        self.run_experiment(self.experiment_base_prompt)
        # self.run_experiment(self.experiment_all_principles_zero_shot)
        # self.run_experiment(self.experiment_all_principles_few_shot)


if __name__ == '__main__':
    experiment_2 = Experiment2()
    experiment_2.run()
