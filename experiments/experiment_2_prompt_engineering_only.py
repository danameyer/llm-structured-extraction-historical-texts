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
        self.prompt_save_dir = os.path.join(base_dir, "prompting", "generated_prompts")
        os.makedirs(self.prompt_save_dir, exist_ok=True)

    def save_prompt_to_file(self, prompt, base_filename):
        file_path = os.path.join(self.prompt_save_dir, base_filename)
        if os.path.exists(file_path):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base, ext = os.path.splitext(base_filename)
            filename = f"{base}_{timestamp}{ext}"
            file_path = os.path.join(self.prompt_save_dir, filename)
        with open(file_path, 'w') as file:
            file.write(prompt)

    def experiment_base_prompt(self):
        prompt = self.prompt_builder.add_base_prompt([self.test_files]).build_prompt()
        self.dialogue.append_message(Message("user", prompt))
        self.save_prompt_to_file(prompt, "base_prompt.txt")

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
        self.dialogue.append_message(Message("user", prompt))
        self.save_prompt_to_file(prompt, "all_principles_zero_shot.txt")

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
        self.dialogue.append_message(Message("user", prompt))
        self.save_prompt_to_file(prompt, "all_principles_few_shot.txt")

    def run(self):
        self.experiment_base_prompt()
        # self.experiment_all_principles_zero_shot()
        # self.experiment_all_principles_few_shot()
        chat_response = self.dialogue.execute_chat_completion_query(
            messages=self.dialogue.message_history
        )
        assistant_message = chat_response.choices[0].message.content
        self.dialogue.append_message(Message("assistant", assistant_message))
        self.dialogue.print_conversation()


if __name__ == '__main__':
    experiment_2 = Experiment2()
    experiment_2.run()
