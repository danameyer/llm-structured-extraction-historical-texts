import os
from pathlib import Path

from dotenv import load_dotenv

from function_calling_components.chat_completion_blueprint import DialogueCompletion
from prompting.prompting_strategies import PromptBuilder


class BaseExperimentPrompting:
    def __init__(self, experiment_dir):
        load_dotenv()
        self.experiment_dir = experiment_dir
        self.prompt_builder = PromptBuilder()
        self.dialogue = DialogueCompletion(model='gpt-3.5-turbo', experiment_dir=experiment_dir)
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        self.test_files = os.path.join(base_dir, "test_data", "test_txt", "sample_text.txt")
        self.demonstrations = os.path.join(base_dir, "test_data", "demonstrations", "demonstration_1.txt")

    def run(self):
        pass
