import os
from pathlib import Path

from dotenv import load_dotenv

from function_calling_components.chat_completion_blueprint import DialogueCompletion
from prompting.prompting_strategies import PromptBuilder


class BaseExperimentFunctionCalling:
    def __init__(self, test_files, demonstrations, experiment_dir):
        load_dotenv()
        self.experiment_dir = experiment_dir
        self.prompt_builder = PromptBuilder()
        self.dialogue = DialogueCompletion(model='gpt-3.5-turbo', experiment_dir=experiment_dir)
        self.test_files = test_files
        self.demonstrations = demonstrations

    def run(self):
        pass
