import os
from pathlib import Path

from dotenv import load_dotenv

from function_calling_components.chat_completion_blueprint import DialogueCompletion
from prompting.prompting_strategies import PromptBuilder


class BaseExperimentFunctionCalling:
    def __init__(self, test_files, demonstrations, experiment_dir, pred_file_name, gpt_model):
        load_dotenv()
        self.experiment_dir = experiment_dir
        self.prompt_builder = PromptBuilder()
        self.dialogue = DialogueCompletion(model=gpt_model, experiment_dir=experiment_dir)
        self.test_files = test_files
        self.demonstrations = demonstrations
        self.pred_file_name = pred_file_name
        self.gpt_model = gpt_model

    def run(self):
        pass
