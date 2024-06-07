import os
from pathlib import Path

from experiments.function_calling.experiment_function_calling_on_plain_text import ExperimentFunctionCallingOnPlainText
from experiments.function_calling.experiment_person_comparison_via_llm import ExperimentPersonComparisonViaLlm
from function_calling_components.chat_completion_blueprint import DialogueCompletion


class ExperimentFunctionCallingPipeline:
    def __init__(self):
        self.dialogue = DialogueCompletion(model='gpt-3.5-turbo')

    def run(self):
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        file_path = os.path.join(base_dir, "test_data", "test_txt", "sample_text.txt")
        experiment_1 = ExperimentFunctionCallingOnPlainText(file_path)
        response = experiment_1.run()
        experiment_2 = ExperimentPersonComparisonViaLlm(response)
        experiment_2.run()


if __name__ == '__main__':
    experiment = ExperimentFunctionCallingPipeline()
    experiment.run()
    