import os
from pathlib import Path

from experiments.function_calling.experiment_function_calling_on_plain_text import ExperimentFunctionCallingOnPlainText
from experiments.function_calling.experiment_person_comparison_via_llm import ExperimentPersonComparisonViaLlm
from function_calling_components.chat_completion_blueprint import DialogueCompletion


class ExperimentFunctionCallingPipeline:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.dialogue = DialogueCompletion(model='gpt-3.5-turbo', experiment_dir=experiment_dir)

    def run(self):
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        file_path = os.path.join(base_dir, "test_data", "test_txt", "sample_text.txt")
        experiment_1 = ExperimentFunctionCallingOnPlainText(file_path, self.experiment_dir)
        response = experiment_1.run()
        experiment_2 = ExperimentPersonComparisonViaLlm(response, self.experiment_dir)
        experiment_2.run()


if __name__ == '__main__':
    experiment = ExperimentFunctionCallingPipeline("/tmp/experiments/function_calling_pipeline")
    experiment.run()
    