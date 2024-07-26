import json
import os
import sys
from datetime import datetime
from pathlib import Path

from jsons import ValidationError

from evaluation.json_comparison.json_comparison import JsonComparison
from experiments.function_calling.function_calling_with_optimised_prompt import \
    ExperimentFunctionCallingWithOptimisedPrompt
from function_calling_components.chat_completion_blueprint import DialogueCompletion
from function_calling_components.chat_file_writer import ChatFileWriter


class ExperimentFunctionCallingEvaluation:

    def __init__(self, gpt_model_name='gpt-3.5-turbo'):
        self.model_name = model_name
        self.dialogue = DialogueCompletion(model=gpt_model_name)

    def get_base_directory(self):
        return Path(os.getenv('PROJECT_BASE_DIR'))

    def get_ground_truth_folder(self, base_dir):
        return os.path.join(base_dir, "test_data", "test_json_diff", "ground_truth")

    def get_demonstrations_folder(self, base_dir):
        return os.path.join(base_dir, "test_data", "demonstrations")

    def get_sample_folder(self, base_dir):
        return os.path.join(base_dir, "test_data", "test_txt")

    def create_predictions_folder(self, base_dir):
        model_predictions_folder = os.path.join(base_dir, "test_data", "test_json_diff", "predictions", self.model_name)
        os.makedirs(model_predictions_folder, exist_ok=True)
        return model_predictions_folder

    def create_output_folder(self, base_dir):
        output_folder = os.path.join(base_dir, "test_data", "test_json_diff", "scores")
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def create_costs_folder(self, base_dir):
        costs_folder = os.path.join(base_dir, "prompting", "costs")
        os.makedirs(costs_folder, exist_ok=True)
        return costs_folder

    def process_files(self, ground_truth_folder, demonstrations_folder, predictions_folder):
        for file_name in os.listdir(ground_truth_folder):

            file_path = os.path.join(ground_truth_folder, file_name)
            if os.path.isfile(file_path):
                try:
                    self.process_single_file(file_path, demonstrations_folder, predictions_folder)
                except ValidationError as e:
                    print(f"""
                    Could not complete file {file_path} because of error: 
                    {e.message}
                    
                    Will continue with next file.
                    """, file=sys.stderr)


    def process_single_file(self, file_path, demonstrations_folder, predictions_folder):
        test_files = [file_path]
        demonstrations = [os.path.join(demonstrations_folder, "demonstration_1.txt")]
        experiment_function_calling = ExperimentFunctionCallingWithOptimisedPrompt(test_files, demonstrations)
        response_json = experiment_function_calling.run()
        response_json_str = json.dumps(response_json, indent=4)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        pred_filename = f'pred_{base_name}.json'
        output_path_response = os.path.join(predictions_folder, pred_filename)
        chat_file_writer = ChatFileWriter()
        chat_file_writer.save_response(response_json_str, output_path_response, timestamp=False, append=False)
        print(f"Processed {os.path.basename(file_path)}: Saved predictions to {pred_filename}")

        cost_summary = experiment_function_calling.dialogue.token_counter.get_cost_summary(pred_filename)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = "cost_summary_" + pred_filename + "_" + timestamp + ".json"
        costs_directory = self.create_costs_folder(self.get_base_directory())
        self.save_costs_to_file(cost_summary, costs_directory, filename)

    def save_costs_to_file(self, costs, directory: str, filename):
        costs_as_string = json.dumps(costs)
        file_path = os.path.join(directory, filename)
        with open(file_path, 'w') as f:
            f.write(costs_as_string)

    def run(self):
        base_dir = self.get_base_directory()
        sample_folder = self.get_sample_folder(base_dir)
        ground_truth_folder = self.get_ground_truth_folder(base_dir)
        demonstrations_folder = self.get_demonstrations_folder(base_dir)
        predictions_folder = self.create_predictions_folder(base_dir)
        output_folder = self.create_output_folder(base_dir)
        self.process_files(sample_folder, demonstrations_folder, predictions_folder)
        json_comparison = JsonComparison()
        json_comparison.perform_json_comparison(ground_truth_folder, predictions_folder, output_folder)


if __name__ == '__main__':
    model_name = 'gpt-3.5-turbo'
    experiment = ExperimentFunctionCallingEvaluation(gpt_model_name=model_name)
    experiment.run()
