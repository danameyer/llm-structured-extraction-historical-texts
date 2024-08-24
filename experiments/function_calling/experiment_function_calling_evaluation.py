import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from jsons import ValidationError

from evaluation.json_comparison.json_comparison import JsonComparison
from experiments.function_calling.function_calling_all_principles_few_shot import ExperimentAllPrinciplesFewShotPrompt
from experiments.function_calling.function_calling_all_principles_zero_shot import ExperimentAllPrinciplesZeroShotPrompt
from experiments.function_calling.function_calling_no_principles_base_prompt import \
    ExperimentFunctionCallingNoPrinciplesBasePrompt
from experiments.function_calling.function_calling_with_optimised_prompt import \
    ExperimentFunctionCallingWithOptimisedPrompt
from function_calling_components.chat_completion_blueprint import DialogueCompletion
from function_calling_components.chat_file_writer import ChatFileWriter


class ExperimentFunctionCallingEvaluation:

    def __init__(self,
                 prompt_experiment_name: str,
                 experiment_dir,
                 gpt_model_name='gpt-3.5-turbo',
                 regenerate_predictions=False):
        self.experiment_dir = experiment_dir
        self.prompt_experiment_name = prompt_experiment_name
        self.model_name = gpt_model_name
        self.regenerate_predictions = regenerate_predictions
        self.dialogue = DialogueCompletion(model=gpt_model_name, experiment_dir=experiment_dir)
        self.failed_files = []

    def init_prompt_experiment(self, demonstrations_files, test_files, gpt_model, pred_file_name):
        if self.prompt_experiment_name == "chain_of_thought":
            return ExperimentFunctionCallingWithOptimisedPrompt(test_files,
                                                                demonstrations_files,
                                                                gpt_model,
                                                                self.experiment_dir,
                                                                pred_file_name)
        elif self.prompt_experiment_name == "all_principles_zero_shot":
            return ExperimentAllPrinciplesZeroShotPrompt(test_files,
                                                         demonstrations_files,
                                                         gpt_model,
                                                         self.experiment_dir,
                                                         pred_file_name)
        elif self.prompt_experiment_name == "all_principles_few_shot":
            return ExperimentAllPrinciplesFewShotPrompt(test_files,
                                                        demonstrations_files,
                                                        gpt_model,
                                                        self.experiment_dir,
                                                        pred_file_name)
        elif self.prompt_experiment_name == "no_principles_base_prompt":
            return ExperimentFunctionCallingNoPrinciplesBasePrompt(test_files,
                                                                   demonstrations_files,
                                                                   gpt_model,
                                                                   self.experiment_dir,
                                                                   pred_file_name)
        elif self.prompt_experiment_name == "best_prompt":
            return ExperimentFunctionCallingWithOptimisedPrompt(test_files,
                                                                demonstrations_files,
                                                                gpt_model,
                                                                self.experiment_dir,
                                                                pred_file_name)
        else:
            raise ValueError("Wrong prompt name: " + self.prompt_experiment_name)

    def get_base_directory(self):
        return Path(os.getenv('PROJECT_BASE_DIR'))

    def get_ground_truth_folder(self, base_dir):
        return os.path.join(base_dir, "evaluation_results", "ground_truth")

    def get_demonstrations_folder(self, base_dir):
        return os.path.join(base_dir, "test_data", "demonstrations")

    def get_sample_folder(self, base_dir):
        return os.path.join(base_dir, "evaluation_results", "text_files")

    def create_predictions_folder(self, base_dir):
        predictions_folder = os.path.join(base_dir, "predictions")
        os.makedirs(predictions_folder, exist_ok=True)
        return predictions_folder

    def create_scores_txt_folder(self, base_dir):
        output_folder = os.path.join(base_dir, "scores_txt")
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def create_scores_json_folder(self, base_dir):
        output_folder = os.path.join(base_dir, "scores_json")
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    def create_costs_folder(self, experiment_dir):
        costs_folder = os.path.join(experiment_dir, "costs")
        os.makedirs(costs_folder, exist_ok=True)
        return costs_folder

    def create_logs_folder(self, experiment_dir):
        logs_folder = os.path.join(experiment_dir, "logs_failed_files")
        os.makedirs(logs_folder, exist_ok=True)
        return logs_folder

    def log_failed_files(self, experiment_dir):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f'failed_files_{timestamp}.txt'
        logs_directory = self.create_logs_folder(experiment_dir)
        log_file_path = os.path.join(logs_directory, log_filename)

        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Failed files count: {len(self.failed_files)}\n\n")
            for failed_file in self.failed_files:
                log_file.write(f"{failed_file}\n")

    def process_files(self, text_files_folder, demonstrations_folder, predictions_folder, gpt_model):
        for file_name in os.listdir(text_files_folder):

            path_to_text_file = os.path.join(text_files_folder, file_name)
            if os.path.isfile(path_to_text_file):
                try:
                    self.process_single_file(path_to_text_file, demonstrations_folder, predictions_folder, gpt_model)
                except ValidationError as e:
                    print(f"""
                    Could not complete file {path_to_text_file} because of error: 
                    {e.message}
                    
                    Will continue with next file.
                    """, file=sys.stderr)
                    self.failed_files.append(path_to_text_file)

        if self.failed_files:
            self.log_failed_files(self.experiment_dir)

    def process_single_file(self, path_to_text_file, demonstrations_folder, predictions_folder, gpt_model):
        base_name = os.path.splitext(os.path.basename(path_to_text_file))[0]
        pred_filename = f'pred_{base_name}.json'
        output_path_response = os.path.join(predictions_folder, pred_filename)

        if not self.regenerate_predictions and os.path.exists(output_path_response):
            print(f"Prediction for {base_name} already exists. Skipping regeneration.")
            return

        test_files = [path_to_text_file]
        demonstrations = [os.path.join(demonstrations_folder, "demonstration_1.txt")]
        experiment_function_calling = self.init_prompt_experiment(demonstrations, test_files, gpt_model=gpt_model,
                                                                  pred_file_name=base_name)
        response_json = experiment_function_calling.run()
        response_json_str = json.dumps(response_json, indent=4)

        chat_file_writer = ChatFileWriter(self.experiment_dir)
        chat_file_writer.save_response(response_json_str, output_path_response, timestamp=False, append=False)
        print(f"Processed {os.path.basename(path_to_text_file)}: Saved predictions to {pred_filename}")

        cost_summary = experiment_function_calling.dialogue.token_counter.get_cost_summary(pred_filename)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = "cost_summary_" + pred_filename + "_" + timestamp + ".json"
        costs_directory = self.create_costs_folder(self.experiment_dir)
        self.save_costs_to_file(cost_summary, costs_directory, filename)

    def save_costs_to_file(self, costs, directory: str, filename):
        costs_as_string = json.dumps(costs)
        file_path = os.path.join(directory, filename)
        with open(file_path, 'w') as f:
            f.write(costs_as_string)

    def run(self, gpt_model, exclusions: List[List[str]]):
        base_dir = self.get_base_directory()
        sample_folder = self.get_sample_folder(base_dir)
        ground_truth_folder = self.get_ground_truth_folder(base_dir)
        demonstrations_folder = self.get_demonstrations_folder(base_dir)
        predictions_folder = self.create_predictions_folder(self.experiment_dir)
        scores_txt_folder = self.create_scores_txt_folder(self.experiment_dir)
        scores_json_folder = self.create_scores_json_folder(self.experiment_dir)
        self.process_files(sample_folder, demonstrations_folder, predictions_folder, gpt_model=gpt_model)
        json_comparison = JsonComparison()
        json_comparison.perform_json_comparison(ground_truth_folder,
                                                predictions_folder,
                                                scores_txt_folder,
                                                scores_json_folder,
                                                exclusions)


def _main():
    model_name = 'gpt-4o-mini'
    prompt_name = 'best_prompt'
    regenerate_predictions = False
    exclusions = [["root['id']"],
                  ["root['id']", "root['cognomen']"],
                  ["root['id']", "root['legal_relationship']"],
                  ["root['id']", "root['place_of_origin']"],
                  ["root['id']", "root['family_relations']"],
                  ["root['id']", "root['title']"],
                  ["root['id']", "root['profession']"]
                  ]

    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    experiment_folder = os.path.join(base_dir, "evaluation_results", "results", prompt_name,
                                     "model_" + model_name)
    os.makedirs(experiment_folder, exist_ok=True)

    experiment = ExperimentFunctionCallingEvaluation(prompt_experiment_name=prompt_name,
                                                     gpt_model_name=model_name,
                                                     regenerate_predictions=regenerate_predictions,
                                                     experiment_dir=experiment_folder)
    experiment.run(model_name, exclusions=exclusions)


if __name__ == '__main__':
    _main()
