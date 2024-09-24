import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from utils.file_reader_util import read_json


class GeneralStructuredOutputRetrival:

    def __init__(self, evaluation_result_path: str, prompts: List[str], model: str):
        self.evaluation_result_path: str = evaluation_result_path
        self.prompts: List[str] = prompts
        self.model: str = model

    def get_number_of_valid_files_per_prompt(self) -> Dict[str, int]:
        valid_files_per_prompt = {}
        for prompt in self.prompts:
            valid_pred_files = self._get_valid_prediction_file_paths(prompt)
            count = len(valid_pred_files)
            valid_files_per_prompt[prompt] = count
        return valid_files_per_prompt

    def get_comparable_scores(self) -> Dict[str, Dict]:
        comparable_pred_files = self._get_comparable_prediction_files()
        comparable_scores = {}
        for prompt in self.prompts:
            comparable_scores[prompt] = {}
            for pred_file in comparable_pred_files:
                score = self._get_scores(prompt=prompt, pred_file=pred_file)
                comparable_scores[prompt][pred_file] = score
        return comparable_scores

    def calc_mean_scores(self, comparable_scores: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        mean_scores = dict()
        for prompt in comparable_scores:
            number_of_pred_files = len(comparable_scores[prompt].keys())
            sum_exact = 0
            sum_fuzzy = 0
            for pred_file in comparable_scores[prompt]:
                exact_score = comparable_scores[prompt][pred_file]["exact_score"]
                fuzzy_score = comparable_scores[prompt][pred_file]["fuzzy_score"]
                sum_exact += exact_score
                sum_fuzzy += fuzzy_score
            mean_exact = sum_exact / number_of_pred_files
            mean_fuzzy = sum_fuzzy / number_of_pred_files
            mean_scores[prompt] = {
                "mean_exact": mean_exact,
                "mean_fuzzy": mean_fuzzy
            }
        return mean_scores



    def _get_comparable_prediction_files(self) -> List[str]:
        comparable_pred_file_sets = list()
        for prompt in self.prompts:
            valid_pred_file_paths = self._get_valid_prediction_file_paths(prompt)
            valid_pred_files = [os.path.basename(x) for x in valid_pred_file_paths]
            valid_pred_files_as_set = set(valid_pred_files)
            comparable_pred_file_sets.append(valid_pred_files_as_set)
        intersection = set.intersection(*comparable_pred_file_sets)
        return intersection

    def _get_scores(self, prompt, pred_file) -> Dict:
        scores_for_prompt = self._get_scores_for_prompt(prompt)
        individual_results = scores_for_prompt["individual_results"]
        filtered_scores = [result for result in individual_results if result["file_name2"] == pred_file and result["exclude_paths"] == ["root['id']"]]
        filtered_score: Dict = filtered_scores[0]

        filtered_score.pop("exclude_paths")
        deep_diff_comparison_results_fuzzy = "deep_diff_comparison_results_fuzzy"
        deepdiff_comparison_results = "deepdiff_comparison_results"
        if deepdiff_comparison_results in filtered_score:
            filtered_score.pop(deepdiff_comparison_results)
        if deep_diff_comparison_results_fuzzy in filtered_score:
            filtered_score.pop(deep_diff_comparison_results_fuzzy)

        return filtered_score

    def _get_scores_for_prompt(self, prompt):
        file = os.path.join(self.evaluation_result_path, "results", prompt, self.model, "scores_json", "results.json")
        scores = read_json(file)
        return scores

    def _get_valid_prediction_file_paths(self, prompt) -> List:
        pred_files = self._get_prediction_file_paths(prompt)
        valid_pred_files = [file for file in pred_files if self._is_prediction_file_valid(file)]
        return valid_pred_files

    def _get_prediction_file_paths(self, prompt: str) -> List[str]:
        directory = os.path.join(self.evaluation_result_path, "results", prompt, self.model, "predictions")
        file_paths = list()
        for filename in os.listdir(directory):
            file_paths.append(os.path.join(directory, filename))
        return file_paths

    def _is_prediction_file_valid(self, pred_file_path) -> bool:
        with open(pred_file_path, "r") as f:
            content = f.read()
            if content.lower().startswith("null"):
                return False
        json_content: Dict = read_json(pred_file_path)
        if json_content:
            if "person_list" in json_content:
                return True
        return False

    @staticmethod
    def plot_successful_json_processings(successes_dict: Dict[str, int], y_max_scale: int = None) -> None:
        x = np.arange(len(successes_dict))
        bar_width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x, list(successes_dict.values()), bar_width, zorder=3)
        ax.set_xlabel('JSON Generation Mode')
        ax.set_ylabel('Successful JSON Processing Count')
        # ax.set_title(f'Runtime per model for {prompt_name}', pad=20)
        ax.set_xticks(x)

        labels = ['json mode\nstructured\noutput', 'tool calling\nno structured\noutput', 'tool calling\nstructured\noutput' ]
        ax.set_xticklabels(labels, rotation=45, ha='right')

        plt.tight_layout()
        if y_max_scale is not None:
            plt.ylim(0, y_max_scale)
        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)

        output_dir = os.path.join('successful_json_processings')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'successful_json_processings.png'), bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_mean_accuracy_for_jsons(accuracy_dict: Dict[str, Dict[str, float]]):
        x = np.arange(len(accuracy_dict.keys()))
        bar_width = 0.35
        fig, ax = plt.subplots()
        mean_exact = [mean_accuracy["mean_exact"] for mean_accuracy in accuracy_dict.values()]
        mean_fuzzy = [mean_accuracy["mean_fuzzy"] for mean_accuracy in accuracy_dict.values()]
        ax.bar(x - bar_width / 2, mean_exact, bar_width, label='Exact Accuracy', zorder=3)
        ax.bar(x + bar_width / 2, mean_fuzzy, bar_width, label='Fuzzy Accuracy', zorder=3)
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        # ax.set_title(f'Exact vs Fuzzy Accuracy by Model for {prompt_name}', pad=20)
        ax.set_xticks(x)
        labels_x_axis = ['json mode\nstructured\noutput', 'tool calling\nno structured\noutput', 'tool calling\nstructured\noutput' ]

        ax.set_xticklabels(labels_x_axis, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
        plt.ylim(0.0, 1.0)
        output_dir_processing_methods = os.path.join('mean_accuracy_different_processing_methods')
        if not os.path.exists(output_dir_processing_methods):
            os.makedirs(output_dir_processing_methods, exist_ok=True)
        plt.savefig(os.path.join(output_dir_processing_methods, f'overall_scores_different_processing_methods.png'),
                    bbox_inches='tight')


def main():
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    evaluation_results_dir = os.path.join(base_dir, "evaluation_results")
    gsor = GeneralStructuredOutputRetrival(
        model="model_gpt-4o-2024-08-06",
        evaluation_result_path=evaluation_results_dir,
        prompts=[
            "best_prompt_no_function_calling",
            "best_prompt_structured_outputs_disabled",
            "best_prompt_structured_outputs_enabled"
        ])

    print("\n\nThese are the scores which can be plotted against one another:")
    comparable_scores = gsor.get_comparable_scores()
    print(json.dumps(comparable_scores, indent=4))

    print("\n\nThese are the mean scores:")
    mean_scores = gsor.calc_mean_scores(comparable_scores)
    print(json.dumps(mean_scores, indent=4))

    print("\n\nThese are the number of valid json files per prompt:")
    valid_files_per_prompt = gsor.get_number_of_valid_files_per_prompt()
    print(json.dumps(valid_files_per_prompt, indent=4))

    gsor.plot_successful_json_processings(valid_files_per_prompt)
    gsor.plot_mean_accuracy_for_jsons(mean_scores)


if __name__ == "__main__":
    main()
