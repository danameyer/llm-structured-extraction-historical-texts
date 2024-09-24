import json
import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

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
        return filtered_scores

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
            person_list: list = json_content.get("person_list")
            if person_list:
                return True
        return False


def main():
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    evaluation_results_dir = os.path.join(base_dir, "evaluation_results")
    gsor = GeneralStructuredOutputRetrival(
        model="model_gpt-4o-2024-08-06",
        evaluation_result_path=evaluation_results_dir,
        prompts=[
            "best_prompt_no_function_calling",

            # nur zum spass, damit man was zum Vergleichen hat ...
            "all_principles_few_shot"
        ])

    print("\n\nThese are the scores which can be plotted against one another:")
    comparable_scores = gsor.get_comparable_scores()
    print(json.dumps(comparable_scores, indent=4))

    print("\n\nThese are the number of valid json files per prompt:")
    valid_files_per_prompt = gsor.get_number_of_valid_files_per_prompt()
    print(json.dumps(valid_files_per_prompt, indent=4))


if __name__ == "__main__":
    main()
