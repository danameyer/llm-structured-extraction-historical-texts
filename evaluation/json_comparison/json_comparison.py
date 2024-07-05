import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from deepdiff import DeepDiff
from deepdiff.helper import CannotCompare
from dotenv import load_dotenv
from thefuzz import fuzz

from evaluation.json_comparison.PersonMatching import PersonMatching
from utils import file_reader_util

from langchain.evaluation import JsonEditDistanceEvaluator


class JsonComparison:

    def __init__(self):
        load_dotenv()

    def sort_list_of_dicts_by_keys(self, lst: List[Dict]) -> List[Dict]:
        """
        Sorts a list of dictionaries by their keys, where each dictionary's keys are sorted alphabetically.

        Parameters:
        lst (List[Dict]): The list of dictionaries to be sorted.

        Returns:
        List[Dict]: A new list of dictionaries with keys sorted alphabetically in each dictionary.
        """
        sorted_list = []
        for d in lst:
            sorted_dict = {key: d[key] for key in sorted(d)}
            sorted_list.append(sorted_dict)

        sorted_list.sort(key=lambda d: list(d.keys()))

        return sorted_list

    def find_matching_person(self,
                             person_list_1: List[Dict],
                             person_list_2: List[Dict]) -> PersonMatching:
        evaluator = JsonEditDistanceEvaluator()
        person_list_1_sorted = self.sort_list_of_dicts_by_keys(person_list_1)
        person_list_2_sorted = self.sort_list_of_dicts_by_keys(person_list_2)
        person_list_1_as_strings = [json.dumps(person) for person in person_list_1_sorted]
        person_list_2_as_strings = [json.dumps(person) for person in person_list_2_sorted]

        matches = list()
        not_found = list()

        for person_a_as_string in person_list_1_as_strings:
            person_a = json.loads(person_a_as_string)
            name_a = person_a.get('name', '')

            matching_person = None
            last_matching_score = 1.0
            for person_b_as_string in person_list_2_as_strings:
                person_b = json.loads(person_b_as_string)
                name_b = person_b.get('name', '')

                if self.fuzzy_compare(name_a, name_b):
                    result = evaluator.evaluate_strings(prediction=person_a_as_string, reference=person_b_as_string)
                    score = result['score']
                    if score < last_matching_score:
                        matching_person = person_b_as_string
                        last_matching_score = score

            if matching_person is None:
                person_a_as_dict = json.loads(person_a_as_string)
                not_found.append(person_a_as_dict)

            else:
                person_a_as_dict = json.loads(person_a_as_string)
                matching_person_as_dict = json.loads(matching_person)
                matches.append((person_a_as_dict, matching_person_as_dict))

        print("This is the list of persons not found", not_found)

        return PersonMatching(matches=matches, not_found=not_found)

    def get_deep_diff(self, json_1, json_2):
        ddiff = DeepDiff(json_1, json_2,
                         ignore_order=True,
                         verbose_level=2,
                         exclude_paths=["root['id']"])
        ddiff_as_json = ddiff.to_json()
        print(ddiff_as_json)
        return ddiff_as_json

    def count_fields(self, obj):
        if isinstance(obj, dict):
            return sum(self.count_fields(v) for v in obj.values()) + len(obj)
        elif isinstance(obj, list):
            return sum(self.count_fields(item) for item in obj)
        else:
            return 0

    def calculate_metric(self, person_matching: PersonMatching, apply_fuzzy=False):
        if not apply_fuzzy:
            metric_values_changed, metric_items_change_count, number_fields_total = self.calculate_diff_metrics(
                person_matching)
        else:
            metric_values_changed, metric_items_change_count, number_fields_total = (
                self.calculate_diff_metrics(
                    person_matching,
                    apply_fuzzy=True))
        metric_persons_added_or_removed = self.calculate_not_found_metric(person_matching)

        number_fields_total += metric_persons_added_or_removed

        print("Number of persons not found:", metric_persons_added_or_removed)

        # Calculate total differences
        all_diffs = (metric_values_changed
                     + metric_persons_added_or_removed
                     + metric_items_change_count)
        matches = number_fields_total - all_diffs

        accuracy_score = self.calculate_accuracy_score(matches, number_fields_total, all_diffs)

        print("Number of fields:", number_fields_total)
        print("Number of matches:", matches)
        print("Number of differences:", all_diffs)

        return accuracy_score

    def calculate_diff_metrics(self,
                               person_matching: PersonMatching,
                               apply_fuzzy=False) -> Tuple[int, int, int]:
        metric_values_changed = 0
        metric_items_change_count = 0
        number_fields_total = 0

        for match in person_matching.matches:
            json_1, json_2 = match
            ddiff_as_json = self.get_deep_diff(json_1, json_2)
            ddiff_as_dict = json.loads(ddiff_as_json)
            number_fields_total += self.count_fields(json_1)

            # Values changed:
            values_changed: Dict = ddiff_as_dict.get("values_changed", {})

            if apply_fuzzy:
                fuzzy_diff_values, value_change_count = self.apply_fuzzy_compare(values_changed)
            else:
                value_change_count = len(values_changed)
                print("This is the number of values changed:", value_change_count)
            metric_values_changed += value_change_count

            # Items added or removed
            iterable_item_added = ddiff_as_dict.get("iterable_item_added", {})
            iterable_item_removed = ddiff_as_dict.get("iterable_item_removed", {})
            metric_items_change_count += len(iterable_item_added) + len(iterable_item_removed)

        return metric_values_changed, metric_items_change_count, number_fields_total

    def calculate_not_found_metric(self, person_matching: PersonMatching) -> int:
        not_found_counts = [self.count_fields(entry) for entry in person_matching.not_found]
        return sum(not_found_counts)

    def calculate_accuracy_score(self, matches: int, number_fields_total: int, all_diffs: int) -> float:
        if number_fields_total == 0:
            return 1.0 if not all_diffs else 0.0
        return matches / number_fields_total

    def fuzzy_compare(self, old_value, new_value, threshold=80):
        if isinstance(old_value, str) and isinstance(new_value, str):
            score = fuzz.ratio(old_value, new_value)
            if score > threshold:
                return True
        return False

    def apply_fuzzy_compare(self, values_changed: Dict, threshold=90) -> Tuple[List[Dict], int]:
        fuzzy_diffs = []
        fuzzy_diffs_counter = 0
        for key, change in values_changed.items():
            old_value = change['old_value']
            new_value = change['new_value']
            if isinstance(old_value, str) and isinstance(new_value, str):
                if not self.fuzzy_compare(old_value, new_value, threshold):
                    fuzzy_diffs.append({
                        'key': key,
                        'old_value': old_value,
                        'new_value': new_value,
                        'similarity': fuzz.ratio(old_value, new_value)
                    })
                    fuzzy_diffs_counter += 1
        return fuzzy_diffs, fuzzy_diffs_counter

    def perform_json_comparison(self, gt_folder, prediction_folder, output_folder):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_folder, f'results_{timestamp}.txt')
        with open(output_file, 'a') as result_file:
            result_file.write(f"\nResults generated on: {datetime.now()}\n\n")

            for gt_filename in os.listdir(gt_folder):
                gt_filepath = os.path.join(gt_folder, gt_filename)
                if os.path.isfile(gt_filepath):
                    base_name = os.path.splitext(gt_filename)[0]
                    pred_filename = f'pred_{base_name}.json'
                    pred_filepath = os.path.join(prediction_folder, pred_filename)

                    if os.path.isfile(pred_filepath):
                        gt_data = file_reader_util.read_json(gt_filepath)
                        pred_data = file_reader_util.read_json(pred_filepath)

                        result_file.write(f"File: {gt_filename} & {pred_filename}\n")
                        deepdiff = self.get_deep_diff(gt_data, pred_data)
                        result_file.write(f"DeepDiff: {deepdiff}\n")
                        matching_person_object = self.find_matching_person(gt_data['person_list'],
                                                                           pred_data['person_list'])
                        exact_score = self.calculate_metric(matching_person_object)
                        result_file.write(f"Exact Metric: {exact_score}\n")
                        fuzzy_score = self.calculate_metric(matching_person_object, apply_fuzzy=True)
                        result_file.write(f"Fuzzy Metric: {fuzzy_score}\n\n")
                    else:
                        result_file.write(f"File: {gt_filename} - JSON file with predicted results not found.\n\n")


if __name__ == '__main__':
    json_comparison = JsonComparison()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    gt_path = os.path.join(base_dir, "test_data", "test_json_diff", "ground_truth")
    pred_path = os.path.join(base_dir, "test_data", "test_json_diff", "predictions")
    output_path = os.path.join(base_dir, "test_data", "test_json_diff", "scores")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    json_comparison.perform_json_comparison(gt_path, pred_path, output_path)
