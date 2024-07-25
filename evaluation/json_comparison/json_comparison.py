import json
import os
from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any
from deepdiff import DeepDiff
from dotenv import load_dotenv
from thefuzz import fuzz
from data_classes.document import create_documents
from data_classes.person import Person
from evaluation.json_comparison.PersonMatching import PersonMatching
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
                             person_list_1: List[Person],
                             person_list_2: List[Person]) -> PersonMatching:
        evaluator = JsonEditDistanceEvaluator()
        matches = []
        not_found = []

        for person_a in person_list_1:
            name_a = person_a.name
            matching_person = None
            last_matching_score = 1.0
            for person_b in person_list_2:
                name_b = person_b.name
                if self.fuzzy_compare(name_a, name_b):
                    person_a_as_dict = person_a.to_dict()
                    person_b_as_dict = person_b.to_dict()
                    person_a_as_string = json.dumps(person_a_as_dict)
                    person_b_as_string = json.dumps(person_b_as_dict)
                    result = evaluator.evaluate_strings(prediction=person_a_as_string, reference=person_b_as_string)
                    score = result['score']
                    if score < last_matching_score:
                        matching_person = person_b
                        last_matching_score = score

            if matching_person is None:
                not_found.append(person_a)
            else:
                matches.append((person_a, matching_person))

        print("This is the list of persons not found", not_found)
        return PersonMatching(matches=matches, not_found=not_found)

    def get_deep_diff(self, person_1: Person, person_2: Person):
        person_1_dict = person_1.__dict__
        person_2_dict = person_2.__dict__
        ddiff = DeepDiff(person_1_dict, person_2_dict,
                         ignore_order=True,
                         verbose_level=2,
                         exclude_paths=["root['id']"])
        ddiff_as_json = ddiff.to_json()
        print(ddiff_as_json)
        return ddiff_as_json

    def count_fields(self, obj):
        if is_dataclass(obj):
            obj = asdict(obj)

        if isinstance(obj, dict):
            return sum(self.count_fields(v) for v in obj.values()) + len(obj)
        elif isinstance(obj, list):
            return sum(self.count_fields(item) for item in obj)
        else:
            return 0

    def calculate_metric(self, person_matching: PersonMatching, apply_fuzzy=False):
        if not apply_fuzzy:
            metric_values_changed, metric_items_change_count, number_fields_total, metric_type_changes, deep_diff_results = self.calculate_diff_metrics(
                person_matching)
        else:
            metric_values_changed, metric_items_change_count, number_fields_total, metric_type_changes, deep_diff_results = (
                self.calculate_diff_metrics(
                    person_matching,
                    apply_fuzzy=True))
        metric_persons_added_or_removed = self.calculate_not_found_metric(person_matching)

        number_fields_total += metric_persons_added_or_removed

        print("Number of persons not found:", metric_persons_added_or_removed)

        # Calculate total differences
        all_diffs = (metric_values_changed
                     + metric_persons_added_or_removed
                     + metric_items_change_count
                     + metric_type_changes)
        matches = number_fields_total - all_diffs

        accuracy_score = self.calculate_accuracy_score(matches, number_fields_total, all_diffs)

        print("Number of fields:", number_fields_total)
        print("Number of matches:", matches)
        print("Number of differences:", all_diffs)

        return accuracy_score, matches, all_diffs, deep_diff_results, number_fields_total

    def calculate_diff_metrics(self,
                               person_matching: PersonMatching,
                               apply_fuzzy=False) -> Tuple[int, int, int, int, List[Dict[str, Any]]]:
        metric_values_changed = 0
        metric_items_change_count = 0
        metric_type_changes = 0
        number_fields_total = 0
        deepdiff_results = []

        for match in person_matching.matches:
            person_1, person_2 = match
            ddiff_as_json = self.get_deep_diff(person_1, person_2)
            ddiff_as_dict = json.loads(ddiff_as_json)
            number_fields_total += self.count_fields(person_1.__dict__)

            # Values changed:
            values_changed: Dict = ddiff_as_dict.get("values_changed", {})

            if apply_fuzzy:
                fuzzy_diff_values, value_change_count = self.apply_fuzzy_compare(values_changed)
                print("This is the number of values changed:", value_change_count)
            else:
                value_change_count = len(values_changed)
                print("This is the number of values changed:", value_change_count)
            metric_values_changed += value_change_count

            # Items added or removed
            iterable_item_added = ddiff_as_dict.get("iterable_item_added", {})
            iterable_item_removed = ddiff_as_dict.get("iterable_item_removed", {})
            metric_items_change_count += len(iterable_item_added) + len(iterable_item_removed)

            # type changes
            type_changes: Dict = ddiff_as_dict.get("type_changes", {})
            metric_type_changes += len(type_changes)

            deepdiff_results.append({
                'name1': person_1.name,
                'name2': person_2.name,
                'deepdiff': ddiff_as_json,
                'values_changed': len(values_changed),
                'iterable_item_added': len(iterable_item_added),
                'iterable_item_removed': len(iterable_item_removed),
                'type_changes': len(type_changes)
            })

        return metric_values_changed, metric_items_change_count, number_fields_total, metric_type_changes, deepdiff_results

    def calculate_not_found_metric(self, person_matching: PersonMatching) -> int:
        not_found_counts = [self.count_fields(person.__dict__) for person in person_matching.not_found]
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

    def apply_fuzzy_compare(self, values_changed: Dict, threshold=80) -> Tuple[List[Dict], int]:
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

            gt_documents = create_documents(gt_folder)
            pred_documents = create_documents(prediction_folder)

            gt_dict = {doc.document_name: doc for doc in gt_documents}
            pred_dict = {doc.document_name: doc for doc in pred_documents}

            for gt_filename, gt_document in gt_dict.items():
                base_name = os.path.splitext(gt_filename)[0]
                pred_filename = f'pred_{base_name}.json'

                if pred_filename in pred_dict:
                    pred_document = pred_dict[pred_filename]
                    comparison_results = []

                    matching_person_object = self.find_matching_person(gt_document.persons, pred_document.persons)
                    exact_score, exact_matches, exact_diffs, deep_diff_results, total_field_count = self.calculate_metric(matching_person_object)

                    fuzzy_score, fuzzy_matches, fuzzy_diffs, deep_diff_results_fuzzy, total_field_count = self.calculate_metric(matching_person_object, apply_fuzzy=True)

                    comparison_results.append({
                        'file_name1': gt_filename,
                        'file_name2': pred_document.document_name,
                        'deepdiff_comparison_results': deep_diff_results,
                        'exact_score': exact_score,
                        'exact_matches': exact_matches,
                        'exact_differences': exact_diffs,
                        'deep_diff_comparison_results_fuzzy': deep_diff_results_fuzzy,
                        'fuzzy_score': fuzzy_score,
                        'fuzzy_matches': fuzzy_matches,
                        'fuzzy_differences': fuzzy_diffs,
                        'total_field_count': total_field_count
                    })

                    # Write formatted results to the file
                    formatted_result = self.format_comparison(comparison_results)
                    result_file.write(formatted_result + '\n')

                else:
                    result_file.write(f"File: {gt_filename} - JSON file with predicted results not found.\n\n")
                    result_file.write("-" * 60 + "\n")

    def format_comparison(self, comparison_results):
        """
        Formats the comparison results into a structured string.
        """
        output = []

        for result in comparison_results:
            # Header for the files being compared
            output.append(f"------------------------------------------------------------")
            output.append(f"FILE: {result['file_name1']} & {result['file_name2']}")
            output.append(f"------------------------------------------------------------\n")

            # Details of each deepdiff comparison
            for comparison in result['deepdiff_comparison_results']:
                name1, name2, deepdiff = comparison['name1'], comparison['name2'], comparison['deepdiff']
                output.append(f"Comparing: {name1} with {name2}")
                output.append(f"- DeepDiff: {deepdiff}")
                output.append(f"- Values Changed: {comparison['values_changed']}")
                output.append(f"- Items Added: {comparison['iterable_item_added']}")
                output.append(f"- Items Removed: {comparison['iterable_item_removed']}")
                output.append(f"- Type Changes: {comparison['type_changes']}\n")

            # Total Field Count
            output.append(f"**Total Field Count: {result['total_field_count']}**\n")

            # Exact Metric
            output.append(f"**Exact Metric: {result['exact_score']}**")
            output.append(f"- Matches: {result['exact_matches']}")
            output.append(f"- Differences: {result['exact_differences']}\n")

            # Fuzzy Metric
            output.append(f"**Fuzzy Metric: {result['fuzzy_score']}**")
            output.append(f"- Matches: {result['fuzzy_matches']}")
            output.append(f"- Differences: {result['fuzzy_differences']}\n")

            output.append(f"============================================================")

        return "\n".join(output)


if __name__ == '__main__':
    json_comparison = JsonComparison()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    gt_path = os.path.join(base_dir, "test_data", "test_json_diff", "ground_truth")
    pred_path = os.path.join(base_dir, "test_data", "test_json_diff", "predictions")
    output_path = os.path.join(base_dir, "test_data", "test_json_diff", "scores")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    json_comparison.perform_json_comparison(gt_path, pred_path, output_path)
