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

    def __init__(self, threshold=80):
        load_dotenv()
        self.threshold = threshold

    def sort_list_of_dicts_by_keys(self, lst: List[Dict]) -> List[Dict]:
        """
        Sorts a list of dictionaries by their keys, where each dictionary's keys are sorted alphabetically.

        Parameters:
        lst (List[Dict]): The list of dictionaries to be sorted.

        Returns:
        List[Dict]: A new list of dictionaries with keys sorted alphabetically in each dictionary.
        """
        # Sort the keys in each dictionary and create a new list with these sorted dictionaries
        sorted_list = []
        for d in lst:
            # Create a new dictionary with keys sorted alphabetically
            sorted_dict = {key: d[key] for key in sorted(d)}
            sorted_list.append(sorted_dict)

        # Sort the list of dictionaries based on their keys
        sorted_list.sort(key=lambda d: list(d.keys()))

        return sorted_list

    def find_matching_person(self, person_list_1: List[Dict], person_list_2: List[Dict], threshold=0.15) -> PersonMatching:
        evaluator = JsonEditDistanceEvaluator()
        person_list_1_sorted = self.sort_list_of_dicts_by_keys(person_list_1)
        person_list_2_sorted = self.sort_list_of_dicts_by_keys(person_list_2)
        person_list_1_as_strings = [json.dumps(person) for person in person_list_1_sorted]
        person_list_2_as_strings = [json.dumps(person) for person in person_list_2_sorted]

        matches = list()
        not_found = list()

        for person_a_as_string in person_list_1_as_strings:
            matching_person = None
            last_matching_score = 1.0
            for person_b_as_string in person_list_2_as_strings:
                result = evaluator.evaluate_strings(prediction=person_a_as_string, reference=person_b_as_string)
                score = result['score']
                if score < threshold and score < last_matching_score:
                    matching_person = person_b_as_string
                    last_matching_score = score

            if matching_person is None:
                person_a_as_dict = json.loads(person_a_as_string)
                not_found.append(person_a_as_dict)

            else:
                person_a_as_dict = json.loads(person_a_as_string)
                matching_person_as_dict = json.loads(matching_person)
                matches.append((person_a_as_dict, matching_person_as_dict))

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

    def calculate_exact_metric(self, person_matching: PersonMatching):
        metric_values_changed = 0
        metric_items_change_count = 0
        number_fields_total = 0
        for match in person_matching.matches:
            json_1, json_2 = match
            ddiff_as_json = self.get_deep_diff(json_1, json_2)
            ddiff_as_dict = json.loads(ddiff_as_json)
            number_fields_total += self.count_fields(json_1)

            # values changed:
            values_changed: Dict = ddiff_as_dict.get("values_changed", {})
            value_change_count = len(values_changed)
            print("This is the number of values changed:", value_change_count)
            metric_values_changed += value_change_count

            # items like legal_relationship and family_relations added or removed
            iterable_item_added = ddiff_as_dict.get("iterable_item_added", {})
            iterable_item_removed = ddiff_as_dict.get("iterable_item_removed", {})
            metric_items_change_count += len(iterable_item_added)
            metric_items_change_count += len(iterable_item_removed)

        # persons not found
        not_found_counts = [self.count_fields(entry) for entry in person_matching.not_found]
        metric_persons_added_or_removed = sum(not_found_counts)
        number_fields_total += metric_persons_added_or_removed
        print("Number of persons not found:", not_found_counts)

        # Calculate total differences
        all_diffs = (metric_values_changed
                     + metric_persons_added_or_removed
                     + metric_items_change_count)
        matches = number_fields_total - all_diffs

        if number_fields_total == 0:
            return 1.0 if not all_diffs else 0.0
        accuracy_score = matches / number_fields_total

        print("Number of fields:", number_fields_total)
        print("Number of matches:", matches)
        print("Number of differences:", all_diffs)

        return accuracy_score

    # def check_nested_entities(self, old_entities: List[Dict], new_entities: List[Dict]):
    #     old_set = set((entry["related_person"], entry["relation_type"]) for entry in old_entities)
    #     new_set = set((entry["related_person"], entry["relation_type"]) for entry in new_entities)
    #
    #     return len(old_set.difference(new_set)) * 2

    # def check_nested_entities_fuzzy(self, old_entities: List[Dict], new_entities: List[Dict]):
    #     def is_entity_in_list(entity, entity_list):
    #         for e in entity_list:
    #             if (entity["related_person"] == e["related_person"] and
    #                     self.fuzzy_compare(entity["relation_type"], e["relation_type"])):
    #                 return True
    #         return False
    #
    #     changes = [
    #         entry for entry in old_entities
    #         if not is_entity_in_list(entry, new_entities)
    #     ]
    #
    #     return 2 * len(changes)

    def fuzzy_compare(self, old_value, new_value):
        if isinstance(old_value, str) and isinstance(new_value, str):
            score = fuzz.ratio(old_value, new_value)
            if score > self.threshold:
                return True
        return False

    # def calculate_fuzzy_metric(self, json_1, json_2):
    #     ddiff_as_json = self.get_deep_diff(json_1, json_2)
    #     ddiff_as_dict = json.loads(ddiff_as_json)
    #     number_fields_total = self.count_fields(json_1)
    #
    #     # values changed:
    #     metric_values_changed = 0
    #     values_changed: Dict = ddiff_as_dict.get("values_changed", {})
    #     for key, entry in values_changed.items():
    #         old_value = entry["old_value"]
    #         new_value = entry["new_value"]
    #
    #         if old_value["id"] != new_value["id"]:
    #             print(f"Value changed at {key}: id from {old_value['id']} to {new_value['id']}")
    #             metric_values_changed += 1
    #
    #         if not self.fuzzy_compare(old_value["name"], new_value["name"]):
    #             print(f"Value changed at {key}: name from {old_value['name']} to {new_value['name']}")
    #             metric_values_changed += 1
    #
    #         if not self.fuzzy_compare(old_value["cognomen"], new_value["cognomen"]):
    #             print(f"Value changed at {key}: cognomen from {old_value['cognomen']} to {new_value['cognomen']}")
    #             metric_values_changed += 1
    #
    #         if not self.fuzzy_compare(old_value["profession"], new_value["profession"]):
    #             print(f"Value changed at {key}: profession from {old_value['profession']} to {new_value['profession']}")
    #             metric_values_changed += 1
    #
    #         if not self.fuzzy_compare(old_value["place_of_origin"], new_value["place_of_origin"]):
    #             print(
    #                 f"Value changed at {key}: place_of_origin from {old_value['place_of_origin']} to {new_value['place_of_origin']}")
    #             metric_values_changed += 1
    #
    #         if not self.fuzzy_compare(old_value["title"], new_value["title"]):
    #             print(f"Value changed at {key}: title from {old_value['title']} to {new_value['title']}")
    #             metric_values_changed += 1
    #
    #         if old_value["family_relations"] != new_value["family_relations"]:
    #             changes = self.check_nested_entities_fuzzy(old_value["family_relations"], new_value["family_relations"])
    #             print(f"Nested changes in family_relations at {key}: {changes} changes")
    #             metric_values_changed += changes
    #
    #         if old_value["legal_relationship"] != new_value["legal_relationship"]:
    #             changes = self.check_nested_entities_fuzzy(old_value["legal_relationship"],
    #                                                        new_value["legal_relationship"])
    #             print(f"Nested changes in legal_relationship at {key}: {changes} changes")
    #             metric_values_changed += changes
    #
    #     # iterable_item_removed
    #     iterable_item_removed = ddiff_as_dict.get("iterable_item_removed", {})
    #     metric_iterable_item_removed = 8 * len(iterable_item_removed)
    #     if metric_iterable_item_removed > 0:
    #         print("Items removed:")
    #         for key in iterable_item_removed:
    #             print(f"Removed: {key}")
    #
    #     # iterable_item_added
    #     iterable_item_added = ddiff_as_dict.get("iterable_item_added", {})
    #     metric_iterable_item_added = 8 * len(iterable_item_added)
    #     if metric_iterable_item_added > 0:
    #         print("Items added:")
    #         for key in iterable_item_added:
    #             print(f"Added: {key}")
    #
    #     # dictionary_item_added
    #     dict_item_added = ddiff_as_dict.get("dictionary_item_added", {})
    #     metric_dict_item_added = 8 * len(dict_item_added)
    #     if metric_dict_item_added > 0:
    #         print("Dictionary items added:")
    #         for key in dict_item_added:
    #             print(f"Added: {key}")
    #
    #     # dictionary_item_removed
    #     dict_item_removed = ddiff_as_dict.get("dictionary_item_removed", {})
    #     metric_dict_item_removed = 8 * len(dict_item_removed)
    #     if metric_dict_item_removed > 0:
    #         print("Dictionary items removed:")
    #         for key in dict_item_removed:
    #             print(f"Removed: {key}")
    #
    #     # Calculate total differences
    #     all_diffs = (metric_values_changed
    #                  + metric_iterable_item_removed
    #                  + metric_iterable_item_added
    #                  + metric_dict_item_added
    #                  + metric_dict_item_removed)
    #     matches = number_fields_total - all_diffs
    #
    #     if number_fields_total == 0:
    #         return 1.0 if not all_diffs else 0.0
    #     accuracy_score = matches / number_fields_total
    #
    #     print("Number of fields:", number_fields_total)
    #     print("Number of matches:", matches)
    #     print("Number of differences:", all_diffs)
    #
    #     return accuracy_score

    # def calculate_fuzzy_metric(self, json_1, json_2):
    #     ddiff_as_json = self.get_deep_diff(json_1, json_2)
    #     ddiff_as_dict = json.loads(ddiff_as_json)
    #
    #     number_fields = self.count_fields(json_1)
    #     dissimilar_items = set()
    #
    #     for key in ['dictionary_item_removed', 'dictionary_item_added', 'iterable_item_removed', 'iterable_item_added']:
    #         if key in ddiff_as_dict:
    #             if isinstance(ddiff_as_dict[key], dict):
    #                 dissimilar_items.update(ddiff_as_dict[key].keys())
    #             elif isinstance(ddiff_as_dict[key], list):
    #                 dissimilar_items.update(range(len(ddiff_as_dict[key])))
    #
    #     if 'values_changed' in ddiff_as_dict:
    #         for item, diff in ddiff_as_dict['values_changed'].items():
    #             old_value = diff['old_value']
    #             new_value = diff['new_value']
    #             if not self.fuzzy_compare(old_value, new_value):
    #                 dissimilar_items.add(item)
    #
    #     matches = number_fields - len(dissimilar_items)
    #     accuracy_score = matches / number_fields
    #
    #     print("dissimilar items:", dissimilar_items)
    #
    #     return accuracy_score

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
                        matching_person_object = self.find_matching_person(gt_data['person_list'], pred_data['person_list'])
                        exact_score = self.calculate_exact_metric(matching_person_object)
                        result_file.write(f"Exact Metric: {exact_score}\n")
                        # fuzzy_score = self.calculate_fuzzy_metric(gt_data, pred_data)
                        # result_file.write(f"Fuzzy Metric: {fuzzy_score}\n\n")
                    else:
                        result_file.write(f"File: {gt_filename} - JSON file with predicted results not found.\n\n")


if __name__ == '__main__':
    json_comparison = JsonComparison()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    path_json1 = os.path.join(base_dir, "test_data", "test_json_diff", "sample_json_original.json")
    json1 = file_reader_util.read_json(path_json1)
    path_json2 = os.path.join(base_dir, "test_data", "test_json_diff", "sample_json_modified.json")
    json2 = file_reader_util.read_json(path_json2)

    # result_exact_matching = json_comparison.get_deep_diff(json1, json2)
    # print(result_exact_matching)

    accuracy = json_comparison.calculate_exact_metric(json1, json2)
    print(accuracy)

    fuzzy_accuracy = json_comparison.calculate_fuzzy_metric(json1, json2)
    print(fuzzy_accuracy)

    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    gt_path = os.path.join(base_dir, "test_data", "test_json_diff", "ground_truth")
    pred_path = os.path.join(base_dir, "test_data", "test_json_diff", "predictions")
    output_path = os.path.join(base_dir, "test_data", "test_json_diff", "scores")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    json_comparison.perform_json_comparison(gt_path, pred_path, output_path)
