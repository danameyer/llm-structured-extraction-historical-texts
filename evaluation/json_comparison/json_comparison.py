import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from deepdiff import DeepDiff
from dotenv import load_dotenv
from thefuzz import fuzz
from utils import file_reader_util


class JsonComparison:

    def __init__(self, threshold=80):
        load_dotenv()
        self.threshold = threshold

    def get_deep_diff(self, json_1, json_2):
        ddiff = DeepDiff(json_1, json_2, ignore_order=True)
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

    def calculate_exact_metric(self, json_1, json_2):
        ddiff_as_json = self.get_deep_diff(json_1, json_2)
        ddiff_as_dict = json.loads(ddiff_as_json)
        number_fields_total = self.count_fields(json_1)

        # values changed:
        metric_values_changed = 0
        values_changed: Dict = ddiff_as_dict.get("values_changed", {})
        for key, entry in values_changed.items():
            old_value = entry["old_value"]
            new_value = entry["new_value"]

            if old_value["id"] != new_value["id"]:
                print(f"Value changed at {key}: id from {old_value['id']} to {new_value['id']}")
                metric_values_changed += 1

            if old_value["name"] != new_value["name"]:
                print(f"Value changed at {key}: name from {old_value['name']} to {new_value['name']}")
                metric_values_changed += 1

            if old_value["cognomen"] != new_value["cognomen"]:
                print(f"Value changed at {key}: cognomen from {old_value['cognomen']} to {new_value['cognomen']}")
                metric_values_changed += 1

            if old_value["profession"] != new_value["profession"]:
                print(f"Value changed at {key}: profession from {old_value['profession']} to {new_value['profession']}")
                metric_values_changed += 1

            if old_value["place_of_origin"] != new_value["place_of_origin"]:
                print(
                    f"Value changed at {key}: place_of_origin from {old_value['place_of_origin']} to {new_value['place_of_origin']}")
                metric_values_changed += 1

            if old_value["title"] != new_value["title"]:
                print(f"Value changed at {key}: title from {old_value['title']} to {new_value['title']}")
                metric_values_changed += 1

            if old_value["family_relations"] != new_value["family_relations"]:
                changes = self.check_nested_entities(old_value["family_relations"], new_value["family_relations"])
                print(f"Nested changes in family_relations at {key}: {changes} changes")
                metric_values_changed += changes

            if old_value["legal_relationship"] != new_value["legal_relationship"]:
                changes = self.check_nested_entities(old_value["legal_relationship"], new_value["legal_relationship"])
                print(f"Nested changes in legal_relationship at {key}: {changes} changes")
                metric_values_changed += changes

        # iterable_item_removed
        iterable_item_removed = ddiff_as_dict.get("iterable_item_removed", {})
        metric_iterable_item_removed = 8 * len(iterable_item_removed)
        if metric_iterable_item_removed > 0:
            print("Items removed:")
            for key in iterable_item_removed:
                print(f"Removed: {key}")

        # iterable_item_added
        iterable_item_added = ddiff_as_dict.get("iterable_item_added", {})
        metric_iterable_item_added = 8 * len(iterable_item_added)
        if metric_iterable_item_added > 0:
            print("Items added:")
            for key in iterable_item_added:
                print(f"Added: {key}")

        # dictionary_item_added
        dict_item_added = ddiff_as_dict.get("dictionary_item_added", {})
        metric_dict_item_added = 8 * len(dict_item_added)
        if metric_dict_item_added > 0:
            print("Dictionary items added:")
            for key in dict_item_added:
                print(f"Added: {key}")

        # dictionary_item_removed
        dict_item_removed = ddiff_as_dict.get("dictionary_item_removed", {})
        metric_dict_item_removed = 8 * len(dict_item_removed)
        if metric_dict_item_removed > 0:
            print("Dictionary items removed:")
            for key in dict_item_removed:
                print(f"Removed: {key}")

        # Calculate total differences
        all_diffs = (metric_values_changed
                     + metric_iterable_item_removed
                     + metric_iterable_item_added
                     + metric_dict_item_added
                     + metric_dict_item_removed)
        matches = number_fields_total - all_diffs

        if number_fields_total == 0:
            return 1.0 if not all_diffs else 0.0
        accuracy_score = matches / number_fields_total

        print("Number of fields:", number_fields_total)
        print("Number of matches:", matches)
        print("Number of differences:", all_diffs)

        return accuracy_score

    def check_nested_entities(self, old_entities: List[Dict], new_entities: List[Dict]):
        old_set = set((entry["related_person"], entry["relation_type"]) for entry in old_entities)
        new_set = set((entry["related_person"], entry["relation_type"]) for entry in new_entities)

        return len(old_set.difference(new_set)) * 2

    def check_nested_entities_fuzzy(self, old_entities: List[Dict], new_entities: List[Dict]):
        def is_entity_in_list(entity, entity_list):
            for e in entity_list:
                if (entity["related_person"] == e["related_person"] and
                        self.fuzzy_compare(entity["relation_type"], e["relation_type"])):
                    return True
            return False

        changes = [
            entry for entry in old_entities
            if not is_entity_in_list(entry, new_entities)
        ]

        return 2 * len(changes)



    # def calculate_exact_metric(self, json_1, json_2):
    #     ddiff_as_json = self.get_deep_diff(json_1, json_2)
    #     ddiff_as_dict = json.loads(ddiff_as_json)
    #     number_fields = self.count_fields(json_1)
    #
    #     all_diffs = set()
    #     for key in ['values_changed', 'dictionary_item_removed', 'dictionary_item_added', 'iterable_item_removed',
    #                 'iterable_item_added']:
    #         if key in ddiff_as_dict:
    #             if isinstance(ddiff_as_dict[key], dict):
    #                 for diff_key, diff_value in ddiff_as_dict[key].items():
    #                     if isinstance(diff_value, dict) and 'old_value' in diff_value and 'new_value' in diff_value:
    #                         if isinstance(diff_value['old_value'], dict) and isinstance(diff_value['new_value'], dict):
    #                             all_diffs.update(diff_value['old_value'].keys())
    #                             all_diffs.update(diff_value['new_value'].keys())
    #                         elif isinstance(diff_value['old_value'], list) and isinstance(diff_value['new_value'],
    #                                                                                       list):
    #                             all_diffs.update(range(len(diff_value['old_value'])))
    #                             all_diffs.update(range(len(diff_value['new_value'])))
    #                         else:
    #                             all_diffs.add(diff_key)
    #                     else:
    #                         all_diffs.add(diff_key)
    #             elif isinstance(ddiff_as_dict[key], list):
    #                 all_diffs.update(range(len(ddiff_as_dict[key])))
    #
    #     matches = number_fields - len(all_diffs)
    #     accuracy_score = matches / number_fields if number_fields != 0 else 0
    #
    #     print("Number of fields:", number_fields)
    #     print("Number of matches:", matches)
    #     print("Number of differences:", len(all_diffs))
    #
    #     return accuracy_score

    def fuzzy_compare(self, old_value, new_value):
        if isinstance(old_value, str) and isinstance(new_value, str):
            score = fuzz.ratio(old_value, new_value)
            if score > self.threshold:
                return True
        return False

    def calculate_fuzzy_metric(self, json_1, json_2):
        ddiff_as_json = self.get_deep_diff(json_1, json_2)
        ddiff_as_dict = json.loads(ddiff_as_json)
        number_fields_total = self.count_fields(json_1)

        # values changed:
        metric_values_changed = 0
        values_changed: Dict = ddiff_as_dict.get("values_changed", {})
        for key, entry in values_changed.items():
            old_value = entry["old_value"]
            new_value = entry["new_value"]

            if old_value["id"] != new_value["id"]:
                print(f"Value changed at {key}: id from {old_value['id']} to {new_value['id']}")
                metric_values_changed += 1

            if not self.fuzzy_compare(old_value["name"], new_value["name"]):
                print(f"Value changed at {key}: name from {old_value['name']} to {new_value['name']}")
                metric_values_changed += 1

            if not self.fuzzy_compare(old_value["cognomen"], new_value["cognomen"]):
                print(f"Value changed at {key}: cognomen from {old_value['cognomen']} to {new_value['cognomen']}")
                metric_values_changed += 1

            if not self.fuzzy_compare(old_value["profession"], new_value["profession"]):
                print(f"Value changed at {key}: profession from {old_value['profession']} to {new_value['profession']}")
                metric_values_changed += 1

            if not self.fuzzy_compare(old_value["place_of_origin"], new_value["place_of_origin"]):
                print(
                    f"Value changed at {key}: place_of_origin from {old_value['place_of_origin']} to {new_value['place_of_origin']}")
                metric_values_changed += 1

            if not self.fuzzy_compare(old_value["title"], new_value["title"]):
                print(f"Value changed at {key}: title from {old_value['title']} to {new_value['title']}")
                metric_values_changed += 1

            if old_value["family_relations"] != new_value["family_relations"]:
                changes = self.check_nested_entities_fuzzy(old_value["family_relations"], new_value["family_relations"])
                print(f"Nested changes in family_relations at {key}: {changes} changes")
                metric_values_changed += changes

            if old_value["legal_relationship"] != new_value["legal_relationship"]:
                changes = self.check_nested_entities_fuzzy(old_value["legal_relationship"], new_value["legal_relationship"])
                print(f"Nested changes in legal_relationship at {key}: {changes} changes")
                metric_values_changed += changes

        # iterable_item_removed
        iterable_item_removed = ddiff_as_dict.get("iterable_item_removed", {})
        metric_iterable_item_removed = 8 * len(iterable_item_removed)
        if metric_iterable_item_removed > 0:
            print("Items removed:")
            for key in iterable_item_removed:
                print(f"Removed: {key}")

        # iterable_item_added
        iterable_item_added = ddiff_as_dict.get("iterable_item_added", {})
        metric_iterable_item_added = 8 * len(iterable_item_added)
        if metric_iterable_item_added > 0:
            print("Items added:")
            for key in iterable_item_added:
                print(f"Added: {key}")

        # dictionary_item_added
        dict_item_added = ddiff_as_dict.get("dictionary_item_added", {})
        metric_dict_item_added = 8 * len(dict_item_added)
        if metric_dict_item_added > 0:
            print("Dictionary items added:")
            for key in dict_item_added:
                print(f"Added: {key}")

        # dictionary_item_removed
        dict_item_removed = ddiff_as_dict.get("dictionary_item_removed", {})
        metric_dict_item_removed = 8 * len(dict_item_removed)
        if metric_dict_item_removed > 0:
            print("Dictionary items removed:")
            for key in dict_item_removed:
                print(f"Removed: {key}")

        # Calculate total differences
        all_diffs = (metric_values_changed
                     + metric_iterable_item_removed
                     + metric_iterable_item_added
                     + metric_dict_item_added
                     + metric_dict_item_removed)
        matches = number_fields_total - all_diffs

        if number_fields_total == 0:
            return 1.0 if not all_diffs else 0.0
        accuracy_score = matches / number_fields_total

        print("Number of fields:", number_fields_total)
        print("Number of matches:", matches)
        print("Number of differences:", all_diffs)

        return accuracy_score

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
                        exact_score = self.calculate_exact_metric(gt_data, pred_data)
                        result_file.write(f"Exact Metric: {exact_score}\n")
                        fuzzy_score = self.calculate_fuzzy_metric(gt_data, pred_data)
                        result_file.write(f"Fuzzy Metric: {fuzzy_score}\n\n")
                    else:
                        result_file.write(f"File: {gt_filename} - JSON file with predicted results not found.\n\n")


if __name__ == '__main__':
    json_comparison = JsonComparison()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    path_json1 = os.path.join(base_dir, "test_data", "test_json_diff", "sample_json_original.json")
    json1 = file_reader_util.read_json(path_json1)
    path_json2 = os.path.join(base_dir, "test_data", "test_json_diff", "sample_json_modified.json")
    json2 = file_reader_util.read_json(path_json2)

    result_exact_matching = json_comparison.get_deep_diff(json1, json2)
    print(result_exact_matching)

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
