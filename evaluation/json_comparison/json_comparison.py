import json
import os
from pathlib import Path
from deepdiff import DeepDiff
from dotenv import load_dotenv
from thefuzz import fuzz
from utils import json_util


class JsonComparison:

    def __init__(self, threshold=80):
        load_dotenv()
        self.threshold = threshold

    def perform_deep_diff(self, json_1, json_2):
        ddiff = DeepDiff(json_1, json_2, ignore_order=True)
        ddiff_as_json = ddiff.to_json()
        return ddiff_as_json

    def count_fields(self, obj):
        if isinstance(obj, dict):
            return sum(self.count_fields(v) for v in obj.values()) + len(obj)
        elif isinstance(obj, list):
            return sum(self.count_fields(item) for item in obj)
        else:
            return 0

    def calculate_exact_metric(self, json_1, json_2):
        ddiff_as_json = self.perform_deep_diff(json_1, json_2)
        ddiff_as_dict = json.loads(ddiff_as_json)
        number_fields = self.count_fields(json_1)

        all_diffs = set()
        for key in ['values_changed', 'dictionary_item_removed', 'dictionary_item_added', 'iterable_item_removed',
                    'iterable_item_added']:
            if key in ddiff_as_dict:
                if isinstance(ddiff_as_dict[key], dict):
                    all_diffs.update(ddiff_as_dict[key].keys())
                elif isinstance(ddiff_as_dict[key], list):
                    all_diffs.update(range(len(ddiff_as_dict[key])))
        matches = number_fields - len(all_diffs)
        accuracy_score = matches / number_fields

        return accuracy_score

    def fuzzy_compare(self, old_value, new_value):
        if isinstance(old_value, str) and isinstance(new_value, str):
            score = fuzz.ratio(old_value, new_value)
            if score > self.threshold:
                return True
        return False

    def calculate_fuzzy_metric(self, json_1, json_2):
        ddiff_as_json = self.perform_deep_diff(json_1, json_2)
        ddiff_as_dict = json.loads(ddiff_as_json)

        number_fields = self.count_fields(json_1)
        dissimilar_items = set()

        for key in ['dictionary_item_removed', 'dictionary_item_added', 'iterable_item_removed', 'iterable_item_added']:
            if key in ddiff_as_dict:
                if isinstance(ddiff_as_dict[key], dict):
                    dissimilar_items.update(ddiff_as_dict[key].keys())
                elif isinstance(ddiff_as_dict[key], list):
                    dissimilar_items.update(range(len(ddiff_as_dict[key])))

        if 'values_changed' in ddiff_as_dict:
            for item, diff in ddiff_as_dict['values_changed'].items():
                old_value = diff['old_value']
                new_value = diff['new_value']
                if not self.fuzzy_compare(old_value, new_value):
                    dissimilar_items.add(item)

        matches = number_fields - len(dissimilar_items)
        accuracy_score = matches / number_fields

        return accuracy_score


if __name__ == '__main__':
    json_comparison = JsonComparison()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    path_json1 = os.path.join(base_dir, "test_data", "test_json_diff", "sample_json_original.json")
    json1 = json_util.read_json(path_json1)
    path_json2 = os.path.join(base_dir, "test_data", "test_json_diff", "sample_json_modified.json")
    json2 = json_util.read_json(path_json2)

    result_exact_matching = json_comparison.perform_deep_diff(json1, json2)
    print(result_exact_matching)

    accuracy = json_comparison.calculate_exact_metric(json1, json2)
    print(accuracy)

    fuzzy_accuracy = json_comparison.calculate_fuzzy_metric(json1, json2)
    print(fuzzy_accuracy)
