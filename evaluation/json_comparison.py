import json
import os
from dataclasses import is_dataclass, asdict
from datetime import datetime
from typing import List, Dict, Tuple, Any
from deepdiff import DeepDiff
from dotenv import load_dotenv
from thefuzz import fuzz
from data_classes.document import create_documents
from data_classes.person import Person
from evaluation.overall_result import OverallResultList
from evaluation.person_matching import PersonMatching
from langchain.evaluation import JsonEditDistanceEvaluator


class JsonComparison:

    def __init__(self):
        load_dotenv()

    @staticmethod
    def _person_dict_for_matching(person: Person) -> Dict:
        """
        Representation used for deciding which predicted person corresponds to which GT person.
        """
        person_dict = person.to_dict()
        person_dict.pop("id", None)

        for relation_field in ["family_relations", "legal_relationship"]:
            person_dict[relation_field] = sorted(
                [{"relation_type": relation["relation_type"]}for relation in person_dict[relation_field]],
                key=lambda relation: (relation["relation_type"]),
            )

        return person_dict

    def _person_distance(self, evaluator, person_a: Person, person_b: Person) -> float:
        person_a_string = json.dumps(self._person_dict_for_matching(person_a), sort_keys=True)
        person_b_string = json.dumps(self._person_dict_for_matching(person_b), sort_keys=True)
        result = evaluator.evaluate_strings(prediction=person_a_string, reference=person_b_string)

        return result["score"]

    def find_matching_person(
            self,
            person_list_1: List[Person],
            person_list_2: List[Person],
            distance_threshold: float = 0.2,
    ) -> PersonMatching:
        evaluator = JsonEditDistanceEvaluator()
        matches = []
        used_gt_indices = set()
        used_prediction_indices = set()

        # Stage 1: Match people whose names correspond.
        name_candidates = []

        for gt_index, person_a in enumerate(person_list_1):
            for pred_index, person_b in enumerate(person_list_2):
                name_a = self.simplify_name(person_a.name)
                name_b = self.simplify_name(person_b.name)

                if self.fuzzy_compare(name_a, name_b):
                    score = self._person_distance(evaluator, person_a, person_b)
                    name_candidates.append((score, gt_index, pred_index))

        # Best matches first.
        name_candidates.sort(key=lambda candidate: candidate[0])

        for (score, gt_index, pred_index) in name_candidates:
            if gt_index in used_gt_indices:
                continue

            if pred_index in used_prediction_indices:
                continue

            matches.append((person_list_1[gt_index], person_list_2[pred_index]))
            used_gt_indices.add(gt_index)
            used_prediction_indices.add(pred_index)


        # Fallback matching for people whose names did not  correspond but whose overall records are close.
        fallback_candidates = []

        for gt_index, person_a in enumerate(person_list_1):
            if gt_index in used_gt_indices:
                continue

            for pred_index, person_b in enumerate(person_list_2):
                if pred_index in used_prediction_indices:
                    continue

                score = self._person_distance(evaluator, person_a, person_b)

                if score < distance_threshold:
                    fallback_candidates.append((score, gt_index, pred_index))

        fallback_candidates.sort(key=lambda candidate: candidate[0])

        for (score, gt_index, pred_index) in fallback_candidates:

            if gt_index in used_gt_indices:
                continue

            if pred_index in used_prediction_indices:
                continue

            matches.append((person_list_1[gt_index], person_list_2[pred_index]))
            used_gt_indices.add(gt_index)
            used_prediction_indices.add(pred_index)

        not_found = [person for index, person in enumerate(person_list_1) if index not in used_gt_indices]
        unmatched_predictions = [person for index, person in enumerate(person_list_2) if index not in used_prediction_indices]

        return PersonMatching(
            matches=matches,
            not_found=not_found,
            unmatched_predictions=unmatched_predictions
        )

    @staticmethod
    def normalize_prediction_relation_targets(person_matching: PersonMatching):
        prediction_id_to_gt_id = {predicted_person.id: gt_person.id for (gt_person, predicted_person) in person_matching.matches}
        all_predicted_people = [predicted_person for _, predicted_person in person_matching.matches]
        all_predicted_people.extend(person_matching.unmatched_predictions)

        for person in all_predicted_people:
            relations = person.family_relations + person.legal_relationship

            for relation in relations:
                related_person = relation.related_person

                if related_person is None:
                    continue

                if not isinstance(related_person, int):
                    continue

                if related_person in prediction_id_to_gt_id:
                    relation.related_person = prediction_id_to_gt_id[related_person]
                else:
                    relation.related_person = f"unmatched_prediction_{related_person}"

    @staticmethod
    def simplify_name(name_a):
        return name_a.replace('.', '').replace('\'', '')

    @staticmethod
    def get_deep_diff(person_1: Person, person_2: Person, exclude_paths: List[str] = None):
        if exclude_paths is None:
            exclude_paths = []

        # Always exclude the 'id' field
        # exclude_paths.append("root['id']")

        person_1_dict = person_1.__dict__
        person_2_dict = person_2.__dict__
        ddiff = DeepDiff(person_1_dict, person_2_dict,
                         ignore_order=True,
                         verbose_level=2,
                         exclude_paths=exclude_paths)
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

    def calculate_metric(self, person_matching: PersonMatching, apply_fuzzy=False, exclude_paths: List[str] = None):
        if not apply_fuzzy:
            metric_values_changed, metric_items_change_count, number_fields_total, metric_type_changes, deep_diff_results = self.calculate_diff_metrics(
                person_matching, exclude_paths=exclude_paths)
        else:
            metric_values_changed, metric_items_change_count, number_fields_total, metric_type_changes, deep_diff_results = (
                self.calculate_diff_metrics(person_matching, apply_fuzzy=True, exclude_paths=exclude_paths))

        number_of_fields_of_missing_persons = self.calculate_not_found_metric(person_matching, exclude_paths)
        number_of_fields_of_extra_persons = self.calculate_unmatched_prediction_metric(person_matching, exclude_paths)
        number_fields_total += (number_of_fields_of_missing_persons + number_of_fields_of_extra_persons)
        all_diffs = (
                metric_values_changed
                + metric_items_change_count
                + metric_type_changes
                + number_of_fields_of_missing_persons
                + number_of_fields_of_extra_persons
        )
        matches = number_fields_total - all_diffs
        accuracy_score = self.calculate_accuracy_score(matches, number_fields_total, all_diffs)

        return (
            accuracy_score,
            matches,
            all_diffs,
            deep_diff_results,
            number_fields_total,
            number_of_fields_of_missing_persons,
            number_of_fields_of_extra_persons,
        )

    def calculate_diff_metrics(self,
                               person_matching: PersonMatching,
                               apply_fuzzy=False,
                               exclude_paths: List[str] = None) -> Tuple[int, int, int, int, List[Dict[str, Any]]]:
        metric_values_changed = 0
        metric_items_change_count = 0
        metric_type_changes = 0
        number_fields_total = 0
        deepdiff_results = []

        for match in person_matching.matches:
            person_1, person_2 = match
            exact_ddiff_as_json = self.get_deep_diff(person_1, person_2, exclude_paths)
            exact_ddiff_as_dict = json.loads(exact_ddiff_as_json)
            number_fields_total += (
                self.count_person_fields(
                    person_1,
                    exclude_paths,
                )
            )

            # Values changed:
            exact_values_changed: Dict = exact_ddiff_as_dict.get("values_changed", {})

            if apply_fuzzy:
                fuzzy_diff_values, value_change_count = self.apply_fuzzy_compare(exact_values_changed)
                ddiff_as_dict = {"values_changed": fuzzy_diff_values}
                print("This is the number of values changed:", value_change_count)
            else:
                value_change_count = len(exact_values_changed)
                ddiff_as_dict = {"values_changed": exact_values_changed}
                print("This is the number of values changed:", value_change_count)
            metric_values_changed += value_change_count

            ddiff_as_dict["iterable_item_added"] = exact_ddiff_as_dict.get("iterable_item_added", {})
            ddiff_as_dict["iterable_item_removed"] = exact_ddiff_as_dict.get("iterable_item_removed", {})
            ddiff_as_dict["type_changes"] = exact_ddiff_as_dict.get("type_changes", {})

            ddiff_as_json = json.dumps(ddiff_as_dict)

            # Items added or removed
            iterable_item_added = ddiff_as_dict.get("iterable_item_added", {})
            iterable_item_removed = ddiff_as_dict.get("iterable_item_removed", {})
            metric_items_change_count += self.count_internal_items(iterable_item_added) + self.count_internal_items(iterable_item_removed)

            # type changes
            # type_changes: Dict = ddiff_as_dict.get("type_changes", {})
            type_changes: Dict = exact_ddiff_as_dict.get("type_changes", {})
            metric_type_changes += len(type_changes)

            deepdiff_results.append({
                'name1': person_1.name,
                'name2': person_2.name,
                'deepdiff': ddiff_as_json,
                'values_changed': value_change_count,
                'iterable_item_added': self.count_internal_items(iterable_item_added),
                'iterable_item_removed': self.count_internal_items(iterable_item_removed),
                'type_changes': len(type_changes)
            })

        return metric_values_changed, metric_items_change_count, number_fields_total, metric_type_changes, deepdiff_results

    @staticmethod
    def count_internal_items(iterable_item_added: Dict[str, Dict]):
        count = 0
        for key in iterable_item_added:
            count += len(iterable_item_added[key])
        return count

    def calculate_not_found_metric(
            self,
            person_matching: PersonMatching,
            exclude_paths=None,
    ) -> int:

        not_found_counts = [
            self.count_person_fields(
                person,
                exclude_paths,
            )
            for person
            in person_matching.not_found
        ]

        return sum(not_found_counts)

    def calculate_unmatched_prediction_metric(
            self,
            person_matching: PersonMatching,
            exclude_paths=None,
    ) -> int:

        extra_counts = [
            self.count_person_fields(
                person,
                exclude_paths,
            )
            for person
            in person_matching.unmatched_predictions
        ]

        return sum(extra_counts)

    def count_person_fields(
            self,
            person: Person,
            exclude_paths=None,
    ) -> int:

        person_dict = person.to_dict()

        if exclude_paths is None:
            exclude_paths = []

        for path in exclude_paths:
            prefix = "root['"
            suffix = "']"

            if (
                    path.startswith(prefix)
                    and path.endswith(suffix)
            ):
                field_name = path[
                             len(prefix):-len(suffix)
                             ]

                person_dict.pop(
                    field_name,
                    None,
                )

        return self.count_fields(person_dict)

    @staticmethod
    def calculate_accuracy_score(matches: int, number_fields_total: int, all_diffs: int) -> float:
        if number_fields_total == 0:
            return 1.0 if not all_diffs else 0.0
        return matches / number_fields_total

    @staticmethod
    def fuzzy_compare(old_value, new_value, threshold=80):
        if isinstance(old_value, str) and isinstance(new_value, str):
            score = fuzz.ratio(old_value, new_value)
            if score > threshold:
                return True
        return False

    def apply_fuzzy_compare(self, values_changed: Dict, threshold=80) -> Tuple[Dict[str, Dict], int]:
        fuzzy_diffs_counter = 0
        deep_diff_formated_fuzzy_changes = dict()
        for key, change in values_changed.items():
            old_value = change["old_value"]
            new_value = change["new_value"]

            if (
                    isinstance(old_value, str)
                    and isinstance(new_value, str)
            ):
                if not self.fuzzy_compare(
                        old_value,
                        new_value,
                        threshold,
                ):
                    deep_diff_formated_fuzzy_changes[
                        key
                    ] = {"old_value": old_value, "new_value": new_value}

                    fuzzy_diffs_counter += 1

            else:
                deep_diff_formated_fuzzy_changes[key] = {"old_value": old_value, "new_value": new_value}
                fuzzy_diffs_counter += 1
        return deep_diff_formated_fuzzy_changes, fuzzy_diffs_counter

    def perform_json_comparison(self,
                                gt_folder,
                                prediction_folder,
                                output_folder,
                                json_output_folder,
                                exclusions_list: List[List[str]],
                                expected_base_names=None,
                                retry_summary=None):
        output_file = os.path.join(output_folder, f'results.txt')
        json_output_file = os.path.join(json_output_folder, f'results.json')
        gt_documents = create_documents(gt_folder)
        pred_documents = create_documents(prediction_folder)
        gt_dict = {doc.document_name: doc for doc in gt_documents}
        pred_dict = {doc.document_name: doc for doc in pred_documents}
        overall_results_list = OverallResultList(exclusions_list)
        comparison_results = {'individual_results': [], 'overall_results': []}
        successful_predictions = 0
        failed_predictions = 0

        for gt_filename, gt_document in gt_dict.items():
            base_name = os.path.splitext(gt_filename)[0]
            if expected_base_names is not None and base_name not in expected_base_names:
                continue

            pred_filename = f'pred_{base_name}.json'

            if pred_filename in pred_dict:
                successful_predictions += 1
                pred_document = pred_dict[pred_filename]
                matching_person_object = self.find_matching_person(gt_document.persons, pred_document.persons)
                self.normalize_prediction_relation_targets(matching_person_object)

                for exclude_paths in exclusions_list:
                    (
                        exact_score,
                        exact_matches,
                        exact_diffs,
                        deep_diff_results,
                        total_field_count,
                        number_of_fields_of_missing_persons,
                        number_of_fields_of_extra_persons,
                    ) = self.calculate_metric(matching_person_object, exclude_paths=exclude_paths)
                    (
                        fuzzy_score,
                        fuzzy_matches,
                        fuzzy_diffs,
                        deep_diff_results_fuzzy,
                        fuzzy_total_field_count,
                        _,
                        _,
                    ) = self.calculate_metric(matching_person_object, apply_fuzzy=True, exclude_paths=exclude_paths)

                    individual_result = {
                        'file_name1': gt_filename,
                        'file_name2': pred_document.document_name,
                        'exclude_paths': exclude_paths,
                        'exact_score': exact_score,
                        'exact_matches': exact_matches,
                        'exact_differences': exact_diffs,
                        'fuzzy_score': fuzzy_score,
                        'fuzzy_matches': fuzzy_matches,
                        'fuzzy_differences': fuzzy_diffs,
                        'total_field_count': total_field_count,
                        'deepdiff_comparison_results': deep_diff_results,
                        'deep_diff_comparison_results_fuzzy': deep_diff_results_fuzzy,
                        "number_of_missing_persons":
                            len(
                                matching_person_object.not_found
                            ),

                        "number_of_extra_persons":
                            len(
                                matching_person_object
                                .unmatched_predictions
                            ),

                        "number_of_fields_of_missing_persons":
                            number_of_fields_of_missing_persons,

                        "number_of_fields_of_extra_persons":
                            number_of_fields_of_extra_persons,
                    }

                    comparison_results['individual_results'].append(individual_result)
                    overall_results_list.add_counts(exclusion_path=exclude_paths,
                                                    fuzzy_count_matches=fuzzy_matches,
                                                    exact_count_matches=exact_matches,
                                                    all_fields_count=total_field_count)

            else:
                failed_predictions += 1

                for exclude_paths in exclusions_list:
                    total_field_count = sum(self.count_person_fields(person, exclude_paths) for person in gt_document.persons)

                    comparison_results["individual_results"].append(
                        {
                            "file_name1": gt_filename,
                            "file_name2": None,
                            "exclude_paths": exclude_paths,
                            "error": "Prediction could not be generated.",
                            "exact_score": 0.0,
                            "exact_matches": 0,
                            "exact_differences": total_field_count,
                            "fuzzy_score": 0.0,
                            "fuzzy_matches": 0,
                            "fuzzy_differences": total_field_count,
                            "total_field_count": total_field_count,
                            "number_of_missing_persons": len(gt_document.persons),
                            "number_of_extra_persons": 0,
                            "number_of_fields_of_missing_persons": total_field_count,
                            "number_of_fields_of_extra_persons": 0,
                        }
                    )

                    overall_results_list.add_counts(
                        exclusion_path=exclude_paths,
                        fuzzy_count_matches=0,
                        exact_count_matches=0,
                        all_fields_count=total_field_count,
                    )

        number_expected = (
                successful_predictions
                + failed_predictions
        )

        comparison_results["generation_summary"] = {
            "expected_predictions": number_expected,
            "successful_predictions": successful_predictions,
            "failed_predictions": failed_predictions,
            "success_rate": successful_predictions / number_expected if number_expected > 0 else 0.0
        }
        if retry_summary is not None:
            comparison_results["retry_summary"] = retry_summary

        comparison_results['overall_results'] = overall_results_list.get_overall_results()

        # Write all results to the file in one go
        with open(output_file, "a", encoding="utf-8") as result_file:
            result_file.write(f"\nResults generated on: {datetime.now()}\n\n")
            formatted_result = self.format_comparison(comparison_results)
            result_file.write(formatted_result)

        # Save the comparison results as a JSON file
        with open(json_output_file, "w", encoding="utf-8") as json_file:
            json_file.write(json.dumps(comparison_results, indent=4, ensure_ascii=False))

    @staticmethod
    def format_comparison(comparison_results):
        """
        Formats the comparison results into a structured string.
        """
        output = []

        # Format individual results
        for result in comparison_results['individual_results']:
            if 'error' in result:
                output.append(f"File: {result['file_name1']} - {result['error']}\n")
                continue

            # Header for the files being compared
            output.append(f"------------------------------------------------------------")
            output.append(f"FILE: {result['file_name1']} & {result['file_name2']}")
            output.append(f"EXCLUDE_PATHS: {result['exclude_paths']}")
            output.append(f"------------------------------------------------------------\n")

            # Total Field Count
            output.append(f"**Total Field Count: {result['total_field_count']}**\n")

            output.append(f"------------------------------------------------------------")
            output.append(f"EXACT METRIC")
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

            # Exact Metric
            output.append(f"**Exact Metric: {result['exact_score']}**")
            output.append(f"- Matches: {result['exact_matches']}")
            output.append(f"- Differences: {result['exact_differences']}\n")

            output.append(f"------------------------------------------------------------")
            output.append(f"FUZZY METRIC")
            output.append(f"------------------------------------------------------------\n")

            # Details of each deepdiff comparison fuzzy
            for comparison in result['deep_diff_comparison_results_fuzzy']:
                name1, name2, deepdiff = comparison['name1'], comparison['name2'], comparison['deepdiff']
                output.append(f"Comparing: {name1} with {name2}")
                output.append(f"- DeepDiff: {deepdiff}")
                output.append(f"- Values Changed: {comparison['values_changed']}")
                output.append(f"- Items Added: {comparison['iterable_item_added']}")
                output.append(f"- Items Removed: {comparison['iterable_item_removed']}")
                output.append(f"- Type Changes: {comparison['type_changes']}\n")

            # Fuzzy Metric
            output.append(f"**Fuzzy Metric: {result['fuzzy_score']}**")
            output.append(f"- Matches: {result['fuzzy_matches']}")
            output.append(f"- Differences: {result['fuzzy_differences']}\n")

            output.append(f"============================================================\n")

        output.append(f"------------------------------------------------------------")
        output.append(f"OVERALL RESULTS")
        output.append(f"------------------------------------------------------------\n")

        # Format overall results
        for overall_results in comparison_results['overall_results']:
            output.append(f"Path Exclusion: {overall_results['overall_path_exclude']}\n")
            output.append(f"Overall Exact Score: {overall_results['overall_exact_score']:.4f}\n")
            output.append(f"Overall Fuzzy Score: {overall_results['overall_fuzzy_score']:.4f}\n")
            output.append(f"Total Exact Matches: {overall_results['total_exact_matches']}\n")
            output.append(f"Total Fuzzy Matches: {overall_results['total_fuzzy_matches']}\n")
            output.append(f"Total Exact Misses: {overall_results['total_exact_misses']}\n")
            output.append(f"Total Fuzzy Misses: {overall_results['total_fuzzy_misses']}\n")
            output.append(f"Total Fields Count: {overall_results['total_fields_count']}\n")
            output.append(f"{'-' * 60}\n\n")

        return "\n".join(output)
