from typing import List


class OverallResultList:
    def __init__(self, exclusions: List[List[str]]):
        self.exclusions = exclusions
        self.overall_results_dict = dict()
        for exclusion in self.exclusions:
            overall_result = OverallResult(exclusion)
            key = self.get_exclusion_path_key(exclusion)
            self.overall_results_dict[key] = overall_result

    def get_exclusion_path_key(self, exclusion_path: List[str]):
        return ",".join(exclusion_path)

    def add_counts(self,
                   exclusion_path: List[str],
                   fuzzy_count_matches: int,
                   exact_count_matches: int,
                   all_fields_count: int):
        key = self.get_exclusion_path_key(exclusion_path)
        self.overall_results_dict[key].add_counts(fuzzy_count_matches=fuzzy_count_matches,
                                                  exact_count_matches=exact_count_matches,
                                                  all_fields_count=all_fields_count)

    def get_overall_results(self):
        results = list()
        for key in self.overall_results_dict:
            result = self.overall_results_dict[key].get_result()
            results.append(result)
        return results


class OverallResult:
    def __init__(self, path_exclude: List[str]):

        self.path_exclude_key = ",".join(path_exclude)

        self.overall_exact_score = 0
        self.overall_fuzzy_score = 0
        self.total_exact_matches = 0
        self.total_fuzzy_matches = 0
        self.total_exact_misses = 0
        self.total_fuzzy_misses = 0
        self.total_fields_count = 0

        self.fuzzy_count_matches = 0
        self.exact_count_matches = 0
        self.all_fields_count = 0

    def add_counts(self, fuzzy_count_matches: int, exact_count_matches: int, all_fields_count: int):
        self.fuzzy_count_matches += fuzzy_count_matches
        self.exact_count_matches += exact_count_matches
        self.all_fields_count += all_fields_count

    def get_result(self):
        overall_score_exact = self.exact_count_matches / self.all_fields_count if self.all_fields_count > 0 else 0
        overall_score_fuzzy = self.fuzzy_count_matches / self.all_fields_count if self.all_fields_count > 0 else 0

        return {
            'overall_path_exclude': self.path_exclude_key,
            'overall_exact_score': overall_score_exact,
            'overall_fuzzy_score': overall_score_fuzzy,
            'total_exact_matches': self.exact_count_matches,
            'total_fuzzy_matches': self.fuzzy_count_matches,
            'total_exact_misses': self.all_fields_count - self.exact_count_matches,
            'total_fuzzy_misses': self.all_fields_count - self.fuzzy_count_matches,
            'total_fields_count': self.all_fields_count
        }

