import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
from collections import defaultdict

from dotenv import load_dotenv


class DataPoint:
    def __init__(self, document_name, total_field_count, fuzzy_score):
        self.document_name = document_name
        self.total_field_count = total_field_count
        self.fuzzy_score = fuzzy_score


class DataPointSet:
    def __init__(self, llm_model: str, prompt_name: str):
        self.llm_model = llm_model
        self.prompt_name = prompt_name
        self.data_points = list()

    def add_data_point(self, data_point: DataPoint):
        self.data_points.append(data_point)


class DataPointSetLibrary:
    def __init__(self):
        self.data_point_sets: Dict[str, DataPointSet] = dict()

    @staticmethod
    def _create_key(llm_model: str, prompt_name: str):
        return f"{llm_model}_{prompt_name}"

    def _find_data_point_set(self, llm_model: str, prompt_name: str):
        _key = self._create_key(llm_model, prompt_name)
        if _key in self.data_point_sets.keys():
            return self.data_point_sets[_key]
        else:
            return None

    def add_data_point(self, data_point: DataPoint, llm_model: str, prompt_name: str):
        data_point_set: DataPointSet = self._find_data_point_set(llm_model, prompt_name)
        if data_point_set:
            data_point_set.add_data_point(data_point)
        else:
            data_point_set = DataPointSet(llm_model, prompt_name)
            data_point_set.add_data_point(data_point)
            key = self._create_key(llm_model, prompt_name)
            self.data_point_sets[key] = data_point_set


class JsonSizeScatterPlot:

    @staticmethod
    def extract_fuzzy_scores_and_field_counts(results_directory, desired_exclusion_path: List[str]):
        # Dictionary to store data for each path exclusion
        path_exclusion_data = defaultdict(lambda: {"fuzzy_scores": [], "total_field_counts": []})

        data_point_library: DataPointSetLibrary = DataPointSetLibrary()

        for prompt_type_folder in os.listdir(results_directory):
            prompt_type_path = os.path.join(results_directory, prompt_type_folder)
            if os.path.isdir(prompt_type_path):
                # Iterate through each model directory in the prompt type directory
                for model_folder in os.listdir(prompt_type_path):
                    model_path = os.path.join(prompt_type_path, model_folder)
                    if os.path.isdir(model_path) and model_folder.startswith('model_'):
                        scores_json_dir = os.path.join(model_path, 'scores_json')
                        if os.path.exists(scores_json_dir):
                            json_files = [file for file in os.listdir(scores_json_dir) if file.endswith('.json')]
                            json_files.sort()

                            if json_files:
                                json_file_path = os.path.join(scores_json_dir, json_files[-1])
                                with open(json_file_path, 'r') as file:
                                    data = json.load(file)

                        # Check if "individual_results" is a list
                        if isinstance(data.get('individual_results'), list):
                            for document in data['individual_results']:
                                document_name = document.get('file_name1', 'Unnamed Document')
                                fuzzy_score = document.get('fuzzy_score')
                                if fuzzy_score is not None:
                                    total_field_count = document['total_field_count']
                                    exclude_path = document.get('exclude_paths')

                                    if exclude_path == desired_exclusion_path:
                                        data_point = DataPoint(document_name, total_field_count, fuzzy_score)
                                        data_point_library.add_data_point(data_point=data_point,
                                                                          llm_model=model_folder,
                                                                          prompt_name=prompt_type_folder)

        return data_point_library

    @staticmethod
    def plot_fuzzy_accuracy_vs_field_count(data_point_set: DataPointSet):
        x_values = [x.total_field_count for x in data_point_set.data_points]
        y_values = [y.fuzzy_score for y in data_point_set.data_points]

        model_name = data_point_set.llm_model
        prompt_name = data_point_set.prompt_name

        title = f'''Fuzzy Accuracy vs. Total Field Count
        Exclusion Path: "root[\'id\']"
        Model: "{model_name}"
        Prompt Name: "{prompt_name}"'''

        plt.title(title)
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, color='blue', alpha=0.7)
        plt.xlabel('Total Field Count')
        plt.ylabel('Fuzzy Accuracy')
        plt.grid(True)
        output_dir = os.path.join('output_scatter_plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'overall_scores_{model_name}_{prompt_name}.png'))
        plt.close()


if __name__ == '__main__':
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    models_dir = os.path.join(base_dir, 'evaluation_results', 'results')

    jsonSizeScatterPlot = JsonSizeScatterPlot()

    # Extract error counts
    data_point_set_library = jsonSizeScatterPlot.extract_fuzzy_scores_and_field_counts(models_dir, ["root['id']"])

    # Plot error counts for each model and prompt type
    for key in data_point_set_library.data_point_sets:
        data_points_for_model_and_prompt = data_point_set_library.data_point_sets[key]
        jsonSizeScatterPlot.plot_fuzzy_accuracy_vs_field_count(data_points_for_model_and_prompt)
