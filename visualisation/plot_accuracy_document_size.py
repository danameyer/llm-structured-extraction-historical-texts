import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt


class DataPoint:
    def __init__(self, document_name, document_size, fuzzy_score):
        self.document_name = document_name
        self.document_size = document_size
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
        key = self._create_key(llm_model, prompt_name)

        if key in self.data_point_sets:
            return self.data_point_sets[key]

        return None

    def add_data_point(self, data_point: DataPoint, llm_model: str, prompt_name: str):
        data_point_set = self._find_data_point_set(llm_model, prompt_name)

        if data_point_set:
            data_point_set.add_data_point(data_point)
        else:
            data_point_set = DataPointSet(llm_model, prompt_name)
            data_point_set.add_data_point(data_point)

            key = self._create_key(llm_model, prompt_name)
            self.data_point_sets[key] = data_point_set


class ScatterPlot:

    @staticmethod
    def extract_fuzzy_scores_and_field_counts(results_directory: str | Path, desired_exclusion_path: List[str]):
        results_directory = Path(results_directory)
        data_point_library = DataPointSetLibrary()

        for prompt_type_path in sorted(results_directory.iterdir()):
            if not prompt_type_path.is_dir():
                continue

            for model_path in sorted(prompt_type_path.iterdir()):
                if not model_path.is_dir() or not model_path.name.startswith('model_'):
                    continue

                json_file_path = model_path/ 'scores_json' / 'results.json'

                if not json_file_path.exists():
                    print(f"No results.json found for {model_path.name}")
                    continue

                with open(json_file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                if not isinstance(data.get('individual_results'), list):
                    continue

                for document in data['individual_results']:
                    document_name = document.get('file_name1')
                    fuzzy_score = document.get('fuzzy_score')
                    total_field_count = document.get('total_field_count')
                    exclude_path = document.get('exclude_paths')

                    if (
                            document_name is not None
                            and fuzzy_score is not None
                            and total_field_count is not None
                            and exclude_path == desired_exclusion_path
                    ):
                        data_point = DataPoint(document_name, total_field_count, fuzzy_score)
                        data_point_library.add_data_point(
                            data_point=data_point,
                            llm_model=model_path.name,
                            prompt_name=prompt_type_path.name,
                        )

        return data_point_library

    @staticmethod
    def extract_fuzzy_accuracy_and_text_length(
            results_directory: str | Path,
            text_directory: str | Path,
            desired_exclusion_path: List[str],
    ):
        results_directory = Path(results_directory)
        text_directory = Path(text_directory)
        data_point_library = DataPointSetLibrary()

        for prompt_type_path in sorted(results_directory.iterdir()):
            if not prompt_type_path.is_dir():
                continue

            for model_path in sorted(prompt_type_path.iterdir()):
                if not model_path.is_dir() or not model_path.name.startswith('model_'):
                    continue

                json_file_path = model_path / 'scores_json' / 'results.json'

                if not json_file_path.exists():
                    print(f"No results.json found for {model_path.name}")
                    continue

                with open(json_file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                if not isinstance(data.get('individual_results'), list):
                    continue

                for document in data['individual_results']:
                    document_name = document.get('file_name1')
                    fuzzy_score = document.get('fuzzy_score')
                    exclude_path = document.get('exclude_paths')

                    if (
                            document_name is None
                            or fuzzy_score is None
                            or exclude_path != desired_exclusion_path
                    ):
                        continue

                    base_document_name = Path(document_name).stem
                    txt_file_path = text_directory / f'{base_document_name}.txt'

                    if not txt_file_path.exists():
                        print(f"No source text found for {document_name}")
                        continue

                    with open(txt_file_path, 'r', encoding='utf-8') as file:
                        txt_content = file.read()

                    total_text_length = len(txt_content.split())
                    data_point = DataPoint(document_name, total_text_length, fuzzy_score)

                    data_point_library.add_data_point(
                        data_point=data_point,
                        llm_model=model_path.name,
                        prompt_name=prompt_type_path.name,
                    )

        return data_point_library

    @staticmethod
    def plot_fuzzy_accuracy_vs_field_count(data_point_set):
        x_values = [data_point.document_size for data_point in data_point_set.data_points]
        y_values = [data_point.fuzzy_score for data_point in data_point_set.data_points]
        model_name = data_point_set.llm_model
        prompt_name = data_point_set.prompt_name
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_values, y_values, alpha=0.7)
        ax.set_xlabel('Total JSON Field Count')
        ax.set_ylabel('Fuzzy Accuracy')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        plt.tight_layout()
        output_dir = Path('output_scatter_plots_json_size')
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            output_dir / f'accuracy_vs_json_size_{model_name}_{prompt_name}.png',
            bbox_inches='tight'
        )
        plt.close(fig)

    @staticmethod
    def plot_fuzzy_accuracy_vs_text_length(data_point_set):
        x_values = [data_point.document_size for data_point in data_point_set.data_points]
        y_values = [data_point.fuzzy_score for data_point in data_point_set.data_points]
        model_name = data_point_set.llm_model
        prompt_name = data_point_set.prompt_name
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_values, y_values, alpha=0.7)
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Fuzzy Accuracy')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        plt.tight_layout()
        output_dir = Path('output_scatter_plots_text_length')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_dir / f'accuracy_vs_text_length_{model_name}_{prompt_name}.png',
            bbox_inches='tight'
        )
        plt.close(fig)