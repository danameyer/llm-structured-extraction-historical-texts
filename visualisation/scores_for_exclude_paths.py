import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv


class ExcludePathsPlot:
    def __init__(self):
        pass

    @staticmethod
    def extract_and_plot_all(models_directory):
        for prompt_type_folder in os.listdir(models_directory):
            prompt_type_path = os.path.join(models_directory, prompt_type_folder)
            if os.path.isdir(prompt_type_path):
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

                                if isinstance(data['overall_results'], list):
                                    exclude_paths = []
                                    exact_scores = []
                                    fuzzy_scores = []

                                    for result in data['overall_results']:
                                        if isinstance(result, dict):
                                            exclude_paths.append(result['overall_path_exclude'])
                                            exact_scores.append(result['overall_exact_score'])
                                            fuzzy_scores.append(result['overall_fuzzy_score'])

                                    # Plot for the current model in the current prompt type
                                    ExcludePathsPlot.plot_scores(
                                        prompt_type_folder,
                                        model_folder,
                                        exclude_paths,
                                        exact_scores,
                                        fuzzy_scores
                                    )

    @staticmethod
    def plot_scores(prompt_type, model_name, exclude_paths, exact_scores, fuzzy_scores):
        extracted_attributes = [
            ', '.join([attr for path in paths.split(',') for attr in re.findall(r"\['(.*?)']", path)])
            for paths in exclude_paths
        ]
        x = np.arange(len(exclude_paths))
        bar_width = 0.35

        fig, ax = plt.subplots()
        ax.bar(x - bar_width / 2, exact_scores, bar_width, label='Exact Accuracy')
        ax.bar(x + bar_width / 2, fuzzy_scores, bar_width, label='Fuzzy Accuracy')

        ax.set_xlabel('Excluded Attributes')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Exact vs Fuzzy Accuracy for {model_name} in {prompt_type}')
        ax.set_xticks(x)
        ax.set_xticklabels(extracted_attributes, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        output_dir = os.path.join('output_plots_attributes_excluded', prompt_type)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{model_name}_scores.png'), bbox_inches='tight')

        plt.close()


if __name__ == '__main__':
    excludePathsPlot = ExcludePathsPlot()
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    models_dir = os.path.join(base_dir, 'evaluation_results', 'results')
    excludePathsPlot.extract_and_plot_all(models_dir)
