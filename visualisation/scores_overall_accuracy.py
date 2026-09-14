import json
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class OverallAccuracy:

    @staticmethod
    def extract_overall_accuracy(models_directory, models_list):
        model_names = []
        exact_scores = []
        fuzzy_scores = []

        for model_folder in models_list:
            model_directory = Path(models_directory) / model_folder

            if not os.path.isdir(model_directory):
                print(f"No model directory found for {model_folder}")
                continue

            json_file_path = os.path.join(model_directory, 'scores_json', 'results.json')

            if not os.path.exists(json_file_path):
                print(f"No results.json found for {model_folder}")
                continue

            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            overall_results = data.get('overall_results', [])

            for result in overall_results:
                if result.get('overall_path_exclude') == "root['id']":
                    model_names.append(model_folder)
                    exact_scores.append(result['overall_exact_score'])
                    fuzzy_scores.append(result['overall_fuzzy_score'])
                    break

        return model_names, exact_scores, fuzzy_scores

    @staticmethod
    def plot_overall_accuracy(model_names, exact_scores, fuzzy_scores, prompt_name):
        x = np.arange(len(model_names))
        bar_width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x - bar_width / 2, exact_scores, bar_width, label='Exact Accuracy', zorder=3)
        ax.bar(x + bar_width / 2, fuzzy_scores, bar_width, label='Fuzzy Accuracy', zorder=3)
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(x)

        truncated_model_names = [
            re.sub(r'^model_(\w+):([a-zA-Z0-9\-.]+)-\d{4}-\d{2}-\d{2}.*', r'\2-\1', name)
            if ':' in name else
            re.sub(r'^model_([a-zA-Z0-9\-.]+).*', r'\1', name)
            for name in model_names
        ]

        ax.set_xticklabels(truncated_model_names, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        output_dir = os.path.join('output_overall_plots')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f'overall_scores_{prompt_name}.png'),
            bbox_inches='tight'
        )

        plt.close(fig)