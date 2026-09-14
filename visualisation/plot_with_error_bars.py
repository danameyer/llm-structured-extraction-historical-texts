import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class PlotWithErrorBars:

    @staticmethod
    def extract_accuracy(models_directory: str | Path, model_list: list[str]):
        models_directory = Path(models_directory)
        exact_scores = {model: [] for model in model_list}
        fuzzy_scores = {model: [] for model in model_list}

        for model_folder in model_list:
            json_file_path = models_directory / model_folder / 'scores_json' / 'results.json'

            if not json_file_path.exists():
                print(f"No results.json found for {model_folder}")
                continue

            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            overall_results = data.get('overall_results', [])

            for result in overall_results:
                if result.get('overall_path_exclude') == "root['id']":
                    exact_scores[model_folder].append(result['overall_exact_score'])
                    fuzzy_scores[model_folder].append(result['overall_fuzzy_score'])
                    break

        return exact_scores, fuzzy_scores

    @staticmethod
    def compute_mean_and_std(exact_scores, fuzzy_scores):
        exact_mean = {}
        exact_std = {}
        fuzzy_mean = {}
        fuzzy_std = {}

        for model, scores in exact_scores.items():
            if not scores:
                raise ValueError(f"No exact scores found for {model}")

            exact_mean[model] = np.mean(scores)
            exact_std[model] = np.std(scores)

        for model, scores in fuzzy_scores.items():
            if not scores:
                raise ValueError(f"No fuzzy scores found for {model}")

            fuzzy_mean[model] = np.mean(scores)
            fuzzy_std[model] = np.std(scores)

        return exact_mean, exact_std, fuzzy_mean, fuzzy_std

    @staticmethod
    def plot_overall_accuracy(model_names, exact_mean, fuzzy_mean, exact_std, fuzzy_std, prompt_name):
        x = np.arange(len(model_names))
        bar_width = 0.35
        fig, ax = plt.subplots()
        exact_mean_values = [exact_mean[model] for model in model_names]
        fuzzy_mean_values = [fuzzy_mean[model] for model in model_names]
        exact_std_values = [exact_std[model] for model in model_names]
        fuzzy_std_values = [fuzzy_std[model] for model in model_names]
        ax.bar(x - bar_width / 2, exact_mean_values, bar_width, yerr=exact_std_values,
               label='Exact Accuracy', zorder=3, capsize=5)
        ax.bar(x + bar_width / 2, fuzzy_mean_values, bar_width, yerr=fuzzy_std_values,
               label='Fuzzy Accuracy', zorder=3, capsize=5)
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(x)
        truncated_model_names = [name.replace("model_", "") for name in model_names]
        ax.set_xticklabels(truncated_model_names, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
        plt.ylim(0.0, 1.0)
        output_dir = Path('output_error_bar_plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_dir / f'overall_scores_error_bars_{prompt_name}.png',
            bbox_inches='tight'
        )
        plt.close(fig)