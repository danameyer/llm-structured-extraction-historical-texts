import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from visualisation.utils import format_model_name


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
        display_model_names = [format_model_name(model) for model in model_names]
        ax.set_xticklabels(display_model_names, rotation=45, ha='right')
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

    @staticmethod
    def plot_prompt_selection_comparison(scores_by_prompt, model_dirs):
        prompt_names = list(scores_by_prompt.keys())

        display_prompt_names = [
            prompt_name
            .replace("_", " ")
            .replace("chain of thought", "Chain of thought")
            .replace("all principles zero shot", "Zero-shot")
            .replace("all principles few shot", "Few-shot")
            .replace("no principles base prompt", "Base prompt")
            for prompt_name in prompt_names
        ]

        display_model_names = [format_model_name(model) for model in model_dirs]

        for score_type, ylabel in [("exact", "Exact Accuracy"), ("fuzzy", "Fuzzy Accuracy")]:
            x = np.arange(len(prompt_names))
            bar_width = 0.8 / len(model_dirs)
            fig, ax = plt.subplots()

            for model_index, model_dir in enumerate(model_dirs):
                scores = [
                    scores_by_prompt[prompt_name]
                    .get(model_dir, {})
                    .get(score_type, np.nan)
                    for prompt_name in prompt_names
                ]

                offset = (model_index - (len(model_dirs) - 1) / 2) * bar_width
                ax.bar(
                    x + offset,
                    scores,
                    bar_width,
                    label=display_model_names[model_index],
                    zorder=3,
                )

            ax.set_xlabel("Prompt")
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(display_prompt_names, rotation=25, ha="right")
            ax.set_ylim(0.0, 1.0)
            ax.legend()
            ax.grid(True, axis="y", linestyle="--", alpha=0.7, zorder=0)
            plt.tight_layout()
            output_dir = Path("output_overall_plots")
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / f"prompt_selection_{score_type}_accuracy.png", bbox_inches="tight")
            plt.close(fig)