import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class ExcludePathsPlot:

    @staticmethod
    def extract_and_plot_all(results_directory: str | Path):
        results_directory = Path(results_directory)

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

                overall_results = data.get('overall_results', [])

                if not overall_results:
                    print(f"No overall results found for {model_path.name}")
                    continue

                exclude_paths = []
                exact_scores = []
                fuzzy_scores = []

                for result in overall_results:
                    exclude_paths.append(result['overall_path_exclude'])
                    exact_scores.append(result['overall_exact_score'])
                    fuzzy_scores.append(result['overall_fuzzy_score'])

                ExcludePathsPlot.plot_scores(
                    prompt_type_path.name,
                    model_path.name,
                    exclude_paths,
                    exact_scores,
                    fuzzy_scores
                )

    @staticmethod
    def format_exclusion_label(exclude_path: str) -> str:
        attributes = re.findall(r"\['(.*?)']", exclude_path)
        attributes = [attribute for attribute in attributes if attribute != 'id']

        if not attributes:
            return 'No exclusions'

        return ', '.join(attribute.replace('_', ' ').capitalize() for attribute in attributes)

    @staticmethod
    def plot_scores(prompt_type, model_name, exclude_paths, exact_scores, fuzzy_scores):
        exclusion_labels = [
            ExcludePathsPlot.format_exclusion_label(exclude_path)
            for exclude_path in exclude_paths
        ]

        x = np.arange(len(exclude_paths))
        bar_width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x - bar_width / 2, exact_scores, bar_width, label='Exact Accuracy', zorder=3)
        ax.bar(x + bar_width / 2, fuzzy_scores, bar_width, label='Fuzzy Accuracy', zorder=3)
        ax.set_xlabel('Excluded Attributes')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(exclusion_labels, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        ax.set_ylim(0.0, 1.0)
        plt.tight_layout()
        output_dir = Path('output_plots_attributes_excluded') / prompt_type
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'{model_name}_scores.png', bbox_inches='tight')
        plt.close(fig)