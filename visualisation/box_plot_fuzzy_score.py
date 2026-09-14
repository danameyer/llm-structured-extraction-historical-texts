import json
import re
from collections import defaultdict
from pathlib import Path
from matplotlib import pyplot as plt


class BoxPlotFuzzyScore:

    @staticmethod
    def extract_fuzzy_scores_by_prompt_and_model(models_directory: str | Path):
        models_directory = Path(models_directory)
        scores_by_prompt_and_model = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for prompt_type_path in sorted(models_directory.iterdir()):
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

                for result in data.get('individual_results', []):
                    exclude_paths = tuple(result.get('exclude_paths', []))
                    fuzzy_score = result.get('fuzzy_score')

                    if fuzzy_score is not None:
                        scores_by_prompt_and_model[prompt_type_path.name][model_path.name][exclude_paths].append(fuzzy_score)

        return scores_by_prompt_and_model

    @staticmethod
    def format_exclusion_label(exclude_paths: tuple[str, ...]) -> str:
        attributes = []

        for path in exclude_paths:
            attributes.extend(re.findall(r"\['(.*?)']", path))

        attributes = [attribute for attribute in attributes if attribute != 'id']

        if not attributes:
            return 'No exclusions'

        return ', '.join(attribute.replace('_', ' ').capitalize() for attribute in attributes)

    @staticmethod
    def plot_fuzzy_scores_by_prompt_and_model(scores_by_prompt_and_model):
        for prompt_type, models_scores in scores_by_prompt_and_model.items():
            for model_name, scores_by_exclude_paths in models_scores.items():
                exclude_paths = list(scores_by_exclude_paths.keys())
                labels = [BoxPlotFuzzyScore.format_exclusion_label(paths) for paths in exclude_paths]
                data = [scores_by_exclude_paths[paths] for paths in exclude_paths]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.boxplot(data, patch_artist=True, medianprops=dict(color='black'))
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel('Fuzzy Accuracy')
                ax.set_ylim(-0.1, 1.1)
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                output_dir = Path('output_boxplots') / prompt_type
                output_dir.mkdir(parents=True, exist_ok=True)
                plot_filename = f'{prompt_type}_{model_name}_fuzzy_accuracy.png'
                plt.savefig(output_dir / plot_filename, bbox_inches='tight')
                plt.close(fig)