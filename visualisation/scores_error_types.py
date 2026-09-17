import json
import re
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from visualisation.utils import format_model_name


class ErrorTypePlot:

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
    def extract_and_count_errors(results_directory: str | Path):
        results_directory = Path(results_directory)
        all_error_counts = {}

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

                error_counts = defaultdict(lambda: defaultdict(int))

                for document in data.get('individual_results', []):
                    exclude_paths = tuple(document.get('exclude_paths', []))

                    if not exclude_paths:
                        continue

                    exclusion_label = ErrorTypePlot.format_exclusion_label(exclude_paths)

                    for comparison_result in document.get('deep_diff_comparison_results_fuzzy', []):
                        for error_type in [
                            'values_changed',
                            'iterable_item_added',
                            'iterable_item_removed',
                            'type_changes',
                        ]:
                            error_counts[exclusion_label][error_type] += comparison_result.get(error_type, 0)

                    error_counts[exclusion_label]['missing_person_fields'] += document.get('number_of_fields_of_missing_persons', 0)
                    error_counts[exclusion_label]['extra_person_fields'] += document.get('number_of_fields_of_extra_persons', 0)

                all_error_counts[(prompt_type_path.name, model_path.name)] = error_counts

        return all_error_counts

    @staticmethod
    def extract_and_count_total_errors_per_model_per_prompt_type(
            results_directory: str | Path,
            model_list: list[str],
            prompt_list: list[str]
    ):
        results_directory = Path(results_directory)
        error_counts_by_prompt_type = defaultdict(lambda: defaultdict(int))

        for prompt_type in prompt_list:
            prompt_type_path = results_directory / prompt_type

            if not prompt_type_path.is_dir():
                continue

            for model_name in model_list:
                json_file_path = prompt_type_path / model_name / 'scores_json' / 'results.json'

                if not json_file_path.exists():
                    print(f"No results.json found for {prompt_type}/{model_name}")
                    continue

                with open(json_file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                total_error_count = 0

                for document in data.get('individual_results', []):
                    if document.get('exclude_paths') != ["root['id']"]:
                        continue

                    total_error_count += document.get('fuzzy_differences', 0)

                error_counts_by_prompt_type[prompt_type][model_name] = total_error_count

        return dict(error_counts_by_prompt_type)

    @staticmethod
    def plot_error_counts(prompt_type, model_name, error_counts):
        error_types = [
            'values_changed',
            'iterable_item_added',
            'iterable_item_removed',
            'type_changes',
            'missing_person_fields',
            'extra_person_fields',
        ]

        error_type_labels = [
            'Values changed',
            'Items added',
            'Items removed',
            'Type changes',
            'Missing-person fields',
            'Extra-person fields',
        ]

        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.12
        num_error_types = len(error_types)
        num_combinations = len(error_counts)
        bar_positions = np.arange(num_combinations) * (num_error_types + 1) * bar_width

        for i, (_, scores) in enumerate(error_counts.items()):
            for j, (error_type, error_type_label) in enumerate(zip(error_types, error_type_labels)):
                count = scores.get(error_type, 0)

                ax.bar(
                    bar_positions[i] + j * bar_width,
                    count,
                    width=bar_width,
                    label=error_type_label if i == 0 else '',
                    zorder=3,
                )

        ax.set_xlabel('Excluded Attributes')
        ax.set_ylabel('Error Count')
        ax.set_xticks(bar_positions + (num_error_types / 2 - 0.5) * bar_width)
        ax.set_xticklabels(list(error_counts.keys()), rotation=45, ha='right')
        ax.legend(title='Error Type', loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        plt.tight_layout()
        output_dir = Path('output_plots_errors') / prompt_type
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'{model_name}_error_counts.png', bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_total_error_counts_by_prompt_type(error_counts_by_prompt_type):
        for prompt_type, error_counts_by_model in error_counts_by_prompt_type.items():
            models = list(error_counts_by_model.keys())
            total_errors = list(error_counts_by_model.values())

            if not models:
                print(f"No error counts to plot for prompt type: {prompt_type}")
                continue

            fig, ax = plt.subplots(figsize=(12, 8))
            bar_positions = np.arange(len(models))
            bar_width = 0.35
            ax.bar(bar_positions, total_errors, bar_width, zorder=3)
            ax.set_xlabel('Models')
            ax.set_ylabel('Total Fuzzy Error Count')
            ax.set_xticks(bar_positions)
            display_model_names = [format_model_name(model) for model in models]
            ax.set_xticklabels(display_model_names, rotation=45, ha='right')
            ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
            plt.tight_layout()
            output_dir = Path('output_plots_total_errors')
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_filename = f'total_error_counts_{prompt_type}.png'
            plt.savefig(output_dir / plot_filename, bbox_inches='tight')
            plt.close(fig)