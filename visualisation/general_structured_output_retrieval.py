import json
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


class GeneralStructuredOutputRetrieval:

    def __init__(
            self,
            results_directory: str | Path,
            prompts: list[str],
            model: str,
    ):
        self.results_directory = Path(results_directory)
        self.prompts = prompts
        self.model = model

    def get_generation_successes(self) -> Dict[str, int]:
        generation_successes = {}

        for prompt in self.prompts:
            results = self._get_results_for_prompt(prompt)
            generation_summary = results.get('generation_summary', {})
            generation_successes[prompt] = generation_summary.get('successful_predictions', 0)

        return generation_successes

    def get_accuracy_scores(self) -> Dict[str, Dict[str, float]]:
        accuracy_scores = {}

        for prompt in self.prompts:
            results = self._get_results_for_prompt(prompt)
            overall_results = results.get('overall_results', [])

            baseline_result = next(
                (
                    result
                    for result in overall_results
                    if result.get('overall_path_exclude') == "root['id']"
                ),
                None,
            )

            if baseline_result is None:
                raise ValueError( f"No baseline overall result found for {prompt}" )

            accuracy_scores[prompt] = {
                'exact_score': baseline_result['overall_exact_score'],
                'fuzzy_score': baseline_result['overall_fuzzy_score'],
            }

        return accuracy_scores

    def _get_results_for_prompt(self, prompt: str) -> dict:
        results_file = self.results_directory / prompt / self.model / 'scores_json' / 'results.json'

        if not results_file.exists():
            raise FileNotFoundError(f"No results.json found for {prompt}/{self.model}")

        with open(results_file, 'r', encoding='utf-8') as file:
            return json.load(file)

    @staticmethod
    def format_prompt_label(prompt: str) -> str:
        labels = {
            'best_prompt_prompt_only_json': 'Prompt-only\nJSON',
            'best_prompt_direct_structured_outputs': 'Structured\noutputs JSON',
            'best_prompt_structured_outputs_disabled': 'Function calling\nstrict=False',
            'best_prompt_structured_outputs_enabled': 'Function calling\nstrict=True',
        }

        return labels.get(prompt, prompt.replace('_', ' '))

    @staticmethod
    def plot_successful_json_processings(successes_dict: Dict[str, int], y_max_scale: int | None = None) -> None:
        x = np.arange(len(successes_dict))
        bar_width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x, list(successes_dict.values()), bar_width, zorder=3)
        ax.set_xlabel('Structured Output Method')
        ax.set_ylabel('Successful Predictions')
        ax.set_xticks(x)

        labels = [GeneralStructuredOutputRetrieval.format_prompt_label(prompt) for prompt in successes_dict]
        ax.set_xticklabels(labels, rotation=45, ha='right')

        if y_max_scale is not None:
            ax.set_ylim(0, y_max_scale)

        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        plt.tight_layout()
        output_dir = Path('successful_json_processings')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'successful_json_processings.png', bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_accuracy(accuracy_dict: Dict[str, Dict[str, float]]):
        x = np.arange(len(accuracy_dict))
        bar_width = 0.35
        exact_scores = [accuracy['exact_score'] for accuracy in accuracy_dict.values()]
        fuzzy_scores = [accuracy['fuzzy_score'] for accuracy in accuracy_dict.values()]
        fig, ax = plt.subplots()
        ax.bar(
            x - bar_width / 2,
            exact_scores,
            bar_width,
            label='Exact Accuracy',
            zorder=3,
        )
        ax.bar(
            x + bar_width / 2,
            fuzzy_scores,
            bar_width,
            label='Fuzzy Accuracy',
            zorder=3,
        )
        ax.set_xlabel('Structured Output Method')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(x)

        labels = [
            GeneralStructuredOutputRetrieval.format_prompt_label(prompt) for prompt in accuracy_dict
        ]

        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        ax.set_ylim(0.0, 1.0)
        plt.tight_layout()
        output_dir = Path('mean_accuracy_different_processing_methods')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'overall_scores_different_processing_methods.png', bbox_inches='tight')
        plt.close(fig)