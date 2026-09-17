from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from visualisation.utils import format_model_name


class ReasoningPlot:

    @staticmethod
    def plot_accuracy(
            model_dirs: list[str],
            baseline_scores: list[float],
            reasoning_scores: list[float],
            score_name: str,
            output_name: str,
    ):
        x = np.arange(len(model_dirs))
        bar_width = 0.35
        fig, ax = plt.subplots()
        ax.bar(
            x - bar_width / 2,
            baseline_scores,
            bar_width,
            label="Baseline",
            zorder=3
        )

        ax.bar(
            x + bar_width / 2,
            reasoning_scores,
            bar_width,
            label="Reasoning enabled",
            zorder=3
        )

        display_model_names = [format_model_name(model_dir) for model_dir in model_dirs]
        ax.set_xlabel("Models")
        ax.set_ylabel(score_name)
        ax.set_xticks(x)
        ax.set_xticklabels(display_model_names, rotation=45, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.7, zorder=0)
        plt.tight_layout()
        output_dir = Path("output_reasoning_plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{output_name}.png", bbox_inches="tight")
        plt.close(fig)