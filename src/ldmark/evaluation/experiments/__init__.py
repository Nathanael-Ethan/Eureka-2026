"""
LDMARK Evaluation - Experiments

Controlled experiments for comparing compression methods on model behavior.
"""

from .compression_comparison import (
    ExperimentConfig,
    ExperimentResult,
    run_compression_comparison_experiment,
    run_predefined_experiments,
    print_experiment_summary,
    save_experiment_results,
    generate_markdown_report,
)

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "run_compression_comparison_experiment",
    "run_predefined_experiments",
    "print_experiment_summary",
    "save_experiment_results",
    "generate_markdown_report",
]