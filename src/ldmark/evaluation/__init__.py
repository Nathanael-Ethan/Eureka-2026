"""
LDMARK Evaluation

Experimental framework for evaluating how LDMARK compression affects model behavior.
This is NOT a benchmark for comparing against other methods or claiming universal quality.
It measures specific, reproducible behavioral differences on a specific model and evaluation suite.
"""

from .models import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationStatus,
    PromptResult,
    CompressionComparisonResult,
    TokenDifference,
    LogitMetrics,
    TaskType,
    ModelSource,
    HardwareInfo,
    SoftwareVersions,
    MetricResult,
)
from .prompts import (
    EvaluationPrompt,
    PromptCategory,
    get_all_prompts,
    get_prompts_by_category,
    get_prompt_suite_summary,
    print_prompt_suite,
    FACTUAL_RECALL_PROMPTS,
    SIMPLE_REASONING_PROMPTS,
    ARITHMETIC_PROMPTS,
    CODE_COMPLETION_PROMPTS,
    INSTRUCTION_FOLLOWING_PROMPTS,
)
from .engine import (
    EvaluationEngine,
    ModelBackend,
    SyntheticModelBackend,
    compare_tokens,
    compute_logit_metrics,
    calculate_perplexity,
    run_synthetic_evaluation,
)
from .reporting import (
    ReportGenerator,
    generate_comparison_table,
    generate_prompt_detail_table,
    generate_full_report,
    save_report,
)
from .experiments.compression_comparison import (
    ExperimentConfig,
    ExperimentResult,
    run_compression_comparison_experiment,
    run_predefined_experiments,
    print_experiment_summary,
    save_experiment_results,
    generate_markdown_report,
)

__all__ = [
    # Models
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationStatus",
    "PromptResult",
    "CompressionComparisonResult",
    "TokenDifference",
    "LogitMetrics",
    "TaskType",
    "PromptCategory",
    "ModelSource",
    "HardwareInfo",
    "SoftwareVersions",
    "MetricResult",
    # Prompts
    "EvaluationPrompt",
    "PromptCategory",
    "get_all_prompts",
    "get_prompts_by_category",
    "get_prompt_suite_summary",
    "print_prompt_suite",
    "FACTUAL_RECALL_PROMPTS",
    "SIMPLE_REASONING_PROMPTS",
    "ARITHMETIC_PROMPTS",
    "CODE_COMPLETION_PROMPTS",
    "INSTRUCTION_FOLLOWING_PROMPTS",
    # Engine
    "EvaluationEngine",
    "ModelBackend",
    "SyntheticModelBackend",
    "compare_tokens",
    "compute_logit_metrics",
    "calculate_perplexity",
    "run_synthetic_evaluation",
    # Reporting
    "ReportGenerator",
    "generate_comparison_table",
    "generate_prompt_detail_table",
    "generate_full_report",
    "save_report",
    # Experiments
    "ExperimentConfig",
    "ExperimentResult",
    "run_compression_comparison_experiment",
    "run_predefined_experiments",
    "print_experiment_summary",
    "save_experiment_results",
    "generate_markdown_report",
]

__version__ = "0.1.0"