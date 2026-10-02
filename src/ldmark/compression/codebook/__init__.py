"""
Codebook-based quantization experimental module for LDMARK.

This module provides non-uniform quantization using learned or data-derived codebooks.
This is an EXPERIMENTAL research subsystem - no claims about model quality improvements.
"""

from .representation import (
    CodebookConfig,
    CodebookTensor,
    CodebookGroup,
    create_codebook_tensor,
    reconstruct_from_codebook,
)
from .generation import (
    generate_codebook_kmeans,
    initialize_codebook_uniform,
    initialize_codebook_percentile,
    initialize_codebook_random,
)
from .analysis import (
    analyze_weight_distribution,
    WeightDistributionStats,
    histogram_analysis,
)
from .storage import (
    calculate_codebook_storage,
    calculate_effective_bits_per_weight,
    StorageAccounting,
)
from .experiment import (
    CodebookExperimentConfig,
    CodebookExperimentResult,
    run_codebook_experiment,
    run_uniform_vs_codebook_experiment,
)

__all__ = [
    # Representation
    "CodebookConfig",
    "CodebookTensor",
    "CodebookGroup",
    "create_codebook_tensor",
    "reconstruct_from_codebook",
    # Generation
    "generate_codebook_kmeans",
    "initialize_codebook_uniform",
    "initialize_codebook_percentile",
    "initialize_codebook_random",
    # Analysis
    "analyze_weight_distribution",
    "WeightDistributionStats",
    "histogram_analysis",
    # Storage
    "calculate_codebook_storage",
    "calculate_effective_bits_per_weight",
    "StorageAccounting",
    # Experiment
    "CodebookExperimentConfig",
    "CodebookExperimentResult",
    "run_codebook_experiment",
    "run_uniform_vs_codebook_experiment",
]