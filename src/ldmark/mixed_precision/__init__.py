"""
LDMARK Mixed-Precision Compilation Engine

Experimental system for investigating mixed-precision strategies.
This is a STRATEGY AND EXPERIMENTATION module - no training, QAT, or inference.
"""

from .plan import (
    MixedPrecisionPlan,
    TensorPrecisionAssignment,
    PrecisionCandidate,
    TensorClassification,
    TensorRole,
    PrecisionType,
    create_mixed_precision_plan,
)
from .classification import (
    classify_tensor,
    classify_model_tensors,
    TensorClassifier,
    TensorMetadata,
)
from .storage import (
    MixedPrecisionStorageModel,
    StorageAccounting,
    calculate_mixed_precision_storage,
    StorageComponent,
)
from .error_analysis import (
    TensorErrorProfile,
    ErrorMeasurement,
    run_sensitivity_experiment,
    measure_tensor_errors,
)
from .search import (
    MixedPrecisionSearcher,
    SearchConstraint,
    SearchResult,
    Configuration,
    greedy_search,
)
from .scenarios import (
    ModelScaleScenario,
    ScenarioResult,
    run_scale_scenarios,
    estimate_parameter_distribution,
)
from .export import (
    export_plan_csv,
    export_plan_json,
    export_search_results_csv,
    export_scenario_results_csv,
)

__all__ = [
    # Plan
    "MixedPrecisionPlan",
    "TensorPrecisionAssignment",
    "PrecisionCandidate",
    "TensorClassification",
    "TensorRole",
    "PrecisionType",
    "create_mixed_precision_plan",
    # Classification
    "classify_tensor",
    "classify_model_tensors",
    "TensorClassifier",
    "TensorMetadata",
    # Storage
    "MixedPrecisionStorageModel",
    "StorageAccounting",
    "calculate_mixed_precision_storage",
    "StorageComponent",
    # Error Analysis
    "TensorErrorProfile",
    "ErrorMeasurement",
    "run_sensitivity_experiment",
    "measure_tensor_errors",
    # Search
    "MixedPrecisionSearcher",
    "SearchConstraint",
    "SearchResult",
    "Configuration",
    "greedy_search",
    # Scenarios
    "ModelScaleScenario",
    "ScenarioResult",
    "run_scale_scenarios",
    "estimate_parameter_distribution",
    # Export
    "export_plan_csv",
    "export_plan_json",
    "export_search_results_csv",
    "export_scenario_results_csv",
]