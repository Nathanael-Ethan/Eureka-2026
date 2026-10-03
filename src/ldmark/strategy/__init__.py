"""
LDMARK Strategy Engine

Hardware-aware compilation strategy engine that determines feasible 
compression configurations from model characteristics, hardware constraints, 
and user requirements.

This is the DECISION LAYER - it determines WHAT SHOULD BE DONE, not HOW TO DO IT.

Does NOT perform compression.
Does NOT rank strategies.
Does NOT claim quality retention.
"""

from .models import (
    Backend,
    BackendCompatibility,
    CompilationStrategy,
    ConstraintConfig,
    FeasibilityState,
    HardwareContext,
    ModelContext,
    RuntimeMemoryFeasibility,
    StorageFeasibility,
    StrategySet,
    WeightRepresentation,
)

from .feasibility import (
    calculate_actual_storage_bytes,
    calculate_effective_bits_per_weight,
    calculate_runtime_memory,
    calculate_storage_bytes,
    build_strategy,
    evaluate_backend_compatibility,
    evaluate_runtime_feasibility,
    evaluate_storage_feasibility,
    get_available_representations,
    KNOWN_FORMATS,
    BACKEND_KERNEL_AVAILABILITY,
)

from .engine import (
    StrategyEngine,
    create_engine,
    evaluate_strategies,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_BATCH_SIZE,
)

from .scenarios import (
    create_synthetic_model_analysis,
    create_synthetic_hardware_profile,
    run_scenario,
    run_all_scenarios,
    analyze_27b_q1_g128,
    verify_internal_consistency,
    SCENARIO_A,
    SCENARIO_B,
    SCENARIO_C,
    SCENARIO_D,
    SCENARIO_E,
    SCENARIO_27B_Q1_G128,
    ALL_SCENARIOS,
)

__all__ = [
    "Backend",
    "BackendCompatibility",
    "CompilationStrategy",
    "ConstraintConfig",
    "FeasibilityState",
    "HardwareContext",
    "ModelContext",
    "RuntimeMemoryFeasibility",
    "StorageFeasibility",
    "StrategySet",
    "WeightRepresentation",
    "calculate_actual_storage_bytes",
    "calculate_effective_bits_per_weight",
    "calculate_runtime_memory",
    "calculate_storage_bytes",
    "build_strategy",
    "evaluate_backend_compatibility",
    "evaluate_runtime_feasibility",
    "evaluate_storage_feasibility",
    "get_available_representations",
    "KNOWN_FORMATS",
    "BACKEND_KERNEL_AVAILABILITY",
    "StrategyEngine",
    "create_engine",
    "evaluate_strategies",
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_BATCH_SIZE",
    "create_synthetic_model_analysis",
    "create_synthetic_hardware_profile",
    "run_scenario",
    "run_all_scenarios",
    "analyze_27b_q1_g128",
    "verify_internal_consistency",
    "SCENARIO_A",
    "SCENARIO_B",
    "SCENARIO_C",
    "SCENARIO_D",
    "SCENARIO_E",
    "SCENARIO_27B_Q1_G128",
    "ALL_SCENARIOS",
]

__version__ = "0.1.0"