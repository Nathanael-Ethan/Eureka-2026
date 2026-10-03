"""
LDMARK Strategy Engine - Main Engine

The decision-making layer that evaluates all feasible compilation strategies
given a model, hardware profile, and user constraints.

Does NOT perform compression. Does NOT rank strategies. Does NOT claim quality retention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.ldmark.analysis.models import ModelAnalysis
from src.ldmark.hardware.profile import HardwareProfile
from src.ldmark.strategy.models import (
    Backend,
    CompilationStrategy,
    ConstraintConfig,
    HardwareContext,
    ModelContext,
    StrategySet,
)
from src.ldmark.strategy.feasibility import (
    build_strategy,
    get_available_representations,
)


DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_BATCH_SIZE = 1


class StrategyEngine:
    """
    Main strategy engine for evaluating compilation strategies.
    
    Receives ModelAnalysis + HardwareProfile + ConstraintConfig
    Produces StrategySet with factual feasibility for all evaluated configurations.
    """
    
    def __init__(
        self,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        include_ldmark_formats: bool = True,
    ):
        self.context_length = context_length
        self.batch_size = batch_size
        self.include_ldmark_formats = include_ldmark_formats
    
    def evaluate(
        self,
        analysis: ModelAnalysis,
        hardware: HardwareProfile,
        constraints: Optional[ConstraintConfig] = None,
        context_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> StrategySet:
        """
        Evaluate all feasible compilation strategies.
        
        Args:
            analysis: ModelAnalysis with parameter counts and architecture info.
            hardware: HardwareProfile of the target system.
            constraints: Optional user constraints.
            context_length: Override default context length for KV cache estimation.
            batch_size: Override default batch size for runtime estimation.
            
        Returns:
            StrategySet with all evaluated strategies and their feasibility.
        """
        constraints = constraints or ConstraintConfig()
        context_length = context_length or self.context_length
        batch_size = batch_size or self.batch_size
        
        model_context = ModelContext.from_model_analysis(analysis)
        hardware_context = HardwareContext.from_hardware_profile(hardware)
        
        representations = get_available_representations(constraints.preferred_precision)
        
        if not self.include_ldmark_formats:
            representations = [r for r in representations if not r.value.startswith("ldmark_")]
        
        strategies = []
        for i, representation in enumerate(representations):
            strategy_id = f"{representation.value}_{i}"
            strategy = build_strategy(
                strategy_id=strategy_id,
                representation=representation,
                model_context=model_context,
                hardware_context=hardware_context,
                constraints=constraints,
                context_length=context_length,
                batch_size=batch_size,
            )
            strategies.append(strategy)
        
        feasible = [s for s in strategies if s.is_fully_feasible]
        storage_only = [s for s in strategies if s.is_storage_feasible and not s.is_runtime_feasible]
        infeasible = [s for s in strategies if not s.is_storage_feasible]
        
        notes = []
        if feasible:
            notes.append(f"Feasible strategies (storage + runtime): {len(feasible)}")
            for s in feasible:
                notes.append(f"  - {s.weight_representation.value}: {s.bits_per_weight:.4f} bits/weight, {s.storage_feasibility.estimated_storage_gb:.4f} GB storage, {s.runtime_feasibility.estimated_total_gb:.4f} GB runtime")
        else:
            notes.append("No strategies are feasible under current constraints (both storage and runtime).")
        
        if storage_only:
            notes.append(f"Storage-feasible but runtime-infeasible: {len(storage_only)}")
            for s in storage_only:
                notes.append(f"  - {s.weight_representation.value}: {s.runtime_feasibility.explanation}")
        
        if infeasible:
            notes.append(f"Storage-infeasible: {len(infeasible)}")
        
        notes.append("Feasibility is based on storage and memory estimates only. Quality impact is not estimated.")
        notes.append("Runtime memory estimates depend on context length and architecture assumptions.")
        notes.append("Backend compatibility does not guarantee kernel availability; indicates known support only.")
        
        return StrategySet(
            model_id=analysis.model_id,
            parameter_count=analysis.parameter_counts.total,
            baseline_dtype="fp32",
            target_hardware=hardware.cpu.model,
            strategies=strategies,
            constraints_used=constraints.to_dict(),
            context_length=context_length,
            batch_size=batch_size,
            timestamp=datetime.now().isoformat(),
            notes=notes,
        )
    
    def evaluate_at_context_lengths(
        self,
        analysis: ModelAnalysis,
        hardware: HardwareProfile,
        constraints: Optional[ConstraintConfig] = None,
        context_lengths: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[int, StrategySet]:
        """
        Evaluate strategies at multiple context lengths.
        
        Explicitly models that runtime memory depends on context length.
        """
        context_lengths = context_lengths or [4096, 16384, 32768, 100000]
        results = {}
        
        for ctx_len in context_lengths:
            results[ctx_len] = self.evaluate(
                analysis=analysis,
                hardware=hardware,
                constraints=constraints,
                context_length=ctx_len,
                batch_size=batch_size,
            )
        
        return results


def create_engine(
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_ldmark_formats: bool = True,
) -> StrategyEngine:
    """Factory function to create a StrategyEngine."""
    return StrategyEngine(
        context_length=context_length,
        batch_size=batch_size,
        include_ldmark_formats=include_ldmark_formats,
    )


def evaluate_strategies(
    analysis: ModelAnalysis,
    hardware: HardwareProfile,
    constraints: Optional[ConstraintConfig] = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_ldmark_formats: bool = True,
) -> StrategySet:
    """
    Convenience function to evaluate strategies.
    
    Args:
        analysis: ModelAnalysis with parameter counts and architecture info.
        hardware: HardwareProfile of the target system.
        constraints: Optional user constraints.
        context_length: Context length for KV cache estimation.
        batch_size: Batch size for runtime estimation.
        include_ldmark_formats: Whether to include LDMARK-specific formats.
        
    Returns:
        StrategySet with all evaluated strategies.
    """
    engine = StrategyEngine(
        context_length=context_length,
        batch_size=batch_size,
        include_ldmark_formats=include_ldmark_formats,
    )
    return engine.evaluate(analysis, hardware, constraints)