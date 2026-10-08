"""
LDMARK Evaluation - Core Models

Structured objects for model quality and behavior evaluation.
This module defines the data structures for evaluation configuration, results, and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import platform
import sys


class TaskType(Enum):
    """Types of evaluation tasks."""
    NEXT_TOKEN_PREDICTION = "next_token_prediction"
    PERPLEXITY = "perplexity"
    EXACT_OUTPUT = "exact_output"
    PROMPT_RESPONSE = "prompt_response"
    CLASSIFICATION = "classification"
    LOGIT_COMPARISON = "logit_comparison"


class ModelSource(Enum):
    """Source of the model being evaluated."""
    HUGGINGFACE = "huggingface"
    LOCAL_PATH = "local_path"
    SYNTHETIC = "synthetic"
    LDMARK_ARTIFACT = "ldmark_artifact"
    UNKNOWN = "unknown"


class EvaluationStatus(Enum):
    """Status of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class HardwareInfo:
    """Hardware information for reproducibility."""
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_architecture: str = ""
    system_ram_gb: float = 0.0
    gpu_model: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    operating_system: str = ""
    python_version: str = ""
    
    @classmethod
    def detect(cls) -> "HardwareInfo":
        """Detect current hardware."""
        import psutil
        return cls(
            cpu_model=platform.processor() or "unknown",
            cpu_cores=psutil.cpu_count(logical=False) or 0,
            cpu_architecture=platform.machine(),
            system_ram_gb=round(psutil.virtual_memory().total / (1024**3), 2),
            operating_system=f"{platform.system()} {platform.release()}",
            python_version=sys.version.split()[0],
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_model": self.cpu_model,
            "cpu_cores": self.cpu_cores,
            "cpu_architecture": self.cpu_architecture,
            "system_ram_gb": self.system_ram_gb,
            "gpu_model": self.gpu_model,
            "gpu_vram_gb": self.gpu_vram_gb,
            "operating_system": self.operating_system,
            "python_version": self.python_version,
        }


@dataclass
class SoftwareVersions:
    """Software versions for reproducibility."""
    ldmark_version: str = "0.0.0"
    numpy_version: str = ""
    torch_version: Optional[str] = None
    transformers_version: Optional[str] = None
    
    @classmethod
    def detect(cls) -> "SoftwareVersions":
        """Detect current software versions."""
        import numpy as np
        versions = cls(
            numpy_version=np.__version__,
        )
        try:
            import torch
            versions.torch_version = torch.__version__
        except ImportError:
            pass
        try:
            import transformers
            versions.transformers_version = transformers.__version__
        except ImportError:
            pass
        try:
            from ldmark import __version__
            versions.ldmark_version = __version__
        except ImportError:
            pass
        return versions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ldmark_version": self.ldmark_version,
            "numpy_version": self.numpy_version,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
        }


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""
    # Model identification
    model_id: str
    model_source: ModelSource
    model_path: str
    baseline_model_id: str
    baseline_model_source: ModelSource
    baseline_model_path: str
    
    # Compression settings being tested
    compression_methods: List[str] = field(default_factory=list)
    
    # Evaluation settings
    task_types: List[TaskType] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)
    prompt_categories: List[str] = field(default_factory=list)
    
    # Generation settings (fixed for reproducibility)
    seed: int = 42
    temperature: float = 0.0
    max_tokens: int = 50
    top_k: int = 1
    top_p: float = 1.0
    
    # Runtime settings
    target_dtype: str = "float16"
    batch_size: int = 1
    
    # Logging
    save_outputs: bool = True
    save_logits: bool = False
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_source": self.model_source.value,
            "model_path": self.model_path,
            "baseline_model_id": self.baseline_model_id,
            "baseline_model_source": self.baseline_model_source.value,
            "baseline_model_path": self.baseline_model_path,
            "compression_methods": self.compression_methods,
            "task_types": [t.value for t in self.task_types],
            "prompts": self.prompts,
            "prompt_categories": self.prompt_categories,
            "seed": self.seed,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "target_dtype": self.target_dtype,
            "batch_size": self.batch_size,
            "save_outputs": self.save_outputs,
            "save_logits": self.save_logits,
            "verbose": self.verbose,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationConfig":
        return cls(
            model_id=data["model_id"],
            model_source=ModelSource(data["model_source"]),
            model_path=data["model_path"],
            baseline_model_id=data["baseline_model_id"],
            baseline_model_source=ModelSource(data["baseline_model_source"]),
            baseline_model_path=data["baseline_model_path"],
            compression_methods=data.get("compression_methods", []),
            task_types=[TaskType(t) for t in data.get("task_types", [])],
            prompts=data.get("prompts", []),
            prompt_categories=data.get("prompt_categories", []),
            seed=data.get("seed", 42),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 50),
            top_k=data.get("top_k", 1),
            top_p=data.get("top_p", 1.0),
            target_dtype=data.get("target_dtype", "float16"),
            batch_size=data.get("batch_size", 1),
            save_outputs=data.get("save_outputs", True),
            save_logits=data.get("save_logits", False),
            verbose=data.get("verbose", False),
        )


@dataclass
class TokenDifference:
    """Token-level difference between baseline and compressed output."""
    position: int
    baseline_token: str
    compressed_token: str
    baseline_token_id: int
    compressed_token_id: int
    match: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "baseline_token": self.baseline_token,
            "compressed_token": self.compressed_token,
            "baseline_token_id": self.baseline_token_id,
            "compressed_token_id": self.compressed_token_id,
            "match": self.match,
        }


@dataclass
class LogitMetrics:
    """Logit-level analysis metrics."""
    mean_absolute_difference: float = 0.0
    max_absolute_difference: float = 0.0
    cosine_similarity: float = 0.0
    top1_agreement: float = 0.0
    top5_agreement: float = 0.0
    top10_agreement: float = 0.0
    kl_divergence: float = 0.0
    js_divergence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_absolute_difference": self.mean_absolute_difference,
            "max_absolute_difference": self.max_absolute_difference,
            "cosine_similarity": self.cosine_similarity,
            "top1_agreement": self.top1_agreement,
            "top5_agreement": self.top5_agreement,
            "top10_agreement": self.top10_agreement,
            "kl_divergence": self.kl_divergence,
            "js_divergence": self.js_divergence,
        }


@dataclass
class PromptResult:
    """Result for a single prompt evaluation."""
    prompt: str
    category: str = ""
    task_type: TaskType = TaskType.PROMPT_RESPONSE
    
    # Baseline output
    baseline_output: str = ""
    baseline_tokens: List[int] = field(default_factory=list)
    baseline_logits: Optional[List[List[float]]] = None
    
    # Compressed output
    compressed_output: str = ""
    compressed_tokens: List[int] = field(default_factory=list)
    compressed_logits: Optional[List[List[float]]] = None
    
    # Comparisons
    output_match: bool = False
    token_differences: List[TokenDifference] = field(default_factory=list)
    token_match_rate: float = 0.0
    output_length_baseline: int = 0
    output_length_compressed: int = 0
    logit_metrics: Optional[LogitMetrics] = None
    perplexity_baseline: Optional[float] = None
    perplexity_compressed: Optional[float] = None
    
    # Metadata
    generation_time_baseline_ms: float = 0.0
    generation_time_compressed_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "category": self.category,
            "task_type": self.task_type.value,
            "baseline_output": self.baseline_output,
            "baseline_tokens": self.baseline_tokens,
            "compressed_output": self.compressed_output,
            "compressed_tokens": self.compressed_tokens,
            "output_match": self.output_match,
            "token_differences": [td.to_dict() for td in self.token_differences],
            "token_match_rate": self.token_match_rate,
            "output_length_baseline": self.output_length_baseline,
            "output_length_compressed": self.output_length_compressed,
            "logit_metrics": self.logit_metrics.to_dict() if self.logit_metrics else None,
            "perplexity_baseline": self.perplexity_baseline,
            "perplexity_compressed": self.perplexity_compressed,
            "generation_time_baseline_ms": self.generation_time_baseline_ms,
            "generation_time_compressed_ms": self.generation_time_compressed_ms,
            "error": self.error,
        }


@dataclass
class CompressionComparisonResult:
    """Results comparing baseline vs compressed for one compression method."""
    compression_method: str
    storage_reduction_ratio: float = 0.0
    compressed_size_bytes: int = 0
    baseline_size_bytes: int = 0
    runtime_memory_estimate_gb: float = 0.0
    
    # Aggregate behavioral metrics
    overall_token_match_rate: float = 0.0
    exact_output_match_rate: float = 0.0
    mean_logit_mae: float = 0.0
    mean_logit_cosine: float = 0.0
    top1_agreement: float = 0.0
    top5_agreement: float = 0.0
    mean_perplexity_ratio: float = 1.0
    
    # Per-prompt results
    prompt_results: List[PromptResult] = field(default_factory=list)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "compression_method": self.compression_method,
            "storage_reduction_ratio": self.storage_reduction_ratio,
            "compressed_size_bytes": self.compressed_size_bytes,
            "baseline_size_bytes": self.baseline_size_bytes,
            "runtime_memory_estimate_gb": self.runtime_memory_estimate_gb,
            "overall_token_match_rate": self.overall_token_match_rate,
            "exact_output_match_rate": self.exact_output_match_rate,
            "mean_logit_mae": self.mean_logit_mae,
            "mean_logit_cosine": self.mean_logit_cosine,
            "top1_agreement": self.top1_agreement,
            "top5_agreement": self.top5_agreement,
            "mean_perplexity_ratio": self.mean_perplexity_ratio,
            "prompt_results": [pr.to_dict() for pr in self.prompt_results],
            "errors": self.errors,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    config: EvaluationConfig
    hardware: HardwareInfo
    software: SoftwareVersions
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: EvaluationStatus = EvaluationStatus.PENDING
    
    # Results per compression method
    comparisons: List[CompressionComparisonResult] = field(default_factory=list)
    
    # Summary
    total_prompts: int = 0
    successful_prompts: int = 0
    failed_prompts: int = 0
    total_time_seconds: float = 0.0
    
    # Limitations and notes
    limitations: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "hardware": self.hardware.to_dict(),
            "software": self.software.to_dict(),
            "timestamp": self.timestamp,
            "status": self.status.value,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "total_prompts": self.total_prompts,
            "successful_prompts": self.successful_prompts,
            "failed_prompts": self.failed_prompts,
            "total_time_seconds": self.total_time_seconds,
            "limitations": self.limitations,
            "notes": self.notes,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_markdown(self) -> str:
        """Generate a markdown report."""
        lines = [
            f"# LDMARK Evaluation Report",
            f"",
            f"**Model:** {self.config.model_id} ({self.config.model_source.value})",
            f"**Baseline:** {self.config.baseline_model_id} ({self.config.baseline_model_source.value})",
            f"**Timestamp:** {self.timestamp}",
            f"**Status:** {self.status.value}",
            f"**Hardware:** {self.hardware.cpu_model}, {self.hardware.system_ram_gb}GB RAM",
            f"**Software:** LDMARK {self.software.ldmark_version}, NumPy {self.software.numpy_version}",
            f"**Seed:** {self.config.seed}, **Temperature:** {self.config.temperature}, **Max Tokens:** {self.config.max_tokens}",
            f"",
            f"## Summary",
            f"",
            f"- **Total Prompts:** {self.total_prompts}",
            f"- **Successful:** {self.successful_prompts}",
            f"- **Failed:** {self.failed_prompts}",
            f"- **Total Time:** {self.total_time_seconds:.2f}s",
            f"",
            f"## Results by Compression Method",
            f"",
        ]
        
        for comp in self.comparisons:
            lines.extend([
                f"### {comp.compression_method}",
                f"",
                f"- **Storage Reduction:** {comp.storage_reduction_ratio:.2f}x",
                f"- **Baseline Size:** {comp.baseline_size_bytes:,} bytes",
                f"- **Compressed Size:** {comp.compressed_size_bytes:,} bytes",
                f"- **Runtime Memory Estimate:** {comp.runtime_memory_estimate_gb:.2f} GB",
                f"- **Overall Token Match Rate:** {comp.overall_token_match_rate:.4f}",
                f"- **Exact Output Match Rate:** {comp.exact_output_match_rate:.4f}",
                f"- **Mean Logit MAE:** {comp.mean_logit_mae:.6f}",
                f"- **Mean Logit Cosine:** {comp.mean_logit_cosine:.6f}",
                f"- **Top-1 Agreement:** {comp.top1_agreement:.4f}",
                f"- **Top-5 Agreement:** {comp.top5_agreement:.4f}",
                f"- **Mean Perplexity Ratio:** {comp.mean_perplexity_ratio:.4f}",
                f"",
                f"| Prompt | Category | Match | Token Match Rate | Output Length (B/C) | Logit MAE |",
                f"|--------|----------|-------|------------------|---------------------|-----------|",
            ])
            
            for pr in comp.prompt_results:
                logit_mae = pr.logit_metrics.mean_absolute_difference if pr.logit_metrics else "N/A"
                lines.append(
                    f"| {pr.prompt[:40]}... | {pr.category} | {pr.output_match} | "
                    f"{pr.token_match_rate:.4f} | {pr.output_length_baseline}/{pr.output_length_compressed} | {logit_mae} |"
                )
            
            lines.append("")
        
        if self.limitations:
            lines.extend(["## Limitations", ""])
            for lim in self.limitations:
                lines.append(f"- {lim}")
            lines.append("")
        
        if self.notes:
            lines.extend(["## Notes", ""])
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")
        
        return "\n".join(lines)


@dataclass
class MetricResult:
    """Individual metric measurement."""
    name: str
    value: float
    unit: str = ""
    description: str = ""
    baseline_value: Optional[float] = None
    compressed_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "description": self.description,
            "baseline_value": self.baseline_value,
            "compressed_value": self.compressed_value,
        }