"""
LDMARK Evaluation - Reporting

Generates JSON and Markdown reports for evaluation results.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional
import json
import os
from datetime import datetime

from .models import (
    EvaluationResult,
    EvaluationConfig,
    CompressionComparisonResult,
    PromptResult,
    LogitMetrics,
    TokenDifference,
    MetricResult,
)


class ReportGenerator:
    """Generates evaluation reports in various formats."""
    
    def __init__(self, result: EvaluationResult):
        self.result = result
    
    def to_json(self, indent: int = 2) -> str:
        """Generate JSON report."""
        return self.result.to_json(indent)
    
    def to_markdown(self) -> str:
        """Generate Markdown report."""
        return self.result.to_markdown()
    
    def save_json(self, filepath: str, indent: int = 2) -> None:
        """Save JSON report to file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.to_json(indent))
    
    def save_markdown(self, filepath: str) -> None:
        """Save Markdown report to file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.to_markdown())
    
    def save_both(self, base_path: str) -> None:
        """Save both JSON and Markdown reports."""
        self.save_json(f"{base_path}.json")
        self.save_markdown(f"{base_path}.md")


def generate_comparison_table(results: List[CompressionComparisonResult]) -> str:
    """Generate a Markdown comparison table for multiple compression methods."""
    if not results:
        return "No results to compare."
    
    lines = [
        "| Method | Storage Reduction | Runtime Mem (GB) | Token Match | Exact Match | Logit MAE | Top-1 | Top-5 | PPL Ratio |",
        "|--------|-------------------|------------------|-------------|-------------|-----------|-------|-------|-----------|",
    ]
    
    for r in results:
        lines.append(
            f"| {r.compression_method} | {r.storage_reduction_ratio:.2f}x | "
            f"{r.runtime_memory_estimate_gb:.2f} | {r.overall_token_match_rate:.4f} | "
            f"{r.exact_output_match_rate:.4f} | {r.mean_logit_mae:.6f} | "
            f"{r.top1_agreement:.4f} | {r.top5_agreement:.4f} | "
            f"{r.mean_perplexity_ratio:.4f} |"
        )
    
    return "\n".join(lines)


def generate_prompt_detail_table(prompt_results: List[PromptResult]) -> str:
    """Generate detailed Markdown table for prompt-level results."""
    if not prompt_results:
        return "No prompt results."
    
    lines = [
        "| Prompt | Category | Match | Token Match Rate | Output Len (B/C) | Logit MAE | Logit Cosine | Top-1 | Top-5 |",
        "|--------|----------|-------|------------------|------------------|-----------|--------------|-------|-------|",
    ]
    
    for pr in prompt_results:
        logit_mae = "N/A"
        logit_cos = "N/A"
        top1 = "N/A"
        top5 = "N/A"
        
        if pr.logit_metrics:
            logit_mae = f"{pr.logit_metrics.mean_absolute_difference:.6f}"
            logit_cos = f"{pr.logit_metrics.cosine_similarity:.6f}"
            top1 = f"{pr.logit_metrics.top1_agreement:.4f}"
            top5 = f"{pr.logit_metrics.top5_agreement:.4f}"
        
        prompt_short = pr.prompt[:40] + "..." if len(pr.prompt) > 40 else pr.prompt
        lines.append(
            f"| {prompt_short} | {pr.category} | {pr.output_match} | "
            f"{pr.token_match_rate:.4f} | {pr.output_length_baseline}/{pr.output_length_compressed} | "
            f"{logit_mae} | {logit_cos} | {top1} | {top5} |"
        )
    
    return "\n".join(lines)


def generate_limitation_section(limitations: List[str]) -> str:
    """Generate limitations section for report."""
    if not limitations:
        return ""
    
    lines = ["## Limitations", ""]
    for lim in limitations:
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)


def generate_reproducibility_section(
    config: EvaluationConfig,
    hardware: Any,
    software: Any,
) -> str:
    """Generate reproducibility section."""
    lines = [
        "## Reproducibility",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Model ID | {config.model_id} |",
        f"| Model Source | {config.model_source.value} |",
        f"| Baseline Model ID | {config.baseline_model_id} |",
        f"| Baseline Model Source | {config.baseline_model_source.value} |",
        f"| Compression Methods | {', '.join(config.compression_methods)} |",
        f"| Task Types | {', '.join([t.value for t in config.task_types])} |",
        f"| Prompt Categories | {', '.join(config.prompt_categories)} |",
        f"| Seed | {config.seed} |",
        f"| Temperature | {config.temperature} |",
        f"| Max Tokens | {config.max_tokens} |",
        f"| Top-K | {config.top_k} |",
        f"| Top-P | {config.top_p} |",
        f"| Target Dtype | {config.target_dtype} |",
        f"| Batch Size | {config.batch_size} |",
        "",
        "### Hardware",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| CPU | {hardware.cpu_model} ({hardware.cpu_cores} cores) |",
        f"| Architecture | {hardware.cpu_architecture} |",
        f"| RAM | {hardware.system_ram_gb} GB |",
        f"| GPU | {hardware.gpu_model or 'N/A'} |",
        f"| GPU VRAM | {hardware.gpu_vram_gb or 'N/A'} GB |",
        f"| OS | {hardware.operating_system} |",
        "",
        "### Software",
        "",
        f"| Package | Version |",
        f"|---------|---------|",
        f"| LDMARK | {software.ldmark_version} |",
        f"| NumPy | {software.numpy_version} |",
        f"| PyTorch | {software.torch_version or 'N/A'} |",
        f"| Transformers | {software.transformers_version or 'N/A'} |",
        "",
    ]
    return "\n".join(lines)


def generate_full_report(
    result: EvaluationResult,
    include_prompt_details: bool = True,
) -> str:
    """Generate a comprehensive full report."""
    lines = [
        "# LDMARK Model Behavior Evaluation Report",
        "",
        f"**Model:** {result.config.model_id} ({result.config.model_source.value})",
        f"**Baseline:** {result.config.baseline_model_id} ({result.config.baseline_model_source.value})",
        f"**Timestamp:** {result.timestamp}",
        f"**Status:** {result.status.value}",
        "",
        "## Executive Summary",
        "",
        f"- **Total Prompts Evaluated:** {result.total_prompts}",
        f"- **Successful:** {result.successful_prompts}",
        f"- **Failed:** {result.failed_prompts}",
        f"- **Total Time:** {result.total_time_seconds:.2f}s",
        "",
    ]
    
    # Comparison table
    if result.comparisons:
        lines.extend([
            "## Compression Method Comparison",
            "",
            generate_comparison_table(result.comparisons),
            "",
        ])
    
    # Per-method details
    for comp in result.comparisons:
        lines.extend([
            f"## {comp.compression_method.upper()}",
            "",
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
            "",
        ])
        
        if comp.errors:
            lines.extend(["### Errors", ""])
            for err in comp.errors:
                lines.append(f"- {err}")
            lines.append("")
        
        if include_prompt_details and comp.prompt_results:
            lines.extend(["### Prompt-Level Details", ""])
            lines.append(generate_prompt_detail_table(comp.prompt_results))
            lines.append("")
    
    # Limitations
    lines.append(generate_limitation_section(result.limitations))
    
    # Reproducibility
    lines.append(generate_reproducibility_section(
        result.config, result.hardware, result.software
    ))
    
    return "\n".join(lines)


def save_report(
    result: EvaluationResult,
    output_dir: str,
    base_name: str = "evaluation_report",
) -> Dict[str, str]:
    """Save report in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    paths = {}
    
    # JSON
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, "w") as f:
        f.write(result.to_json())
    paths["json"] = json_path
    
    # Markdown (summary)
    md_path = os.path.join(output_dir, f"{base_name}.md")
    with open(md_path, "w") as f:
        f.write(result.to_markdown())
    paths["markdown_summary"] = md_path
    
    # Full Markdown report
    full_md_path = os.path.join(output_dir, f"{base_name}_full.md")
    with open(full_md_path, "w") as f:
        f.write(generate_full_report(result))
    paths["markdown_full"] = full_md_path
    
    return paths