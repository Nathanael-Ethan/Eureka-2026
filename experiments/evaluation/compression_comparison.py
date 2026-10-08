"""
LDMARK Evaluation - Experiments

Controlled experiments for comparing compression methods on model behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time
import json

from src.ldmark.evaluation.engine import (
    EvaluationEngine,
    run_synthetic_evaluation,
    SyntheticModelBackend,
)
from src.ldmark.evaluation.models import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationStatus,
    ModelSource,
    TaskType,
    PromptCategory,
    HardwareInfo,
    SoftwareVersions,
)
from src.ldmark.evaluation.prompts import get_prompt_suite_summary


@dataclass
class ExperimentConfig:
    """Configuration for a compression comparison experiment."""
    name: str
    description: str = ""
    compression_methods: List[str] = field(default_factory=list)
    prompt_categories: List[str] = field(default_factory=list)
    custom_prompts: List[str] = field(default_factory=list)
    seed: int = 42
    temperature: float = 0.0
    max_tokens: int = 50
    iterations: int = 1  # Number of times to repeat for variance estimation


@dataclass
class ExperimentResult:
    """Result of a compression comparison experiment."""
    config: ExperimentConfig
    evaluation_results: List[EvaluationResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    total_time_seconds: float = 0.0
    status: EvaluationStatus = EvaluationStatus.PENDING
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "name": self.config.name,
                "description": self.config.description,
                "compression_methods": self.config.compression_methods,
                "prompt_categories": self.config.prompt_categories,
                "custom_prompts": self.config.custom_prompts,
                "seed": self.config.seed,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "iterations": self.config.iterations,
            },
            "evaluation_results": [r.to_dict() for r in self.evaluation_results],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_time_seconds": self.total_time_seconds,
            "status": self.status.value,
            "errors": self.errors,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def run_compression_comparison_experiment(
    config: ExperimentConfig,
) -> ExperimentResult:
    """
    Run a controlled compression comparison experiment.
    
    Compares FP16 baseline against multiple compression methods
    using the evaluation framework.
    """
    result = ExperimentResult(config=config)
    result.start_time = time.time()
    result.status = EvaluationStatus.RUNNING
    
    if not config.compression_methods:
        config.compression_methods = ["int8", "int4", "binary", "ternary"]
    
    if not config.prompt_categories:
        config.prompt_categories = [c.value for c in PromptCategory]
    
    try:
        for iteration in range(config.iterations):
            iter_seed = config.seed + iteration
            
            eval_config = EvaluationConfig(
                model_id=f"synthetic-model-iter{iteration}",
                model_source=ModelSource.SYNTHETIC,
                model_path="/synthetic/compressed",
                baseline_model_id=f"synthetic-model-iter{iteration}",
                baseline_model_source=ModelSource.SYNTHETIC,
                baseline_model_path="/synthetic/baseline",
                compression_methods=config.compression_methods,
                prompt_categories=config.prompt_categories,
                custom_prompts=config.custom_prompts,
                seed=iter_seed,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            
            engine = EvaluationEngine(eval_config)
            
            # Load baseline
            baseline = SyntheticModelBackend(seed=eval_config.seed)
            engine.load_baseline(baseline)
            
            # Load compressed variants
            for method in config.compression_methods:
                noise_factor = {"int8": 1.01, "int4": 1.05, "binary": 1.15, "ternary": 1.10}.get(method, 1.0)
                compressed = SyntheticModelBackend(seed=eval_config.seed, compression_factor=noise_factor)
                engine.load_compressed(method, compressed)
            
            eval_result = engine.run()
            result.evaluation_results.append(eval_result)
        
        result.status = EvaluationStatus.COMPLETED
        
    except Exception as e:
        result.status = EvaluationStatus.FAILED
        result.errors.append(str(e))
    
    result.end_time = time.time()
    result.total_time_seconds = result.end_time - result.start_time
    
    return result


def run_predefined_experiments() -> Dict[str, ExperimentResult]:
    """Run a set of predefined experiments."""
    
    experiments = {
        "fp16_vs_int8": ExperimentConfig(
            name="fp16_vs_int8",
            description="Compare FP16 baseline vs INT8 compression",
            compression_methods=["int8"],
            prompt_categories=[c.value for c in PromptCategory],
            seed=42,
        ),
        "fp16_vs_int4": ExperimentConfig(
            name="fp16_vs_int4",
            description="Compare FP16 baseline vs INT4 compression",
            compression_methods=["int4"],
            prompt_categories=[c.value for c in PromptCategory],
            seed=42,
        ),
        "fp16_vs_binary": ExperimentConfig(
            name="fp16_vs_binary",
            description="Compare FP16 baseline vs Binary compression",
            compression_methods=["binary"],
            prompt_categories=[c.value for c in PromptCategory],
            seed=42,
        ),
        "fp16_vs_ternary": ExperimentConfig(
            name="fp16_vs_ternary",
            description="Compare FP16 baseline vs Ternary compression",
            compression_methods=["ternary"],
            prompt_categories=[c.value for c in PromptCategory],
            seed=42,
        ),
        "all_compression_methods": ExperimentConfig(
            name="all_compression_methods",
            description="Compare all compression methods against FP16 baseline",
            compression_methods=["int8", "int4", "binary", "ternary"],
            prompt_categories=[c.value for c in PromptCategory],
            seed=42,
        ),
        "factual_only": ExperimentConfig(
            name="factual_only",
            description="Evaluate only factual recall tasks",
            compression_methods=["int8", "int4", "binary", "ternary"],
            prompt_categories=["factual_recall"],
            seed=42,
        ),
        "reasoning_only": ExperimentConfig(
            name="reasoning_only",
            description="Evaluate only simple reasoning tasks",
            compression_methods=["int8", "int4", "binary", "ternary"],
            prompt_categories=["simple_reasoning"],
            seed=42,
        ),
        "arithmetic_only": ExperimentConfig(
            name="arithmetic_only",
            description="Evaluate only arithmetic tasks",
            compression_methods=["int8", "int4", "binary", "ternary"],
            prompt_categories=["arithmetic"],
            seed=42,
        ),
    }
    
    results = {}
    for name, config in experiments.items():
        print(f"\nRunning experiment: {name}")
        result = run_compression_comparison_experiment(config)
        results[name] = result
        print(f"  Status: {result.status.value}, Time: {result.total_time_seconds:.2f}s")
    
    return results


def print_experiment_summary(results: Dict[str, ExperimentResult]) -> None:
    """Print summary of experiment results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    
    for name, result in results.items():
        print(f"\n## {name}")
        print(f"  Status: {result.status.value}")
        print(f"  Time: {result.total_time_seconds:.2f}s")
        print(f"  Iterations: {result.config.iterations}")
        
        if result.evaluation_results:
            eval_result = result.evaluation_results[0]
            print(f"  Prompts: {eval_result.total_prompts}")
            print(f"  Successful: {eval_result.successful_prompts}")
            
            for comp in eval_result.comparisons:
                print(f"\n  {comp.compression_method.upper()}:")
                print(f"    Storage Reduction: {comp.storage_reduction_ratio:.2f}x")
                print(f"    Runtime Memory: {comp.runtime_memory_estimate_gb:.2f} GB")
                print(f"    Token Match Rate: {comp.overall_token_match_rate:.4f}")
                print(f"    Exact Match Rate: {comp.exact_output_match_rate:.4f}")
                print(f"    Logit MAE: {comp.mean_logit_mae:.6f}")
                print(f"    Logit Cosine: {comp.mean_logit_cosine:.6f}")
                print(f"    Top-1 Agreement: {comp.top1_agreement:.4f}")
                print(f"    Top-5 Agreement: {comp.top5_agreement:.4f}")
                print(f"    Perplexity Ratio: {comp.mean_perplexity_ratio:.4f}")
                
                if comp.errors:
                    print(f"    Errors: {len(comp.errors)}")


def save_experiment_results(
    results: Dict[str, ExperimentResult],
    output_dir: str,
) -> None:
    """Save experiment results to JSON files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, result in results.items():
        filepath = os.path.join(output_dir, f"{name}_results.json")
        with open(filepath, "w") as f:
            f.write(result.to_json())
        print(f"Saved: {filepath}")
    
    # Save combined summary
    summary = {
        "timestamp": time.time(),
        "experiments": {name: result.to_dict() for name, result in results.items()},
    }
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


def generate_markdown_report(
    results: Dict[str, ExperimentResult],
    output_path: str,
) -> None:
    """Generate a comprehensive markdown report."""
    with open(output_path, "w") as f:
        f.write("# LDMARK Compression Behavior Evaluation Report\n\n")
        f.write(f"Generated: {time.ctime()}\n\n")
        
        f.write("## Prompt Suite\n\n")
        summary = get_prompt_suite_summary()
        f.write(f"- **Total Prompts:** {summary['total']}\n")
        for cat, count in summary.items():
            if cat != "total":
                f.write(f"- **{cat.replace('_', ' ').title()}:** {count}\n")
        f.write("\n")
        
        f.write("## Experiments\n\n")
        
        for name, result in results.items():
            f.write(f"### {name}\n\n")
            f.write(f"**Description:** {result.config.description}\n\n")
            f.write(f"**Status:** {result.status.value}\n\n")
            f.write(f"**Time:** {result.total_time_seconds:.2f}s\n\n")
            
            if result.evaluation_results:
                eval_result = result.evaluation_results[0]
                f.write(f"**Total Prompts:** {eval_result.total_prompts}\n")
                f.write(f"**Successful:** {eval_result.successful_prompts}\n\n")
                
                f.write("| Method | Storage Reduction | Runtime Mem (GB) | Token Match | Exact Match | Logit MAE | Top-1 | Top-5 | PPL Ratio |\n")
                f.write("|--------|-------------------|------------------|-------------|-------------|-----------|-------|-------|-----------|\n")
                
                for comp in eval_result.comparisons:
                    f.write(f"| {comp.compression_method} | {comp.storage_reduction_ratio:.2f}x | "
                            f"{comp.runtime_memory_estimate_gb:.2f} | {comp.overall_token_match_rate:.4f} | "
                            f"{comp.exact_output_match_rate:.4f} | {comp.mean_logit_mae:.6f} | "
                            f"{comp.top1_agreement:.4f} | {comp.top5_agreement:.4f} | "
                            f"{comp.mean_perplexity_ratio:.4f} |\n")
            
            f.write("\n")
        
        f.write("\n## Limitations\n\n")
        f.write("1. **Synthetic backend only** - Results do not represent real model behavior\n")
        f.write("2. **Deterministic tokens** - Uses character-level tokenization, not real tokenizer\n")
        f.write("3. **Synthetic logits** - Logits are generated, not from actual model forward pass\n")
        f.write("4. **Small prompt suite** - Not comprehensive behavioral evaluation\n")
        f.write("5. **No real perplexity** - Uses synthetic logits for calculation\n")
        f.write("6. **Single model** - Results specific to synthetic test model\n\n")
        
        f.write("## Reproducibility\n\n")
        f.write("```json\n")
        f.write("{\n")
        f.write('  "seed": 42,\n')
        f.write('  "temperature": 0.0,\n')
        f.write('  "max_tokens": 50,\n')
        f.write('  "top_k": 1,\n')
        f.write('  "top_p": 1.0\n')
        f.write("}\n")
        f.write("```\n")


if __name__ == "__main__":
    print("Running predefined LDMARK compression comparison experiments...")
    results = run_predefined_experiments()
    print_experiment_summary(results)
    
    # Save results
    save_experiment_results(results, "experiments/evaluation/results")
    generate_markdown_report(results, "experiments/evaluation/results/report.md")
    print("\nResults saved to experiments/evaluation/results/")