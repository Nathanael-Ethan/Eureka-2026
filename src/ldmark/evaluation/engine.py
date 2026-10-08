"""
LDMARK Evaluation - Core Engine

Main evaluation engine for running model quality and behavior evaluations.
Supports both synthetic (deterministic) and real model evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
import time
import numpy as np

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
)
from .prompts import (
    EvaluationPrompt,
    PromptCategory,
    get_all_prompts,
    get_prompts_by_category,
)
from src.ldmark.compression.metrics import calculate_error_metrics
from src.ldmark.runtime.memory import estimate_decompressed_size


class ModelBackend:
    """Abstract base for model backends."""
    
    def load(self, model_path: str, **kwargs) -> bool:
        """Load model from path. Returns success."""
        raise NotImplementedError
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
        top_k: int = 1,
        top_p: float = 1.0,
        seed: int = 42,
    ) -> Tuple[str, List[int], Optional[List[List[float]]]]:
        """Generate text from prompt. Returns (text, tokens, logits)."""
        raise NotImplementedError
    
    def get_logits(self, prompt: str) -> Optional[List[List[float]]]:
        """Get logits for a prompt if available."""
        raise NotImplementedError
    
    def get_model_size_bytes(self) -> int:
        """Get model size in bytes."""
        raise NotImplementedError
    
    def close(self) -> None:
        """Release resources."""
        pass


class SyntheticModelBackend(ModelBackend):
    """
    Synthetic model backend for deterministic testing without real models.
    
    This backend produces deterministic outputs based on the prompt content,
    allowing evaluation framework testing without downloading models.
    """
    
    def __init__(self, seed: int = 42, compression_factor: float = 1.0):
        self.seed = seed
        self.compression_factor = compression_factor  # Simulate compression effects
        self._rng = np.random.RandomState(seed)
        self._loaded = False
    
    def load(self, model_path: str, **kwargs) -> bool:
        self._loaded = True
        return True
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
        top_k: int = 1,
        top_p: float = 1.0,
        seed: int = 42,
    ) -> Tuple[str, List[int], Optional[List[List[float]]]]:
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        
        # Deterministic generation based on prompt hash
        prompt_hash = hash(prompt) % 10000
        rng = np.random.RandomState(prompt_hash + seed)
        
        # Simple deterministic completion logic
        completion = self._deterministic_completion(prompt, max_tokens, rng)
        tokens = [ord(c) for c in completion]  # Fake token IDs
        
        # Generate fake logits (for testing logit comparison)
        logits = None
        if max_tokens > 0:
            vocab_size = 1000
            logits = []
            for _ in range(len(completion)):
                logit = rng.randn(vocab_size) * 0.1
                # Make the chosen token have higher logit
                if tokens:
                    token_idx = tokens[-1] % vocab_size
                    logit[token_idx] += 5.0
                logits.append(logit.tolist())
        
        return completion, tokens, logits
    
    def _deterministic_completion(self, prompt: str, max_tokens: int, rng: np.random.RandomState) -> str:
        """Generate deterministic completion based on prompt."""
        prompt_lower = prompt.lower().strip()
        
        # Factual recall
        if "capital of france" in prompt_lower:
            return " Paris"
        elif "water boils" in prompt_lower:
            return " 100 degrees Celsius"
        elif "largest planet" in prompt_lower:
            return " Jupiter"
        elif "romeo and juliet" in prompt_lower:
            return " William Shakespeare"
        elif "chemical symbol for gold" in prompt_lower:
            return " Au"
        
        # Reasoning
        elif "all cats are animals" in prompt_lower:
            return " living things"
        elif "taller than mary" in prompt_lower:
            return " John"
        elif "square has four" in prompt_lower:
            return " the square"
        elif "2, 4, 6, 8" in prompt_lower:
            return " 10"
        elif "rains" in prompt_lower and "wet" in prompt_lower:
            return " Not necessarily, the ground could be wet for other reasons."
        
        # Arithmetic
        elif "2 + 2" in prompt_lower:
            return " 4"
        elif "15 - 7" in prompt_lower:
            return " 8"
        elif "6 * 7" in prompt_lower or "6 times 7" in prompt_lower:
            return " 42"
        elif "100 / 4" in prompt_lower:
            return " 25"
        elif "12 times 12" in prompt_lower:
            return " 144"
        elif "(5 + 3) * 2" in prompt_lower:
            return " 16"
        
        # Code
        elif "def add" in prompt_lower:
            return " a + b"
        elif "for i in range" in prompt_lower:
            return ""
        elif "class myclass" in prompt_lower:
            return " 0"
        elif "json." in prompt_lower:
            return " dumps(data)"
        elif "factorial" in prompt_lower:
            return " n - 1)"
        
        # Instruction following
        elif "list three colors" in prompt_lower:
            return " Red, blue, green"
        elif "using the word 'hello'" in prompt_lower:
            return " Hello world!"
        elif "count from 1 to 5" in prompt_lower:
            return " 1, 2, 3, 4, 5"
        elif "reverse" in prompt_lower and "hello" in prompt_lower:
            return " olleh"
        elif "yes or no" in prompt_lower and "water wet" in prompt_lower:
            return " Yes"
        
        # Default: generate some deterministic text
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        completion_words = []
        for i in range(min(max_tokens, 10)):
            idx = (prompt_hash + i) % len(words)
            completion_words.append(words[idx])
        return " " + " ".join(completion_words)
    
    def get_logits(self, prompt: str) -> Optional[List[List[float]]]:
        """Return fake logits for testing."""
        prompt_hash = hash(prompt) % 10000
        rng = np.random.RandomState(prompt_hash)
        vocab_size = 1000
        seq_len = 5
        logits = []
        for _ in range(seq_len):
            logit = rng.randn(vocab_size) * 0.1
            logits.append(logit.tolist())
        return logits
    
    def get_model_size_bytes(self) -> int:
        """Fake model size."""
        return 100_000_000  # 100MB fake size


def compare_tokens(
    baseline_tokens: List[int],
    compressed_tokens: List[int],
    baseline_vocab: Optional[Dict[int, str]] = None,
    compressed_vocab: Optional[Dict[int, str]] = None,
) -> Tuple[List[TokenDifference], float]:
    """
    Compare token sequences and compute match rate.
    
    Returns token differences and match rate.
    """
    differences = []
    matches = 0
    max_len = max(len(baseline_tokens), len(compressed_tokens))
    
    for i in range(max_len):
        base_token = baseline_tokens[i] if i < len(baseline_tokens) else None
        comp_token = compressed_tokens[i] if i < len(compressed_tokens) else None
        
        match = base_token == comp_token
        if match:
            matches += 1
        
        # Handle None tokens (out of range) with "<EOS>" string
        if base_token is None:
            base_str = "<EOS>"
        elif baseline_vocab:
            base_str = baseline_vocab.get(base_token, str(base_token))
        else:
            base_str = str(base_token)
        
        if comp_token is None:
            comp_str = "<EOS>"
        elif compressed_vocab:
            comp_str = compressed_vocab.get(comp_token, str(comp_token))
        else:
            comp_str = str(comp_token)
        
        differences.append(TokenDifference(
            position=i,
            baseline_token=base_str,
            compressed_token=comp_str,
            baseline_token_id=base_token if base_token is not None else -1,
            compressed_token_id=comp_token if comp_token is not None else -1,
            match=match,
        ))
    
    match_rate = matches / max_len if max_len > 0 else 1.0
    return differences, match_rate


def compute_logit_metrics(
    baseline_logits: List[List[float]],
    compressed_logits: List[List[float]],
) -> LogitMetrics:
    """
    Compute logit-level metrics between baseline and compressed logits.
    
    Args:
        baseline_logits: List of logit arrays for each position [seq_len, vocab_size]
        compressed_logits: Same format
    
    Returns:
        LogitMetrics with various comparison metrics
    """
    if not baseline_logits or not compressed_logits:
        return LogitMetrics()
    
    min_len = min(len(baseline_logits), len(compressed_logits))
    if min_len == 0:
        return LogitMetrics()
    
    metrics = LogitMetrics()
    mae_values = []
    max_diff_values = []
    cos_sims = []
    top1_matches = 0
    top5_matches = 0
    top10_matches = 0
    kl_divs = []
    js_divs = []
    
    for i in range(min_len):
        base = np.array(baseline_logits[i], dtype=np.float64)
        comp = np.array(compressed_logits[i], dtype=np.float64)
        
        # Ensure same length
        min_vocab = min(len(base), len(comp))
        base = base[:min_vocab]
        comp = comp[:min_vocab]
        
        if min_vocab == 0:
            continue
        
        # Mean absolute difference
        diff = np.abs(base - comp)
        mae_values.append(np.mean(diff))
        max_diff_values.append(np.max(diff))
        
        # Cosine similarity
        norm_base = np.linalg.norm(base)
        norm_comp = np.linalg.norm(comp)
        if norm_base > 0 and norm_comp > 0:
            cos_sims.append(float(np.dot(base, comp) / (norm_base * norm_comp)))
        else:
            cos_sims.append(0.0)
        
        # Top-k agreement
        base_top1 = int(np.argmax(base))
        comp_top1 = int(np.argmax(comp))
        if base_top1 == comp_top1:
            top1_matches += 1
        
        base_top5 = set(np.argsort(base)[-5:])
        comp_top5 = set(np.argsort(comp)[-5:])
        top5_matches += len(base_top5 & comp_top5)
        
        base_top10 = set(np.argsort(base)[-10:])
        comp_top10 = set(np.argsort(comp)[-10:])
        top10_matches += len(base_top10 & comp_top10)
        
        # KL divergence (with small epsilon for numerical stability)
        eps = 1e-10
        base_prob = np.exp(base - np.max(base))
        base_prob = base_prob / (np.sum(base_prob) + eps)
        comp_prob = np.exp(comp - np.max(comp))
        comp_prob = comp_prob / (np.sum(comp_prob) + eps)
        
        kl = np.sum(base_prob * np.log((base_prob + eps) / (comp_prob + eps)))
        kl_divs.append(float(kl))
        
        # JS divergence
        m = 0.5 * (base_prob + comp_prob)
        js = 0.5 * np.sum(base_prob * np.log((base_prob + eps) / (m + eps))) + \
             0.5 * np.sum(comp_prob * np.log((comp_prob + eps) / (m + eps)))
        js_divs.append(float(js))
    
    metrics.mean_absolute_difference = float(np.mean(mae_values)) if mae_values else 0.0
    metrics.max_absolute_difference = float(np.max(max_diff_values)) if max_diff_values else 0.0
    metrics.cosine_similarity = float(np.mean(cos_sims)) if cos_sims else 0.0
    metrics.top1_agreement = top1_matches / min_len if min_len > 0 else 0.0
    metrics.top5_agreement = top5_matches / (min_len * 5) if min_len > 0 else 0.0
    metrics.top10_agreement = top10_matches / (min_len * 10) if min_len > 0 else 0.0
    metrics.kl_divergence = float(np.mean(kl_divs)) if kl_divs else 0.0
    metrics.js_divergence = float(np.mean(js_divs)) if js_divs else 0.0
    
    return metrics


def calculate_perplexity(logits: List[List[float]], target_tokens: List[int]) -> float:
    """
    Calculate perplexity from logits and target tokens.
    
    Args:
        logits: List of logit arrays [seq_len, vocab_size]
        target_tokens: Target token IDs
    
    Returns:
        Perplexity value
    """
    if not logits or not target_tokens:
        return float('inf')
    
    min_len = min(len(logits), len(target_tokens))
    if min_len == 0:
        return float('inf')
    
    total_log_prob = 0.0
    for i in range(min_len):
        logit = np.array(logits[i], dtype=np.float64)
        target = target_tokens[i]
        
        if target >= len(logit):
            return float('inf')
        
        # Softmax
        log_probs = logit - np.max(logit)
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs)))
        total_log_prob += log_probs[target]
    
    avg_log_prob = total_log_prob / min_len
    perplexity = float(np.exp(-avg_log_prob))
    return perplexity


class EvaluationEngine:
    """
    Main evaluation engine for running model behavior evaluations.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.hardware = HardwareInfo.detect()
        self.software = SoftwareVersions.detect()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Backends
        self.baseline_backend: Optional[ModelBackend] = None
        self.compressed_backends: Dict[str, ModelBackend] = {}
    
    def load_baseline(self, backend: ModelBackend) -> bool:
        """Load baseline model."""
        self.baseline_backend = backend
        return backend.load(self.config.baseline_model_path)
    
    def load_compressed(self, compression_method: str, backend: ModelBackend) -> bool:
        """Load compressed model for a specific compression method."""
        self.compressed_backends[compression_method] = backend
        return backend.load(self.config.model_path)
    
    def evaluate_prompt(
        self,
        prompt: EvaluationPrompt,
        baseline_backend: ModelBackend,
        compressed_backend: ModelBackend,
    ) -> PromptResult:
        """Evaluate a single prompt on both baseline and compressed models."""
        result = PromptResult(
            prompt=prompt.prompt,
            category=prompt.category.value,
            task_type=TaskType.PROMPT_RESPONSE,
        )
        
        # Generation settings
        gen_kwargs = {
            "max_tokens": prompt.max_tokens or self.config.max_tokens,
            "temperature": prompt.temperature if prompt.temperature is not None else self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "seed": self.config.seed,
        }
        
        try:
            # Baseline generation
            base_start = time.perf_counter()
            base_output, base_tokens, base_logits = baseline_backend.generate(
                prompt.prompt, **gen_kwargs
            )
            result.generation_time_baseline_ms = (time.perf_counter() - base_start) * 1000
            result.baseline_output = base_output
            result.baseline_tokens = base_tokens
            result.baseline_logits = base_logits
            
            # Compressed generation
            comp_start = time.perf_counter()
            comp_output, comp_tokens, comp_logits = compressed_backend.generate(
                prompt.prompt, **gen_kwargs
            )
            result.generation_time_compressed_ms = (time.perf_counter() - comp_start) * 1000
            result.compressed_output = comp_output
            result.compressed_tokens = comp_tokens
            result.compressed_logits = comp_logits
            
            # Compare outputs
            result.output_match = base_output.strip() == comp_output.strip()
            result.output_length_baseline = len(base_tokens)
            result.output_length_compressed = len(comp_tokens)
            
            # Token-level comparison
            token_diffs, match_rate = compare_tokens(base_tokens, comp_tokens)
            result.token_differences = token_diffs
            result.token_match_rate = match_rate
            
            # Logit comparison if available
            if base_logits and comp_logits:
                result.logit_metrics = compute_logit_metrics(base_logits, comp_logits)
            
            # Perplexity if logits and tokens available
            if base_logits and base_tokens:
                result.perplexity_baseline = calculate_perplexity(base_logits, base_tokens)
            if comp_logits and comp_tokens:
                result.perplexity_compressed = calculate_perplexity(comp_logits, comp_tokens)
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def evaluate_compression_method(
        self,
        compression_method: str,
        prompts: List[EvaluationPrompt],
    ) -> CompressionComparisonResult:
        """Evaluate a single compression method against baseline."""
        if not self.baseline_backend:
            raise RuntimeError("Baseline model not loaded")
        
        compressed_backend = self.compressed_backends.get(compression_method)
        if not compressed_backend:
            raise RuntimeError(f"Compressed backend for {compression_method} not loaded")
        
        result = CompressionComparisonResult(compression_method=compression_method)
        
        # Estimate storage
        result.baseline_size_bytes = self.baseline_backend.get_model_size_bytes()
        result.compressed_size_bytes = compressed_backend.get_model_size_bytes()
        if result.baseline_size_bytes > 0:
            result.storage_reduction_ratio = result.baseline_size_bytes / result.compressed_size_bytes
        
        # Runtime memory estimate
        # For synthetic backends, use a reasonable tensor shape based on model size
        # Approximate: model_size_bytes / target_dtype_size = num_elements
        import numpy as np
        target_dtype_size = np.dtype(self.config.target_dtype).itemsize
        estimated_elements = result.compressed_size_bytes // target_dtype_size if result.compressed_size_bytes > 0 else 1000000
        # Use a square-ish shape for estimation
        estimated_shape = (int(estimated_elements ** 0.5), int(estimated_elements ** 0.5))
        result.runtime_memory_estimate_gb = estimate_decompressed_size(
            estimated_shape, self.config.target_dtype
        ) / (1024**3)
        
        # Evaluate each prompt
        token_match_rates = []
        logit_maes = []
        logit_cosines = []
        top1_agreements = []
        top5_agreements = []
        perplexity_ratios = []
        exact_matches = 0
        
        for prompt in prompts:
            pr = self.evaluate_prompt(prompt, self.baseline_backend, compressed_backend)
            result.prompt_results.append(pr)
            
            if not pr.error:
                token_match_rates.append(pr.token_match_rate)
                exact_matches += 1 if pr.output_match else 0
                
                if pr.logit_metrics:
                    logit_maes.append(pr.logit_metrics.mean_absolute_difference)
                    logit_cosines.append(pr.logit_metrics.cosine_similarity)
                    top1_agreements.append(pr.logit_metrics.top1_agreement)
                    top5_agreements.append(pr.logit_metrics.top5_agreement)
                
                if pr.perplexity_baseline and pr.perplexity_compressed:
                    ratio = pr.perplexity_compressed / pr.perplexity_baseline
                    if ratio > 0 and ratio < float('inf'):
                        perplexity_ratios.append(ratio)
            else:
                result.errors.append(f"Prompt failed: {pr.prompt[:50]}... - {pr.error}")
        
        # Aggregate metrics
        if token_match_rates:
            result.overall_token_match_rate = float(np.mean(token_match_rates))
        result.exact_output_match_rate = exact_matches / len(prompts) if prompts else 0.0
        result.mean_logit_mae = float(np.mean(logit_maes)) if logit_maes else 0.0
        result.mean_logit_cosine = float(np.mean(logit_cosines)) if logit_cosines else 0.0
        result.top1_agreement = float(np.mean(top1_agreements)) if top1_agreements else 0.0
        result.top5_agreement = float(np.mean(top5_agreements)) if top5_agreements else 0.0
        result.mean_perplexity_ratio = float(np.mean(perplexity_ratios)) if perplexity_ratios else 1.0
        
        return result
    
    def run(self) -> EvaluationResult:
        """Run complete evaluation."""
        self.start_time = time.perf_counter()
        
        eval_result = EvaluationResult(
            config=self.config,
            hardware=self.hardware,
            software=self.software,
            status=EvaluationStatus.RUNNING,
        )
        
        # Get prompts to evaluate
        if self.config.prompts:
            # Use custom prompts
            prompts = [
                EvaluationPrompt(prompt=p, category=PromptCategory.INSTRUCTION_FOLLOWING)
                for p in self.config.prompts
            ]
        else:
            # Use prompt suite
            prompts = []
            for cat in self.config.prompt_categories:
                cat_enum = PromptCategory(cat) if isinstance(cat, str) else cat
                prompts.extend(get_prompts_by_category(cat_enum))
            
            if not prompts:
                prompts = get_all_prompts()
        
        # Filter by task types if specified
        if self.config.task_types:
            prompts = [p for p in prompts if p.task_type in self.config.task_types]
        
        # Run evaluation for each compression method
        for method in self.config.compression_methods:
            try:
                print(f"Evaluating {method}...")
                comp_result = self.evaluate_compression_method(method, prompts)
                eval_result.comparisons.append(comp_result)
            except Exception as e:
                error_result = CompressionComparisonResult(
                    compression_method=method,
                    errors=[f"Evaluation failed: {e}"],
                )
                eval_result.comparisons.append(error_result)
        
        # Summary
        eval_result.total_prompts = len(prompts) * len(self.config.compression_methods)
        eval_result.successful_prompts = sum(
            len([pr for pr in c.prompt_results if not pr.error])
            for c in eval_result.comparisons
        )
        eval_result.failed_prompts = sum(
            len([pr for pr in c.prompt_results if pr.error])
            for c in eval_result.comparisons
        )
        
        self.end_time = time.perf_counter()
        eval_result.total_time_seconds = self.end_time - self.start_time
        eval_result.status = EvaluationStatus.COMPLETED
        
        # Add limitations
        eval_result.limitations.extend([
            "Evaluation uses synthetic backend - results are not representative of real model behavior",
            "Token matching uses character-level tokenization, not real tokenizer",
            "Logits are synthetically generated, not from actual model forward pass",
            "Small prompt suite - not comprehensive behavioral evaluation",
            "No real perplexity calculation - uses synthetic logits",
        ])
        
        return eval_result


def run_synthetic_evaluation(
    compression_methods: List[str] = None,
    prompt_categories: List[str] = None,
    seed: int = 42,
) -> EvaluationResult:
    """
    Convenience function to run a synthetic evaluation.
    
    Uses SyntheticModelBackend for both baseline and compressed models.
    """
    if compression_methods is None:
        compression_methods = ["int8", "int4", "binary", "ternary"]
    
    config = EvaluationConfig(
        model_id="synthetic-test-model",
        model_source=ModelSource.SYNTHETIC,
        model_path="/synthetic/compressed",
        baseline_model_id="synthetic-test-model",
        baseline_model_source=ModelSource.SYNTHETIC,
        baseline_model_path="/synthetic/baseline",
        compression_methods=compression_methods,
        prompt_categories=prompt_categories or [c.value for c in PromptCategory],
        seed=seed,
    )
    
    engine = EvaluationEngine(config)
    
    # Load baseline
    baseline = SyntheticModelBackend(seed=config.seed)
    engine.load_baseline(baseline)
    
    # Load compressed variants
    for method in compression_methods:
        # Simulate compression by adding noise factor
        noise_factor = {"int8": 1.01, "int4": 1.05, "binary": 1.15, "ternary": 1.10}.get(method, 1.0)
        compressed = SyntheticModelBackend(seed=config.seed, compression_factor=noise_factor)
        engine.load_compressed(method, compressed)
    
    return engine.run()