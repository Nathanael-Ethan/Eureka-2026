"""
Tests for LDMARK Evaluation Framework.

All tests are deterministic and do NOT require downloading large models.
Uses synthetic backends exclusively.
"""

import sys
import os
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

from src.ldmark.evaluation.models import (
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
from src.ldmark.evaluation.experiments import (
    ExperimentConfig,
    ExperimentResult,
)
from src.ldmark.evaluation.prompts import (
    EvaluationPrompt,
    PromptCategory,
    get_all_prompts,
    get_prompts_by_category,
    get_prompt_suite_summary,
    FACTUAL_RECALL_PROMPTS,
    SIMPLE_REASONING_PROMPTS,
    ARITHMETIC_PROMPTS,
    CODE_COMPLETION_PROMPTS,
    INSTRUCTION_FOLLOWING_PROMPTS,
)
from src.ldmark.evaluation.engine import (
    EvaluationEngine,
    ModelBackend,
    SyntheticModelBackend,
    compare_tokens,
    compute_logit_metrics,
    calculate_perplexity,
    run_synthetic_evaluation,
)
from src.ldmark.evaluation.reporting import (
    ReportGenerator,
    generate_comparison_table,
    generate_prompt_detail_table,
    generate_full_report,
    save_report,
)


class TestEvaluationModels:
    def test_evaluation_config_creation(self):
        config = EvaluationConfig(
            model_id="test-model",
            model_source=ModelSource.SYNTHETIC,
            model_path="/test/model",
            baseline_model_id="test-baseline",
            baseline_model_source=ModelSource.SYNTHETIC,
            baseline_model_path="/test/baseline",
        )
        assert config.model_id == "test-model"
        assert config.model_source == ModelSource.SYNTHETIC
        assert config.seed == 42
        assert config.temperature == 0.0
        assert config.max_tokens == 50
    
    def test_evaluation_config_to_dict(self):
        config = EvaluationConfig(
            model_id="test-model",
            model_source=ModelSource.SYNTHETIC,
            model_path="/test/model",
            baseline_model_id="test-baseline",
            baseline_model_source=ModelSource.SYNTHETIC,
            baseline_model_path="/test/baseline",
            compression_methods=["int8", "int4"],
            seed=123,
        )
        d = config.to_dict()
        assert d["model_id"] == "test-model"
        assert d["model_source"] == "synthetic"
        assert d["compression_methods"] == ["int8", "int4"]
        assert d["seed"] == 123
    
    def test_evaluation_config_from_dict(self):
        d = {
            "model_id": "test-model",
            "model_source": "synthetic",
            "model_path": "/test/model",
            "baseline_model_id": "test-baseline",
            "baseline_model_source": "synthetic",
            "baseline_model_path": "/test/baseline",
            "compression_methods": ["int8"],
            "seed": 42,
        }
        config = EvaluationConfig.from_dict(d)
        assert config.model_id == "test-model"
        assert config.model_source == ModelSource.SYNTHETIC
        assert config.compression_methods == ["int8"]
        assert config.seed == 42
    
    def test_prompt_result(self):
        result = PromptResult(
            prompt="test prompt",
            category="factual_recall",
            task_type=TaskType.PROMPT_RESPONSE,
            baseline_output="expected",
            compressed_output="expected",
            output_match=True,
            token_match_rate=1.0,
        )
        assert result.output_match
        assert result.token_match_rate == 1.0
    
    def test_logit_metrics(self):
        metrics = LogitMetrics(
            mean_absolute_difference=0.01,
            cosine_similarity=0.99,
            top1_agreement=0.95,
        )
        d = metrics.to_dict()
        assert d["mean_absolute_difference"] == 0.01
        assert d["cosine_similarity"] == 0.99
        assert d["top1_agreement"] == 0.95
    
    def test_compression_comparison_result(self):
        result = CompressionComparisonResult(
            compression_method="int8",
            storage_reduction_ratio=2.0,
            overall_token_match_rate=0.9,
            exact_output_match_rate=0.8,
        )
        assert result.compression_method == "int8"
        assert result.storage_reduction_ratio == 2.0
    
    def test_evaluation_result(self):
        config = EvaluationConfig(
            model_id="test",
            model_source=ModelSource.SYNTHETIC,
            model_path="/test",
            baseline_model_id="baseline",
            baseline_model_source=ModelSource.SYNTHETIC,
            baseline_model_path="/baseline",
        )
        hw = HardwareInfo()
        sw = SoftwareVersions()
        result = EvaluationResult(config=config, hardware=hw, software=sw)
        assert result.config == config
        assert result.status == EvaluationStatus.PENDING
    
    def test_token_difference(self):
        td = TokenDifference(
            position=0,
            baseline_token="the",
            compressed_token="the",
            baseline_token_id=1,
            compressed_token_id=1,
            match=True,
        )
        assert td.match
        assert td.position == 0
    
    def test_hardware_info_detect(self):
        hw = HardwareInfo.detect()
        assert hw.cpu_cores > 0
        assert hw.system_ram_gb > 0
        assert hw.python_version != ""
    
    def test_software_versions_detect(self):
        sw = SoftwareVersions.detect()
        assert sw.numpy_version != ""
        assert sw.ldmark_version != ""


class TestPrompts:
    def test_factual_recall_prompts(self):
        prompts = FACTUAL_RECALL_PROMPTS
        assert len(prompts) == 5
        for p in prompts:
            assert isinstance(p, EvaluationPrompt)
            assert p.category == PromptCategory.FACTUAL_RECALL
            assert p.expected_tokens is not None
    
    def test_simple_reasoning_prompts(self):
        prompts = SIMPLE_REASONING_PROMPTS
        assert len(prompts) == 5
        for p in prompts:
            assert p.category == PromptCategory.SIMPLE_REASONING
    
    def test_arithmetic_prompts(self):
        prompts = ARITHMETIC_PROMPTS
        assert len(prompts) == 6
        for p in prompts:
            assert p.category == PromptCategory.ARITHMETIC
    
    def test_code_completion_prompts(self):
        prompts = CODE_COMPLETION_PROMPTS
        assert len(prompts) == 5
        for p in prompts:
            assert p.category == PromptCategory.CODE_COMPLETION
    
    def test_instruction_following_prompts(self):
        prompts = INSTRUCTION_FOLLOWING_PROMPTS
        assert len(prompts) == 5
        for p in prompts:
            assert p.category == PromptCategory.INSTRUCTION_FOLLOWING
    
    def test_all_prompts(self):
        all_prompts = get_all_prompts()
        assert len(all_prompts) == 26
        
        categories = set(p.category for p in all_prompts)
        assert len(categories) == 5
    
    def test_get_prompts_by_category(self):
        factual = get_prompts_by_category(PromptCategory.FACTUAL_RECALL)
        assert len(factual) == 5
        
        arithmetic = get_prompts_by_category(PromptCategory.ARITHMETIC)
        assert len(arithmetic) == 6
    
    def test_get_prompt_suite_summary(self):
        summary = get_prompt_suite_summary()
        assert summary["factual_recall"] == 5
        assert summary["simple_reasoning"] == 5
        assert summary["arithmetic"] == 6
        assert summary["code_completion"] == 5
        assert summary["instruction_following"] == 5
        assert summary["total"] == 26


class TestTokenComparison:
    def test_identical_tokens(self):
        baseline = [1, 2, 3, 4, 5]
        compressed = [1, 2, 3, 4, 5]
        diffs, rate = compare_tokens(baseline, compressed)
        assert rate == 1.0
        assert all(d.match for d in diffs)
    
    def test_different_tokens(self):
        baseline = [1, 2, 3, 4, 5]
        compressed = [1, 2, 9, 4, 5]
        diffs, rate = compare_tokens(baseline, compressed)
        assert rate == 0.8
        assert not diffs[2].match
        assert diffs[2].baseline_token_id == 3
        assert diffs[2].compressed_token_id == 9
    
    def test_different_lengths(self):
        baseline = [1, 2, 3]
        compressed = [1, 2, 3, 4, 5]
        diffs, rate = compare_tokens(baseline, compressed)
        assert rate == 0.6  # 3 matches out of 5
        assert diffs[3].baseline_token == "<EOS>"
        assert diffs[4].baseline_token == "<EOS>"
    
    def test_empty_tokens(self):
        baseline = []
        compressed = []
        diffs, rate = compare_tokens(baseline, compressed)
        assert rate == 1.0
        assert len(diffs) == 0


class TestLogitMetrics:
    def test_identical_logits(self):
        # Use larger vocab size to test top-k properly
        baseline = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01],
        ]
        compressed = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01],
        ]
        metrics = compute_logit_metrics(baseline, compressed)
        assert metrics.mean_absolute_difference == 0.0
        assert metrics.cosine_similarity == 1.0
        assert metrics.top1_agreement == 1.0
        assert metrics.top5_agreement == 1.0
        assert metrics.top10_agreement == 1.0
    
    def test_different_logits(self):
        baseline = [[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]]
        compressed = [[3.0, 2.0, 1.0], [0.5, 1.0, 2.0]]
        metrics = compute_logit_metrics(baseline, compressed)
        assert metrics.mean_absolute_difference > 0
        assert metrics.cosine_similarity < 1.0
    
    def test_empty_logits(self):
        metrics = compute_logit_metrics([], [])
        assert metrics.mean_absolute_difference == 0.0
        assert metrics.cosine_similarity == 0.0
    
    def test_mismatched_lengths(self):
        baseline = [[1.0, 2.0], [2.0, 1.0], [3.0, 0.5]]
        compressed = [[1.0, 2.0], [2.0, 1.0]]
        metrics = compute_logit_metrics(baseline, compressed)
        # Should only compare first 2 positions
        assert metrics.top1_agreement == 1.0


class TestPerplexity:
    def test_perfect_perplexity(self):
        # Logits where target token has probability 1.0
        logits = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
        tokens = [0, 1]
        ppl = calculate_perplexity(logits, tokens)
        assert abs(ppl - 1.0) < 0.01
    
    def test_higher_perplexity(self):
        # Uniform logits
        logits = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        tokens = [0, 1]
        ppl = calculate_perplexity(logits, tokens)
        assert ppl > 1.0
    
    def test_mismatched_lengths(self):
        logits = [[1.0, 2.0], [2.0, 1.0], [3.0, 0.5]]
        tokens = [0, 1]
        ppl = calculate_perplexity(logits, tokens)
        # Should work with min length
        assert ppl > 0
    
    def test_invalid_target(self):
        logits = [[1.0, 2.0]]
        tokens = [5]  # Out of vocab
        ppl = calculate_perplexity(logits, tokens)
        assert ppl == float('inf')


class TestSyntheticModelBackend:
    def test_load(self):
        backend = SyntheticModelBackend(seed=42)
        assert backend.load("/fake/path")
    
    def test_deterministic_generation(self):
        backend = SyntheticModelBackend(seed=42)
        backend.load("/fake/path")
        
        out1, tokens1, logits1 = backend.generate("The capital of France is", max_tokens=10, seed=42)
        out2, tokens2, logits2 = backend.generate("The capital of France is", max_tokens=10, seed=42)
        
        assert out1 == out2
        assert tokens1 == tokens2
    
    def test_different_prompts_different_outputs(self):
        backend = SyntheticModelBackend(seed=42)
        backend.load("/fake/path")
        
        out1, _, _ = backend.generate("The capital of France is", max_tokens=10, seed=42)
        out2, _, _ = backend.generate("Water boils at", max_tokens=10, seed=42)
        
        assert out1 != out2
        assert "Paris" in out1
        assert "100" in out2 or "degrees" in out2
    
    def test_get_logits(self):
        backend = SyntheticModelBackend(seed=42)
        backend.load("/fake/path")
        
        logits = backend.get_logits("test prompt")
        assert logits is not None
        assert len(logits) > 0
        assert len(logits[0]) > 0
    
    def test_model_size(self):
        backend = SyntheticModelBackend(seed=42)
        size = backend.get_model_size_bytes()
        assert size > 0


class TestEvaluationEngine:
    def test_basic_evaluation(self):
        config = EvaluationConfig(
            model_id="test-model",
            model_source=ModelSource.SYNTHETIC,
            model_path="/synthetic/compressed",
            baseline_model_id="test-model",
            baseline_model_source=ModelSource.SYNTHETIC,
            baseline_model_path="/synthetic/baseline",
            compression_methods=["int8", "int4"],
            prompt_categories=["factual_recall"],
            seed=42,
        )
        
        engine = EvaluationEngine(config)
        
        baseline = SyntheticModelBackend(seed=42)
        engine.load_baseline(baseline)
        
        for method in ["int8", "int4"]:
            compressed = SyntheticModelBackend(seed=42)
            engine.load_compressed(method, compressed)
        
        result = engine.run()
        
        assert result.status == EvaluationStatus.COMPLETED
        assert len(result.comparisons) == 2
        assert result.total_prompts > 0
        assert result.successful_prompts > 0
        assert result.total_time_seconds > 0
    
    def test_evaluate_single_method(self):
        config = EvaluationConfig(
            model_id="test-model",
            model_source=ModelSource.SYNTHETIC,
            model_path="/synthetic/compressed",
            baseline_model_id="test-model",
            baseline_model_source=ModelSource.SYNTHETIC,
            baseline_model_path="/synthetic/baseline",
            compression_methods=["int8"],
            prompt_categories=["arithmetic"],
            seed=42,
        )
        
        engine = EvaluationEngine(config)
        baseline = SyntheticModelBackend(seed=42)
        engine.load_baseline(baseline)
        compressed = SyntheticModelBackend(seed=42)
        engine.load_compressed("int8", compressed)
        
        comp_result = engine.evaluate_compression_method("int8", get_prompts_by_category(PromptCategory.ARITHMETIC))
        
        assert comp_result.compression_method == "int8"
        assert comp_result.storage_reduction_ratio > 0
        assert len(comp_result.prompt_results) == 6  # 6 arithmetic prompts


class TestRunSyntheticEvaluation:
    def test_run_synthetic(self):
        result = run_synthetic_evaluation(
            compression_methods=["int8", "int4"],
            prompt_categories=["factual_recall"],
            seed=42,
        )
        
        assert isinstance(result, EvaluationResult)
        assert result.status == EvaluationStatus.COMPLETED
        assert len(result.comparisons) == 2


class TestReporting:
    def test_generate_comparison_table(self):
        comp1 = CompressionComparisonResult(
            compression_method="int8",
            storage_reduction_ratio=2.0,
            runtime_memory_estimate_gb=1.0,
            overall_token_match_rate=0.95,
            exact_output_match_rate=0.80,
            mean_logit_mae=0.01,
            top1_agreement=0.90,
            top5_agreement=0.95,
            mean_perplexity_ratio=1.02,
        )
        comp2 = CompressionComparisonResult(
            compression_method="int4",
            storage_reduction_ratio=4.0,
            runtime_memory_estimate_gb=0.5,
            overall_token_match_rate=0.85,
            exact_output_match_rate=0.60,
            mean_logit_mae=0.05,
            top1_agreement=0.75,
            top5_agreement=0.85,
            mean_perplexity_ratio=1.10,
        )
        
        table = generate_comparison_table([comp1, comp2])
        assert "int8" in table
        assert "int4" in table
        assert "2.00x" in table
        assert "4.00x" in table
    
    def test_generate_prompt_detail_table(self):
        pr = PromptResult(
            prompt="Test prompt",
            category="factual_recall",
            output_match=True,
            token_match_rate=1.0,
            output_length_baseline=5,
            output_length_compressed=5,
            logit_metrics=LogitMetrics(
                mean_absolute_difference=0.01,
                cosine_similarity=0.99,
                top1_agreement=1.0,
                top5_agreement=1.0,
            ),
        )
        
        table = generate_prompt_detail_table([pr])
        assert "Test prompt" in table
        assert "factual_recall" in table
        assert "True" in table
    
    def test_save_report(self):
        config = EvaluationConfig(
            model_id="test",
            model_source=ModelSource.SYNTHETIC,
            model_path="/test",
            baseline_model_id="baseline",
            baseline_model_source=ModelSource.SYNTHETIC,
            baseline_model_path="/baseline",
        )
        hw = HardwareInfo()
        sw = SoftwareVersions()
        result = EvaluationResult(config=config, hardware=hw, software=sw)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_report(result, tmpdir, "test_report")
            
            assert "json" in paths
            assert "markdown_summary" in paths
            assert "markdown_full" in paths
            
            assert os.path.exists(paths["json"])
            assert os.path.exists(paths["markdown_summary"])
            assert os.path.exists(paths["markdown_full"])
    
    def test_report_generator(self):
        config = EvaluationConfig(
            model_id="test",
            model_source=ModelSource.SYNTHETIC,
            model_path="/test",
            baseline_model_id="baseline",
            baseline_model_source=ModelSource.SYNTHETIC,
            baseline_model_path="/baseline",
        )
        hw = HardwareInfo()
        sw = SoftwareVersions()
        result = EvaluationResult(config=config, hardware=hw, software=sw)
        
        generator = ReportGenerator(result)
        json_str = generator.to_json()
        md_str = generator.to_markdown()
        
        assert "test" in json_str
        assert "LDMARK Evaluation Report" in md_str


class TestExperiments:
    def test_experiment_config(self):
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment",
            compression_methods=["int8"],
            prompt_categories=["factual_recall"],
        )
        assert config.name == "test_experiment"
        assert config.compression_methods == ["int8"]
    
    def test_experiment_result(self):
        config = ExperimentConfig(name="test")
        result = ExperimentResult(config=config)
        assert result.config.name == "test"
        assert result.status == EvaluationStatus.PENDING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])