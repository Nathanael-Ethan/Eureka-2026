"""
Tests for LDMARK codebook-based quantization.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.ldmark.compression.codebook.representation import (
    CodebookConfig,
    CodebookTensor,
    CodebookGroup,
    create_codebook_tensor,
    reconstruct_from_codebook,
)
from src.ldmark.compression.codebook.generation import (
    generate_codebook_kmeans,
    initialize_codebook_uniform,
    initialize_codebook_percentile,
    initialize_codebook_random,
    initialize_codebook_kmeanspp,
    assign_to_nearest_centroid,
    recompute_centroids,
    generate_codebook_groupwise,
)
from src.ldmark.compression.codebook.analysis import (
    analyze_weight_distribution,
    WeightDistributionStats,
    histogram_analysis,
    compute_optimal_codebook_size_estimate,
)
from src.ldmark.compression.codebook.storage import (
    calculate_codebook_storage,
    calculate_effective_bits_per_weight,
    calculate_uniform_storage_comparison,
    StorageAccounting,
)
from src.ldmark.compression.codebook.experiment import (
    CodebookExperimentConfig,
    CodebookExperimentResult,
    run_codebook_experiment,
    run_uniform_vs_codebook_experiment,
    run_binary_codebook_experiment,
    generate_test_tensor,
)


# ---------- Representation Tests ----------

class TestCodebookConfig:
    def test_basic_config(self):
        config = CodebookConfig(codebook_size=16, group_size=None)
        assert config.codebook_size == 16
        assert config.group_size is None
        assert config.index_bit_width == 4
        assert not config.is_group_wise
    
    def test_group_wise_config(self):
        config = CodebookConfig(codebook_size=16, group_size=128)
        assert config.group_size == 128
        assert config.is_group_wise
    
    def test_index_bit_widths(self):
        assert CodebookConfig(codebook_size=2).index_bit_width == 1
        assert CodebookConfig(codebook_size=4).index_bit_width == 2
        assert CodebookConfig(codebook_size=8).index_bit_width == 3
        assert CodebookConfig(codebook_size=16).index_bit_width == 4
        assert CodebookConfig(codebook_size=32).index_bit_width == 5
        assert CodebookConfig(codebook_size=256).index_bit_width == 8
    
    def test_invalid_config(self):
        with pytest.raises(ValueError):
            CodebookConfig(codebook_size=1)
        with pytest.raises(ValueError):
            CodebookConfig(codebook_size=257)
        with pytest.raises(ValueError):
            CodebookConfig(codebook_size=16, group_size=0)
        with pytest.raises(ValueError):
            CodebookConfig(codebook_size=16, max_iterations=0)
        with pytest.raises(ValueError):
            CodebookConfig(codebook_size=16, initialization="invalid")


class TestCodebookGroup:
    def test_basic_group(self):
        codebook = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        indices = np.array([0, 2, 4, 1, 3], dtype=np.uint8)
        
        group = CodebookGroup(
            codebook=codebook,
            indices=indices,
            start_idx=0,
            end_idx=5,
        )
        
        assert group.size == 5
        assert group.codebook_size == 5
        reconstructed = group.reconstruct()
        expected = np.array([-1.0, 0.0, 1.0, -0.5, 0.5])
        assert np.allclose(reconstructed, expected)
    
    def test_group_validation(self):
        codebook = np.array([-1.0, 1.0])
        indices = np.array([0, 1, 2])  # Out of bounds
        
        with pytest.raises(ValueError):
            CodebookGroup(
                codebook=codebook,
                indices=indices,
                start_idx=0,
                end_idx=3,
            )


class TestCodebookTensor:
    def test_global_codebook_tensor(self):
        tensor = np.random.randn(100).astype(np.float32)
        config = CodebookConfig(codebook_size=16, group_size=None)
        
        # Create with dummy data
        cb_tensor = create_codebook_tensor(tensor, config)
        
        assert cb_tensor.num_groups == 1
        assert cb_tensor.codebook_size == 16
        assert cb_tensor.total_elements == 100
    
    def test_group_wise_codebook_tensor(self):
        tensor = np.random.randn(256).astype(np.float32)
        config = CodebookConfig(codebook_size=8, group_size=128)
        
        cb_tensor = create_codebook_tensor(tensor, config)
        
        assert cb_tensor.num_groups == 2
        assert cb_tensor.codebook_size == 8
        assert cb_tensor.total_elements == 256
    
    def test_reconstruction(self):
        tensor = np.array([-1.0, 0.0, 1.0, -0.5, 0.5], dtype=np.float32)
        config = CodebookConfig(codebook_size=4, group_size=None)
        
        # Create codebook that matches values
        codebook = np.array([-1.0, -0.5, 0.0, 1.0], dtype=np.float32)
        indices = np.array([0, 2, 3, 1, 2], dtype=np.uint8)  # Correct indices
        
        cb_tensor = create_codebook_tensor(tensor, config, [codebook], [indices])
        reconstructed = reconstruct_from_codebook(cb_tensor)
        
        expected = codebook[indices]
        assert np.allclose(reconstructed, expected)


# ---------- Generation Tests ----------

class TestCodebookInitialization:
    def test_uniform_init(self):
        values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        codebook = initialize_codebook_uniform(values, 4, seed=42)
        
        assert len(codebook) == 4
        assert np.allclose(codebook, [-1.0, -0.333, 0.333, 1.0], atol=1e-3)
    
    def test_uniform_init_constant(self):
        values = np.ones(10, dtype=np.float32) * 0.5
        codebook = initialize_codebook_uniform(values, 4, seed=42)
        
        assert np.allclose(codebook, 0.5)
    
    def test_percentile_init(self):
        values = np.linspace(-1, 1, 100).astype(np.float32)
        codebook = initialize_codebook_percentile(values, 4, seed=42)
        
        assert len(codebook) == 4
        assert codebook[0] <= codebook[1] <= codebook[2] <= codebook[3]
    
    def test_random_init(self):
        values = np.random.randn(100).astype(np.float32)
        codebook = initialize_codebook_random(values, 8, seed=42)
        
        assert len(codebook) == 8
        assert np.all(np.isin(codebook, values))
    
    def test_kmeanspp_init(self):
        values = np.random.randn(100).astype(np.float32)
        codebook = initialize_codebook_kmeanspp(values, 8, seed=42)
        
        assert len(codebook) == 8
        # Should be sorted
        assert np.all(codebook[:-1] <= codebook[1:])


class TestAssignmentAndRecompute:
    def test_assign_to_nearest(self):
        values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        centroids = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        
        indices = assign_to_nearest_centroid(values, centroids)
        
        # -0.5 is equidistant from -1.0 and 0.0, picks first (0)
        # 0.5 is equidistant from 0.0 and 1.0, picks first (1)
        expected = np.array([0, 0, 1, 1, 2], dtype=np.uint8)
        assert np.array_equal(indices, expected)
    
    def test_recompute_centroids(self):
        values = np.array([-1.0, -0.9, 0.0, 0.1, 1.0, 0.9], dtype=np.float32)
        indices = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint8)
        
        centroids = recompute_centroids(values, indices, 3)
        
        assert np.isclose(centroids[0], -0.95)
        assert np.isclose(centroids[1], 0.05)
        assert np.isclose(centroids[2], 0.95)


class TestKMeansGeneration:
    def test_generate_codebook_kmeans_basic(self):
        values = np.random.randn(1000).astype(np.float32)
        config = CodebookConfig(codebook_size=16, seed=42)
        
        codebook, indices, errors = generate_codebook_kmeans(values, config)
        
        assert len(codebook) == 16
        assert len(indices) == len(values)
        assert len(errors) > 0
        assert all(e >= 0 for e in errors)
        assert errors[-1] <= errors[0]  # Error should not increase
    
    def test_generate_codebook_small_tensor(self):
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        config = CodebookConfig(codebook_size=4, seed=42)
        
        codebook, indices, errors = generate_codebook_kmeans(values, config)
        
        assert len(codebook) == 4
        assert len(indices) == 3
    
    def test_generate_codebook_deterministic(self):
        values = np.random.randn(500).astype(np.float32)
        config = CodebookConfig(codebook_size=8, seed=123)
        
        cb1, idx1, _ = generate_codebook_kmeans(values, config)
        cb2, idx2, _ = generate_codebook_kmeans(values, config)
        
        assert np.allclose(cb1, cb2)
        assert np.array_equal(idx1, idx2)
    
    def test_generate_codebook_different_seeds(self):
        values = np.random.randn(500).astype(np.float32)
        config1 = CodebookConfig(codebook_size=8, seed=1)
        config2 = CodebookConfig(codebook_size=8, seed=2)
        
        cb1, _, _ = generate_codebook_kmeans(values, config1)
        cb2, _, _ = generate_codebook_kmeans(values, config2)
        
        # Different seeds may produce different codebooks
        # (but not guaranteed to be different)


class TestGroupwiseGeneration:
    def test_generate_codebook_groupwise(self):
        values = np.random.randn(1000).astype(np.float32)
        config = CodebookConfig(codebook_size=8, group_size=128, seed=42)
        
        results = generate_codebook_groupwise(values, config)
        
        assert len(results) == 8  # 1000 / 128 = 7.8 -> 8 groups
        
        for codebook, indices, errors in results:
            assert len(codebook) == 8
            assert len(indices) <= 128
            assert len(errors) > 0
    
    def test_generate_codebook_groupwise_invalid(self):
        config = CodebookConfig(codebook_size=8, group_size=None)
        
        with pytest.raises(ValueError):
            generate_codebook_groupwise(np.ones(100), config)


# ---------- Analysis Tests ----------

class TestWeightDistributionAnalysis:
    def test_analyze_gaussian(self):
        tensor = np.random.randn(1000).astype(np.float32)
        stats = analyze_weight_distribution(tensor)
        
        assert stats.count == 1000
        assert stats.min_val < stats.max_val
        assert isinstance(stats.percentiles, dict)
        assert 50 in stats.percentiles  # median
        assert "bins" in stats.histogram
    
    def test_analyze_empty(self):
        tensor = np.array([], dtype=np.float32)
        stats = analyze_weight_distribution(tensor)
        
        assert stats.count == 0
    
    def test_analyze_constant(self):
        tensor = np.ones(100, dtype=np.float32) * 0.5
        stats = analyze_weight_distribution(tensor)
        
        assert stats.min_val == 0.5
        assert stats.max_val == 0.5
        assert stats.std == 0.0
        assert stats.skew == 0.0
    
    def test_histogram_analysis(self):
        tensor = np.random.randn(1000).astype(np.float32)
        hist = histogram_analysis(tensor, bins=20)
        
        assert len(hist["counts"]) == 20
        assert len(hist["bin_edges"]) == 21
        assert len(hist["bin_centers"]) == 20
        assert hist["total_count"] == 1000
        assert "bin_width" in hist
        assert "range" in hist
    
    def test_optimal_codebook_estimate(self):
        tensor = np.random.randn(1000).astype(np.float32)
        estimate = compute_optimal_codebook_size_estimate(tensor, target_bits=4.0)
        
        assert "entropy_estimate" in estimate
        assert "estimated_codebook_size" in estimate
        assert estimate["estimated_codebook_size"] == 16  # 2^4


# ---------- Storage Tests ----------

class TestStorageAccounting:
    def test_calculate_codebook_storage_global(self):
        tensor = np.random.randn(1000).astype(np.float32)
        config = CodebookConfig(codebook_size=16, group_size=None, seed=42)
        
        codebook, indices, _ = generate_codebook_kmeans(tensor, config)
        cb_tensor = create_codebook_tensor(tensor, config, [codebook], [indices])
        
        storage = calculate_codebook_storage(cb_tensor)
        
        assert storage.codebook_size == 16
        assert storage.group_count == 1
        assert storage.index_bit_width == 4
        assert storage.total_bytes > 0
        assert storage.compression_ratio > 1
    
    def test_calculate_codebook_storage_groupwise(self):
        tensor = np.random.randn(1000).astype(np.float32)
        config = CodebookConfig(codebook_size=8, group_size=128, seed=42)
        
        group_results = generate_codebook_groupwise(tensor, config)
        codebooks = [r[0] for r in group_results]
        indices_list = [r[1] for r in group_results]
        cb_tensor = create_codebook_tensor(tensor, config, codebooks, indices_list)
        
        storage = calculate_codebook_storage(cb_tensor)
        
        assert storage.codebook_size == 8
        assert storage.group_count == 8
        assert storage.index_bit_width == 3
        assert storage.codebook_bytes == 8 * 8 * 4  # 8 groups * 8 entries * 4 bytes
    
    def test_effective_bits_per_weight(self):
        # Global codebook
        bpw = calculate_effective_bits_per_weight(
            total_elements=1000,
            index_bit_width=4,
            num_codebooks=1,
            codebook_size=16,
        )
        # 1000*4 + 1*16*32 = 4000 + 512 = 4512 bits -> 4.512 bpw
        assert abs(bpw - 4.512) < 0.01
        
        # Group-wise
        bpw = calculate_effective_bits_per_weight(
            total_elements=1000,
            index_bit_width=3,
            num_codebooks=8,
            codebook_size=8,
        )
        # 1000*3 + 8*8*32 = 3000 + 2048 = 5048 bits -> 5.048 bpw
        assert abs(bpw - 5.048) < 0.01
    
    def test_uniform_storage_comparison(self):
        # 1000 FP32 weights = 4000 bytes
        uniform = calculate_uniform_storage_comparison(4000, 4, 128, 16)
        
        assert "total_bytes" in uniform
        assert "bits_per_weight" in uniform
        assert "compression_ratio" in uniform
        assert uniform["bits_per_weight"] > 4.0  # Includes scale overhead


# ---------- Experiment Tests ----------

class TestExperimentConfig:
    def test_config_creation(self):
        config = CodebookExperimentConfig(
            experiment_name="test",
            tensor_shape=(100, 100),
            source_dtype=np.float32,
            codebook_size=16,
            group_size=128,
        )
        
        assert config.codebook_size == 16
        assert config.group_size == 128
    
    def test_config_to_codebook_config(self):
        config = CodebookExperimentConfig(
            experiment_name="test",
            tensor_shape=(100,),
            source_dtype=np.float32,
            codebook_size=16,
            group_size=128,
            max_iterations=10,
            tolerance=1e-3,
            seed=123,
            initialization="kmeans++",
        )
        
        cb_config = config.to_codebook_config()
        
        assert cb_config.codebook_size == 16
        assert cb_config.group_size == 128
        assert cb_config.max_iterations == 10
        assert cb_config.tolerance == 1e-3
        assert cb_config.seed == 123
        assert cb_config.initialization == "kmeans++"
    
    def test_custom_tensor_required(self):
        with pytest.raises(ValueError):
            CodebookExperimentConfig(
                experiment_name="test",
                tensor_shape=(10,),
                source_dtype=np.float32,
                codebook_size=4,
                tensor_generator="custom",
            )


class TestTensorGeneration:
    def test_random_normal(self):
        tensor = generate_test_tensor((100,), np.float32, "random_normal", seed=42)
        assert tensor.shape == (100,)
        assert tensor.dtype == np.float32
    
    def test_random_uniform(self):
        tensor = generate_test_tensor((100,), np.float32, "random_uniform", seed=42)
        assert np.all(tensor >= -1) and np.all(tensor <= 1)
    
    def test_heavy_tail(self):
        tensor = generate_test_tensor((1000,), np.float32, "heavy_tail", seed=42)
        # Heavy tail should have higher kurtosis
        stats = analyze_weight_distribution(tensor)
        assert stats.kurtosis > 0  # Heavy tailed
    
    def test_transformer_like(self):
        tensor = generate_test_tensor((1000,), np.float32, "transformer_like", seed=42)
        stats = analyze_weight_distribution(tensor)
        # Should have some outliers
        assert stats.max_val > 3 * stats.std
    
    def test_custom_tensor(self):
        custom = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = generate_test_tensor((3,), np.float32, "custom", seed=42, custom=custom)
        assert np.array_equal(tensor, custom)


class TestCodebookExperiment:
    def test_run_codebook_experiment_global(self):
        config = CodebookExperimentConfig(
            experiment_name="test_global",
            tensor_shape=(256,),
            source_dtype=np.float32,
            codebook_size=16,
            group_size=None,
            seed=42,
            tensor_generator="random_normal",
        )
        
        result = run_codebook_experiment(config)
        
        assert result.experiment_id is not None
        assert result.codebook_error is not None
        assert result.codebook_storage is not None
        assert result.uniform_error is not None
        assert result.uniform_storage is not None
        assert result.weight_distribution.count == 256
    
    def test_run_codebook_experiment_groupwise(self):
        config = CodebookExperimentConfig(
            experiment_name="test_groupwise",
            tensor_shape=(256,),
            source_dtype=np.float32,
            codebook_size=8,
            group_size=128,
            seed=42,
            tensor_generator="random_normal",
        )
        
        result = run_codebook_experiment(config)
        
        assert result.codebook_storage.group_count == 2
        assert result.codebook_storage.codebook_size == 8
    
    def test_run_uniform_vs_codebook_experiment(self):
        tensor = np.random.randn(512).astype(np.float32)
        
        results = run_uniform_vs_codebook_experiment(
            tensor,
            codebook_sizes=[4, 8],
            group_sizes=[None, 128],
            seed=42,
        )
        
        # 2 codebook sizes * 2 group sizes = 4 experiments
        assert len(results) == 4
        
        for r in results:
            assert r.codebook_error is not None
            assert r.uniform_error is not None
    
    def test_run_binary_codebook_experiment(self):
        tensor = np.random.randn(256).astype(np.float32)
        
        result = run_binary_codebook_experiment(tensor, seed=42)
        
        assert "fixed_binary" in result
        assert "learned_binary" in result
        assert "improvement" in result
        
        # Learned should have different values than {-1, +1}
        learned_cbs = result["learned_binary"]["codebooks"]
        assert len(learned_cbs) > 0
        for cb in learned_cbs:
            assert len(cb) == 2
            # Should not be exactly [-1, 1]
            assert not (np.isclose(cb[0], -1.0) and np.isclose(cb[1], 1.0))


# ---------- Edge Cases ----------

class TestEdgeCases:
    def test_empty_tensor(self):
        config = CodebookConfig(codebook_size=4)
        values = np.array([], dtype=np.float32)
        
        codebook, indices, errors = generate_codebook_kmeans(values, config)
        assert len(codebook) == 0
        assert len(indices) == 0
    
    def test_constant_tensor(self):
        values = np.ones(100, dtype=np.float32) * 0.5
        config = CodebookConfig(codebook_size=4, seed=42)
        
        codebook, indices, errors = generate_codebook_kmeans(values, config)
        
        # All values should map to same index
        assert len(np.unique(indices)) == 1
    
    def test_single_value_tensor(self):
        values = np.array([42.0], dtype=np.float32)
        config = CodebookConfig(codebook_size=4, seed=42)
        
        codebook, indices, _ = generate_codebook_kmeans(values, config)
        assert len(indices) == 1
        assert indices[0] == 0
    
    def test_non_divisible_tensor_groupwise(self):
        # 100 elements, group_size=128 -> 1 group
        values = np.random.randn(100).astype(np.float32)
        config = CodebookConfig(codebook_size=8, group_size=128, seed=42)
        
        results = generate_codebook_groupwise(values, config)
        assert len(results) == 1
        
        # 200 elements, group_size=128 -> 2 groups
        values = np.random.randn(200).astype(np.float32)
        results = generate_codebook_groupwise(values, config)
        assert len(results) == 2
        assert results[0][1].shape[0] == 128
        assert results[1][1].shape[0] == 72


if __name__ == "__main__":
    pytest.main([__file__, "-v"])