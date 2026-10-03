"""
Weight distribution analysis utilities for LDMARK codebook experiments.

Provides statistical analysis of weight distributions to understand
suitability for non-uniform quantization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class WeightDistributionStats:
    """Statistical summary of weight distribution."""
    count: int
    min_val: float
    max_val: float
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    skew: float
    kurtosis: float
    percentiles: Dict[int, float]  # e.g., {1: -0.5, 99: 0.5}
    histogram: Dict[str, Any]  # bins, counts, bin_edges
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "min": self.min_val,
            "max": self.max_val,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "percentiles": self.percentiles,
            "histogram": self.histogram,
        }


def analyze_weight_distribution(
    tensor: np.ndarray,
    percentiles: Optional[List[int]] = None,
    histogram_bins: int = 50,
) -> WeightDistributionStats:
    """
    Analyze the distribution of weight values in a tensor.
    
    Args:
        tensor: Input tensor
        percentiles: List of percentiles to compute (default: 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)
        histogram_bins: Number of histogram bins
        
    Returns:
        WeightDistributionStats with comprehensive statistics
    """
    if percentiles is None:
        percentiles = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    
    flat = tensor.flatten().astype(np.float64)
    n = len(flat)
    
    if n == 0:
        return WeightDistributionStats(
            count=0, min_val=0, max_val=0, mean=0, std=0, median=0,
            q25=0, q75=0, skew=0, kurtosis=0, percentiles={}, histogram={}
        )
    
    # Basic statistics
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    median = float(np.median(flat))
    q25 = float(np.percentile(flat, 25))
    q75 = float(np.percentile(flat, 75))
    
    # Skewness and kurtosis
    if std > 0:
        skew = float(np.mean(((flat - mean) / std) ** 3))
        kurtosis = float(np.mean(((flat - mean) / std) ** 4)) - 3
    else:
        skew = 0.0
        kurtosis = 0.0
    
    # Percentiles
    perc_vals = {}
    for p in percentiles:
        perc_vals[p] = float(np.percentile(flat, p))
    
    # Histogram
    counts, bin_edges = np.histogram(flat, bins=histogram_bins)
    histogram = {
        "bins": len(counts),
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "density": (counts / n).tolist(),
    }
    
    return WeightDistributionStats(
        count=n,
        min_val=min_val,
        max_val=max_val,
        mean=mean,
        std=std,
        median=median,
        q25=q25,
        q75=q75,
        skew=skew,
        kurtosis=kurtosis,
        percentiles=perc_vals,
        histogram=histogram,
    )


def histogram_analysis(
    tensor: np.ndarray,
    bins: int = 50,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute histogram of tensor values.
    
    Args:
        tensor: Input tensor
        bins: Number of bins
        range_min: Minimum value (default: tensor min)
        range_max: Maximum value (default: tensor max)
        
    Returns:
        Dictionary with histogram data
    """
    flat = tensor.flatten()
    if range_min is None:
        range_min = float(np.min(flat))
    if range_max is None:
        range_max = float(np.max(flat))
    
    if range_min == range_max:
        counts = np.array([len(flat)])
        bin_edges = np.array([range_min, range_min + 1])
    else:
        counts, bin_edges = np.histogram(flat, bins=bins, range=(range_min, range_max))
    
    return {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
        "bin_width": float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 0,
        "total_count": int(np.sum(counts)),
        "range": [float(range_min), float(range_max)],
    }


def compute_optimal_codebook_size_estimate(
    tensor: np.ndarray,
    target_bits: float = 4.0,
) -> Dict[str, Any]:
    """
    Estimate optimal codebook size based on weight distribution.
    
    Uses heuristic: effective entropy of distribution.
    
    Args:
        tensor: Input tensor
        target_bits: Target bits per weight
        
    Returns:
        Dictionary with estimates
    """
    flat = tensor.flatten()
    n = len(flat)
    
    if n == 0:
        return {"estimated_codebook_size": 0, "entropy_estimate": 0}
    
    # Estimate entropy using histogram
    stats = analyze_weight_distribution(tensor)
    hist = stats.histogram
    counts = np.array(hist["counts"])
    probs = counts / n
    probs = probs[probs > 0]
    
    entropy = -np.sum(probs * np.log2(probs))
    
    # Estimate codebook size for target bits
    # Using high-rate quantization theory: R ≈ H + log2(sqrt(12)*Δ) for uniform
    # For codebook: R ≈ log2(K) where K is codebook size
    # So K ≈ 2^R
    estimated_k = int(np.ceil(2 ** target_bits))
    
    return {
        "entropy_estimate": float(entropy),
        "estimated_codebook_size": estimated_k,
        "target_bits": target_bits,
        "theoretical_min_bits": entropy,
    }


def compare_distributions(
    tensor1: np.ndarray,
    tensor2: np.ndarray,
    name1: str = "tensor1",
    name2: str = "tensor2",
) -> Dict[str, Any]:
    """
    Compare two weight distributions.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        name1: Name for first tensor
        name2: Name for second tensor
        
    Returns:
        Comparison dictionary
    """
    stats1 = analyze_weight_distribution(tensor1)
    stats2 = analyze_weight_distribution(tensor2)
    
    return {
        name1: stats1.to_dict(),
        name2: stats2.to_dict(),
        "difference": {
            "mean_diff": stats1.mean - stats2.mean,
            "std_diff": stats1.std - stats2.std,
            "min_diff": stats1.min_val - stats2.min_val,
            "max_diff": stats1.max_val - stats2.max_val,
            "skew_diff": stats1.skew - stats2.skew,
            "kurtosis_diff": stats1.kurtosis - stats2.kurtosis,
        },
    }