"""
Codebook generation algorithms for LDMARK experimental compression.

Provides deterministic methods for generating codebooks from weight data.
"""

import numpy as np
from typing import List, Optional, Tuple
from .representation import CodebookConfig, CodebookGroup


def initialize_codebook_uniform(
    values: np.ndarray,
    codebook_size: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Initialize codebook with uniform spacing between min and max.
    
    Args:
        values: Input weight values
        codebook_size: Number of codebook entries
        seed: Random seed for reproducibility
        
    Returns:
        Codebook array of shape (codebook_size,)
    """
    rng = np.random.default_rng(seed)
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmin == vmax:
        return np.full(codebook_size, vmin, dtype=np.float32)
    return np.linspace(vmin, vmax, codebook_size, dtype=np.float32)


def initialize_codebook_percentile(
    values: np.ndarray,
    codebook_size: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Initialize codebook using percentiles of the data distribution.
    
    Args:
        values: Input weight values
        codebook_size: Number of codebook entries
        seed: Random seed for reproducibility
        
    Returns:
        Codebook array of shape (codebook_size,)
    """
    percentiles = np.linspace(0, 100, codebook_size)
    return np.percentile(values, percentiles).astype(np.float32)


def initialize_codebook_random(
    values: np.ndarray,
    codebook_size: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Initialize codebook by randomly sampling from values.
    
    Args:
        values: Input weight values
        codebook_size: Number of codebook entries
        seed: Random seed for reproducibility
        
    Returns:
        Codebook array of shape (codebook_size,)
    """
    rng = np.random.default_rng(seed)
    flat = values.flatten()
    if len(flat) <= codebook_size:
        return flat.astype(np.float32)
    sampled = rng.choice(flat, size=codebook_size, replace=False)
    return np.sort(sampled).astype(np.float32)


def initialize_codebook_kmeanspp(
    values: np.ndarray,
    codebook_size: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Initialize codebook using k-means++ style initialization.
    
    Args:
        values: Input weight values
        codebook_size: Number of codebook entries
        seed: Random seed for reproducibility
        
    Returns:
        Codebook array of shape (codebook_size,)
    """
    rng = np.random.default_rng(seed)
    flat = values.flatten().astype(np.float32)
    n = len(flat)
    
    if n <= codebook_size:
        return np.sort(flat)[:codebook_size]
    
    # Pick first centroid randomly
    centroids = [flat[rng.integers(n)]]
    
    for _ in range(1, codebook_size):
        # Compute distances to nearest centroid
        dists = np.min([(flat - c) ** 2 for c in centroids], axis=0)
        # Choose next centroid with probability proportional to squared distance
        probs = dists / np.sum(dists)
        next_idx = rng.choice(n, p=probs)
        centroids.append(flat[next_idx])
    
    return np.sort(np.array(centroids, dtype=np.float32))


def assign_to_nearest_centroid(
    values: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Assign each value to its nearest centroid.
    
    Args:
        values: Input values (any shape)
        centroids: Centroid values (1D array)
        
    Returns:
        Indices array of same shape as values
    """
    flat = values.flatten()
    # Compute distances: (n_values, n_centroids)
    dists = np.abs(flat[:, np.newaxis] - centroids[np.newaxis, :])
    indices = np.argmin(dists, axis=1).astype(np.uint8)
    return indices.reshape(values.shape)


def recompute_centroids(
    values: np.ndarray,
    indices: np.ndarray,
    codebook_size: int,
) -> np.ndarray:
    """
    Recompute centroids as mean of assigned values.
    
    Args:
        values: Original values
        indices: Assigned indices
        codebook_size: Number of centroids
        
    Returns:
        Updated centroids array
    """
    flat = values.flatten()
    idx_flat = indices.flatten()
    new_centroids = np.zeros(codebook_size, dtype=np.float32)
    counts = np.zeros(codebook_size, dtype=np.int32)
    
    for i, val in zip(idx_flat, flat):
        new_centroids[i] += val
        counts[i] += 1
    
    # Handle empty clusters by keeping old centroid value
    for k in range(codebook_size):
        if counts[k] > 0:
            new_centroids[k] /= counts[k]
    
    return new_centroids


def compute_codebook_error(
    values: np.ndarray,
    centroids: np.ndarray,
    indices: np.ndarray,
) -> float:
    """
    Compute reconstruction error (MSE) for current codebook assignment.
    
    Args:
        values: Original values
        centroids: Current centroids
        indices: Assigned indices
        
    Returns:
        Mean squared error
    """
    flat = values.flatten()
    idx_flat = indices.flatten()
    reconstructed = centroids[idx_flat]
    mse = float(np.mean((flat - reconstructed) ** 2))
    return mse


def generate_codebook_kmeans(
    values: np.ndarray,
    config: CodebookConfig,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Generate codebook using k-means style iterative algorithm.
    
    Args:
        values: Input tensor values
        config: Codebook configuration
        
    Returns:
        Tuple of (codebook, indices, error_history)
    """
    rng = np.random.default_rng(config.seed)
    flat = values.flatten().astype(np.float32)
    
    if len(flat) == 0:
        return np.array([]), np.array([], dtype=np.uint8), []
    
    if len(flat) <= config.codebook_size:
        # Fewer values than codebook entries - use unique values
        unique = np.unique(flat)
        if len(unique) >= config.codebook_size:
            codebook = unique[:config.codebook_size]
        else:
            codebook = np.pad(unique, (0, config.codebook_size - len(unique)), 
                             mode='edge')
        indices = assign_to_nearest_centroid(flat, codebook)
        return codebook, indices, [0.0]
    
    # Initialize centroids
    if config.initialization == "uniform":
        centroids = initialize_codebook_uniform(flat, config.codebook_size, config.seed)
    elif config.initialization == "percentile":
        centroids = initialize_codebook_percentile(flat, config.codebook_size, config.seed)
    elif config.initialization == "random":
        centroids = initialize_codebook_random(flat, config.codebook_size, config.seed)
    elif config.initialization == "kmeans++":
        centroids = initialize_codebook_kmeanspp(flat, config.codebook_size, config.seed)
    else:
        raise ValueError(f"Unknown initialization: {config.initialization}")
    
    error_history = []
    
    for iteration in range(config.max_iterations):
        # Assign
        indices = assign_to_nearest_centroid(flat, centroids)
        
        # Compute error
        error = compute_codebook_error(flat, centroids, indices)
        error_history.append(error)
        
        # Check convergence
        if iteration > 0 and abs(error_history[-2] - error) < config.tolerance:
            break
        
        # Recompute
        new_centroids = recompute_centroids(flat, indices, config.codebook_size)
        
        # Handle empty clusters by reinitializing
        empty_mask = np.bincount(indices, minlength=config.codebook_size) == 0
        if np.any(empty_mask):
            # Reinitialize empty centroids from random data points
            empty_count = np.sum(empty_mask)
            if empty_count > 0:
                # Use values far from existing centroids
                dists = np.min([(flat - c) ** 2 for c in centroids], axis=0)
                far_indices = np.argsort(dists)[-empty_count:]
                new_centroids[empty_mask] = flat[far_indices]
        
        centroids = new_centroids
    
    # Final assignment
    indices = assign_to_nearest_centroid(flat, centroids)
    final_error = compute_codebook_error(flat, centroids, indices)
    if error_history:
        error_history[-1] = final_error
    else:
        error_history.append(final_error)
    
    return centroids.astype(np.float32), indices, error_history


def generate_codebook_groupwise(
    values: np.ndarray,
    config: CodebookConfig,
) -> List[Tuple[np.ndarray, np.ndarray, List[float]]]:
    """
    Generate codebooks for each group separately.
    
    Args:
        values: Input tensor values
        config: Codebook configuration (must be group-wise)
        
    Returns:
        List of (codebook, indices, error_history) per group
    """
    if not config.is_group_wise:
        raise ValueError("Config must be group-wise for group-wise codebook generation")
    
    flat = values.flatten().astype(np.float32)
    group_size = config.group_size
    num_groups = (len(flat) + group_size - 1) // group_size
    
    results = []
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, len(flat))
        group_values = flat[start:end]
        
        # Create group config with modified seed
        group_config = CodebookConfig(
            codebook_size=config.codebook_size,
            group_size=None,  # Single group
            max_iterations=config.max_iterations,
            tolerance=config.tolerance,
            seed=config.seed + g,
            initialization=config.initialization,
        )
        
        codebook, indices, errors = generate_codebook_kmeans(group_values, group_config)
        results.append((codebook, indices, errors))
    
    return results