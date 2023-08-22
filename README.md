# Fast Vector Similarity Library

## Introduction

The Fast Vector Similarity Library is designed to provide efficient computation of various similarity measures between vectors. It is suitable for tasks such as data analysis, machine learning, and statistics, where measuring the relationship between vectors is essential. Written in Rust, the library offers a high-performance solution and can be easily integrated with Python through provided bindings.

## Features

### Similarity Measures

The library implements several popular similarity measures, including:

1. **Spearman's Rank-Order Correlation (`spearman_rho`)**
2. **Kendall's Tau Rank Correlation (`kendall_tau`)**
3. **Approximate Distance Correlation (`approximate_distance_correlation`)**
4. **Jensen-Shannon Similarity (`jensen_shannon_similarity`)**
5. **Hoeffding's D Measure (`hoeffding_d`)**

### Bootstrapping Technique

The library supports bootstrapping for robust similarity computation. This technique involves randomly resampling the dataset to estimate the distribution of the similarity measures, providing more confidence in the results.

### Performance Optimizations

To achieve high efficiency, the library leverages:

- **Parallel Computing**: Using the `rayon` crate, computations are parallelized across available CPU cores.
- **Vectorized Operations**: The library leverages efficient vectorized operations provided by the `ndarray` crate.

## Python Bindings

The library includes Python bindings to enable seamless integration with Python code. Functions `py_compute_vector_similarity_stats` and `py_compute_bootstrapped_similarity_stats` are exposed, allowing you to compute similarity statistics and bootstrapped similarity statistics, respectively.

## Installation

### Rust

Include the library in your Rust project by adding it to your Cargo.toml file.

### Python

You can install the Python bindings for the Fast Vector Similarity Library directly from PyPI using the following command:

```bash
pip install fast_vector_similarity
```

This command will download and install the package, making it available for use in your Python projects.


### Example

```python
import time
import numpy as np
import json
import fast_vector_similarity as fvs

def main():
    length_of_test_vectors = 15000
    print(f"Generating 2 test vectors of length {length_of_test_vectors}...")
    vector_1 = np.linspace(0., length_of_test_vectors - 1, length_of_test_vectors)
    vector_2 = vector_1 ** 0.2 + np.random.rand(length_of_test_vectors)
    print("Generated vector_1 using linear spacing and vector_2 using vector_1 with a power of 0.2 and some random noise.\n")

    similarity_measure = "all" # Or specify a particular measure
    params = {
        "vector_1": vector_1.tolist(),
        "vector_2": vector_2.tolist(),
        "similarity_measure": similarity_measure
    }
    
    # Time the exact similarity calculation
    print("Computing Exact Similarity Measures...")
    start_time_exact = time.time()
    similarity_stats_str = fvs.py_compute_vector_similarity_stats(json.dumps(params))
    similarity_stats_json = json.loads(similarity_stats_str)
    elapsed_time_exact = time.time() - start_time_exact
    print(f"Time taken for exact calculation: {elapsed_time_exact:.5f} seconds")

    # Print results
    print("_______________________________________________________________________________________________________________________________________________\n")
    print("Spearman's rho:", similarity_stats_json["spearman_rho"])
    print("Kendall's tau:", similarity_stats_json["kendall_tau"])
    print("Distance Correlation:", similarity_stats_json["approximate_distance_correlation"])
    print("Jensen-Shannon Similarity:", similarity_stats_json["jensen_shannon_similarity"])
    print("Hoeffding's D:", similarity_stats_json["hoeffing_d"])
    print("_______________________________________________________________________________________________________________________________________________\n")

    # Bootstrapped calculations
    number_of_bootstraps = 2000
    n = 15
    sample_size = int(length_of_test_vectors / n)

    print(f"Computing Bootstrapped Similarity Measures with {number_of_bootstraps} bootstraps and a sample size of {sample_size}...")
    start_time_bootstrapped = time.time()
    params_bootstrapped = {
        "x": vector_1.tolist(),
        "y": vector_2.tolist(),
        "sample_size": sample_size,
        "number_of_bootstraps": number_of_bootstraps,
        "similarity_measure": similarity_measure
    }
    bootstrapped_similarity_stats_str = fvs.py_compute_bootstrapped_similarity_stats(json.dumps(params_bootstrapped))
    bootstrapped_similarity_stats_json = json.loads(bootstrapped_similarity_stats_str)
    elapsed_time_bootstrapped = time.time() - start_time_bootstrapped
    print(f"Time taken for bootstrapped calculation: {elapsed_time_bootstrapped:.5f} seconds")

    time_difference = abs(elapsed_time_exact - elapsed_time_bootstrapped)
    print(f"Time difference between exact and robust bootstrapped calculations: {time_difference:.5f} seconds")
    
    # Print bootstrapped results
    print("_______________________________________________________________________________________________________________________________________________\n")
    print("Number of Bootstrap Iterations:", bootstrapped_similarity_stats_json["number_of_bootstraps"])
    print("Bootstrap Sample Size:", bootstrapped_similarity_stats_json["sample_size"])
    print("\nRobust Spearman's rho:", bootstrapped_similarity_stats_json["spearman_rho"])
    print("Robust Kendall's tau:", bootstrapped_similarity_stats_json["kendall_tau"])
    print("Robust Distance Correlation:", bootstrapped_similarity_stats_json["approximate_distance_correlation"])
    print("Robust Jensen-Shannon Similarity:", bootstrapped_similarity_stats_json["jensen_shannon_similarity"])
    print("Robust Hoeffding's D:", bootstrapped_similarity_stats_json["hoeffing_d"])
    print("_______________________________________________________________________________________________________________________________________________\n")

```

## Usage

### Rust

You can utilize the core functionality in Rust by calling functions like `compute_vector_similarity_stats` and `compute_bootstrapped_similarity_stats` with appropriate parameters.

### Python

Install the provided Python package and use the functions as demonstrated in the example above.

## About the Different Similarity Measures

1. **Spearman's Rank-Order Correlation (`spearman_rho`)**:
   Spearman's rho assesses the strength and direction of the monotonic relationship between two ranked variables. Unlike Pearson's correlation, it does not assume linearity and is less sensitive to outliers, making it suitable for non-linear relationships.

2. **Kendall's Tau Rank Correlation (`kendall_tau`)**:
   Kendall's tau measures the ordinal association between two variables. It's valuable for its ability to handle ties and its interpretability as a probability. Unlike other correlation measures, it's based on the difference between concordant and discordant pairs, making it robust and versatile.

3. **Approximate Distance Correlation (`approximate_distance_correlation`)**:
   Distance correlation quantifies both linear and non-linear dependencies between variables. It has the powerful property of being zero if and only if the variables are independent. This makes it a more comprehensive measure of association than traditional correlation metrics.

4. **Jensen-Shannon Similarity (`jensen_shannon_similarity`)**:
   Jensen-Shannon Similarity is derived from the Jensen-Shannon Divergence, a symmetrical and smoothed version of the Kullback-Leibler divergence. It quantifies the similarity between two probability distributions and is especially useful in comparing distributions that may have non-overlapping support.

5. **Hoeffding's D Measure (`hoeffding_d`)**:
   Hoeffding's D is a non-parametric measure that detects complex non-linear relationships between variables. The D statistic is robust against various alternatives to independence, including non-monotonic relationships. It's particularly useful when the nature of the relationship between variables is unknown or unconventional.

Each of these measures has unique properties and applicability, providing a comprehensive toolkit for understanding the relationships between variables in different contexts. By including both classical and more specialized measures, the library offers flexibility and depth in analyzing vector similarities.

## Conclusion

The Fast Vector Similarity Library offers an extensive set of tools for computing various similarity measures between vectors. With the inclusion of performance optimizations and Python bindings, it can be an essential part of any data analysis or machine learning pipeline.

For further details and customization, please refer to the source code.