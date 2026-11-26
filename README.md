# Fast Vector Similarity Library

## Introduction

The Fast Vector Similarity Library is a high-performance Rust-based tool for efficiently computing similarity measures between vectors. It is ideal for data analysis, machine learning, and statistical tasks where comparing vectors is essential. The library now includes several advanced measures, performance optimizations, and Python bindings, allowing seamless integration with Python workflows.

## Features

### Similarity Measures

The library implements a range of classical and modern similarity measures, including:

1. **Spearman's Rank-Order Correlation (`spearman_rho`)**
2. **Kendall's Tau Rank Correlation (`kendall_tau`)** (optimized for faster computation with large datasets)
3. **Approximate Distance Correlation (`approximate_distance_correlation`)** (linear working storage)
4. **Jensen-Shannon Dependency Measure (`jensen_shannon_dependency_measure`)** (revised for improved utility in dependency measurement)
5. **Hoeffding's D Measure (`hoeffding_d`)**
6. **Normalized Mutual Information (`normalized_mutual_information`)** (newly introduced for analyzing variable dependence)

### Bootstrapping Technique

The library includes robust bootstrapping functionality to estimate the distribution of similarity measures. Bootstrapping offers improved confidence in the results by randomly resampling the dataset multiple times.

For each measure, the summary removes `floor(B / 4)` scores from each tail, where
`B` is the number of bootstrap replicates. It reports the mean and population
standard deviation of the remaining scores. Both tails lose the same number of
scores, including when `B` is not divisible by four; with two or three replicates,
all scores are retained.

### Performance Optimizations

Several enhancements have been introduced for optimal efficiency:

- **Parallel Processing**: Utilizing the `rayon` crate for parallel computation, ensuring that operations scale with the number of CPU cores.
- **Efficient Algorithms**: Algorithms like merge sort are used for inversion counting, which improves the speed of measures like Kendall's Tau.
- **Vectorized Operations**: Many functions leverage vectorized operations using the `ndarray` crate to maximize performance in Rust.

### Benchmarking and Verification

The library now includes a benchmarking suite that verifies the correctness of the numerical results while measuring performance gains from recent improvements. This ensures that any changes in computational speed do not affect accuracy (except in intended changes like the Jensen-Shannon measure).

## Python Bindings

Seamless integration with Python is possible via bindings that expose core functionality. The library provides two key functions for Python users:

- `py_compute_vector_similarity_stats`: For computing various vector similarity measures.
- `py_compute_bootstrapped_similarity_stats`: For bootstrapping-based similarity calculations.

Both functions return results in JSON format, making them easy to work with in Python environments.

## Installation

### Rust

Add the library to your Rust project by including it in your `Cargo.toml` file.

### Python

The Python bindings can be installed directly from PyPI:

```bash
pip install fast_vector_similarity
```

## Use with Text Embedding Vectors from LLMs

This library is highly compatible with modern language models like Llama2, enabling easy analysis of text embeddings. It integrates with the output of services like [Llama2 Embeddings FastAPI Service](https://github.com/Dicklesworthstone/llama_embeddings_fastapi_service) and can handle high-dimensional embeddings (e.g., 4096-dimensional vectors).

### Example Workflow

1. **Load Embeddings into a DataFrame**: Convert text embeddings from a JSON format into a Pandas DataFrame.
2. **Compute Similarities**: Use the Fast Vector Similarity Library to compute similarity measures between embeddings, leveraging the optimized functions.
3. **Analyze Results**: Generate a ranked list of most similar vectors based on measures like Hoeffding's D.

### Example Python Code

Here’s a Python snippet demonstrating the use of the library with large embedding vectors:

```python
import time
import numpy as np
import json
import pandas as pd
import fast_vector_similarity as fvs
from random import choice

def convert_embedding_json_to_pandas_df(file_path):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Extract the text and embeddings
    texts = [item['text'] for item in data]
    embeddings = [item['embedding'] for item in data]
    # Determine the total number of vectors and the dimensions of each vector
    total_vectors = len(embeddings)
    vector_dimensions = len(embeddings[0]) if total_vectors > 0 else 0
    # Print the total number of vectors and dimensions
    print(f"Total number of vectors: {total_vectors}")
    print(f"Dimensions of each vector: {vector_dimensions}")
    # Convert the embeddings into a DataFrame
    df = pd.DataFrame(embeddings, index=texts)
    return df

def apply_fvs_to_vector(row_embedding, query_embedding):
    params = {
        "vector_1": query_embedding.tolist(),
        "vector_2": row_embedding.tolist(),
        "similarity_measure": "all"
    }
    similarity_stats_str = fvs.py_compute_vector_similarity_stats(json.dumps(params))
    return json.loads(similarity_stats_str)

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
    print("Jensen-Shannon Dependency Measure:", similarity_stats_json["jensen_shannon_dependency_measure"])
    print("Normalized Mutual Information:", similarity_stats_json["normalized_mutual_information"])
    print("Hoeffding's D:", similarity_stats_json["hoeffding_d"])
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
    print("Robust Jensen-Shannon  Dependency Measure:", bootstrapped_similarity_stats_json["jensen_shannon_dependency_measure"])
    print("Robust Normalized Mutual Information:", bootstrapped_similarity_stats_json["normalized_mutual_information"])
    print("Robust Hoeffding's D:", bootstrapped_similarity_stats_json["hoeffding_d"])
    print("_______________________________________________________________________________________________________________________________________________\n")

    # Compute the differences between exact and bootstrapped results
    measures = ["spearman_rho", "kendall_tau", "approximate_distance_correlation", "jensen_shannon_dependency_measure", "normalized_mutual_information", "hoeffding_d"]
    for measure in measures:
        exact_value = similarity_stats_json[measure]
        bootstrapped_value = bootstrapped_similarity_stats_json[measure]
        absolute_difference = abs(exact_value - bootstrapped_value)
        percentage_difference = (absolute_difference / exact_value) * 100

        print(f"\nDifference between exact and bootstrapped {measure}: {absolute_difference}")
        print(f"Difference as % of the exact value: {percentage_difference:.2f}%")

    print("Now testing with a larger dataset, using sentence embedddings from Llama2 (4096-dimensional vectors) on some Shakespeare Sonnets...")
    # Load the embeddings into a DataFrame
    input_file_path = "sample_input_files/Shakespeare_Sonnets_small.json"
    embeddings_df = convert_embedding_json_to_pandas_df(input_file_path)
    
    # Select a random row for the query embedding
    query_embedding_index = choice(embeddings_df.index)
    query_embedding = embeddings_df.loc[query_embedding_index]
    print(f"Selected query embedding for sentence: `{query_embedding_index}`")

    # Remove the selected row from the DataFrame
    embeddings_df = embeddings_df.drop(index=query_embedding_index)

    # Apply the function to each row of embeddings_df
    json_outputs = embeddings_df.apply(lambda row: apply_fvs_to_vector(row, query_embedding), axis=1)

    # Create a DataFrame from the list of JSON outputs
    vector_similarity_results_df = pd.DataFrame.from_records(json_outputs)
    vector_similarity_results_df.index = embeddings_df.index

    # Add the required columns to the DataFrame
    columns = ["spearman_rho", "kendall_tau", "approximate_distance_correlation", "jensen_shannon_dependency_measure", "normalized_mutual_information", "hoeffding_d"]
    vector_similarity_results_df = vector_similarity_results_df[columns]
    
    # Sort the DataFrame by the hoeffding_d column in descending order
    vector_similarity_results_df = vector_similarity_results_df.sort_values(by="hoeffding_d", ascending=False)
    
    print("\nTop 10 most similar embedding results by Hoeffding's D:")
    print(vector_similarity_results_df.head(10))

if __name__ == "__main__":
    main()

```

## Usage

### In Rust

The core functions can be used directly within Rust projects. For example, use `compute_vector_similarity_stats` or `compute_bootstrapped_similarity_stats` with appropriate parameters for efficient computations.

### In Python

Install the Python package and use the exposed functions to compute vector similarity or perform bootstrapped analysis, as demonstrated in the example above.

### Non-finite and undefined values

Input vectors must have equal lengths; the Rust APIs assert this precondition, and
the Python wrappers raise `ValueError` when it is violated. The Rust scalar
similarity functions return
`f64::NAN` if either vector is empty or contains `NaN`, positive infinity, or negative
infinity. They do not discard, replace, or rank non-finite observations. Validation
covers the complete vectors before distance-correlation subsampling or bootstrapping.

The Rust JSON statistics functions represent undefined scores as `null`, while
preserving the requested measure names and metadata. For bootstrap results, an
invalid source vector or any undefined replicate makes both the corresponding
average and `stdev_<measure>` value `null`; undefined replicates are not silently
removed or trimmed away. Empty inputs, zero-sized bootstrap samples, and fewer than
two bootstrap replicates also produce `null` summaries.

Finite inputs can still yield an undefined statistic. Spearman's rho and Kendall's
tau-b are undefined when either vector is constant or there are fewer than two
observations. Hoeffding's D requires at least five observations, including within
each bootstrap sample. NMI is undefined when both vectors are constant because
their total marginal entropy is zero. These results are `NaN` directly and `null`
through the JSON APIs, including bootstrapping. Sampling with replacement can create a
constant replicate even from nonconstant source vectors.

The Python wrappers accept JSON arrays of finite numbers. They raise `ValueError`
for malformed JSON (including `NaN`, `Infinity`, and out-of-range numeric literals),
missing or non-array vectors, `null` or nonnumeric vector elements, unequal lengths,
and non-positive or invalid bootstrap counts/sizes. Undefined results from valid
finite inputs are returned as JSON `null`, which `json.loads` converts to `None`.

### Finite scales and array layouts

NMI and the Jensen-Shannon dependency measure use the same equal-width binning
rule. Constant vectors occupy one bin; nonconstant vectors are normalized by
their actual range. This preserves tiny and subnormal differences and handles
opposite-sign finite extremes without adding a fixed epsilon to the range.

Distance correlation normalizes each sampled vector independently before computing
pairwise distances. Translation and scaling preserve the statistic, while bounded
coordinates keep the distance variances stable across tiny and huge overall scales.
A first pass computes row means; a second pass recomputes centered distances and
accumulates covariance and variance with compensated sums. This preserves the
biased centered-distance estimate using two linear row-mean arrays, including the
centered diagonal contributions. The existing convention of zero distance
correlation for a constant sample is preserved.

All Rust entry points taking `Array1<f64>` support reversed and strided owned
arrays. Computation follows logical element order, preserving observation pairs
and ignoring unused backing-storage elements. Contiguous data is borrowed when a
rank function needs a slice; other layouts are copied in logical order.

JSON float parsing uses Serde's `float_roundtrip` mode so serialized `f64` values
retain their distinctions, including adjacent representable values. This matters
for both rank ties and finite-range normalization at the Python boundary.

---

## Detailed Overview of Similarity Measures

### 1. **Spearman's Rank-Order Correlation (`spearman_rho`)**

Spearman’s Rank-Order Correlation is a non-parametric measure of the strength and direction of the monotonic relationship between two variables. Unlike Pearson's correlation, which measures linear relationships, Spearman's correlation can capture non-linear monotonic relationships. This makes it useful in many real-world applications where variables have complex relationships but still follow a consistent directional trend.

**How It Works**:
- First, both input vectors are converted into ranks, where the lowest value is assigned rank 1, the second-lowest rank 2, and so on. If ties are present, the average rank for the tied values is computed.
- Once the ranks are assigned, the measure reduces to computing the Pearson correlation on these ranks. However, the key difference lies in its robustness to non-linearity.
  
The implementation computes Pearson correlation of the average ranks, so tied
observations receive the correct normalization. It does not use the squared-rank-
difference shortcut, which assumes there are no ties.

**Optimizations in Our Implementation**:
- **Parallel Sorting**: The library uses parallel sorting with the `rayon` crate to assign ranks, ensuring that this operation scales efficiently even for large datasets.
- **Efficient Rank Calculation**: The average rank computation in the presence of ties is optimized with a direct look-up mechanism, minimizing redundant operations when processing multiple tied values in sequence.

**Why It Stands Out**:
- **Robust Against Outliers**: Since it uses ranks rather than raw data values, Spearman's correlation is less sensitive to outliers.
- **Monotonic Relationships**: It captures monotonic trends, making it suitable for many practical scenarios where linear correlation fails but directional trends exist.

### 2. **Kendall's Tau Rank Correlation (`kendall_tau`)**

Kendall’s Tau is a rank-based measure that evaluates the strength of ordinal association between two variables by comparing the relative ordering of data points. Positive scores favor concordant pairs, while negative scores favor discordant pairs.

**How It Works**:
- Kendall’s Tau compares all possible pairs of observations. For each pair, if the ranks of both variables agree (i.e., are concordant), the count of concordant pairs is increased. If the ranks disagree (i.e., are discordant), the discordant count is incremented.
- The implementation returns **tau-b**, normalizing the difference between concordant and discordant pairs with corrections for ties in either coordinate. Pairs tied in both coordinates are excluded from the concordant and discordant counts.

**Optimizations in Our Implementation**:
- **Inversion Counting Using Merge Sort**: The implementation sorts observation pairs by both coordinates, then counts strict inversions in the second coordinate. This excludes ties in the first coordinate from discordant pairs and runs in O(n log n) time with O(n) storage. Pair counts use wide integers before floating-point normalization.
- **Parallel Sorting**: The `rayon` crate is used to parallelize the sorting process, allowing the calculation to scale effectively on multi-core systems.

**Why It Stands Out**:
- **Tie Handling**: Kendall’s Tau inherently handles ties (when two elements have the same rank), making it more robust for datasets with tied values compared to other rank-based measures.
- **Pairwise Interpretation**: Tau-b summarizes concordant versus discordant ordering, with normalization that accounts for ties in each variable.
- **Fast with Inversion Counting**: Our use of merge sort for counting inversions dramatically improves the performance, making it feasible for large datasets where traditional implementations would be too slow.

### 3. **Approximate Distance Correlation (`approximate_distance_correlation`)**

Distance correlation measures both linear and non-linear dependence. This function
computes the biased sample estimator on up to 1,000 paired observations. Larger
inputs use a random subset, so their estimates can vary between calls.

**How It Works**:

- A first pass over observation pairs computes the row means of the two absolute-distance matrices.
- A second pass recomputes each centered distance and accumulates covariance and both variances. Off-diagonal pairs contribute twice, and centered diagonal entries contribute once.
- Normalizing these moments gives the same biased centered-distance statistic as the full matrix formulation, up to floating-point rounding.

**Optimizations in Our Implementation**:

- **Linear Working Storage**: The calculation keeps two row-mean arrays and a few scalar accumulators. At 1,000 sampled observations, those arrays use 16,000 bytes, compared with 16,000,000 bytes for two dense distance matrices; sampled input buffers are additional. This reduces memory use when many bootstrap replicates run concurrently.
- **Compensated Moment Sums**: Covariance and both variances use compensated accumulation, limiting rounding loss in signed cross-products.
- **Bounded Sampling**: The function samples at most 1,000 paired observations without replacement. Complete-input validation remains in place before sampling. With `n` input observations and `m = min(n, 1000)` sampled observations, time is O(n + m²) and auxiliary storage is O(m).

### 4. **Jensen-Shannon Dependency Measure (`jensen_shannon_dependency_measure`)**

The Jensen-Shannon Dependency Measure is derived from the Jensen-Shannon Divergence (JSD), a symmetric and smooth version of Kullback-Leibler divergence. JSD quantifies the similarity between two probability distributions. In the context of vector similarity, we use this measure to assess how much the distribution of values in one vector depends on the distribution in another.

**How It Works**:
- Both vectors are discretized into bins, creating histograms that approximate their probability distributions.
- We compute the joint distribution and the marginal distributions of the two vectors. The Jensen-Shannon Divergence is calculated as the difference between the joint and marginal entropies.
- The Jensen-Shannon Dependency Measure is defined as the difference between the observed JSD and a baseline JSD derived from shuffled data, normalized to lie between 0 and 1.

**Optimizations in Our Implementation**:
- **Efficient Histogram Binning**: The vectors are discretized into bins using optimized techniques that ensure minimal overhead, even for large vectors.
- **Parallel Computation of JSD**: The computation of the joint and marginal distributions is fully parallelized, which significantly speeds up the calculation.
- **Shuffling for Baseline Estimation**: The shuffled baseline is computed by randomizing the order of one vector and re-computing the JSD. This baseline helps differentiate between real dependencies and coincidental overlaps in the distributions.

**Why It Stands Out**:
- **Smooth and Symmetric**: Unlike KL-divergence, which can be asymmetric and undefined when one distribution has zero probabilities, JSD is symmetric and always well-defined.
- **Useful for Non-Overlapping Distributions**: It is especially powerful for comparing distributions with non-overlapping support, where other similarity measures might fail.
- **Dependency Measure**: The inclusion of a baseline JSD (derived from shuffled data) makes this a robust measure of dependency, filtering out spurious similarities.

### 5. **Hoeffding's D Measure (`hoeffding_d`)**

Hoeffding’s D is a powerful non-parametric measure that detects complex and potentially non-linear relationships between variables. Unlike traditional correlation measures, Hoeffding’s D is designed to detect general dependencies without assuming any specific form of the relationship.

**How It Works**:
- Hoeffding's D measures the joint ranks of two variables. It counts how many times a pair of observations follows a consistent pattern across the two variables.
- The statistic is based on the empirical distribution of the ranks and includes terms that account for joint and marginal distributions.

**Optimizations in Our Implementation**:
- **Parallel Computation of Ranks**: Like Spearman’s rho, Hoeffding’s D involves computing ranks, and this is optimized using parallel sorting algorithms.
- **Joint Ranks in O(n log n) Time**: A Fenwick tree counts observations with smaller ranks in both coordinates. Equal ranks in the first coordinate are processed as one group before the tree is updated, preserving the half-weight contributions from ties in one coordinate and quarter-weight contributions from ties in both. The complete calculation uses O(n) storage and retains the same tie-adjusted statistic without scanning every pair of observations.

**Why It Stands Out**:
- **Non-Parametric and General**: Hoeffding’s D can detect relationships where other correlation measures fail, especially when the relationship between variables is neither linear nor monotonic.
- **Sensitivity to Complex Patterns**: It is particularly effective when the relationship between variables is complex or unknown, providing a more general-purpose measure of dependence.

### 6. **Normalized Mutual Information (`normalized_mutual_information`)**

Normalized Mutual Information (NMI) is a measure of the mutual dependence between two variables. It is based on the concept of entropy from information theory and quantifies how much information one variable provides about another.

**How It Works**:
- First, both vectors are discretized into bins (histograms). The joint and marginal distributions are then computed.
- The mutual information is calculated as the difference between the joint entropy and the sum of the marginal entropies.
- The result is normalized to lie between 0 and 1, where 1 indicates perfect dependence and 0 indicates no dependence.

**Optimizations in Our Implementation**:
- **Efficient Histogram Calculation**: The binning process is optimized to handle large datasets efficiently, ensuring that the mutual information can be computed quickly even for high-dimensional vectors.
- **Parallel Entropy Calculation**: The calculation of joint and marginal entropies is parallelized, reducing the time required for large-scale datasets.

**Why It Stands Out**:
- **Interpretable and Scalable**: NMI is easy to interpret and particularly useful for

 comparing variables with different distributions, making it a versatile tool for high-dimensional data.
- **Handles Non-Linear Relationships**: Like distance correlation, NMI captures both linear and non-linear dependencies but does so in a way that is grounded in information theory, providing a complementary perspective on data dependence.

## Bootstrapping Technique for Robust Estimation

Bootstrapping is a statistical method that improves the reliability of similarity estimates by resampling the dataset. The Fast Vector Similarity Library offers this feature for robust estimation of similarity measures.

### Advantages of Bootstrapping

1. **Robustness to Outliers**: By resampling the data, the technique reduces the influence of outliers, providing more reliable estimates.
2. **Model-Free Estimation**: It makes no assumptions about the underlying data distribution, making it suitable for diverse datasets.
3. **Confidence Intervals**: Bootstrapping allows the construction of confidence intervals, adding interpretability to the results.
4. **Deeper Insights**: By examining the distribution of similarity measures across bootstrap samples, bootstrapping offers a richer understanding of the underlying relationships.

---

Thanks for your interest in my open-source project! I hope you find it useful. You might also find my commercial web apps useful, and I would really appreciate it if you checked them out:

**[YoutubeTranscriptOptimizer.com](https://youtubetranscriptoptimizer.com)** makes it really quick and easy to paste in a YouTube video URL and have it automatically generate not just a really accurate direct transcription, but also a super polished and beautifully formatted written document that can be used independently of the video.

The document basically sticks to the same material as discussed in the video, but it sounds much more like a real piece of writing and not just a transcript. It also lets you optionally generate quizzes based on the contents of the document, which can be either multiple choice or short-answer quizzes, and the multiple choice quizzes get turned into interactive HTML files that can be hosted and easily shared, where you can actually take the quiz and it will grade your answers and score the quiz for you.

**[FixMyDocuments.com](https://fixmydocuments.com/)** lets you submit any kind of document— PDFs (including scanned PDFs that require OCR), MS Word and Powerpoint files, images, audio files (mp3, m4a, etc.) —and turn them into highly optimized versions in nice markdown formatting, from which HTML and PDF versions are automatically generated. Once converted, you can also edit them directly in the site using the built-in markdown editor, where it saves a running revision history and regenerates the PDF/HTML versions.

In addition to just getting the optimized version of the document, you can also generate many other kinds of "derived documents" from the original: interactive multiple-choice quizzes that you can actually take and get graded on; slick looking presentation slides as PDF or HTML (using LaTeX and Reveal.js), an in-depth summary, a concept mind map (using Mermaid diagrams) and outline, custom lesson plans where you can select your target audience, a readability analysis and grade-level versions of your original document (good for simplifying concepts for students), Anki Flashcards that you can import directly into the Anki app or use on the site in a nice interface, and more.
