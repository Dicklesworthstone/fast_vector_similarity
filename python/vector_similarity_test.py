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

    # Compute the differences between exact and bootstrapped results
    measures = ["spearman_rho", "kendall_tau", "approximate_distance_correlation", "jensen_shannon_similarity", "hoeffing_d"]
    for measure in measures:
        exact_value = similarity_stats_json[measure]
        bootstrapped_value = bootstrapped_similarity_stats_json[measure]
        absolute_difference = abs(exact_value - bootstrapped_value)
        percentage_difference = (absolute_difference / exact_value) * 100

        print(f"\nDifference between exact and bootstrapped {measure}: {absolute_difference}")
        print(f"Difference as % of the exact value: {percentage_difference:.2f}%")

if __name__ == "__main__":
    main()
