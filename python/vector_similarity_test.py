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
