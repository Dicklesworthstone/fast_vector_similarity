use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray::prelude::*;
use rayon::prelude::*;
use serde_json::json;
use statrs::distribution::{Continuous, Normal as StatsrsNormal};
use rand::seq::IteratorRandom; // To select random indices
use std::error::Error;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use serde_json::Value;

fn average_rank(data: &[f64]) -> Vec<f64> {
    let mut ranks: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
    ranks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut rank_values = vec![0.0; data.len()];
    let mut i = 0;
    while i < ranks.len() {
        let mut total_rank = i as f64 + 1.0; // Start rank
        let mut count = 1;
        while i + count < ranks.len() && ranks[i].1 == ranks[i + count].1 {
            total_rank += (i + count) as f64 + 1.0;
            count += 1;
        }
        let average_rank = total_rank / count as f64;
        for j in 0..count {
            rank_values[ranks[i + j].0] = average_rank;
        }
        i += count;
    }
    rank_values
}

fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let rank_x = average_rank(x);
    let rank_y = average_rank(y);

    let d_squared_sum: f64 = rank_x.iter()
        .zip(rank_y.iter())
        .map(|(rx, ry)| (rx - ry).powi(2))
        .sum();
    1.0 - (6.0 * d_squared_sum) / (n * (n * n - 1.0))
}

fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut concordant_count = 0;
    let mut discordant_count = 0;
    for i in 0..n {
        for j in i + 1..n {
            let x_concordant = (x[i] > x[j]) as i64 - (x[i] < x[j]) as i64;
            let y_concordant = (y[i] > y[j]) as i64 - (y[i] < y[j]) as i64;
            let concordance = x_concordant * y_concordant;

            if concordance > 0 {
                concordant_count += 1;
            } else if concordance < 0 {
                discordant_count += 1;
            }
        }
    }
    (concordant_count as f64 - discordant_count as f64) / (concordant_count as f64 + discordant_count as f64)
}

fn kernel_density(data: &[f64], x: f64, bandwidth: f64) -> f64 {
    let normal = StatsrsNormal::new(0.0, bandwidth).unwrap();
    let density: f64 = data.par_iter().map(|&value| normal.pdf((x - value) / bandwidth)).sum();
    density / (data.len() as f64 * bandwidth)
}

fn jensen_shannon_similarity(x: &Array1<f64>, y: &Array1<f64>, bandwidth: f64) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    let x_slice = x.as_slice().unwrap();
    let y_slice = y.as_slice().unwrap();
    let jsd: f64 = x_slice.par_iter().enumerate().map(|(i, &xi)| {
        let p_x = kernel_density(x_slice, xi, bandwidth);
        let p_y = kernel_density(y_slice, y_slice[i], bandwidth);
        let m = 0.5 * (p_x + p_y);
        if m > 0.0 {
            0.5 * (p_x * (p_x / m).ln() + p_y * (p_y / m).ln()) / n
        } else {
            0.0
        }
    }).sum();
    let jenson_shannon_similarity = 1.0 - jsd;
    jenson_shannon_similarity
}

fn double_centering(matrix: &mut Array2<f64>) {
    let n = matrix.nrows();
    let row_means = matrix.mean_axis(Axis(1)).unwrap();
    let grand_mean = row_means.mean().unwrap();
    let row_means_with_axis = row_means.insert_axis(Axis(1));
    let row_means_broadcasted = row_means_with_axis.broadcast((n, n)).unwrap();
    let col_means_broadcasted = row_means_broadcasted.t();
    *matrix -= &row_means_broadcasted;
    *matrix -= &col_means_broadcasted;
    *matrix += grand_mean;
}

fn distance_matrix_one_d(data: &Array1<f64>) -> Array2<f64> {
    let data_column = data.view().insert_axis(Axis(1));
    let data_row = data.view().insert_axis(Axis(0));
    let distance = &data_column - &data_row;
    distance.mapv(f64::abs)
}

fn approximate_distance_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let subset_size = 1000;
    let actual_subset_size = std::cmp::min(subset_size, x.len());
    let subset_indices: Vec<usize> = (0..x.len()).choose_multiple(&mut rand::thread_rng(), actual_subset_size);
    let x_subset: Array1<f64> = Array1::from(subset_indices.iter().map(|&i| x[i]).collect::<Vec<_>>());
    let y_subset: Array1<f64> = Array1::from(subset_indices.iter().map(|&i| y[i]).collect::<Vec<_>>());
    let mut a_matrix = distance_matrix_one_d(&x_subset);
    let mut b_matrix = distance_matrix_one_d(&y_subset);
    double_centering(&mut a_matrix);
    double_centering(&mut b_matrix);
    let distance_covariance_sq = (a_matrix.clone() * b_matrix.clone()).sum() / (actual_subset_size * actual_subset_size) as f64;
    let distance_variance_x_sq = (a_matrix.clone() * a_matrix).sum() / (actual_subset_size * actual_subset_size) as f64;
    let distance_variance_y_sq = (b_matrix.clone() * b_matrix).sum() / (actual_subset_size * actual_subset_size) as f64;
    if distance_variance_x_sq == 0.0 || distance_variance_y_sq == 0.0 {
        return 0.0;
    }
    // Return the computed distance correlation
    (distance_covariance_sq / (distance_variance_x_sq * distance_variance_y_sq).sqrt()).sqrt()
}

fn hoeffd_inner_loop_func(i: usize, r: &Array1<f64>, s: &Array1<f64>) -> f64 {
    let mut q_i = 1.0 + r.iter().zip(s.iter()).filter(|&(r_val, s_val)| *r_val < r[i] && *s_val < s[i]).count() as f64;
    q_i += 0.25 * (r.iter().zip(s.iter()).filter(|&(r_val, s_val)| *r_val == r[i] && *s_val == s[i]).count() as f64 - 1.0);
    q_i += 0.5 * r.iter().zip(s.iter()).filter(|&(r_val, s_val)| *r_val == r[i] && *s_val < s[i]).count() as f64;
    q_i += 0.5 * r.iter().zip(s.iter()).filter(|&(r_val, s_val)| *r_val < r[i] && *s_val == s[i]).count() as f64;
    q_i
}

fn exact_hoeffdings_d_func(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    let r = Array1::from(average_rank(&x.to_vec()));
    let s = Array1::from(average_rank(&y.to_vec()));
    let q: Array1<f64> = Array1::from((0..x.len()).into_par_iter().map(|i| hoeffd_inner_loop_func(i, &r, &s)).collect::<Vec<f64>>());
    let d1: f64 = (q.clone() - 1.0).mapv(|v| v * (v - 2.0)).sum();
    let d2: f64 = (r.clone() - 1.0).mapv(|v| v * (v - 2.0)).iter().zip((s.clone() - 1.0).mapv(|v| v * (v - 2.0)).iter()).map(|(a, b)| a * b).sum();
    let d3: f64 = (r.clone() - 2.0).iter().zip((s.clone() - 2.0).iter()).zip((q.clone() - 1.0).iter()).map(|((a, b), c)| a * b * c).sum();
    let d: f64 = 30.0 * ((n - 2.0) * (n - 3.0) * d1 + d2 - 2.0 * (n - 2.0) * d3)
        / (n * (n - 1.0) * (n - 2.0) * (n - 3.0) * (n - 4.0));
    d
}

fn skewness(data: &Array1<f64>) -> f64 {
    let mean = data.mean().unwrap();
    let variance = data.var(0.);
    let std_dev = variance.sqrt();
    let n = data.len() as f64;
    let sum_cubed_deviations: f64 = data.iter().map(|&x| (x - mean).powi(3)).sum();
    sum_cubed_deviations / (n * std_dev.powi(3))
}

pub fn compute_vector_similarity_stats(
    vector_1: &Array1<f64>,
    vector_2: &Array1<f64>,
    similarity_measure: Option<&str>,
) -> serde_json::Value {
    assert_eq!(vector_1.len(), vector_2.len());
    let input_vector_dimensions = vector_1.len();

    // Standard deviation for both vectors
    let std_dev_1 = vector_1.std(0.);
    let std_dev_2 = vector_2.std(0.);

    // Silverman's rule for both vectors
    let bandwidth_1 = (4.0 * std_dev_1.powf(5.0) / (3.0 * (input_vector_dimensions as f64))).powf(1.0 / 5.0);
    let bandwidth_2 = (4.0 * std_dev_2.powf(5.0) / (3.0 * (input_vector_dimensions as f64))).powf(1.0 / 5.0);
    // Use the weighted average of the bandwidths where we weight the bandwidths by the skewness of the vectors, since the skewness is a measure of how non-Gaussian the distribution is and the bandwidth is a measure of the spread of the distribution
    let skewness_1 = skewness(&vector_1);
    let skewness_2 = skewness(&vector_2);
    let total_skewness = skewness_1.abs() + skewness_2.abs();
    let weight_1 = skewness_1.abs() / total_skewness;
    let weight_2 = skewness_2.abs() / total_skewness;
    let bandwidth = bandwidth_1 * weight_1 + bandwidth_2 * weight_2;

    let similarity_measure = similarity_measure.unwrap_or("all");
    let mut computations: Vec<(Box<dyn Fn() -> f64 + Send>, &str)> = Vec::new();
    if similarity_measure == "spearman_rho" || similarity_measure == "all" {
        computations.push((Box::new(|| spearman_rho(&vector_1.to_vec(), &vector_2.to_vec())), "spearman_rho"));
    }
    if similarity_measure == "kendall_tau" || similarity_measure == "all" {
        computations.push((Box::new(|| kendall_tau(&vector_1.to_vec(), &vector_2.to_vec())), "kendall_tau"));
    }
    if similarity_measure == "approximate_distance_correlation" || similarity_measure == "all" {
        computations.push((Box::new(|| approximate_distance_correlation(&vector_1, &vector_2)), "approximate_distance_correlation"));
    }
    if similarity_measure == "jensen_shannon_similarity" || similarity_measure == "all" {
        computations.push((Box::new(|| jensen_shannon_similarity(&vector_1, &vector_2, bandwidth)), "jensen_shannon_similarity"));
    }
    if similarity_measure == "hoeffding_d" || similarity_measure == "all" {
        computations.push((Box::new(|| exact_hoeffdings_d_func(&vector_1, &vector_2)), "hoeffding_d"));
    }
    let mut similarity_stats = json!({
        "input_vector_dimensions": input_vector_dimensions,
    });
    // Collect the results into a temporary vector
    let results: Vec<(&str, f64)> = computations.into_par_iter().map(|(func, key)| (key, func())).collect();
    // Update the similarity_stats JSON object
    for (key, value) in results {
        similarity_stats[key] = json!(value);
    }
    similarity_stats
}

fn generate_bootstrap_sample_func(original_length_of_input: usize, sample_size: usize) -> Array1<usize> {
    Array1::random(sample_size, Uniform::new(0, original_length_of_input))
}

fn compute_average_and_stdev_of_25th_to_75th_percentile_func(input_vector: &Array1<f64>) -> (f64, f64) {
    let mut sorted_vector = input_vector.to_vec();
    sorted_vector.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let percentile_25 = sorted_vector[input_vector.len() / 4];
    let percentile_75 = sorted_vector[3 * input_vector.len() / 4];
    let trimmed_vector: Array1<f64> = input_vector
        .iter()
        .filter(|&&x| x > percentile_25 && x < percentile_75)
        .cloned()
        .collect();
    let trimmed_vector_avg = trimmed_vector.mean().unwrap();
    let trimmed_vector_stdev = trimmed_vector.std(0.);
    (trimmed_vector_avg, trimmed_vector_stdev)
}

pub fn compute_bootstrapped_similarity_stats(
    x: &Array1<f64>,
    y: &Array1<f64>,
    sample_size: usize,
    number_of_bootstraps: usize,
    similarity_measure: Option<&str>,
) -> serde_json::Value {
    let original_length_of_input = x.len();
    assert_eq!(original_length_of_input, y.len());
    let apply_bootstrap_similarity_stats_func = |_| {
        let bootstrap_sample_indices = generate_bootstrap_sample_func(original_length_of_input, sample_size);
        let x_bootstrap_sample: Array1<f64> = bootstrap_sample_indices.iter().map(|&idx| x[idx]).collect();
        let y_bootstrap_sample: Array1<f64> = bootstrap_sample_indices.iter().map(|&idx| y[idx]).collect();
        compute_vector_similarity_stats(&x_bootstrap_sample, &y_bootstrap_sample, similarity_measure)
    };
    let list_of_similarity_stats: Vec<serde_json::Value> = (0..number_of_bootstraps)
        .into_par_iter()
        .map(apply_bootstrap_similarity_stats_func)
        .collect();
    let similarity_measures = vec![
        "spearman_rho",
        "kendall_tau",
        "approximate_distance_correlation",
        "jensen_shannon_similarity",
        "hoeffding_d",
    ];
    let mut bootstrapped_similarity_stats = json!({
        "number_of_bootstraps": number_of_bootstraps,
        "sample_size": sample_size,
    });
    for key in similarity_measures {
        if similarity_measure.is_none() || similarity_measure == Some(key) || similarity_measure == Some("all") {
            let values: Vec<f64> = list_of_similarity_stats.iter().map(|item| item[key].as_f64().unwrap()).collect();
            let (robust_average, stdev) = compute_average_and_stdev_of_25th_to_75th_percentile_func(&Array1::from(values));
            bootstrapped_similarity_stats[key] = json!(robust_average);
            bootstrapped_similarity_stats[format!("stdev_{}", key)] = json!(stdev);
        }
    }
    bootstrapped_similarity_stats
}

#[pyfunction]
fn py_compute_vector_similarity_stats(_py: Python, json_params: &str) -> PyResult<String> {
    let run = || -> Result<String, Box<dyn Error>> {
        let params: Value = serde_json::from_str(json_params)?;

        let vector_1_vec: Vec<f64> = params["vector_1"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let vector_1 = Array1::from(vector_1_vec);
        let vector_2_vec: Vec<f64> = params["vector_2"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let vector_2 = Array1::from(vector_2_vec);

        let similarity_measure = params["similarity_measure"].as_str();

        let result = compute_vector_similarity_stats(&vector_1, &vector_2, similarity_measure);

        let result_str = serde_json::to_string(&result)?;
        Ok(result_str)
    };

    match run() {
        Ok(result_str) => Ok(result_str),
        Err(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("An error occurred")),
    }
}

#[pyfunction]
fn py_compute_bootstrapped_similarity_stats(_py: Python, json_params: &str) -> PyResult<String> {
    let run = || -> Result<String, Box<dyn Error>> {
        let params: Value = serde_json::from_str(json_params)?;

        let x_vec: Vec<f64> = params["x"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let x = Array1::from(x_vec);
        let y_vec: Vec<f64> = params["y"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let y = Array1::from(y_vec);

        let sample_size = params["sample_size"].as_u64().unwrap() as usize;
        let number_of_bootstraps = params["number_of_bootstraps"].as_u64().unwrap() as usize;
        let similarity_measure = params["similarity_measure"].as_str();

        let result = compute_bootstrapped_similarity_stats(&x, &y, sample_size, number_of_bootstraps, similarity_measure);

        let result_str = serde_json::to_string(&result)?;
        Ok(result_str)
    };

    match run() {
        Ok(result_str) => Ok(result_str),
        Err(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("An error occurred")),
    }
}

#[pymodule]
fn fast_vector_similarity(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_compute_vector_similarity_stats, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_bootstrapped_similarity_stats, m)?)?;
    Ok(())
}
