use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray::prelude::*;
use ndarray::Array1;
use rayon::prelude::*;
use serde_json::json;
use rand::prelude::SliceRandom;
use rand::seq::IteratorRandom;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use serde_json::Value;

fn has_non_finite<'a>(mut values: impl Iterator<Item = &'a f64>) -> bool {
    values.any(|value| !value.is_finite())
}

fn average_rank(data: &[f64]) -> Vec<f64> {
    let mut ranks: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
    ranks.par_sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
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

/// Returns NaN for empty inputs or any NaN or infinite observation.
pub fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
    if x.is_empty() || y.is_empty() || has_non_finite(x.iter().chain(y.iter())) {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let rank_x = average_rank(x);
    let rank_y = average_rank(y);

    let d_squared_sum: f64 = rank_x.iter()
        .zip(rank_y.iter())
        .map(|(rx, ry)| (rx - ry).powi(2))
        .sum();
    1.0 - (6.0 * d_squared_sum) / (n * (n * n - 1.0))
}

fn count_inversions(arr: &mut [f64]) -> usize {
    fn merge_sort(arr: &mut [f64], temp: &mut [f64]) -> usize {
        let n = arr.len();
        if n <= 1 {
            return 0;
        }
        let mid = n / 2;
        let mut inv_count = merge_sort(&mut arr[..mid], &mut temp[..mid]);
        inv_count += merge_sort(&mut arr[mid..], &mut temp[mid..]);
        inv_count += merge(arr, temp, mid);
        inv_count
    }

    fn merge(arr: &mut [f64], temp: &mut [f64], mid: usize) -> usize {
        let mut i = 0;
        let mut j = mid;
        let mut k = 0;
        let mut inv_count = 0;

        while i < mid && j < arr.len() {
            if arr[i] <= arr[j] {
                temp[k] = arr[i];
                i += 1;
            } else {
                temp[k] = arr[j];
                inv_count += mid - i;
                j += 1;
            }
            k += 1;
        }
        while i < mid {
            temp[k] = arr[i];
            i += 1;
            k += 1;
        }
        while j < arr.len() {
            temp[k] = arr[j];
            j += 1;
            k += 1;
        }
        arr.copy_from_slice(&temp[..arr.len()]);
        inv_count
    }

    let n = arr.len();
    let mut temp = vec![0.0; n];
    merge_sort(arr, &mut temp)
}

/// Returns NaN for empty inputs or any NaN or infinite observation.
pub fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    if x.is_empty() || has_non_finite(x.iter().chain(y.iter())) {
        return f64::NAN;
    }
    let n = x.len() as f64;

    let rank_x = average_rank(x);
    let rank_y = average_rank(y);

    // Create an array of ranks sorted by rank_x
    let mut pairs: Vec<(f64, f64)> = rank_x
        .iter()
        .zip(rank_y.iter())
        .map(|(&rx, &ry)| (rx, ry))
        .collect();

    // Sort pairs by rank_x
    pairs.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Extract rank_y in the order of sorted rank_x
    let sorted_rank_y: Vec<f64> = pairs.iter().map(|&(_, ry)| ry).collect();

    // Count inversions in sorted_rank_y using a more efficient algorithm
    let inversions = count_inversions(&mut sorted_rank_y.clone());

    let total_pairs = n * (n - 1.0) / 2.0;
    let tau = 1.0 - 2.0 * inversions as f64 / total_pairs;

    tau
}

/// Returns NaN for empty inputs or any NaN or infinite observation.
pub fn jensen_shannon_dependency_measure(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    assert_eq!(x.len(), y.len());
    if x.is_empty() || has_non_finite(x.iter().chain(y.iter())) {
        return f64::NAN;
    }
    let _n = x.len() as f64;
    let num_bins: usize = 20;

    let compute_jsd = |x: &Array1<f64>, y: &Array1<f64>, num_bins: usize| -> f64 {
        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut joint_hist = vec![vec![0f64; num_bins]; num_bins];
        let mut x_hist = vec![0f64; num_bins];
        let mut y_hist = vec![0f64; num_bins];

        x.iter().zip(y.iter()).for_each(|(&xi, &yi)| {
            let x_bin = ((xi - x_min) / (x_max - x_min + 1e-10) * num_bins as f64).floor() as usize;
            let y_bin = ((yi - y_min) / (y_max - y_min + 1e-10) * num_bins as f64).floor() as usize;
            let x_bin = x_bin.min(num_bins - 1);
            let y_bin = y_bin.min(num_bins - 1);

            joint_hist[x_bin][y_bin] += 1.0;
            x_hist[x_bin] += 1.0;
            y_hist[y_bin] += 1.0;
        });

        let p_x: Vec<f64> = x_hist.iter().map(|&count| count / _n).collect();
        let p_y: Vec<f64> = y_hist.iter().map(|&count| count / _n).collect();
        let p_xy: Vec<Vec<f64>> = joint_hist.iter().map(|row| row.iter().map(|&count| count / _n).collect()).collect();

        let mut p_x_p_y = vec![vec![0f64; num_bins]; num_bins];
        for i in 0..num_bins {
            for j in 0..num_bins {
                p_x_p_y[i][j] = p_x[i] * p_y[j];
            }
        }

        let mut m = vec![vec![0f64; num_bins]; num_bins];
        for i in 0..num_bins {
            for j in 0..num_bins {
                m[i][j] = 0.5 * (p_xy[i][j] + p_x_p_y[i][j]);
            }
        }

        let mut kl1 = 0.0;
        let mut kl2 = 0.0;
        for i in 0..num_bins {
            for j in 0..num_bins {
                if p_xy[i][j] > 0.0 {
                    kl1 += p_xy[i][j] * (p_xy[i][j] / m[i][j]).ln();
                }
                if p_x_p_y[i][j] > 0.0 {
                    kl2 += p_x_p_y[i][j] * (p_x_p_y[i][j] / m[i][j]).ln();
                }
            }
        }

        0.5 * (kl1 + kl2) / std::f64::consts::LN_2
    };

    let jsd_observed = compute_jsd(x, y, num_bins);

    // Shuffle y to estimate expected JSD under independence
    let mut y_shuffled_vec = y.to_vec();
    let mut rng = rand::thread_rng();
    y_shuffled_vec.shuffle(&mut rng);
    let y_shuffled = Array1::from(y_shuffled_vec);

    let jsd_independent = compute_jsd(x, &y_shuffled, num_bins);

    let dependency = (jsd_observed - jsd_independent) / (1.0 - jsd_independent);
    dependency.max(0.0).min(1.0) // Ensure the value is between 0 and 1
}

fn distance_matrix_one_d(data: &Array1<f64>) -> Array2<f64> {
    let n = data.len();
    let mut distance = Array2::<f64>::zeros((n, n));
    distance.axis_iter_mut(Axis(0)).enumerate().for_each(|(i, mut row)| {
        row.iter_mut().enumerate().for_each(|(j, elem)| {
            *elem = (data[i] - data[j]).abs();
        });
    });
    distance
}

/// Returns NaN for empty inputs or any NaN or infinite observation, even outside
/// the randomly selected subset.
pub fn approximate_distance_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    if x.is_empty() || y.is_empty() || has_non_finite(x.iter().chain(y.iter())) {
        return f64::NAN;
    }
    let subset_size = 1000.min(x.len());
    let mut rng = rand::thread_rng();
    let subset_indices: Vec<usize> = (0..x.len()).choose_multiple(&mut rng, subset_size);
    let x_subset = x.select(Axis(0), &subset_indices);
    let y_subset = y.select(Axis(0), &subset_indices);

    let mut a_matrix = distance_matrix_one_d(&x_subset);
    let mut b_matrix = distance_matrix_one_d(&y_subset);

    // Optimize double centering using vectorized operations
    fn double_centering(matrix: &mut Array2<f64>) {
        let row_means = matrix.mean_axis(Axis(1)).unwrap();
        let col_means = matrix.mean_axis(Axis(0)).unwrap();
        let grand_mean = row_means.mean().unwrap();

        ndarray::Zip::from(matrix.rows_mut())
            .and(&row_means)
            .for_each(|mut row, &row_mean| {
                row -= &col_means;
                row += grand_mean;
                row -= row_mean;
            });
    }

    double_centering(&mut a_matrix);
    double_centering(&mut b_matrix);

    let distance_covariance = (&a_matrix * &b_matrix).mean().unwrap();
    let distance_variance_x = (&a_matrix * &a_matrix).mean().unwrap();
    let distance_variance_y = (&b_matrix * &b_matrix).mean().unwrap();

    if distance_variance_x <= 0.0 || distance_variance_y <= 0.0 {
        return 0.0;
    }

    (distance_covariance / (distance_variance_x * distance_variance_y).sqrt()).sqrt()
}

fn dot_product(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Returns NaN for empty inputs or any NaN or infinite observation.
pub fn exact_hoeffdings_d_func(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    if x.is_empty() || y.is_empty() || has_non_finite(x.iter().chain(y.iter())) {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let r = Array1::from(average_rank(x.as_slice().unwrap()));
    let s = Array1::from(average_rank(y.as_slice().unwrap()));

    // Precompute necessary arrays
    let r_minus = &r - 1.0;
    let s_minus = &s - 1.0;

    let q_vec: Vec<f64> = (0..x.len())
        .into_par_iter()
        .map(|i| {
            let ri = r[i];
            let si = s[i];
            let less_than = r.iter().zip(s.iter()).filter(|&(r_val, s_val)| *r_val < ri && *s_val < si).count() as f64;
            let equal_both = r.iter().zip(s.iter()).filter(|&(r_val, s_val)| *r_val == ri && *s_val == si).count() as f64 - 1.0;
            let equal_r = r.iter().zip(s.iter()).filter(|&(r_val, s_val)| *r_val == ri && *s_val < si).count() as f64;
            let equal_s = r.iter().zip(s.iter()).filter(|&(r_val, s_val)| *r_val < ri && *s_val == si).count() as f64;
            1.0 + less_than + 0.25 * equal_both + 0.5 * (equal_r + equal_s)
        })
        .collect();

    let q = Array1::from(q_vec);
    let q_minus = &q - 1.0;

    let d1 = q_minus.mapv(|v| v * (v - 1.0)).sum();
    let r_component = r_minus.mapv(|v| v * (v - 1.0));
    let s_component = s_minus.mapv(|v| v * (v - 1.0));

    let d2 = dot_product(&r_component, &s_component);
    let r_minus2 = &r - 2.0;
    let s_minus2 = &s - 2.0;
    let d3 = r_minus2
        .iter()
        .zip(s_minus2.iter())
        .zip(q_minus.iter())
        .map(|((&a, &b), &c)| a * b * c)
        .sum::<f64>();

    let denom = n * (n - 1.0) * (n - 2.0) * (n - 3.0) * (n - 4.0);
    let numerator = 30.0 * ((n - 2.0) * (n - 3.0) * d1 + d2 - 2.0 * (n - 2.0) * d3);
    numerator / denom
}

/// Returns NaN for empty inputs, any NaN or infinite observation, or zero total
/// marginal entropy (both vectors constant). Inputs must have equal lengths.
pub fn normalized_mutual_information(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    assert_eq!(x.len(), y.len());
    if x.is_empty() || has_non_finite(x.iter().chain(y.iter())) {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let num_bins = 20;

    // Compute bin edges for x and y
    let (x_min, x_max) = x.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min, max), &value| (min.min(value), max.max(value)),
    );
    let (y_min, y_max) = y.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min, max), &value| (min.min(value), max.max(value)),
    );

    // Initialize histograms using ndarray
    let mut joint_hist = Array2::<usize>::zeros((num_bins, num_bins));
    let mut x_hist = Array1::<usize>::zeros(num_bins);
    let mut y_hist = Array1::<usize>::zeros(num_bins);

    // Assign data to bins and compute histograms
    x.iter().zip(y.iter()).for_each(|(&xi, &yi)| {
        // Find bin index for xi and yi
        let x_bin = ((xi - x_min) / (x_max - x_min + 1e-10) * num_bins as f64).floor() as usize;
        let y_bin = ((yi - y_min) / (y_max - y_min + 1e-10) * num_bins as f64).floor() as usize;
        let x_bin = x_bin.min(num_bins - 1);
        let y_bin = y_bin.min(num_bins - 1);

        joint_hist[(x_bin, y_bin)] += 1;
        x_hist[x_bin] += 1;
        y_hist[y_bin] += 1;
    });

    // Compute entropies
    let h_x = x_hist
        .iter()
        .filter_map(|&count| {
            if count > 0 {
                let p = count as f64 / n;
                Some(-p * p.ln())
            } else {
                None
            }
        })
        .sum::<f64>();

    let h_y = y_hist
        .iter()
        .filter_map(|&count| {
            if count > 0 {
                let p = count as f64 / n;
                Some(-p * p.ln())
            } else {
                None
            }
        })
        .sum::<f64>();

    let h_xy = joint_hist
        .iter()
        .filter_map(|&count| {
            if count > 0 {
                let p = count as f64 / n;
                Some(-p * p.ln())
            } else {
                None
            }
        })
        .sum::<f64>();

    let mi = h_x + h_y - h_xy;

    // Compute NMI using average of entropies
    let nmi = mi / ((h_x + h_y) / 2.0);

    nmi
}

/// Computes the requested measures, representing undefined results as JSON null.
/// Empty inputs or any non-finite observation make every requested measure null.
pub fn compute_vector_similarity_stats(
    vector_1: &Array1<f64>,
    vector_2: &Array1<f64>,
    similarity_measure: Option<&str>,
) -> serde_json::Value {
    assert_eq!(vector_1.len(), vector_2.len());
    let input_vector_dimensions = vector_1.len();
    let invalid_input =
        vector_1.is_empty() || has_non_finite(vector_1.iter().chain(vector_2.iter()));

    let similarity_measure = similarity_measure.unwrap_or("all");
    let mut computations: Vec<(Box<dyn Fn() -> f64 + Send + Sync>, &str)> = Vec::new();
    if similarity_measure == "spearman_rho" || similarity_measure == "all" {
        computations.push((Box::new(|| spearman_rho(vector_1.as_slice().unwrap(), vector_2.as_slice().unwrap())), "spearman_rho"));
    }
    if similarity_measure == "kendall_tau" || similarity_measure == "all" {
        computations.push((Box::new(|| kendall_tau(vector_1.as_slice().unwrap(), vector_2.as_slice().unwrap())), "kendall_tau"));
    }
    if similarity_measure == "approximate_distance_correlation" || similarity_measure == "all" {
        computations.push((Box::new(|| approximate_distance_correlation(&vector_1, &vector_2)), "approximate_distance_correlation"));
    }
    if similarity_measure == "jensen_shannon_dependency_measure" || similarity_measure == "all" {
        computations.push((Box::new(|| jensen_shannon_dependency_measure(&vector_1, &vector_2)), "jensen_shannon_dependency_measure"));
    }
    if similarity_measure == "normalized_mutual_information" || similarity_measure == "all" {
        computations.push((Box::new(|| normalized_mutual_information(&vector_1, &vector_2)), "normalized_mutual_information"));
    }
    if similarity_measure == "hoeffding_d" || similarity_measure == "all" {
        computations.push((Box::new(|| exact_hoeffdings_d_func(&vector_1, &vector_2)), "hoeffding_d"));
    }
    let mut similarity_stats = json!({
        "input_vector_dimensions": input_vector_dimensions,
    });
    // Collect the results into a temporary vector
    let results: Vec<(&str, f64)> = computations
        .into_par_iter()
        .map(|(func, key)| (key, if invalid_input { f64::NAN } else { func() }))
        .collect();
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
    // Undefined replicates invalidate the summary; dropping or trimming them
    // would silently bias the estimate. Fewer than two values leave an empty trim.
    if input_vector.len() < 2 || has_non_finite(input_vector.iter()) {
        return (f64::NAN, f64::NAN);
    }
    let mut sorted_vector = input_vector.to_vec();
    sorted_vector.par_sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lower_idx = input_vector.len() / 4;
    let upper_idx = 3 * input_vector.len() / 4;
    let trimmed_vector = &sorted_vector[lower_idx..upper_idx];
    let trimmed_array = Array1::from(trimmed_vector.to_vec());
    let trimmed_vector_avg = trimmed_array.mean().unwrap();
    let trimmed_vector_stdev = trimmed_array.std(0.);
    (trimmed_vector_avg, trimmed_vector_stdev)
}

/// Computes bootstrap summaries, using JSON null for undefined means and stdevs.
/// The complete inputs are validated before sampling; any non-finite observation
/// or undefined replicate invalidates the corresponding summary.
pub fn compute_bootstrapped_similarity_stats(
    x: &Array1<f64>,
    y: &Array1<f64>,
    sample_size: usize,
    number_of_bootstraps: usize,
    similarity_measure: Option<&str>,
) -> serde_json::Value {
    let original_length_of_input = x.len();
    assert_eq!(original_length_of_input, y.len());
    let invalid_input = x.is_empty()
        || sample_size == 0
        || has_non_finite(x.iter().chain(y.iter()));
    let apply_bootstrap_similarity_stats_func = |_| {
        let bootstrap_sample_indices = generate_bootstrap_sample_func(original_length_of_input, sample_size);
        let x_bootstrap_sample: Array1<f64> = bootstrap_sample_indices.iter().map(|&idx| x[idx]).collect();
        let y_bootstrap_sample: Array1<f64> = bootstrap_sample_indices.iter().map(|&idx| y[idx]).collect();
        compute_vector_similarity_stats(&x_bootstrap_sample, &y_bootstrap_sample, similarity_measure)
    };
    let list_of_similarity_stats: Vec<serde_json::Value> = if invalid_input {
        Vec::new()
    } else {
        (0..number_of_bootstraps)
            .into_par_iter()
            .map(apply_bootstrap_similarity_stats_func)
            .collect()
    };
    let similarity_measures = vec![
        "spearman_rho",
        "kendall_tau",
        "approximate_distance_correlation",
        "jensen_shannon_dependency_measure",
        "hoeffding_d",
        "normalized_mutual_information",
    ];
    let mut bootstrapped_similarity_stats = json!({
        "number_of_bootstraps": number_of_bootstraps,
        "sample_size": sample_size,
    });
    for key in similarity_measures {
        if similarity_measure.is_none() || similarity_measure == Some(key) || similarity_measure == Some("all") {
            let values: Vec<f64> = list_of_similarity_stats
                .iter()
                .map(|item| item[key].as_f64().unwrap_or(f64::NAN))
                .collect();
            let (robust_average, stdev) = compute_average_and_stdev_of_25th_to_75th_percentile_func(&Array1::from(values));
            bootstrapped_similarity_stats[key] = json!(robust_average);
            bootstrapped_similarity_stats[format!("stdev_{}", key)] = json!(stdev);
        }
    }
    bootstrapped_similarity_stats
}

fn parse_json_params(json_params: &str) -> PyResult<Value> {
    serde_json::from_str(json_params)
        .map_err(|error| PyValueError::new_err(format!("Invalid JSON parameters: {error}")))
}

fn parse_vector(params: &Value, name: &str) -> PyResult<Array1<f64>> {
    let values = params[name].as_array().ok_or_else(|| {
        PyValueError::new_err(format!("{name} must be an array of finite numbers"))
    })?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    PyValueError::new_err(format!("{name}[{index}] must be a finite number"))
                })
        })
        .collect()
}

fn parse_positive_size(params: &Value, name: &str) -> PyResult<usize> {
    params[name]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be a positive integer")))
}

fn check_vector_lengths(x: &Array1<f64>, y: &Array1<f64>) -> PyResult<()> {
    if x.len() != y.len() {
        return Err(PyValueError::new_err("Input vectors must have equal lengths"));
    }
    Ok(())
}

#[pyfunction]
fn py_compute_vector_similarity_stats(_py: Python, json_params: &str) -> PyResult<String> {
    let params = parse_json_params(json_params)?;
    let vector_1 = parse_vector(&params, "vector_1")?;
    let vector_2 = parse_vector(&params, "vector_2")?;
    check_vector_lengths(&vector_1, &vector_2)?;
    let similarity_measure = params.get("similarity_measure").and_then(|v| v.as_str());

    let result = compute_vector_similarity_stats(&vector_1, &vector_2, similarity_measure);
    Ok(result.to_string())
}

#[pyfunction]
fn py_compute_bootstrapped_similarity_stats(_py: Python, json_params: &str) -> PyResult<String> {
    let params = parse_json_params(json_params)?;
    let x = parse_vector(&params, "x")?;
    let y = parse_vector(&params, "y")?;
    check_vector_lengths(&x, &y)?;
    let sample_size = parse_positive_size(&params, "sample_size")?;
    let number_of_bootstraps = parse_positive_size(&params, "number_of_bootstraps")?;
    let similarity_measure = params.get("similarity_measure").and_then(|v| v.as_str());

    let result = compute_bootstrapped_similarity_stats(
        &x,
        &y,
        sample_size,
        number_of_bootstraps,
        similarity_measure,
    );
    Ok(result.to_string())
}

#[pymodule]
fn fast_vector_similarity(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_compute_vector_similarity_stats, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_bootstrapped_similarity_stats, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::compute_average_and_stdev_of_25th_to_75th_percentile_func;
    use ndarray::Array1;

    #[test]
    fn bootstrap_summary_propagates_non_finite_replicates_before_trimming() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for index in 0..4 {
                let mut values = Array1::from(vec![0.25, 0.5, 0.75, 1.0]);
                values[index] = invalid;
                let (mean, stdev) =
                    compute_average_and_stdev_of_25th_to_75th_percentile_func(&values);
                assert!(mean.is_nan(), "non-finite replicate must invalidate the mean");
                assert!(
                    stdev.is_nan(),
                    "non-finite replicate must invalidate the stdev"
                );
            }
        }
    }
}
