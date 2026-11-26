mod distance;
mod hoeffding;

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
use std::borrow::Cow;

fn has_non_finite<'a>(mut values: impl Iterator<Item = &'a f64>) -> bool {
    values.any(|value| !value.is_finite())
}

// A unit-interval coordinate system for nonempty, finite observations. Subtract
// before dividing to preserve nearby values at large offsets. Only halve when
// the range overflows; always halving would lose subnormal observations.
struct FiniteRange {
    minimum: f64,
    width: f64,
    halve: bool,
}

impl FiniteRange {
    fn new(values: &Array1<f64>) -> Self {
        let (minimum, maximum) = values.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(min, max), &value| (min.min(value), max.max(value)),
        );
        let width = maximum - minimum;
        if width.is_infinite() {
            Self {
                minimum: minimum * 0.5,
                width: maximum * 0.5 - minimum * 0.5,
                halve: true,
            }
        } else {
            Self { minimum, width, halve: false }
        }
    }

    fn position(&self, value: f64) -> f64 {
        if self.width == 0.0 {
            return 0.0;
        }
        let value = if self.halve { value * 0.5 } else { value };
        ((value - self.minimum) / self.width).clamp(0.0, 1.0)
    }

    fn bin(&self, value: f64, num_bins: usize) -> usize {
        ((self.position(value) * num_bins as f64) as usize).min(num_bins - 1)
    }
}

fn logical_values(values: &Array1<f64>) -> Cow<'_, [f64]> {
    match values.as_slice() {
        Some(slice) => Cow::Borrowed(slice),
        // Owned arrays may have negative or non-unit strides. Preserve logical
        // order so observations in the two vectors remain paired correctly.
        None => Cow::Owned(values.iter().copied().collect()),
    }
}

fn average_rank(data: &[f64]) -> Vec<f64> {
    let mut ranks: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
    ranks.par_sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank_values = vec![0.0; data.len()];
    let mut i = 0;
    while i < ranks.len() {
        let mut end = i + 1;
        while end < ranks.len() && ranks[i].1 == ranks[end].1 {
            end += 1;
        }
        let average_rank = (i as f64 + end as f64 + 1.0) / 2.0;
        for &(index, _) in &ranks[i..end] {
            rank_values[index] = average_rank;
        }
        i = end;
    }
    rank_values
}

/// Spearman correlation is Pearson correlation of average ranks, including ties.
/// Returns NaN for fewer than two observations, a constant vector, or any
/// non-finite observation. Inputs must have equal lengths.
pub fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    if x.len() < 2 || has_non_finite(x.iter().chain(y.iter())) {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let rank_x = average_rank(x);
    let rank_y = average_rank(y);

    // Average ranks preserve the mean (n + 1) / 2 even when observations tie.
    // The shortcut based on squared rank differences is only valid without ties.
    let mean_rank = (n + 1.0) / 2.0;
    let (mut covariance, mut variance_x, mut variance_y) = (0.0, 0.0, 0.0);
    for (&rx, &ry) in rank_x.iter().zip(rank_y.iter()) {
        let dx = rx - mean_rank;
        let dy = ry - mean_rank;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    if variance_x == 0.0 || variance_y == 0.0 {
        return f64::NAN;
    }
    (covariance / variance_x.sqrt() / variance_y.sqrt()).clamp(-1.0, 1.0)
}

fn count_inversions(arr: &mut [f64]) -> u128 {
    fn merge_sort(arr: &mut [f64], temp: &mut [f64]) -> u128 {
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

    fn merge(arr: &mut [f64], temp: &mut [f64], mid: usize) -> u128 {
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
                inv_count += (mid - i) as u128;
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

fn pair_count(n: usize) -> u128 {
    let n = n as u128;
    n * n.saturating_sub(1) / 2
}

fn count_tied_pairs<T>(values: &[T], equal: impl Fn(&T, &T) -> bool) -> u128 {
    let mut ties = 0;
    let mut start = 0;
    while start < values.len() {
        let mut end = start + 1;
        while end < values.len() && equal(&values[start], &values[end]) {
            end += 1;
        }
        ties += pair_count(end - start);
        start = end;
    }
    ties
}

/// Kendall's tau-b, correcting for ties in either or both coordinates.
/// Runs in O(n log n) time with O(n) auxiliary storage. Returns NaN for fewer
/// than two observations, a constant vector, or any non-finite observation.
pub fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    if x.len() < 2 || has_non_finite(x.iter().chain(y.iter())) {
        return f64::NAN;
    }
    let mut pairs: Vec<(f64, f64)> = x.iter().copied().zip(y.iter().copied()).collect();
    // Sorting by y within each x tie prevents those pairs from being counted as
    // discordant. Numeric comparison also keeps -0.0 and +0.0 in the same group.
    pairs.par_sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    let ties_x = count_tied_pairs(&pairs, |a, b| a.0 == b.0);
    let ties_xy = count_tied_pairs(&pairs, |a, b| a == b);
    let mut sorted_y: Vec<f64> = pairs.iter().map(|&(_, y)| y).collect();
    let discordant = count_inversions(&mut sorted_y);
    let ties_y = count_tied_pairs(&sorted_y, |a, b| a == b);

    let total_pairs = pair_count(x.len());
    let untied_x = total_pairs - ties_x;
    let untied_y = total_pairs - ties_y;
    if untied_x == 0 || untied_y == 0 {
        return f64::NAN;
    }
    let comparable = untied_x - (ties_y - ties_xy);
    let concordant = comparable - discordant;
    // Subtract integer counts before conversion to retain small differences.
    let numerator = if concordant >= discordant {
        (concordant - discordant) as f64
    } else {
        -((discordant - concordant) as f64)
    };
    (numerator / (untied_x as f64).sqrt() / (untied_y as f64).sqrt()).clamp(-1.0, 1.0)
}

/// Returns NaN for empty inputs or any NaN or infinite observation.
pub fn jensen_shannon_dependency_measure(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    assert_eq!(x.len(), y.len());
    if x.is_empty() || has_non_finite(x.iter().chain(y.iter())) {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let num_bins: usize = 20;
    let x_range = FiniteRange::new(x);
    let y_range = FiniteRange::new(y);
    let x_bins: Vec<usize> = x.iter().map(|&value| x_range.bin(value, num_bins)).collect();
    let mut y_bins: Vec<usize> = y.iter().map(|&value| y_range.bin(value, num_bins)).collect();

    let compute_jsd = |y_bins: &[usize]| -> f64 {

        let mut joint_hist = vec![vec![0f64; num_bins]; num_bins];
        let mut x_hist = vec![0f64; num_bins];
        let mut y_hist = vec![0f64; num_bins];

        x_bins.iter().zip(y_bins.iter()).for_each(|(&x_bin, &y_bin)| {
            joint_hist[x_bin][y_bin] += 1.0;
            x_hist[x_bin] += 1.0;
            y_hist[y_bin] += 1.0;
        });

        let p_x: Vec<f64> = x_hist.iter().map(|&count| count / n).collect();
        let p_y: Vec<f64> = y_hist.iter().map(|&count| count / n).collect();
        let p_xy: Vec<Vec<f64>> = joint_hist.iter().map(|row| row.iter().map(|&count| count / n).collect()).collect();

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

    let jsd_observed = compute_jsd(&y_bins);

    // Shuffling bin labels is equivalent to shuffling y, and reuses the same
    // discretization without repeating range scans or floating-point binning.
    let mut rng = rand::thread_rng();
    y_bins.shuffle(&mut rng);

    let jsd_independent = compute_jsd(&y_bins);

    let dependency = (jsd_observed - jsd_independent) / (1.0 - jsd_independent);
    dependency.max(0.0).min(1.0) // Ensure the value is between 0 and 1
}

/// Returns NaN for empty inputs or any NaN or infinite observation, even outside
/// the randomly selected subset. Uses O(m) working storage for at most 1000
/// sampled observations while preserving the biased centered-distance estimate.
pub fn approximate_distance_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    assert_eq!(x.len(), y.len());
    if x.is_empty() || y.is_empty() || has_non_finite(x.iter().chain(y.iter())) {
        return f64::NAN;
    }
    let subset_size = 1000.min(x.len());
    let mut rng = rand::thread_rng();
    let subset_indices: Vec<usize> = (0..x.len()).choose_multiple(&mut rng, subset_size);
    let mut x_subset = x.select(Axis(0), &subset_indices);
    let mut y_subset = y.select(Axis(0), &subset_indices);
    // Distance correlation is invariant under independent translation and
    // positive rescaling. Bounded coordinates keep distances and their squares
    // finite without collapsing tiny, nonconstant vectors to zero variance.
    let x_range = FiniteRange::new(&x_subset);
    let y_range = FiniteRange::new(&y_subset);
    x_subset.mapv_inplace(|value| x_range.position(value));
    y_subset.mapv_inplace(|value| y_range.position(value));

    distance::correlation(&logical_values(&x_subset), &logical_values(&y_subset))
}

fn dot_product(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Computes Hoeffding's D in O(n log n) time with O(n) storage.
/// Returns NaN for fewer than five observations or any NaN or infinite value.
pub fn exact_hoeffdings_d_func(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    assert_eq!(x.len(), y.len());
    if x.len() < 5 || has_non_finite(x.iter().chain(y.iter())) {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let r = Array1::from(average_rank(&logical_values(x)));
    let s = Array1::from(average_rank(&logical_values(y)));

    // Precompute necessary arrays
    let r_minus = &r - 1.0;
    let s_minus = &s - 1.0;

    let q = hoeffding::joint_ranks(&r, &s);
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

    let x_range = FiniteRange::new(x);
    let y_range = FiniteRange::new(y);

    // Initialize histograms using ndarray
    let mut joint_hist = Array2::<usize>::zeros((num_bins, num_bins));
    let mut x_hist = Array1::<usize>::zeros(num_bins);
    let mut y_hist = Array1::<usize>::zeros(num_bins);

    // Assign data to bins and compute histograms
    x.iter().zip(y.iter()).for_each(|(&xi, &yi)| {
        // Find bin index for xi and yi
        let x_bin = x_range.bin(xi, num_bins);
        let y_bin = y_range.bin(yi, num_bins);

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
    if h_x + h_y == 0.0 {
        return f64::NAN;
    }
    (mi / ((h_x + h_y) / 2.0)).clamp(0.0, 1.0)
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
        computations.push((Box::new(|| spearman_rho(&logical_values(vector_1), &logical_values(vector_2))), "spearman_rho"));
    }
    if similarity_measure == "kendall_tau" || similarity_measure == "all" {
        computations.push((Box::new(|| kendall_tau(&logical_values(vector_1), &logical_values(vector_2))), "kendall_tau"));
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
    // would silently bias the estimate. Require at least two bootstrap replicates.
    if input_vector.len() < 2 || has_non_finite(input_vector.iter()) {
        return (f64::NAN, f64::NAN);
    }
    let mut sorted_vector = input_vector.to_vec();
    sorted_vector.par_sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Round the trim count down and remove the same count from each tail.
    // Computing 3 * len / 4 as an exclusive upper bound drops one extra high
    // score when len is not divisible by four, biasing the summary downward.
    let lower_idx = input_vector.len() / 4;
    let upper_idx = input_vector.len() - lower_idx;
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
    use super::{compute_average_and_stdev_of_25th_to_75th_percentile_func, FiniteRange};
    use ndarray::Array1;

    #[test]
    fn histogram_bins_preserve_scale_and_offset_invariance() {
        let offset = 1e300_f64.to_bits();
        let tiny = f64::MIN_POSITIVE;
        let fixtures = [
            Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0]),
            Array1::from(vec![0.0, tiny, 2.0 * tiny, 3.0 * tiny, 4.0 * tiny]),
            Array1::from_iter((0..5).map(f64::from_bits)),
            Array1::from_iter((0..5).map(|i| f64::from_bits(offset + i))),
            Array1::from(vec![
                -f64::MAX, -f64::MAX / 2.0, 0.0, f64::MAX / 2.0, f64::MAX,
            ]),
        ];
        for values in fixtures {
            let range = FiniteRange::new(&values);
            let bins: Vec<_> = values.iter().map(|&value| range.bin(value, 20)).collect();
            assert_eq!(bins, [0, 5, 10, 15, 19], "values: {values:?}");
        }
    }

    #[test]
    fn json_input_preserves_finite_float_roundtrips() {
        let offset = 1e300_f64.to_bits();
        let mut values = vec![
            0.0, -0.0, f64::from_bits(1), f64::MIN_POSITIVE,
            f64::MAX, -f64::MAX, 3.9287532173373315e299,
        ];
        values.extend((0..8).map(|i| f64::from_bits(offset + i)));
        let encoded = serde_json::to_string(&values).unwrap();
        let decoded: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        for (index, value) in values.iter().enumerate() {
            assert_eq!(decoded[index].as_f64().unwrap().to_bits(), value.to_bits());
        }
    }

    #[test]
    fn histogram_bins_handle_constant_values_and_signed_zero() {
        for values in [
            Array1::from(vec![-0.0, 0.0, -0.0, 0.0]),
            Array1::from_elem(4, f64::MAX),
            Array1::from_elem(4, f64::from_bits(1)),
        ] {
            let range = FiniteRange::new(&values);
            assert!(values.iter().all(|&value| range.bin(value, 20) == 0));
        }
    }

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

    fn bootstrap_summary_of(values: &[f64]) -> (f64, f64) {
        compute_average_and_stdev_of_25th_to_75th_percentile_func(&Array1::from(values.to_vec()))
    }

    fn assert_bootstrap_pair(actual: (f64, f64), expected: (f64, f64)) {
        assert!(
            actual.0.is_finite() && (actual.0 - expected.0).abs() <= 1e-12,
            "mean: expected {}, got {}",
            expected.0,
            actual.0
        );
        assert!(
            actual.1.is_finite() && (actual.1 - expected.1).abs() <= 1e-12,
            "population stdev: expected {}, got {}",
            expected.1,
            actual.1
        );
    }

    #[test]
    fn bootstrap_summary_trims_equal_numbers_from_both_tails() {
        // Expected retained samples are explicit, independent of index arithmetic.
        // Tuple entries are (input observations, expected mean, population variance).
        let cases: &[(&[f64], f64, f64)] = &[
            // B=2: retain [1, 3].
            (&[3.0, 1.0], 2.0, 1.0),
            // B=3: retain [1, 2, 3].
            (&[3.0, 1.0, 2.0], 2.0, 2.0 / 3.0),
            // B=5: retain [1, 2, 3].
            (&[40.0, 3.0, -20.0, 1.0, 2.0], 2.0, 2.0 / 3.0),
            // B=6: retain [0, 2, 4, 6].
            (&[70.0, 4.0, -50.0, 2.0, 6.0, 0.0], 3.0, 5.0),
            // B=7: retain [-2, -1, 0, 1, 2].
            (&[110.0, 1.0, -90.0, -1.0, 2.0, -2.0, 0.0], 0.0, 2.0),
            // B=4: retain [1, 3], preserving the existing divisible-by-four rule.
            (&[40.0, 3.0, -10.0, 1.0], 2.0, 1.0),
            // B=8: retain [0, 2, 4, 6].
            (&[90.0, 4.0, -100.0, 0.0, 70.0, 6.0, -80.0, 2.0], 3.0, 5.0),
            // B=12: retain [1, 2, 3, 4, 5, 6].
            (
                &[300.0, 3.0, -100.0, 5.0, 100.0, 1.0, -300.0, 6.0, 200.0, 2.0, -200.0, 4.0],
                3.5,
                35.0 / 12.0,
            ),
            // Ties at trim boundaries: retain [0, 1, 1, 1, 2].
            (&[2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0], 1.0, 2.0 / 5.0),
            // Constant observations retain their value and zero spread.
            (&[2.0; 7], 2.0, 0.0),
        ];
        for &(values, mean, variance) in cases {
            assert_bootstrap_pair(bootstrap_summary_of(values), (mean, variance.sqrt()));
        }
    }

    #[test]
    fn bootstrap_summary_sign_reversal_preserves_spread_and_reverses_mean() {
        for values in [
            vec![-1.0, 1.0],
            vec![-1.0, 0.0, 1.0],
            vec![101.0, -9.0, 1.0, 4.0, 8.0, -12.0, 17.0],
        ] {
            let (mean, stdev) = bootstrap_summary_of(&values);
            let reflected: Vec<f64> = values.iter().map(|&value| -value).collect();
            assert_bootstrap_pair(bootstrap_summary_of(&reflected), (-mean, stdev));
        }
    }

    #[test]
    fn bootstrap_summary_is_independent_of_observation_order() {
        let values = [101.0, -9.0, 1.0, 4.0, 8.0, -12.0, 17.0];
        // Retain [-9, 1, 4, 8, 17]: mean = 21/5, variance = 1814/25.
        let expected = (21.0 / 5.0, (1814.0_f64 / 25.0).sqrt());
        for shift in 0..values.len() {
            let mut permuted = values;
            permuted.rotate_left(shift);
            assert_bootstrap_pair(bootstrap_summary_of(&permuted), expected);
            permuted.reverse();
            assert_bootstrap_pair(bootstrap_summary_of(&permuted), expected);
        }
    }

    #[test]
    fn bootstrap_summary_respects_affine_changes_including_negative_scale() {
        let values = [101.0, -9.0, 1.0, 4.0, 8.0, -12.0, 17.0];
        // Explicit baseline moments of retained [-9, 1, 4, 8, 17].
        let expected_mean = 21.0 / 5.0;
        let expected_stdev = (1814.0_f64 / 25.0).sqrt();
        for (scale, offset) in [(3.0_f64, 11.0), (-2.0, 8.0), (1.0, -32.0)] {
            let transformed: Vec<f64> = values.iter().map(|&value| scale * value + offset).collect();
            assert_bootstrap_pair(
                bootstrap_summary_of(&transformed),
                (scale * expected_mean + offset, scale.abs() * expected_stdev),
            );
        }
    }

    #[test]
    fn bootstrap_summary_rejects_non_finite_values_even_in_trimmed_tails() {
        let original = [-100.0, -2.0, -1.0, 0.0, 1.0, 2.0, 100.0];
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for index in 0..original.len() {
                let mut values = original;
                values[index] = invalid;
                let (mean, stdev) = bootstrap_summary_of(&values);
                assert!(mean.is_nan(), "invalid replicate at {index} must invalidate the mean");
                assert!(stdev.is_nan(), "invalid replicate at {index} must invalidate the spread");
            }
        }
    }

    #[test]
    fn bootstrap_summary_requires_at_least_two_observations() {
        for values in [vec![], vec![7.0]] {
            let (mean, stdev) = bootstrap_summary_of(&values);
            assert!(mean.is_nan());
            assert!(stdev.is_nan());
        }
    }
}
