use fast_vector_similarity::{
    compute_bootstrapped_similarity_stats, compute_vector_similarity_stats, exact_hoeffdings_d_func,
};
use ndarray::{s, Array1};
use serde_json::Value;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

fn twice_midranks(values: &[f64]) -> Vec<i128> {
    values
        .iter()
        .map(|&value| {
            let lower = values.iter().filter(|&&other| other < value).count() as i128;
            let equal = values.iter().filter(|&&other| other == value).count() as i128;
            2 * lower + equal + 1
        })
        .collect()
}

/// Independent quadratic reference for small inputs. Midranks are represented
/// in half-units, joint ranks in quarter-units, and all three statistic moments
/// in sixteenth-units. Integer arithmetic is exact until the final division.
fn quadratic_hoeffding(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    if x.len() < 5 {
        return f64::NAN;
    }
    let r = twice_midranks(x);
    let s = twice_midranks(y);
    let (mut d1, mut d2, mut d3) = (0_i128, 0_i128, 0_i128);
    for i in 0..x.len() {
        let mut q = 4_i128;
        for j in 0..x.len() {
            if j == i {
                continue;
            }
            let x_weight = if x[j] < x[i] {
                2
            } else if x[j] == x[i] {
                1
            } else {
                0
            };
            let y_weight = if y[j] < y[i] {
                2
            } else if y[j] == y[i] {
                1
            } else {
                0
            };
            q += x_weight * y_weight;
        }
        d1 += (q - 4) * (q - 8);
        d2 += (r[i] - 2) * (r[i] - 4) * (s[i] - 2) * (s[i] - 4);
        d3 += (r[i] - 4) * (s[i] - 4) * (q - 4);
    }
    let n = x.len() as i128;
    let numerator = 30 * ((n - 2) * (n - 3) * d1 + d2 - 2 * (n - 2) * d3);
    let denominator = 16 * n * (n - 1) * (n - 2) * (n - 3) * (n - 4);
    numerator as f64 / denominator as f64
}

#[test]
fn small_tied_inputs_match_an_exact_quadratic_reference() {
    // A deterministic generator exercises different tie group sizes, duplicate
    // pairs, observation orders and half-integer midranks without test RNG.
    for n in 5..=12 {
        for seed in 0..32_u64 {
            let mut state = seed + 97 * n as u64;
            let mut next = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                state >> 32
            };
            let x: Vec<f64> = (0..n).map(|_| (next() % 5) as f64 - 2.0).collect();
            let y: Vec<f64> = (0..n).map(|_| (next() % 4) as f64 - 1.0).collect();
            let expected = quadratic_hoeffding(&x, &y);
            let x = Array1::from(x);
            let y = Array1::from(y);
            assert_close(exact_hoeffdings_d_func(&x, &y), expected, 1e-12);
            assert_close(exact_hoeffdings_d_func(&y, &x), expected, 1e-12);
        }
    }
}

#[test]
fn constants_joint_duplicates_and_signed_zero_preserve_tie_semantics() {
    let fixtures = [
        (vec![1.0; 8], vec![2.0; 8]),
        (vec![1.0; 8], vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
        (
            vec![-0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
            vec![1.0, 1.0, -0.0, 0.0, 2.0, 2.0, 2.0, 3.0],
        ),
    ];
    for (x_values, y_values) in fixtures {
        let expected = quadratic_hoeffding(&x_values, &y_values);
        let x = Array1::from(x_values);
        let y = Array1::from(y_values);
        assert_close(exact_hoeffdings_d_func(&x, &y), expected, 1e-12);
        assert_close(exact_hoeffdings_d_func(&y, &x), expected, 1e-12);
        let positive_zeros = x.mapv(|value| if value == 0.0 { 0.0 } else { value });
        assert_close(
            exact_hoeffdings_d_func(&positive_zeros, &y),
            expected,
            1e-12,
        );
    }
}

#[test]
fn strided_observations_match_the_logical_order_reference() {
    let x_values = [-0.0, 2.0, 1.0, 0.0, 3.0, 2.0, 4.0, 1.0];
    let y_values = [1.0, 3.0, 2.0, -0.0, 0.0, 1.0, 4.0, 2.0];
    let expected = quadratic_hoeffding(&x_values, &y_values);
    let x = Array1::from_iter(x_values.iter().flat_map(|&value| [value, f64::NAN]))
        .slice_move(s![..;2]);
    let y = Array1::from_iter(y_values.iter().rev().copied()).slice_move(s![..;-1]);
    assert!(x.as_slice().is_none());
    assert!(y.as_slice().is_none());
    assert_close(exact_hoeffdings_d_func(&x, &y), expected, 1e-12);
    assert_close(exact_hoeffdings_d_func(&y, &x), expected, 1e-12);
}

#[test]
fn strictly_monotone_unique_data_has_unit_hoeffding_d() {
    for n in [5, 6, 12, 31] {
        let x = Array1::from_iter((0..n).map(|i| i as f64));
        let increasing = x.mapv(|value| value * value + 7.0);
        let decreasing = x.mapv(|value| -(value * value + 7.0));
        assert_close(exact_hoeffdings_d_func(&x, &increasing), 1.0, 1e-12);
        assert_close(exact_hoeffdings_d_func(&x, &decreasing), 1.0, 1e-12);
    }
}

#[test]
fn fewer_than_five_observations_are_undefined_directly_and_in_json() {
    for n in 0..5 {
        for x in [
            Array1::from_iter((0..n).map(|i| i as f64)),
            Array1::from_elem(n, 2.0),
        ] {
            assert!(exact_hoeffdings_d_func(&x, &x).is_nan());
            let result = compute_vector_similarity_stats(&x, &x, Some("hoeffding_d"));
            assert_eq!(result.get("hoeffding_d"), Some(&Value::Null));
            let bootstrap =
                compute_bootstrapped_similarity_stats(&x, &x, n, 8, Some("hoeffding_d"));
            assert_eq!(bootstrap.get("hoeffding_d"), Some(&Value::Null));
            assert_eq!(bootstrap.get("stdev_hoeffding_d"), Some(&Value::Null));
        }
    }
}

#[test]
fn selected_all_and_bootstrap_statistics_use_the_public_hoeffding_path() {
    let x_values = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0];
    let y_values = [3.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 4.0];
    let expected = quadratic_hoeffding(&x_values, &y_values);
    let x = Array1::from(x_values.to_vec());
    let y = Array1::from(y_values.to_vec());
    for selection in [None, Some("all"), Some("hoeffding_d")] {
        let result = compute_vector_similarity_stats(&x, &y, selection);
        assert_eq!(result["input_vector_dimensions"], x_values.len());
        assert_close(result["hoeffding_d"].as_f64().unwrap(), expected, 1e-12);
    }

    let result = compute_bootstrapped_similarity_stats(&x, &y, 12, 16, Some("hoeffding_d"));
    assert_eq!(result["sample_size"], 12);
    assert_eq!(result["number_of_bootstraps"], 16);
    assert_eq!(result.as_object().unwrap().len(), 4);
    assert!(result["hoeffding_d"].as_f64().unwrap().is_finite());
    let deviation = result["stdev_hoeffding_d"].as_f64().unwrap();
    assert!(deviation.is_finite() && deviation >= 0.0);
}

#[test]
fn one_hundred_thousand_unique_observations_remain_practical() {
    // No timing threshold: this workload detects accidentally restoring the
    // quadratic counting path while checking both monotone directions.
    let n = 100_000;
    let x = Array1::from_iter((0..n).map(|i| i as f64));
    let reversed = Array1::from_iter((0..n).rev().map(|i| i as f64));
    assert_close(exact_hoeffdings_d_func(&x, &x), 1.0, 1e-9);
    let result = compute_vector_similarity_stats(&x, &reversed, Some("hoeffding_d"));
    assert_eq!(result["input_vector_dimensions"], n);
    assert_close(result["hoeffding_d"].as_f64().unwrap(), 1.0, 1e-9);
}
