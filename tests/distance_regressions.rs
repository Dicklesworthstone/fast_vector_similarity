use fast_vector_similarity::{
    approximate_distance_correlation, compute_bootstrapped_similarity_stats,
    compute_vector_similarity_stats,
};
use ndarray::{s, Array1};
use serde_json::Value;

/// Exact raw-distance moments for small integer fixtures, independently of
/// floating-point centering. All three centered sums have the same n^2
/// denominator, which cancels in the normalized coefficient.
fn integer_moment_reference(x: &[i64], y: &[i64]) -> f64 {
    assert_eq!(x.len(), y.len());
    assert!((1..=64).contains(&x.len()));

    let mut row_x = vec![0_i128; x.len()];
    let mut row_y = vec![0_i128; y.len()];
    let (mut raw_xy, mut raw_xx, mut raw_yy) = (0_i128, 0_i128, 0_i128);

    for i in 0..x.len() {
        for j in 0..x.len() {
            let dx = (i128::from(x[i]) - i128::from(x[j])).abs();
            let dy = (i128::from(y[i]) - i128::from(y[j])).abs();
            row_x[i] += dx;
            row_y[i] += dy;
            raw_xy += dx * dy;
            raw_xx += dx * dx;
            raw_yy += dy * dy;
        }
    }

    let n = x.len() as i128;
    let total_x: i128 = row_x.iter().sum();
    let total_y: i128 = row_y.iter().sum();
    let rows_xy: i128 = row_x.iter().zip(&row_y).map(|(&a, &b)| a * b).sum();
    let rows_xx: i128 = row_x.iter().map(|&a| a * a).sum();
    let rows_yy: i128 = row_y.iter().map(|&b| b * b).sum();

    let covariance = n * n * raw_xy - 2 * n * rows_xy + total_x * total_y;
    let variance_x = n * n * raw_xx - 2 * n * rows_xx + total_x * total_x;
    let variance_y = n * n * raw_yy - 2 * n * rows_yy + total_y * total_y;

    assert!(covariance >= 0 && variance_x >= 0 && variance_y >= 0);
    if variance_x == 0 || variance_y == 0 {
        return 0.0;
    }

    (covariance as f64 / (variance_x as f64).sqrt() / (variance_y as f64).sqrt()).sqrt()
}

fn assert_coefficient(actual: f64, expected: f64) {
    assert!(
        actual.is_finite() && (0.0..=1.0).contains(&actual),
        "invalid coefficient {actual}"
    );
    if expected == 0.0 {
        // Near zero, the final square root amplifies last-bit covariance
        // rounding. Compare its square against the exact zero moment instead.
        assert!(
            actual * actual <= 1e-14,
            "expected zero dependence, got {actual}"
        );
    } else {
        assert!(
            (actual - expected).abs() <= 1e-11,
            "expected {expected}, got {actual}"
        );
    }
}

fn signed_zero_array(values: &[i64]) -> Array1<f64> {
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if value == 0 && index % 2 == 0 {
                -0.0
            } else {
                value as f64
            }
        })
        .collect()
}

fn reversed_storage(values: &[f64]) -> Array1<f64> {
    let result = Array1::from_iter(values.iter().rev().copied()).slice_move(s![..;-1]);
    assert!(result.as_slice().is_none());
    result
}

fn strided_storage(values: &[f64]) -> Array1<f64> {
    let result =
        Array1::from_iter(values.iter().flat_map(|&value| [value, f64::NAN])).slice_move(s![..;2]);
    assert!(result.as_slice().is_none());
    result
}

#[test]
fn distance_matches_exact_integer_moments_for_small_tied_and_nonlinear_inputs() {
    let mut fixtures = vec![
        (vec![0], vec![1]),
        (vec![-1, 1], vec![2, -2]),
        (vec![-1, 0, 1], vec![1, 0, 1]),
        (vec![0, 0, 1, 1], vec![0, 1, 0, 1]),
        (vec![-2, -1, 0, 1, 2], vec![4, 1, 0, 1, 4]),
        (vec![0, 0, 1, 2, 2, 3, 3], vec![3, 1, 0, 2, 3, 0, 1]),
        (vec![-4; 8], vec![1, 0, 1, 2, 2, 0, -1, -1]),
        (vec![1, 0, 1, 2, 2, 0, -1, -1], vec![5; 8]),
        (vec![4; 17], vec![-3; 17]),
    ];

    for n in [5, 7, 16, 31, 64] {
        for seed in 0..8 {
            let x = (0..n)
                .map(|i| ((i * i + 7 * i + seed) % 13) as i64 - 6)
                .collect();
            let y = (0..n)
                .map(|i| ((i * 13 + seed * (i + 1) + 3) % 11) as i64 - 5)
                .collect();
            fixtures.push((x, y));
        }
    }

    for (x_values, y_values) in fixtures {
        let expected = integer_moment_reference(&x_values, &y_values);
        let x = signed_zero_array(&x_values);
        let y = signed_zero_array(&y_values);
        assert_coefficient(approximate_distance_correlation(&x, &y), expected);
        assert_coefficient(approximate_distance_correlation(&y, &x), expected);
    }
}

#[test]
fn distance_preserves_negative_affine_changes_and_logical_array_order() {
    let x_values = [0, -2, 1, 3, 1, 0, 4, -1, -2, 2, 3, 0, 1, 4, -1, 2, 0];
    let y_values = [2, 0, -1, 2, 3, 1, 0, 4, -1, 3, 2, 4, 0, 1, 2, -1, 3];
    let expected = integer_moment_reference(&x_values, &y_values);

    for (x_scale, x_offset, y_scale, y_offset) in [
        (1.0, 0.0, 1.0, 0.0),
        (-3.0, 7.0, 2.0, -4.0),
        (2.0, 16.0, -5.0, 32.0),
    ] {
        let x: Vec<f64> = x_values
            .iter()
            .map(|&v| v as f64 * x_scale + x_offset)
            .collect();
        let y: Vec<f64> = y_values
            .iter()
            .map(|&v| v as f64 * y_scale + y_offset)
            .collect();

        for (array_x, array_y) in [
            (Array1::from(x.clone()), Array1::from(y.clone())),
            (reversed_storage(&x), Array1::from(y.clone())),
            (Array1::from(x.clone()), reversed_storage(&y)),
            (strided_storage(&x), reversed_storage(&y)),
            (reversed_storage(&x), strided_storage(&y)),
            // Reordering observation pairs together preserves DCor.
            (
                Array1::from_iter(x.iter().rev().copied()),
                Array1::from_iter(y.iter().rev().copied()),
            ),
        ] {
            assert_coefficient(
                approximate_distance_correlation(&array_x, &array_y),
                expected,
            );
        }
    }
}

#[test]
fn distance_dispatch_matches_the_reference_for_selected_default_and_all_measures() {
    let x_values = [0, 0, 1, 1, 2, 2, 3, 4];
    let y_values = [3, 1, 2, 0, 0, 2, 1, 4];
    let expected = integer_moment_reference(&x_values, &y_values);
    let x = signed_zero_array(&x_values);
    let y = signed_zero_array(&y_values);

    for selection in [None, Some("all"), Some("approximate_distance_correlation")] {
        let result = compute_vector_similarity_stats(&x, &y, selection);
        assert_eq!(result["input_vector_dimensions"], x_values.len());
        let coefficient = result["approximate_distance_correlation"].as_f64().unwrap();
        assert_coefficient(coefficient, expected);
    }
}

#[test]
fn distance_bootstrap_constant_coordinate_is_zero_for_every_possible_sample() {
    let constant = Array1::from_elem(8, 3.0);
    let variable = Array1::from(vec![0.0, -1.0, 2.0, 4.0, 3.0, 2.0, 1.0, -2.0]);

    for (x, y) in [(&constant, &variable), (&variable, &constant)] {
        for selection in [Some("approximate_distance_correlation"), Some("all")] {
            // Assertions hold for every draw, not just likely samples.
            let result = compute_bootstrapped_similarity_stats(x, y, 8, 2, selection);
            assert_eq!(result["sample_size"], 8);
            assert_eq!(result["number_of_bootstraps"], 2);
            assert_eq!(
                result["approximate_distance_correlation"].as_f64(),
                Some(0.0)
            );
            assert_eq!(
                result["stdev_approximate_distance_correlation"].as_f64(),
                Some(0.0)
            );
        }
    }
}

#[test]
fn distance_full_thousand_observation_subset_preserves_known_relationships() {
    // The complete 1000 observations are selected: randomness changes their
    // order only. Each binary contingency cell appears exactly 250 times.
    let x = Array1::from_iter((0..1000).map(|i| ((i / 2) % 2) as f64));
    let independent = Array1::from_iter((0..1000).map(|i| (i % 2) as f64));
    let independent_score = approximate_distance_correlation(&x, &independent);

    assert!(independent_score.is_finite() && independent_score.abs() <= 1e-12);
    assert_coefficient(approximate_distance_correlation(&x, &x), 1.0);

    let linear = Array1::from_iter((0..1000).map(|i| i as f64));
    let decreasing = linear.mapv(|value| 17.0 - 3.0 * value);
    assert_coefficient(approximate_distance_correlation(&linear, &decreasing), 1.0);
}

#[test]
fn distance_empty_and_non_finite_inputs_remain_undefined_through_public_paths() {
    let empty = Array1::<f64>::zeros(0);
    assert!(approximate_distance_correlation(&empty, &empty).is_nan());
    let empty_stats =
        compute_vector_similarity_stats(&empty, &empty, Some("approximate_distance_correlation"));
    assert_eq!(
        empty_stats.get("approximate_distance_correlation"),
        Some(&Value::Null)
    );

    let finite = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut contaminated = finite.clone();
        contaminated[2] = invalid;
        for (x, y) in [(&contaminated, &finite), (&finite, &contaminated)] {
            assert!(approximate_distance_correlation(x, y).is_nan());
            let stats =
                compute_vector_similarity_stats(x, y, Some("approximate_distance_correlation"));
            assert_eq!(
                stats.get("approximate_distance_correlation"),
                Some(&Value::Null)
            );
            let bootstrap = compute_bootstrapped_similarity_stats(
                x,
                y,
                8,
                4,
                Some("approximate_distance_correlation"),
            );
            assert_eq!(
                bootstrap.get("approximate_distance_correlation"),
                Some(&Value::Null)
            );
            assert_eq!(
                bootstrap.get("stdev_approximate_distance_correlation"),
                Some(&Value::Null)
            );
        }
    }
}
