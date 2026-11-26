use fast_vector_similarity::{
    approximate_distance_correlation, compute_bootstrapped_similarity_stats,
    compute_vector_similarity_stats, normalized_mutual_information,
};
use ndarray::Array1;
use serde_json::Value;

fn assert_close(actual: f64, expected: f64, context: &str) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= 1e-12,
        "{context}: expected {expected}, got {actual}"
    );
}

fn reversed(values: &Array1<f64>) -> Array1<f64> {
    values.iter().rev().copied().collect()
}

fn extreme_fixtures() -> Vec<(&'static str, Array1<f64>)> {
    let large_offset = 1e300_f64.to_bits();
    vec![
        ("ordinary", Array1::from(vec![-2.0, -1.0, 0.0, 1.0, 2.0])),
        (
            "tiny",
            Array1::from(vec![-2e-300, -1e-300, 0.0, 1e-300, 2e-300]),
        ),
        ("subnormal", Array1::from_iter((0..5).map(f64::from_bits))),
        (
            "opposite finite extremes",
            Array1::from(vec![
                -f64::MAX,
                -f64::MAX / 2.0,
                0.0,
                f64::MAX / 2.0,
                f64::MAX,
            ]),
        ),
        (
            "adjacent large values",
            Array1::from_iter((0..5).map(|offset| f64::from_bits(large_offset + offset))),
        ),
    ]
}

#[test]
fn self_and_reversed_similarity_survive_tiny_and_extreme_finite_ranges() {
    for (name, x) in extreme_fixtures() {
        let reverse = reversed(&x);
        for y in [&x, &reverse] {
            assert_close(normalized_mutual_information(&x, y), 1.0, name);
            assert_close(approximate_distance_correlation(&x, y), 1.0, name);
        }
    }
}

#[test]
fn nmi_preserves_independence_dependence_and_constants_at_every_scale() {
    let adjacent = 1e300_f64.to_bits();
    for (low, high) in [
        (0.0, 1.0),
        (0.0, 1e-12),
        (0.0, 1e-300),
        (0.0, f64::from_bits(1)),
        (-f64::MAX, f64::MAX),
        (f64::from_bits(adjacent), f64::from_bits(adjacent + 1)),
    ] {
        let x = Array1::from(vec![low, low, high, high]);
        let independent = Array1::from(vec![low, high, low, high]);
        let reverse = Array1::from(vec![high, high, low, low]);
        let constant = Array1::from_elem(4, low);
        assert_close(normalized_mutual_information(&x, &x), 1.0, "NMI identity");
        assert_close(
            normalized_mutual_information(&x, &reverse),
            1.0,
            "NMI reversed categories",
        );
        assert_close(
            normalized_mutual_information(&x, &independent),
            0.0,
            "NMI independent contingency",
        );
        assert_close(
            normalized_mutual_information(&constant, &x),
            0.0,
            "NMI one constant marginal",
        );
        assert!(normalized_mutual_information(&constant, &constant).is_nan());
    }
}

#[test]
fn nmi_preserves_a_nontrivial_information_ratio_under_affine_changes() {
    // Four equiprobable X categories determine two equiprobable Y categories:
    // H(X) = ln(4), H(Y) = I(X;Y) = ln(2), so average-entropy NMI = 2/3.
    let x = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    let y = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    for (x_scale, y_scale, x_offset, y_offset) in [
        (1.0, 1.0, 0.0, 0.0),
        (1.0, -2.0, 16.0, 8.0),
        (1e-300, 1e300, 0.0, 0.0),
        (1e300, -1e-300, 0.0, 0.0),
    ] {
        let scaled_x = x.mapv(|v| v * x_scale + x_offset);
        let scaled_y = y.mapv(|v| v * y_scale + y_offset);
        assert_close(
            normalized_mutual_information(&scaled_x, &scaled_y),
            2.0 / 3.0,
            "NMI nontrivial affine invariance",
        );
    }
    for origin in [0, 1e300_f64.to_bits()] {
        let adjacent_x = x.mapv(|v| f64::from_bits(origin + v as u64));
        let adjacent_y = y.mapv(|v| f64::from_bits(origin + v as u64));
        assert_close(
            normalized_mutual_information(&adjacent_x, &adjacent_y),
            2.0 / 3.0,
            "NMI subnormal or adjacent-large information ratio",
        );
    }
}

#[test]
fn distance_correlation_avoids_overflow_and_underflow_in_normalization() {
    for magnitude in [1e80, 1e-100, 1e-200, 1e-300] {
        let x = Array1::from(vec![-magnitude, 0.0, magnitude]);
        assert_close(
            approximate_distance_correlation(&x, &x),
            1.0,
            "distance self-correlation at extreme magnitude",
        );
    }
}

#[test]
fn distance_correlation_matches_an_independent_exact_moment_reference() {
    // For X = [-2,-1,0,1,2], Y = X^2, direct integer sums of uncentered
    // pairwise distances give these biased centered second moments:
    // covariance = 280/625, variance_x = 760/625, variance_y = 1456/625.
    // Their common denominator cancels in the normalized expression.
    let expected = (280.0_f64 / (760.0_f64 * 1456.0).sqrt()).sqrt();
    let x = Array1::from(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    let y = Array1::from(vec![4.0, 1.0, 0.0, 1.0, 4.0]);
    for (x_scale, y_scale, x_offset, y_offset) in [
        (1.0, 1.0, 0.0, 0.0),
        (-2.0, 4.0, 16.0, -8.0),
        (1e80, 1e80, 0.0, 0.0),
        (1e-100, 1e-100, 0.0, 0.0),
        (1e-300, 1e300, 0.0, 0.0),
        (-1e300, 1e-300, 0.0, 0.0),
    ] {
        let scaled_x = x.mapv(|v| v * x_scale + x_offset);
        let scaled_y = y.mapv(|v| v * y_scale + y_offset);
        assert_close(
            approximate_distance_correlation(&scaled_x, &scaled_y),
            expected,
            "distance nonlinear reference with independent rescaling",
        );
    }

    let adjacent = 1e300_f64.to_bits();
    for origin in [0, adjacent] {
        let shifted_x = Array1::from_iter((0..5).map(|offset| f64::from_bits(origin + offset)));
        let shifted_y = Array1::from_iter(
            [4, 1, 0, 1, 4]
                .into_iter()
                .map(|offset| f64::from_bits(origin + offset)),
        );
        assert_close(
            approximate_distance_correlation(&shifted_x, &shifted_y),
            expected,
            "distance subnormal or adjacent-large nonlinear reference",
        );
    }

    let extreme_x = Array1::from(vec![
        -f64::MAX,
        -f64::MAX / 2.0,
        0.0,
        f64::MAX / 2.0,
        f64::MAX,
    ]);
    let extreme_y = Array1::from(vec![
        f64::MAX,
        f64::MAX / 4.0,
        0.0,
        f64::MAX / 4.0,
        f64::MAX,
    ]);
    assert_close(
        approximate_distance_correlation(&extreme_x, &extreme_y),
        expected,
        "distance nonlinear reference spanning opposite finite extremes",
    );
}

#[test]
fn distance_correlation_preserves_independent_and_constant_cases() {
    for (low, high) in [(0.0, 1e-300), (-f64::MAX, f64::MAX)] {
        let x = Array1::from(vec![low, low, high, high]);
        let y = Array1::from(vec![low, high, low, high]);
        let constant = Array1::from_elem(4, low);
        assert_close(
            approximate_distance_correlation(&x, &y),
            0.0,
            "distance independent contingency",
        );
        assert_close(
            approximate_distance_correlation(&constant, &x),
            0.0,
            "distance constant input",
        );
        assert_close(
            approximate_distance_correlation(&x, &constant),
            0.0,
            "distance other constant input",
        );
    }
}

#[test]
fn selected_json_scores_preserve_finite_extreme_range_results() {
    for (name, x) in extreme_fixtures() {
        for measure in [
            "normalized_mutual_information",
            "approximate_distance_correlation",
        ] {
            let result = compute_vector_similarity_stats(&x, &x, Some(measure));
            assert_eq!(result["input_vector_dimensions"], x.len());
            assert_eq!(result.as_object().unwrap().len(), 2);
            let score = result[measure]
                .as_f64()
                .expect("finite score must not become null");
            assert_close(score, 1.0, name);
        }
    }
}

#[test]
fn bootstrap_extreme_range_results_remain_defined_except_degenerate_nmi_samples() {
    for (name, x) in extreme_fixtures() {
        for measure in [
            "normalized_mutual_information",
            "approximate_distance_correlation",
        ] {
            let result = compute_bootstrapped_similarity_stats(&x, &x, 8, 8, Some(measure));
            assert_eq!(result["sample_size"], 8);
            assert_eq!(result["number_of_bootstraps"], 8);
            assert_eq!(result.as_object().unwrap().len(), 4);
            let deviation_key = format!("stdev_{measure}");
            if measure == "normalized_mutual_information" && result[measure].is_null() {
                // Sampling with replacement can draw a constant replicate,
                // which correctly makes the entire NMI summary undefined.
                assert_eq!(result.get(measure), Some(&Value::Null));
                assert_eq!(result.get(&deviation_key), Some(&Value::Null));
                continue;
            }
            let score = result[measure]
                .as_f64()
                .expect("expected a finite bootstrap score");
            let stdev = result[&deviation_key]
                .as_f64()
                .expect("expected a finite bootstrap stdev");
            assert!(
                score.is_finite() && (0.0..=1.0 + 1e-12).contains(&score),
                "{name}: {result}"
            );
            assert!(
                stdev.is_finite() && (0.0..=0.5 + 1e-12).contains(&stdev),
                "{name}: {result}"
            );
            if measure == "normalized_mutual_information" {
                assert_close(score, 1.0, name);
                assert_close(stdev, 0.0, name);
            }
        }
    }
}
