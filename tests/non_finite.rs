use fast_vector_similarity::{
    approximate_distance_correlation, compute_bootstrapped_similarity_stats,
    compute_vector_similarity_stats, exact_hoeffdings_d_func, jensen_shannon_dependency_measure,
    kendall_tau, normalized_mutual_information, spearman_rho,
};
use ndarray::Array1;
use serde_json::Value;

const MEASURES: [&str; 6] = [
    "spearman_rho",
    "kendall_tau",
    "approximate_distance_correlation",
    "jensen_shannon_dependency_measure",
    "hoeffding_d",
    "normalized_mutual_information",
];

fn scalar_results(x: &Array1<f64>, y: &Array1<f64>) -> [(&'static str, f64); 6] {
    [
        (
            "spearman_rho",
            spearman_rho(x.as_slice().unwrap(), y.as_slice().unwrap()),
        ),
        (
            "kendall_tau",
            kendall_tau(x.as_slice().unwrap(), y.as_slice().unwrap()),
        ),
        (
            "approximate_distance_correlation",
            approximate_distance_correlation(x, y),
        ),
        (
            "jensen_shannon_dependency_measure",
            jensen_shannon_dependency_measure(x, y),
        ),
        ("hoeffding_d", exact_hoeffdings_d_func(x, y)),
        (
            "normalized_mutual_information",
            normalized_mutual_information(x, y),
        ),
    ]
}

fn assert_null_measure(result: &Value, measure: &str) {
    assert_eq!(
        result.get(measure),
        Some(&Value::Null),
        "{measure}: {result}"
    );
}

#[test]
fn issue_2_nan_reproducer_returns_nan() {
    let x = Array1::from(vec![1.0, f64::NAN, 2.0]);
    let y = Array1::from(vec![1.0, 2.0, 3.0]);
    assert!(normalized_mutual_information(&x, &y).is_nan());
}

#[test]
fn scalar_metrics_reject_every_non_finite_class_in_either_vector() {
    let finite = Array1::from(vec![-3.0, -1.0, 0.0, 2.0, 4.0, 7.0]);
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for position in 0..finite.len() {
            let mut contaminated = finite.clone();
            contaminated[position] = invalid;
            for (x, y) in [(&contaminated, &finite), (&finite, &contaminated)] {
                for (measure, result) in scalar_results(x, y) {
                    assert!(
                        result.is_nan(),
                        "{measure} accepted {invalid} at index {position}: {result}"
                    );
                }
            }
        }
    }
}

#[test]
fn empty_and_entirely_non_finite_vectors_are_undefined() {
    for values in [
        vec![],
        vec![f64::NAN; 6],
        vec![f64::INFINITY; 6],
        vec![f64::NEG_INFINITY; 6],
    ] {
        let x = Array1::from(values);
        for (measure, result) in scalar_results(&x, &x) {
            assert!(result.is_nan(), "{measure}: {result}");
        }
        let stats = compute_vector_similarity_stats(&x, &x, None);
        assert_eq!(stats["input_vector_dimensions"], x.len());
        for measure in MEASURES {
            assert_null_measure(&stats, measure);
        }
    }
}

#[test]
fn distance_correlation_checks_input_before_random_subsampling() {
    let finite = Array1::from_iter((0..2001).map(|i| i as f64));
    let mut contaminated = finite.clone();
    contaminated[2000] = f64::NAN;
    // Invalidity must not depend on whether the random 1000-element subset
    // happens to include the contaminated observation.
    for _ in 0..16 {
        assert!(approximate_distance_correlation(&contaminated, &finite).is_nan());
        assert!(approximate_distance_correlation(&finite, &contaminated).is_nan());
    }
}

#[test]
fn nmi_empty_singleton_and_constant_inputs_remain_undefined() {
    for values in [vec![], vec![1.0], vec![2.0; 6]] {
        let x = Array1::from(values);
        assert!(normalized_mutual_information(&x, &x).is_nan());
    }
}

#[test]
fn nmi_preserves_finite_dependence_and_independence_results() {
    let ascending = Array1::from(vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
    let descending = Array1::from(vec![3.0, 2.0, 1.0, 0.0, -1.0, -2.0]);
    for other in [&ascending, &descending] {
        assert!((normalized_mutual_information(&ascending, other) - 1.0).abs() < 1e-12);
    }

    let x = Array1::from(vec![0.0, 0.0, 1.0, 1.0]);
    let independent = Array1::from(vec![0.0, 1.0, 0.0, 1.0]);
    assert!(normalized_mutual_information(&x, &independent).abs() < 1e-12);
    assert!((normalized_mutual_information(&x, &x) - 1.0).abs() < 1e-12);
    assert!(normalized_mutual_information(&Array1::zeros(4), &x).abs() < 1e-12);

    let signed_zeros = Array1::from(vec![-0.0, 0.0, 1.0, 1.0]);
    assert!((normalized_mutual_information(&signed_zeros, &x) - 1.0).abs() < 1e-12);
}

#[test]
fn ordinary_finite_input_still_produces_finite_scalar_scores() {
    let x = Array1::from(vec![-3.0, -1.0, 0.0, 2.0, 4.0, 7.0]);
    for (measure, result) in scalar_results(&x, &x) {
        assert!(result.is_finite(), "{measure}: {result}");
    }
    assert!((spearman_rho(x.as_slice().unwrap(), x.as_slice().unwrap()) - 1.0).abs() < 1e-12);
    assert!((kendall_tau(x.as_slice().unwrap(), x.as_slice().unwrap()) - 1.0).abs() < 1e-12);
}

#[test]
fn json_stats_preserve_requested_keys_and_metadata_for_invalid_input() {
    let finite = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut contaminated = finite.clone();
        contaminated[2] = invalid;
        for (x, y) in [(&contaminated, &finite), (&finite, &contaminated)] {
            for selection in [None, Some("all")] {
                let result = compute_vector_similarity_stats(x, y, selection);
                assert_eq!(result["input_vector_dimensions"], finite.len());
                assert_eq!(result.as_object().unwrap().len(), 7);
                for measure in MEASURES {
                    assert_null_measure(&result, measure);
                }
            }
            for measure in MEASURES {
                let result = compute_vector_similarity_stats(x, y, Some(measure));
                assert_eq!(result["input_vector_dimensions"], finite.len());
                assert_eq!(result.as_object().unwrap().len(), 2);
                assert_null_measure(&result, measure);
            }
        }
    }
}

#[test]
fn bootstrap_validates_original_vectors_before_sampling() {
    let finite = Array1::from_iter((0..128).map(|i| i as f64));
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut contaminated = finite.clone();
        contaminated[127] = invalid;
        for (x, y) in [(&contaminated, &finite), (&finite, &contaminated)] {
            for selection in [None, Some("all"), Some("spearman_rho")] {
                let result = compute_bootstrapped_similarity_stats(x, y, 2, 8, selection);
                assert_eq!(result["sample_size"], 2);
                assert_eq!(result["number_of_bootstraps"], 8);
                let selected: &[&str] = if selection == Some("spearman_rho") {
                    &MEASURES[..1]
                } else {
                    &MEASURES
                };
                assert_eq!(result.as_object().unwrap().len(), 2 + 2 * selected.len());
                for &measure in selected {
                    assert_null_measure(&result, measure);
                    assert_null_measure(&result, &format!("stdev_{measure}"));
                }
            }
        }
    }
}

#[test]
fn bootstrap_propagates_undefined_finite_samples_without_panicking() {
    let constants = Array1::from(vec![2.0; 6]);
    let result = compute_bootstrapped_similarity_stats(
        &constants,
        &constants,
        6,
        8,
        Some("normalized_mutual_information"),
    );
    assert_null_measure(&result, "normalized_mutual_information");
    assert_null_measure(&result, "stdev_normalized_mutual_information");

    let finite = Array1::from(vec![1.0, 2.0, 3.0]);
    let result = compute_bootstrapped_similarity_stats(&finite, &finite, 1, 8, Some("all"));
    for measure in [
        "spearman_rho",
        "kendall_tau",
        "hoeffding_d",
        "normalized_mutual_information",
    ] {
        assert_null_measure(&result, measure);
        assert_null_measure(&result, &format!("stdev_{measure}"));
    }
    for measure in [
        "approximate_distance_correlation",
        "jensen_shannon_dependency_measure",
    ] {
        assert_eq!(result[measure].as_f64(), Some(0.0));
        assert_eq!(result[format!("stdev_{measure}")].as_f64(), Some(0.0));
    }
}

#[test]
fn empty_bootstrap_inputs_and_aggregates_return_null() {
    let finite = Array1::from(vec![1.0, 2.0, 3.0]);
    let empty = Array1::<f64>::zeros(0);
    for (x, sample_size, count) in [
        (&empty, 4, 8),
        (&finite, 0, 8),
        (&finite, 3, 0),
        (&finite, 3, 1),
    ] {
        let result = compute_bootstrapped_similarity_stats(x, x, sample_size, count, Some("all"));
        for measure in MEASURES {
            assert_null_measure(&result, measure);
            assert_null_measure(&result, &format!("stdev_{measure}"));
        }
    }
}

#[test]
fn finite_bootstrap_results_remain_numeric() {
    let x = Array1::from(vec![-3.0, -1.0, 0.0, 2.0, 4.0, 7.0]);
    let result = compute_bootstrapped_similarity_stats(&x, &x, 8, 8, Some("spearman_rho"));
    assert_eq!(result["spearman_rho"].as_f64(), Some(1.0));
    assert_eq!(result["stdev_spearman_rho"].as_f64(), Some(0.0));
    assert_eq!(result.as_object().unwrap().len(), 4);
}
