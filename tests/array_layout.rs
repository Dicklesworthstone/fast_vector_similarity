use fast_vector_similarity::{
    approximate_distance_correlation, compute_bootstrapped_similarity_stats,
    compute_vector_similarity_stats, exact_hoeffdings_d_func, jensen_shannon_dependency_measure,
    kendall_tau, normalized_mutual_information, spearman_rho,
};
use ndarray::{s, Array1};

const MEASURES: [&str; 6] = [
    "spearman_rho",
    "kendall_tau",
    "approximate_distance_correlation",
    "jensen_shannon_dependency_measure",
    "hoeffding_d",
    "normalized_mutual_information",
];

fn reversed_storage(values: &[f64]) -> Array1<f64> {
    let storage = Array1::from_iter(values.iter().rev().copied());
    let result = storage.slice_move(s![..;-1]);
    assert!(result.as_slice().is_none());
    result
}

fn strided_storage(values: &[f64]) -> Array1<f64> {
    // Padding is outside the logical array. It must neither contaminate
    // validation nor be treated as additional observations.
    let storage = Array1::from_iter(values.iter().flat_map(|&value| [value, f64::NAN]));
    let result = storage.slice_move(s![..;2]);
    assert!(result.as_slice().is_none());
    result
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    assert!(
        actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= 1e-12,
        "{context}: expected {expected}, got {actual}"
    );
}

fn array_metrics(x: &Array1<f64>, y: &Array1<f64>) -> [(&'static str, f64); 4] {
    [
        ("hoeffding_d", exact_hoeffdings_d_func(x, y)),
        (
            "normalized_mutual_information",
            normalized_mutual_information(x, y),
        ),
        (
            "approximate_distance_correlation",
            approximate_distance_correlation(x, y),
        ),
        (
            "jensen_shannon_dependency_measure",
            jensen_shannon_dependency_measure(x, y),
        ),
    ]
}

#[test]
fn all_array_metrics_and_dispatchers_use_logical_observation_order() {
    // Every observation occupies a distinct histogram bin. JSD is therefore
    // invariant to its shuffled baseline for this fixture, avoiding RNG-based
    // comparison failures while still exercising its strided input path.
    let x_values = [3.0, 0.0, 5.0, 1.0, 4.0, 2.0];
    let y_values = [5.0, 1.0, 4.0, 2.0, 0.0, 3.0];
    let dense_x = Array1::from(x_values.to_vec());
    let dense_y = Array1::from(y_values.to_vec());
    let expected_metrics = array_metrics(&dense_x, &dense_y);
    let expected_stats = compute_vector_similarity_stats(&dense_x, &dense_y, Some("all"));

    for (x, y) in [
        (reversed_storage(&x_values), dense_y.clone()),
        (dense_x.clone(), reversed_storage(&y_values)),
        (strided_storage(&x_values), dense_y.clone()),
        (dense_x.clone(), strided_storage(&y_values)),
        (reversed_storage(&x_values), strided_storage(&y_values)),
        (strided_storage(&x_values), reversed_storage(&y_values)),
    ] {
        for ((measure, actual), (expected_name, expected)) in
            array_metrics(&x, &y).into_iter().zip(expected_metrics)
        {
            assert_eq!(measure, expected_name);
            assert_close(actual, expected, measure);
        }
        for selection in [None, Some("all")] {
            let actual = compute_vector_similarity_stats(&x, &y, selection);
            assert_eq!(actual["input_vector_dimensions"], x_values.len());
            assert_eq!(actual.as_object().unwrap().len(), 7);
            for measure in MEASURES {
                assert_close(
                    actual[measure]
                        .as_f64()
                        .expect("strided result must be numeric"),
                    expected_stats[measure].as_f64().unwrap(),
                    measure,
                );
            }
        }
        for measure in MEASURES {
            let actual = compute_vector_similarity_stats(&x, &y, Some(measure));
            assert_eq!(actual["input_vector_dimensions"], x_values.len());
            assert_eq!(actual.as_object().unwrap().len(), 2);
            assert_close(
                actual[measure]
                    .as_f64()
                    .expect("selected strided result must be numeric"),
                expected_stats[measure].as_f64().unwrap(),
                measure,
            );
        }
    }
}

#[test]
fn bootstrap_accepts_strided_inputs_with_non_finite_unused_padding() {
    let values = [3.0, 0.0, 5.0, 1.0, 4.0, 2.0];
    let x = strided_storage(&values);
    let y = reversed_storage(&values);
    let result = compute_bootstrapped_similarity_stats(
        &x,
        &y,
        8,
        8,
        Some("approximate_distance_correlation"),
    );
    assert_eq!(result["sample_size"], 8);
    assert_eq!(result["number_of_bootstraps"], 8);
    let score = result["approximate_distance_correlation"]
        .as_f64()
        .expect("unused non-finite padding must not invalidate the bootstrap");
    let stdev = result["stdev_approximate_distance_correlation"]
        .as_f64()
        .unwrap();
    // A constant sample has distance correlation zero; every other self-sample
    // has correlation one. Both possibilities are valid random draws.
    assert!((0.0..=1.0 + 1e-12).contains(&score));
    assert!((0.0..=0.5 + 1e-12).contains(&stdev));
}

#[test]
fn every_scalar_and_dispatcher_rejects_unequal_lengths() {
    type Scalar = fn(&Array1<f64>, &Array1<f64>) -> f64;
    let scalars: [(&str, Scalar); 6] = [
        ("spearman_rho", |x, y| {
            spearman_rho(x.as_slice().unwrap(), y.as_slice().unwrap())
        }),
        ("kendall_tau", |x, y| {
            kendall_tau(x.as_slice().unwrap(), y.as_slice().unwrap())
        }),
        (
            "approximate_distance_correlation",
            approximate_distance_correlation,
        ),
        (
            "jensen_shannon_dependency_measure",
            jensen_shannon_dependency_measure,
        ),
        ("hoeffding_d", exact_hoeffdings_d_func),
        (
            "normalized_mutual_information",
            normalized_mutual_information,
        ),
    ];
    let full = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let shorter = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    let empty = Array1::<f64>::zeros(0);
    for (x, y) in [
        (&full, &shorter),
        (&shorter, &full),
        (&full, &empty),
        (&empty, &full),
    ] {
        for (name, scalar) in scalars {
            assert!(
                std::panic::catch_unwind(|| scalar(x, y)).is_err(),
                "{name} must reject unequal lengths before calculation"
            );
        }
        assert!(
            std::panic::catch_unwind(|| compute_vector_similarity_stats(x, y, None)).is_err(),
            "statistics dispatcher must reject unequal lengths"
        );
        assert!(
            std::panic::catch_unwind(|| {
                compute_bootstrapped_similarity_stats(x, y, 6, 8, None)
            })
            .is_err(),
            "bootstrap dispatcher must reject unequal lengths"
        );
    }
}
