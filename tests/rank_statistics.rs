use fast_vector_similarity::{
    compute_bootstrapped_similarity_stats, compute_vector_similarity_stats, kendall_tau,
    spearman_rho,
};
use ndarray::Array1;
use serde_json::Value;

type RankMetric = fn(&[f64], &[f64]) -> f64;

const RANK_METRICS: [(&str, RankMetric); 2] =
    [("spearman_rho", spearman_rho), ("kendall_tau", kendall_tau)];

// Independent quadratic oracles: no sorting, inversion counting, or production
// rank helpers. Average tied ranks and tau-b are defined in the SciPy docs:
// https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
// https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
fn oracle_midranks(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|&value| {
            let less = values.iter().filter(|&&other| other < value).count();
            let equal = values.iter().filter(|&&other| other == value).count();
            less as f64 + (equal as f64 + 1.0) / 2.0
        })
        .collect()
}

fn oracle_spearman(x: &[f64], y: &[f64]) -> f64 {
    if x.len() < 2 {
        return f64::NAN;
    }
    let rx = oracle_midranks(x);
    let ry = oracle_midranks(y);
    let mean_x = rx.iter().sum::<f64>() / rx.len() as f64;
    let mean_y = ry.iter().sum::<f64>() / ry.len() as f64;
    let covariance: f64 = rx
        .iter()
        .zip(&ry)
        .map(|(&a, &b)| (a - mean_x) * (b - mean_y))
        .sum();
    let variance_x: f64 = rx.iter().map(|&a| (a - mean_x).powi(2)).sum();
    let variance_y: f64 = ry.iter().map(|&b| (b - mean_y).powi(2)).sum();
    if variance_x == 0.0 || variance_y == 0.0 {
        f64::NAN
    } else {
        covariance / (variance_x * variance_y).sqrt()
    }
}

fn oracle_kendall(x: &[f64], y: &[f64]) -> f64 {
    let (mut concordant, mut discordant, mut ties_x_only, mut ties_y_only) = (0, 0, 0, 0);
    for i in 0..x.len() {
        for j in (i + 1)..x.len() {
            if x[i] == x[j] && y[i] == y[j] {
                continue;
            } else if x[i] == x[j] {
                ties_x_only += 1;
            } else if y[i] == y[j] {
                ties_y_only += 1;
            } else if (x[i] < x[j]) == (y[i] < y[j]) {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let non_tied = (concordant + discordant) as f64;
    let denominator = ((non_tied + ties_x_only as f64) * (non_tied + ties_y_only as f64)).sqrt();
    if denominator == 0.0 {
        f64::NAN
    } else {
        (concordant as f64 - discordant as f64) / denominator
    }
}

fn assert_score(name: &str, actual: f64, expected: f64, x: &[f64], y: &[f64]) {
    if expected.is_nan() {
        assert!(
            actual.is_nan(),
            "{name}: expected NaN, got {actual}; x={x:?}, y={y:?}"
        );
    } else {
        assert!(
            actual.is_finite() && (actual - expected).abs() < 1e-12,
            "{name}: expected {expected}, got {actual}; x={x:?}, y={y:?}"
        );
        assert!(
            (-1.0..=1.0).contains(&actual),
            "{name} out of range: {actual}"
        );
    }
}

fn ternary_vector(mut code: usize, length: usize) -> Vec<f64> {
    (0..length)
        .map(|_| {
            let value = (code % 3) as f64 - 1.0;
            code /= 3;
            value
        })
        .collect()
}

#[test]
fn rank_metrics_match_independent_oracles_for_all_small_ternary_vectors() {
    let mut cases = 0;
    for length in 1..=4 {
        let vectors: Vec<_> = (0..3_usize.pow(length as u32))
            .map(|code| ternary_vector(code, length))
            .collect();
        for x in &vectors {
            for y in &vectors {
                assert_score(
                    "spearman_rho",
                    spearman_rho(x, y),
                    oracle_spearman(x, y),
                    x,
                    y,
                );
                assert_score("kendall_tau", kendall_tau(x, y), oracle_kendall(x, y), x, y);
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 7_380);
}

#[test]
fn tied_examples_match_known_coefficients() {
    let cases: &[(&[f64], &[f64], f64, f64)] = &[
        (&[1.0, 1.0, 2.0, 2.0], &[2.0, 2.0, 1.0, 1.0], -1.0, -1.0),
        (
            &[1.0, 1.0, 2.0],
            &[1.0, 2.0, 3.0],
            0.8660254037844387,
            0.8164965809277261,
        ),
        (
            &[1.0, 2.0, 3.0],
            &[1.0, 1.0, 2.0],
            0.8660254037844387,
            0.8164965809277261,
        ),
        (&[1.0, 1.0, 2.0, 2.0], &[1.0, 2.0, 1.0, 2.0], 0.0, 0.0),
        // Published example in the SciPy kendalltau documentation.
        (
            &[12.0, 2.0, 1.0, 12.0, 2.0],
            &[1.0, 4.0, 7.0, 1.0, 0.0],
            -0.5407380704358752,
            -0.47140452079103173,
        ),
        (
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[5.0, 6.0, 7.0, 8.0, 7.0],
            0.8207826816681233,
            0.7378647873726218,
        ),
    ];
    for &(x, y, rho, tau) in cases {
        assert_score("spearman_rho", spearman_rho(x, y), rho, x, y);
        assert_score("kendall_tau", kendall_tau(x, y), tau, x, y);
    }
}

#[test]
fn signed_zeros_are_one_tie_group() {
    let x = [-0.0, 0.0, 1.0];
    let y = [2.0, 1.0, 3.0];
    for (a, b) in [(&x, &y), (&y, &x)] {
        assert_score(
            "spearman_rho",
            spearman_rho(a, b),
            oracle_spearman(a, b),
            a,
            b,
        );
        assert_score("kendall_tau", kendall_tau(a, b), oracle_kendall(a, b), a, b);
    }
    let zeros = [-0.0, 0.0, -0.0, 0.0];
    for (name, metric) in RANK_METRICS {
        assert_score(name, metric(&zeros, &zeros), f64::NAN, &zeros, &zeros);
    }
}

#[test]
fn coefficients_preserve_symmetry_sign_and_joint_permutations() {
    let x = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0];
    let y = [4.0, 1.0, 4.0, 2.0, 2.0, 1.0, 3.0, 0.0];
    let order = [2, 0, 7, 4, 6, 1, 5, 3];
    let px: Vec<_> = order.iter().map(|&i| x[i]).collect();
    let py: Vec<_> = order.iter().map(|&i| y[i]).collect();
    let increasing_x: Vec<_> = x.iter().map(|&value| 3.0 * value + 5.0).collect();
    let decreasing_y: Vec<_> = y.iter().map(|&value| -2.0 * value - 7.0).collect();
    let duplicated_x: Vec<_> = x.iter().flat_map(|&value| [value, value]).collect();
    let duplicated_y: Vec<_> = y.iter().flat_map(|&value| [value, value]).collect();
    for (name, metric, expected) in [
        (
            "spearman_rho",
            spearman_rho as RankMetric,
            oracle_spearman(&x, &y),
        ),
        (
            "kendall_tau",
            kendall_tau as RankMetric,
            oracle_kendall(&x, &y),
        ),
    ] {
        assert_score(name, metric(&x, &y), expected, &x, &y);
        assert_score(name, metric(&y, &x), expected, &y, &x);
        assert_score(name, metric(&px, &py), expected, &px, &py);
        assert_score(name, metric(&increasing_x, &y), expected, &increasing_x, &y);
        assert_score(
            name,
            metric(&x, &decreasing_y),
            -expected,
            &x,
            &decreasing_y,
        );
        assert_score(
            name,
            metric(&duplicated_x, &duplicated_y),
            expected,
            &duplicated_x,
            &duplicated_y,
        );
    }
}

#[test]
fn constant_empty_and_singleton_rank_inputs_are_undefined() {
    let cases: &[(&[f64], &[f64])] = &[
        (&[], &[]),
        (&[1.0], &[2.0]),
        (&[2.0; 6], &[2.0; 6]),
        (&[2.0; 6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        (&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2.0; 6]),
    ];
    for &(x, y) in cases {
        for (name, metric) in RANK_METRICS {
            assert_score(name, metric(x, y), f64::NAN, x, y);
        }
    }
}

#[test]
#[should_panic]
fn spearman_rejects_unequal_lengths_instead_of_truncating_pairs() {
    spearman_rho(&[1.0, 2.0, 3.0], &[1.0, 2.0]);
}

#[test]
fn large_reversed_inputs_preserve_exact_rank_direction() {
    // 4,999,950,000 discordant pairs exceed u32::MAX. This also exercises the
    // scalable path without a machine-dependent wall-clock assertion.
    let x: Vec<_> = (0..100_000).map(|index| index as f64).collect();
    let y: Vec<_> = x.iter().rev().copied().collect();
    for (name, metric) in RANK_METRICS {
        let actual = metric(&x, &y);
        assert!((actual + 1.0).abs() < 1e-12, "{name}: {actual}");
    }
}

fn selected(name: &str, selection: Option<&str>) -> bool {
    selection.is_none() || selection == Some("all") || selection == Some(name)
}

fn assert_null(stats: &Value, key: &str) {
    assert_eq!(stats.get(key), Some(&Value::Null), "{key}: {stats}");
}

#[test]
fn json_all_and_selected_statistics_use_correct_tied_rank_scores() {
    let x = Array1::from(vec![12.0, 2.0, 1.0, 12.0, 2.0]);
    let y = Array1::from(vec![1.0, 4.0, 7.0, 1.0, 0.0]);
    for selection in [None, Some("all"), Some("spearman_rho"), Some("kendall_tau")] {
        let stats = compute_vector_similarity_stats(&x, &y, selection);
        assert_eq!(stats["input_vector_dimensions"], x.len());
        let expected_keys = if selection.is_none() || selection == Some("all") {
            7
        } else {
            2
        };
        assert_eq!(stats.as_object().unwrap().len(), expected_keys);
        for (name, expected) in [
            ("spearman_rho", -0.5407380704358752),
            ("kendall_tau", -0.47140452079103173),
        ] {
            if selected(name, selection) {
                let actual = stats[name].as_f64().expect("finite tied-rank coefficient");
                assert_score(
                    name,
                    actual,
                    expected,
                    x.as_slice().unwrap(),
                    y.as_slice().unwrap(),
                );
            } else {
                assert!(stats.get(name).is_none());
            }
        }
    }
}

#[test]
fn json_and_bootstrap_rank_statistics_propagate_constant_input_as_null() {
    let constant = Array1::from(vec![2.0; 6]);
    let varying = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    for (x, y) in [
        (&constant, &constant),
        (&constant, &varying),
        (&varying, &constant),
    ] {
        for selection in [None, Some("all"), Some("spearman_rho"), Some("kendall_tau")] {
            let stats = compute_vector_similarity_stats(x, y, selection);
            let bootstrapped = compute_bootstrapped_similarity_stats(x, y, 6, 4, selection);
            assert_eq!(stats["input_vector_dimensions"], 6);
            assert_eq!(bootstrapped["sample_size"], 6);
            assert_eq!(bootstrapped["number_of_bootstraps"], 4);
            for (name, _) in RANK_METRICS {
                let stdev_key = format!("stdev_{name}");
                if selected(name, selection) {
                    assert_null(&stats, name);
                    assert_null(&bootstrapped, name);
                    assert_null(&bootstrapped, &stdev_key);
                } else {
                    assert!(stats.get(name).is_none());
                    assert!(bootstrapped.get(name).is_none());
                    assert!(bootstrapped.get(&stdev_key).is_none());
                }
            }
        }
    }
}
