//! Biased distance correlation of finite, independently normalized coordinates.
//!
//! Center distances explicitly in two passes, using O(n) working storage. Keep
//! diagonal contributions and both halves of the symmetric distance matrices.

#[derive(Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let corrected = value - self.correction;
        let next = self.sum + corrected;
        self.correction = (next - self.sum) - corrected;
        self.sum = next;
    }
}

/// The caller must validate finite inputs and normalize each coordinate to
/// [0, 1]. Constant coordinates are allowed. Input lengths must be equal.
pub(crate) fn correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    if x.is_empty() {
        return f64::NAN;
    }
    let n = x.len();
    let scale = n as f64;
    let mut mean_x = vec![0.0; n];
    let mut mean_y = vec![0.0; n];

    // Each off-diagonal distance contributes to two row sums; diagonal raw
    // distances are zero. These sums contain only nonnegative bounded terms.
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = (x[i] - x[j]).abs();
            let dy = (y[i] - y[j]).abs();
            mean_x[i] += dx;
            mean_x[j] += dx;
            mean_y[i] += dy;
            mean_y[j] += dy;
        }
    }
    for mean in &mut mean_x {
        *mean /= scale;
    }
    for mean in &mut mean_y {
        *mean /= scale;
    }
    let grand_x = mean_x.iter().sum::<f64>() / scale;
    let grand_y = mean_y.iter().sum::<f64>() / scale;

    let mut covariance = CompensatedSum::default();
    let mut variance_x = CompensatedSum::default();
    let mut variance_y = CompensatedSum::default();
    for i in 0..n {
        // Centered diagonal entries are generally nonzero even though the raw
        // distance diagonals vanish. Omitting them changes the estimator.
        let a = grand_x - 2.0 * mean_x[i];
        let b = grand_y - 2.0 * mean_y[i];
        covariance.add(a * b);
        variance_x.add(a * a);
        variance_y.add(b * b);

        for j in (i + 1)..n {
            let a = (x[i] - x[j]).abs() - mean_x[i] - mean_x[j] + grand_x;
            let b = (y[i] - y[j]).abs() - mean_y[i] - mean_y[j] + grand_y;
            // The lower triangle has identical contributions. Compensated
            // accumulation limits cancellation in the signed cross-products.
            covariance.add(2.0 * (a * b));
            variance_x.add(2.0 * (a * a));
            variance_y.add(2.0 * (b * b));
        }
    }

    if variance_x.sum <= 0.0 || variance_y.sum <= 0.0 {
        return 0.0;
    }
    (covariance.sum / variance_x.sum.sqrt() / variance_y.sum.sqrt())
        .clamp(0.0, 1.0)
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::correlation as centered_distance_correlation;

    // Deliberately retain the original dense formulation in this small oracle:
    // compute full matrices, row/column means separately, and every centered
    // entry. No upper-triangle optimization or production summation helper.
    fn dense_reference(x: &[f64], y: &[f64]) -> f64 {
        fn centered_matrix(values: &[f64]) -> Vec<Vec<f64>> {
            let n = values.len();
            let mut matrix: Vec<Vec<f64>> = values
                .iter()
                .map(|&a| values.iter().map(|&b| (a - b).abs()).collect())
                .collect();
            let rows: Vec<f64> = matrix
                .iter()
                .map(|row| row.iter().sum::<f64>() / n as f64)
                .collect();
            let columns: Vec<f64> = (0..n)
                .map(|j| matrix.iter().map(|row| row[j]).sum::<f64>() / n as f64)
                .collect();
            let grand = rows.iter().sum::<f64>() / n as f64;
            for i in 0..n {
                for j in 0..n {
                    matrix[i][j] = matrix[i][j] - columns[j] + grand - rows[i];
                }
            }
            matrix
        }
        let a = centered_matrix(x);
        let b = centered_matrix(y);
        let (mut covariance, mut variance_x, mut variance_y) = (0.0, 0.0, 0.0);
        for i in 0..x.len() {
            for j in 0..x.len() {
                covariance += a[i][j] * b[i][j];
                variance_x += a[i][j] * a[i][j];
                variance_y += b[i][j] * b[i][j];
            }
        }
        if variance_x <= 0.0 || variance_y <= 0.0 {
            return 0.0;
        }
        (covariance / variance_x.sqrt() / variance_y.sqrt())
            .clamp(0.0, 1.0)
            .sqrt()
    }

    fn assert_matches_dense(x: &[f64], y: &[f64]) {
        let actual = centered_distance_correlation(x, y);
        let expected = dense_reference(x, y);
        assert!(actual.is_finite() && (0.0..=1.0).contains(&actual));
        // Compare squared coefficients near zero: taking a square root can
        // magnify last-bit covariance cancellation in either formulation.
        assert!(
            (actual * actual - expected * expected).abs() <= 1e-12,
            "expected {expected}, got {actual}; x={x:?}, y={y:?}"
        );
        if expected > 1e-6 {
            assert!(
                (actual - expected).abs() <= 1e-12,
                "expected {expected}, got {actual}; x={x:?}, y={y:?}"
            );
        }
    }

    #[test]
    fn centered_pairs_match_dense_oracle_for_all_small_ternary_vectors() {
        for n in 1..=4 {
            let vectors: Vec<Vec<f64>> = (0..3_usize.pow(n as u32))
                .map(|mut code| {
                    (0..n)
                        .map(|_| {
                            let value = (code % 3) as f64 / 2.0;
                            code /= 3;
                            value
                        })
                        .collect()
                })
                .collect();
            for x in &vectors {
                for y in &vectors {
                    assert_matches_dense(x, y);
                }
            }
        }
    }

    #[test]
    fn retains_diagonals_constants_independence_and_nonlinear_dependence() {
        for (x, y, expected) in [
            (&[0.0, 1.0][..], &[0.0, 1.0][..], 1.0),
            (&[0.0, 1.0][..], &[1.0, 0.0][..], 1.0),
            (&[0.0][..], &[1.0][..], 0.0),
            (&[0.0; 4][..], &[0.0, 0.0, 1.0, 1.0][..], 0.0),
            (&[0.0, 0.0, 1.0, 1.0][..], &[0.0, 1.0, 0.0, 1.0][..], 0.0),
            (
                &[0.0, 0.25, 0.5, 0.75, 1.0][..],
                &[1.0, 0.25, 0.0, 0.25, 1.0][..],
                0.5159234568589328,
            ),
        ] {
            let actual = centered_distance_correlation(x, y);
            assert!(
                (actual - expected).abs() <= 1e-12,
                "expected {expected}, got {actual}"
            );
            assert_matches_dense(x, y);
        }
        assert!(centered_distance_correlation(&[], &[]).is_nan());
    }

    #[test]
    fn varied_continuous_tied_and_unbalanced_inputs_match_dense() {
        let mut state = 0x4f9d_617c_b53a_2801_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        for case in 0..128 {
            let n = 2 + case % 31;
            let mut x: Vec<_> = (0..n).map(|_| next()).collect();
            let mut y: Vec<_> = (0..n).map(|_| next()).collect();
            if case % 3 == 0 {
                x.iter_mut()
                    .for_each(|value| *value = (*value * 4.0).floor() / 4.0);
                y.iter_mut()
                    .for_each(|value| *value = (*value * 4.0).floor() / 4.0);
            } else if case % 3 == 1 {
                x.iter_mut().for_each(|value| *value *= 1e-300);
                y.iter_mut().for_each(|value| *value *= 1e-200);
                x[0] = 1.0;
                y[n - 1] = 1.0;
            }
            assert_matches_dense(&x, &y);
            assert_matches_dense(&y, &x);
        }
    }

    #[test]
    fn largest_public_subset_keeps_self_and_reversed_scores() {
        let x: Vec<_> = (0..1000).map(|i| i as f64 / 999.0).collect();
        let reversed: Vec<_> = x.iter().rev().copied().collect();
        for y in [&x, &reversed] {
            let score = centered_distance_correlation(&x, y);
            assert!((score - 1.0).abs() <= 1e-12, "got {score}");
        }
        let mut single_outlier = vec![0.0; 1000];
        single_outlier[999] = 1.0;
        let score = centered_distance_correlation(&single_outlier, &single_outlier);
        assert!((score - 1.0).abs() <= 1e-12, "got {score}");
    }
}
