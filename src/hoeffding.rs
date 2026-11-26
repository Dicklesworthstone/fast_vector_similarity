//! Joint midranks for Hoeffding's D, using an O(n log n) dominance-count sweep.

use ndarray::Array1;
use rayon::prelude::*;

struct Fenwick {
    counts: Vec<usize>,
}

impl Fenwick {
    fn new(max_rank: usize) -> Self {
        Self {
            counts: vec![0; max_rank + 1],
        }
    }

    /// Number of inserted observations whose integer rank is at most `rank`.
    fn prefix(&self, mut rank: usize) -> usize {
        let mut count = 0;
        while rank != 0 {
            count += self.counts[rank];
            rank &= rank - 1;
        }
        count
    }

    fn insert(&mut self, mut rank: usize, count: usize) {
        debug_assert!(rank > 0 && rank < self.counts.len());
        while rank < self.counts.len() {
            self.counts[rank] += count;
            rank += rank & rank.wrapping_neg();
        }
    }
}

/// Computes joint ranks from the average ranks of two equally sized vectors.
///
/// Each input must contain the finite average ranks of n observations, in their
/// original observation order. Average ranks lie in [1, n], and distinct rank
/// groups differ by at least one. Truncating a rank to usize therefore preserves
/// its strict order and tie groups, without an additional compression sort.
pub(crate) fn joint_ranks(r: &Array1<f64>, s: &Array1<f64>) -> Array1<f64> {
    assert_eq!(r.len(), s.len());
    let n = r.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.par_sort_unstable_by(|&left, &right| {
        r[left]
            .total_cmp(&r[right])
            .then_with(|| s[left].total_cmp(&s[right]))
    });

    let mut previous_x = Fenwick::new(n);
    let mut q = Array1::zeros(n);
    let mut group_start = 0;
    while group_start < n {
        let x_rank = r[order[group_start]];
        let mut group_end = group_start + 1;
        while group_end < n && r[order[group_end]] == x_rank {
            group_end += 1;
        }

        let mut joint_start = group_start;
        while joint_start < group_end {
            let y_rank = s[order[joint_start]];
            let mut joint_end = joint_start + 1;
            while joint_end < group_end && s[order[joint_end]] == y_rank {
                joint_end += 1;
            }

            let y_index = y_rank as usize;
            debug_assert!(y_rank.is_finite() && y_index > 0 && y_index <= n);
            let less_both = previous_x.prefix(y_index - 1);
            let lower_x_equal_y = previous_x.prefix(y_index) - less_both;
            let equal_x_lower_y = joint_start - group_start;
            let equal_both_except_self = joint_end - joint_start - 1;

            // The tree contains only strictly smaller x ranks. Equal-x and
            // equal-(x,y) contributions come from this sorted group itself.
            let joint_rank = 1.0
                + less_both as f64
                + 0.25 * equal_both_except_self as f64
                + 0.5 * (equal_x_lower_y + lower_x_equal_y) as f64;
            for &index in &order[joint_start..joint_end] {
                q[index] = joint_rank;
            }
            joint_start = joint_end;
        }

        // Delay every insertion until the entire equal-x group has been
        // queried. Insert duplicates together to avoid redundant tree updates.
        let mut joint_start = group_start;
        while joint_start < group_end {
            let y_rank = s[order[joint_start]];
            let mut joint_end = joint_start + 1;
            while joint_end < group_end && s[order[joint_end]] == y_rank {
                joint_end += 1;
            }
            previous_x.insert(y_rank as usize, joint_end - joint_start);
            joint_start = joint_end;
        }
        group_start = group_end;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::joint_ranks;
    use ndarray::Array1;

    fn quadratic_average_ranks(values: &[f64]) -> Array1<f64> {
        values
            .iter()
            .map(|&value| {
                let lower = values.iter().filter(|&&other| other < value).count();
                let equal = values.iter().filter(|&&other| other == value).count();
                lower as f64 + (equal + 1) as f64 / 2.0
            })
            .collect()
    }

    fn quadratic_joint_ranks(r: &Array1<f64>, s: &Array1<f64>) -> Array1<f64> {
        (0..r.len())
            .map(|i| {
                // Each earlier coordinate contributes weight 1, each equal
                // coordinate weight 1/2. Exclude the observation itself and
                // multiply the two weights, independently of the sweep logic.
                let mut quarter_units = 4_usize;
                for j in 0..r.len() {
                    if j == i {
                        continue;
                    }
                    let x_weight = if r[j] < r[i] {
                        2
                    } else if r[j] == r[i] {
                        1
                    } else {
                        0
                    };
                    let y_weight = if s[j] < s[i] {
                        2
                    } else if s[j] == s[i] {
                        1
                    } else {
                        0
                    };
                    quarter_units += x_weight * y_weight;
                }
                quarter_units as f64 / 4.0
            })
            .collect()
    }

    #[test]
    fn joint_ranks_match_quadratic_weights_for_exhaustive_binary_pairs() {
        for n in 0..=5 {
            for x_bits in 0..(1_usize << n) {
                let x: Vec<f64> = (0..n).map(|i| ((x_bits >> i) & 1) as f64).collect();
                let r = quadratic_average_ranks(&x);
                for y_bits in 0..(1_usize << n) {
                    let y: Vec<f64> = (0..n).map(|i| ((y_bits >> i) & 1) as f64).collect();
                    let s = quadratic_average_ranks(&y);
                    assert_eq!(
                        joint_ranks(&r, &s),
                        quadratic_joint_ranks(&r, &s),
                        "x={x:?}, y={y:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn joint_ranks_batch_equal_x_groups_and_weight_joint_duplicates() {
        let r = quadratic_average_ranks(&[1.0, 1.0, 1.0, 2.0, 2.0, 3.0]);
        let s = quadratic_average_ranks(&[1.0, 2.0, 2.0, 1.0, 2.0, 2.0]);
        let expected = Array1::from(vec![1.0, 1.75, 1.75, 1.5, 3.5, 4.5]);
        assert_eq!(quadratic_joint_ranks(&r, &s), expected);
        assert_eq!(joint_ranks(&r, &s), expected);
    }
}
