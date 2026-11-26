"""Python boundary regressions for tie-preserving Hoeffding statistics.

Run with: python -m unittest discover -s python -p test_hoeffding.py
"""

import json
import math
import unittest

import fast_vector_similarity as fvs


MEASURE = "hoeffding_d"
STDEV = "stdev_hoeffding_d"


def pairwise_hoeffding_reference(x, y):
    """Small O(n^2) oracle using raw pair comparisons, with no sorting/tree."""
    n = len(x)
    if n < 5:
        return math.nan
    ranks = []
    for i in range(n):
        r = s = q = 1.0
        for j in range(n):
            if i == j:
                continue
            wx = 1.0 if x[j] < x[i] else 0.5 if x[j] == x[i] else 0.0
            wy = 1.0 if y[j] < y[i] else 0.5 if y[j] == y[i] else 0.0
            r += wx
            s += wy
            q += wx * wy
        ranks.append((r, s, q))
    d1 = sum((q - 1.0) * (q - 2.0) for _, _, q in ranks)
    d2 = sum((r - 1.0) * (r - 2.0) * (s - 1.0) * (s - 2.0) for r, s, _ in ranks)
    d3 = sum((r - 2.0) * (s - 2.0) * (q - 1.0) for r, s, q in ranks)
    numerator = 30.0 * ((n - 2) * (n - 3) * d1 + d2 - 2.0 * (n - 2) * d3)
    denominator = n * (n - 1) * (n - 2) * (n - 3) * (n - 4)
    return numerator / denominator


class HoeffdingTests(unittest.TestCase):
    def statistics(self, x, y, selection):
        params = {"vector_1": x, "vector_2": y}
        if selection is not None:
            params["similarity_measure"] = selection
        result = json.loads(fvs.py_compute_vector_similarity_stats(json.dumps(params)))
        self.assertEqual(result["input_vector_dimensions"], len(x))
        self.assertEqual(len(result), 2 if selection == MEASURE else 7)
        self.assertIn(MEASURE, result)
        return result

    def bootstrap(self, x, y, size, selection):
        params = {"x": x, "y": y, "sample_size": size, "number_of_bootstraps": 8}
        if selection is not None:
            params["similarity_measure"] = selection
        result = json.loads(fvs.py_compute_bootstrapped_similarity_stats(json.dumps(params)))
        self.assertEqual(result["sample_size"], size)
        self.assertEqual(result["number_of_bootstraps"], 8)
        self.assertEqual(len(result), 4 if selection == MEASURE else 14)
        self.assertIn(MEASURE, result)
        self.assertIn(STDEV, result)
        return result

    def test_selected_default_and_all_match_independent_pairwise_reference(self):
        cases = [
            ("minimum untied sample", [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]),
            ("reverse", list(range(8)), list(reversed(range(8)))),
            ("mixed untied order", list(range(8)), [3, 0, 7, 2, 5, 1, 6, 4]),
            (
                "ties and signed zero",
                [-0.0, 0.0, 1, 1, 1, 2, 2, 3],
                [1, 1, -0.0, 0.0, 2, 2, 2, 3],
            ),
            ("both constant", [2] * 8, [3] * 8),
        ]
        for name, x, y in cases:
            expected = pairwise_hoeffding_reference(x, y)
            for selection in (None, "all", MEASURE):
                with self.subTest(fixture=name, selection=selection):
                    result = self.statistics(x, y, selection)
                    self.assertIsNotNone(result[MEASURE])
                    self.assertTrue(math.isfinite(result[MEASURE]))
                    self.assertAlmostEqual(result[MEASURE], expected, delta=1e-12)

    def test_samples_below_five_are_null_in_both_wrappers(self):
        for size in range(5):
            x = list(range(size))
            y = list(reversed(x))
            for selection in (None, "all", MEASURE):
                with self.subTest(wrapper="exact", size=size, selection=selection):
                    self.assertIsNone(self.statistics(x, y, selection)[MEASURE])
        x = list(range(8))
        y = [3, 0, 7, 2, 5, 1, 6, 4]
        for size in range(1, 5):
            for selection in (None, "all", MEASURE):
                with self.subTest(wrapper="bootstrap", size=size, selection=selection):
                    result = self.bootstrap(x, y, size, selection)
                    self.assertIsNone(result[MEASURE])
                    self.assertIsNone(result[STDEV])

    def test_supported_bootstrap_samples_have_finite_summaries(self):
        cases = [
            ("untied source", list(range(8)), [3, 0, 7, 2, 5, 1, 6, 4]),
            ("tied source", [0, 0, 1, 1, 2, 2, 3, 3], [3, 1, 3, 1, 2, 0, 2, 0]),
            ("constant source", [2] * 8, [3] * 8),
        ]
        for name, x, y in cases:
            for size in (5, 8):
                for selection in (MEASURE, "all"):
                    with self.subTest(fixture=name, size=size, selection=selection):
                        result = self.bootstrap(x, y, size, selection)
                        self.assertIsNotNone(result[MEASURE])
                        self.assertIsNotNone(result[STDEV])
                        self.assertTrue(math.isfinite(result[MEASURE]))
                        self.assertTrue(math.isfinite(result[STDEV]))
                        self.assertGreaterEqual(result[STDEV], 0.0)
                        # Do not require a random sample's exact score, or apply
                        # untied-data bounds to the preserved finite tie formula.


if __name__ == "__main__":
    unittest.main()
