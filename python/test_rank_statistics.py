"""Regression tests for tied and constant rank statistics in the Python API.

Run with: python -m unittest discover -s python -p test_rank_statistics.py
"""

import json
import math
import unittest

import fast_vector_similarity as fvs


RANK_MEASURES = ("spearman_rho", "kendall_tau")


class RankStatisticsTests(unittest.TestCase):
    def assert_rank_score(self, actual, expected):
        self.assertIsNotNone(actual)
        self.assertTrue(math.isfinite(actual))
        self.assertGreaterEqual(actual, -1.0)
        self.assertLessEqual(actual, 1.0)
        self.assertAlmostEqual(actual, expected, delta=1e-12)

    def test_tied_scores_for_all_and_selected_measures(self):
        # The final tau value is the worked example in SciPy's official docs:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
        cases = [
            ([1, 1, 2, 2], [2, 2, 1, 1], -1.0, -1.0),
            ([1, 1, 2], [1, 2, 3], 0.8660254037844387, 0.8164965809277261),
            ([1, 2, 3], [1, 1, 2], 0.8660254037844387, 0.8164965809277261),
            ([1, 1, 2, 2], [1, 2, 1, 2], 0.0, 0.0),
            ([-0.0, 0.0, 1.0], [2, 1, 3], 0.8660254037844387, 0.8164965809277261),
            ([12, 2, 1, 12, 2], [1, 4, 7, 1, 0], -0.5407380704358752, -0.47140452079103173),
        ]
        for x, y, rho, tau in cases:
            expected = dict(zip(RANK_MEASURES, (rho, tau)))
            for selection in (None, "all", *RANK_MEASURES):
                with self.subTest(x=x, y=y, selection=selection):
                    params = {"vector_1": x, "vector_2": y}
                    if selection is not None:
                        params["similarity_measure"] = selection
                    result = json.loads(fvs.py_compute_vector_similarity_stats(json.dumps(params)))
                    self.assertEqual(result["input_vector_dimensions"], len(x))
                    self.assertEqual(len(result), 2 if selection in RANK_MEASURES else 7)
                    for measure in RANK_MEASURES:
                        if selection in (None, "all", measure):
                            self.assert_rank_score(result[measure], expected[measure])
                        else:
                            self.assertNotIn(measure, result)

    def test_joint_permutation_does_not_change_rank_scores(self):
        x = [1, 1, 2, 2, 3, 3, 3, 4]
        y = [4, 1, 4, 2, 2, 1, 3, 0]
        order = [2, 0, 7, 4, 6, 1, 5, 3]
        for measure in RANK_MEASURES:
            with self.subTest(measure=measure):
                def score(a, b):
                    params = {"vector_1": a, "vector_2": b, "similarity_measure": measure}
                    return json.loads(fvs.py_compute_vector_similarity_stats(json.dumps(params)))[measure]

                expected = score(x, y)
                self.assert_rank_score(score(y, x), expected)
                self.assert_rank_score(score([x[i] for i in order], [y[i] for i in order]), expected)
                self.assert_rank_score(score(x, [-value for value in y]), -expected)

    def test_constant_empty_and_singleton_inputs_return_null_through_both_wrappers(self):
        cases = [
            ([], []),
            ([1], [2]),
            ([2] * 6, [2] * 6),
            ([2] * 6, list(range(6))),
            (list(range(6)), [2] * 6),
            ([-0.0, 0.0, -0.0, 0.0], [0.0] * 4),
        ]
        for x, y in cases:
            entrypoints = [
                (fvs.py_compute_vector_similarity_stats, {"vector_1": x, "vector_2": y}, False),
                (
                    fvs.py_compute_bootstrapped_similarity_stats,
                    {"x": x, "y": y, "sample_size": 6, "number_of_bootstraps": 4},
                    True,
                ),
            ]
            for function, base, is_bootstrap in entrypoints:
                for selection in (None, "all", *RANK_MEASURES):
                    with self.subTest(x=x, y=y, function=function.__name__, selection=selection):
                        params = dict(base)
                        if selection is not None:
                            params["similarity_measure"] = selection
                        result = json.loads(function(json.dumps(params)))
                        if is_bootstrap:
                            self.assertEqual(result["sample_size"], 6)
                            self.assertEqual(result["number_of_bootstraps"], 4)
                        else:
                            self.assertEqual(result["input_vector_dimensions"], len(x))
                        for measure in RANK_MEASURES:
                            if selection in (None, "all", measure):
                                self.assertIn(measure, result)
                                self.assertIsNone(result[measure])
                                if is_bootstrap:
                                    self.assertIn("stdev_" + measure, result)
                                    self.assertIsNone(result["stdev_" + measure])
                            else:
                                self.assertNotIn(measure, result)
                                self.assertNotIn("stdev_" + measure, result)

    def test_tied_bootstrap_outputs_preserve_valid_numeric_or_null_summaries(self):
        # Sampling may produce a constant replicate. Check the documented null
        # propagation instead of assuming a particular random sample or score.
        for selection in ("all", *RANK_MEASURES):
            with self.subTest(selection=selection):
                params = {
                    "x": [1, 1, 2, 2, 3, 3],
                    "y": [3, 3, 2, 2, 1, 1],
                    "sample_size": 6,
                    "number_of_bootstraps": 4,
                    "similarity_measure": selection,
                }
                result = json.loads(fvs.py_compute_bootstrapped_similarity_stats(json.dumps(params)))
                self.assertEqual(result["sample_size"], 6)
                self.assertEqual(result["number_of_bootstraps"], 4)
                for measure in RANK_MEASURES:
                    if selection not in ("all", measure):
                        self.assertNotIn(measure, result)
                        continue
                    self.assertIn(measure, result)
                    self.assertIn("stdev_" + measure, result)
                    mean = result[measure]
                    stdev = result["stdev_" + measure]
                    if mean is None:
                        self.assertIsNone(stdev)
                    else:
                        self.assertTrue(math.isfinite(mean))
                        self.assertGreaterEqual(mean, -1.0)
                        self.assertLessEqual(mean, 1.0)
                        self.assertIsNotNone(stdev)
                        self.assertTrue(math.isfinite(stdev))
                        self.assertGreaterEqual(stdev, 0.0)


if __name__ == "__main__":
    unittest.main()
