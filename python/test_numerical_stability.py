"""Finite-range regressions through the Python extension's JSON boundary.

Run with: python -m unittest discover -s python -p test_numerical_stability.py
"""

import json
import math
import sys
import unittest

import fast_vector_similarity as fvs


NMI = "normalized_mutual_information"
DISTANCE = "approximate_distance_correlation"
JSD = "jensen_shannon_dependency_measure"


def adjacent_values(origin, count=5):
    values = [origin]
    for _ in range(count - 1):
        values.append(math.nextafter(values[-1], math.inf))
    return values


def extreme_fixtures():
    maximum = sys.float_info.max
    return [
        ("ordinary", [-2.0, -1.0, 0.0, 1.0, 2.0]),
        ("tiny", [-2e-300, -1e-300, 0.0, 1e-300, 2e-300]),
        ("subnormal", adjacent_values(0.0)),
        ("opposite finite extremes", [-maximum, -maximum / 2.0, 0.0, maximum / 2.0, maximum]),
        ("adjacent large values", adjacent_values(1e300)),
    ]


class NumericalStabilityTests(unittest.TestCase):
    def assert_unit_score(self, actual):
        self.assertIsNotNone(actual)
        self.assertTrue(math.isfinite(actual))
        self.assertGreaterEqual(actual, 0.0)
        self.assertLessEqual(actual, 1.0)

    def assert_close(self, actual, expected):
        self.assert_unit_score(actual)
        self.assertAlmostEqual(actual, expected, delta=1e-12)

    def statistics(self, x, y, selection):
        params = {"vector_1": x, "vector_2": y}
        if selection is not None:
            params["similarity_measure"] = selection
        # Use actual stdlib serialization, including its shortest decimal
        # representations of subnormals and adjacent representable floats.
        result = json.loads(
            fvs.py_compute_vector_similarity_stats(json.dumps(params, allow_nan=False))
        )
        self.assertEqual(result["input_vector_dimensions"], len(x))
        self.assertEqual(len(result), 7 if selection in (None, "all") else 2)
        if selection in (None, "all"):
            # The shuffled JSD baseline is random; only its finite range is a
            # deterministic expectation for these nonempty finite inputs.
            self.assert_unit_score(result[JSD])
        return result

    def test_self_and_reversed_scores_survive_extreme_finite_ranges(self):
        for name, x in extreme_fixtures():
            for direction, y in [("self", x), ("reversed", list(reversed(x)))]:
                for selection in (None, "all", NMI, DISTANCE):
                    with self.subTest(fixture=name, direction=direction, selection=selection):
                        result = self.statistics(x, y, selection)
                        for measure in (NMI, DISTANCE):
                            if selection in (None, "all", measure):
                                self.assert_close(result[measure], 1.0)
                            else:
                                self.assertNotIn(measure, result)

    def test_nonlinear_distance_matches_reference_under_independent_rescaling(self):
        # X=[-2,-1,0,1,2], Y=X^2 have centered second moments 280/625,
        # 760/625, and 1456/625. The resulting distance correlation is:
        expected = 0.5159234568589328
        x = [-2.0, -1.0, 0.0, 1.0, 2.0]
        y = [4.0, 1.0, 0.0, 1.0, 4.0]
        cases = []
        for x_scale, y_scale, x_offset, y_offset in [
            (1.0, 1.0, 0.0, 0.0),
            (-2.0, 4.0, 16.0, -8.0),
            (1e80, 1e80, 0.0, 0.0),
            (1e-100, 1e-100, 0.0, 0.0),
            (1e-300, 1e300, 0.0, 0.0),
            (-1e300, 1e-300, 0.0, 0.0),
        ]:
            name = (x_scale, y_scale, x_offset, y_offset)
            cases.append((
                name,
                [value * x_scale + x_offset for value in x],
                [value * y_scale + y_offset for value in y],
            ))
        for name, origin in [("subnormal steps", 0.0), ("adjacent large steps", 1e300)]:
            levels = adjacent_values(origin)
            cases.append((name, levels, [levels[index] for index in [4, 1, 0, 1, 4]]))
        maximum = sys.float_info.max
        cases.append((
            "opposite finite extremes",
            [-maximum, -maximum / 2.0, 0.0, maximum / 2.0, maximum],
            [maximum, maximum / 4.0, 0.0, maximum / 4.0, maximum],
        ))
        for name, scaled_x, scaled_y in cases:
            for selection in (DISTANCE, "all"):
                with self.subTest(fixture=name, selection=selection):
                    result = self.statistics(scaled_x, scaled_y, selection)
                    self.assert_close(result[DISTANCE], expected)

    def test_nontrivial_nmi_information_ratio_survives_mixed_scales(self):
        # Four equiprobable X categories determine two equiprobable Y
        # categories, giving average-entropy normalized information of 2/3.
        x = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        cases = []
        for x_scale, y_scale, x_offset, y_offset in [
            (1.0, 1.0, 0.0, 0.0),
            (1.0, -2.0, 16.0, 8.0),
            (1e-300, 1e300, 0.0, 0.0),
            (1e300, -1e-300, 0.0, 0.0),
        ]:
            name = (x_scale, y_scale, x_offset, y_offset)
            cases.append((
                name,
                [value * x_scale + x_offset for value in x],
                [value * y_scale + y_offset for value in y],
            ))
        for name, origin in [("subnormal categories", 0.0), ("adjacent large categories", 1e300)]:
            levels = adjacent_values(origin, 4)
            cases.append((
                name,
                [levels[int(value)] for value in x],
                [levels[int(value)] for value in y],
            ))
        for name, scaled_x, scaled_y in cases:
            for selection in (NMI, "all"):
                with self.subTest(fixture=name, selection=selection):
                    result = self.statistics(scaled_x, scaled_y, selection)
                    self.assert_close(result[NMI], 2.0 / 3.0)

    def test_independent_and_constant_cases_preserve_their_meaning_at_every_scale(self):
        large = adjacent_values(1e300, 2)
        for low, high in [
            (0.0, 1.0),
            (0.0, 1e-300),
            (0.0, math.nextafter(0.0, math.inf)),
            (-sys.float_info.max, sys.float_info.max),
            tuple(large),
        ]:
            x = [low, low, high, high] * 2
            independent = [low, high, low, high] * 2
            constant = [low] * len(x)
            for case, a, b in [
                ("independent", x, independent),
                ("constant x", constant, x),
                ("constant y", x, constant),
            ]:
                for measure in (NMI, DISTANCE):
                    with self.subTest(low=low, high=high, case=case, measure=measure):
                        result = self.statistics(a, b, measure)
                        self.assert_close(result[measure], 0.0)
            with self.subTest(low=low, high=high, case="both constant"):
                result = self.statistics(constant, constant, "all")
                self.assertIsNone(result[NMI])
                self.assert_close(result[DISTANCE], 0.0)

    def test_bootstrap_extreme_scores_are_finite_or_correctly_undefined(self):
        for name, x in extreme_fixtures():
            for selection in (NMI, DISTANCE, "all"):
                with self.subTest(fixture=name, selection=selection):
                    params = {
                        "x": x,
                        "y": x,
                        "sample_size": 8,
                        "number_of_bootstraps": 8,
                        "similarity_measure": selection,
                    }
                    result = json.loads(
                        fvs.py_compute_bootstrapped_similarity_stats(
                            json.dumps(params, allow_nan=False)
                        )
                    )
                    self.assertEqual(result["sample_size"], 8)
                    self.assertEqual(result["number_of_bootstraps"], 8)
                    self.assertEqual(len(result), 14 if selection == "all" else 4)
                    for measure in (NMI, DISTANCE, JSD):
                        if selection not in ("all", measure):
                            self.assertNotIn(measure, result)
                            continue
                        self.assertIn(measure, result)
                        self.assertIn("stdev_" + measure, result)
                        mean = result[measure]
                        stdev = result["stdev_" + measure]
                        if measure == NMI and mean is None:
                            # A sample drawn with replacement can be constant;
                            # its undefined NMI invalidates the full summary.
                            self.assertIsNone(stdev)
                            continue
                        self.assert_unit_score(mean)
                        self.assertIsNotNone(stdev)
                        self.assertTrue(math.isfinite(stdev))
                        self.assertGreaterEqual(stdev, 0.0)
                        self.assertLessEqual(stdev, 1.0)


if __name__ == "__main__":
    unittest.main()
