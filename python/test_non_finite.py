"""Regression coverage for the JSON boundaries of the built Python extension.

Run with: python -m unittest discover -s python -p test_non_finite.py
"""

import json
import unittest

import fast_vector_similarity as fvs


class NonFiniteInputTests(unittest.TestCase):
    def cases(self):
        return [
            (
                fvs.py_compute_vector_similarity_stats,
                {"vector_1": [1.0, 2.0, 3.0], "vector_2": [1.0, 2.0, 3.0]},
                ("vector_1", "vector_2"),
            ),
            (
                fvs.py_compute_bootstrapped_similarity_stats,
                {
                    "x": [1.0, 2.0, 3.0],
                    "y": [1.0, 2.0, 3.0],
                    "sample_size": 3,
                    "number_of_bootstraps": 8,
                },
                ("x", "y"),
            ),
        ]

    def test_non_finite_json_tokens_raise_value_error(self):
        for function, base, names in self.cases():
            for name in names:
                for invalid in [float("nan"), float("inf"), float("-inf")]:
                    with self.subTest(function=function.__name__, vector=name, value=invalid):
                        params = dict(base)
                        params[name] = [1.0, invalid, 3.0]
                        with self.assertRaises(ValueError):
                            function(json.dumps(params))

    def test_null_and_non_numeric_vector_members_raise_value_error(self):
        for function, base, names in self.cases():
            for name in names:
                for invalid in [None, "NaN", "Infinity", True, {}, []]:
                    with self.subTest(function=function.__name__, vector=name, value=invalid):
                        params = dict(base)
                        params[name] = [1.0, invalid, 3.0]
                        with self.assertRaisesRegex(ValueError, name):
                            function(json.dumps(params))

    def test_out_of_range_json_numbers_raise_value_error(self):
        for function, base, names in self.cases():
            for name in names:
                for invalid in ["1e400", "-1e400"]:
                    with self.subTest(function=function.__name__, vector=name, value=invalid):
                        params = dict(base)
                        params[name] = [1.0, "overflow", 3.0]
                        payload = json.dumps(params).replace('"overflow"', invalid)
                        with self.assertRaises(ValueError):
                            function(payload)

    def test_missing_and_non_array_vectors_raise_value_error(self):
        for function, base, names in self.cases():
            for name in names:
                params = dict(base)
                del params[name]
                with self.subTest(function=function.__name__, vector=name, missing=True):
                    with self.assertRaisesRegex(ValueError, name):
                        function(json.dumps(params))
                for invalid in [None, 1, "NaN", {}]:
                    with self.subTest(function=function.__name__, vector=name, value=invalid):
                        params = dict(base)
                        params[name] = invalid
                        with self.assertRaisesRegex(ValueError, name):
                            function(json.dumps(params))

    def test_finite_input_returns_normal_json(self):
        for function, base, _ in self.cases():
            with self.subTest(function=function.__name__):
                params = dict(base, similarity_measure="spearman_rho")
                result = json.loads(function(json.dumps(params)))
                if result["spearman_rho"] is None:
                    # A bootstrap replicate can be constant even when the
                    # source is not. Such replicates must not be discarded.
                    self.assertEqual(function.__name__, "py_compute_bootstrapped_similarity_stats")
                    self.assertIsNone(result["stdev_spearman_rho"])
                else:
                    self.assertAlmostEqual(result["spearman_rho"], 1.0)

    def test_mismatched_vector_lengths_raise_value_error(self):
        for function, base, names in self.cases():
            with self.subTest(function=function.__name__):
                params = dict(base)
                params[names[0]] = [1.0]
                with self.assertRaises(ValueError):
                    function(json.dumps(params))

    def test_invalid_bootstrap_counts_raise_value_error(self):
        function, base, _ = self.cases()[1]
        for name in ["sample_size", "number_of_bootstraps"]:
            params = dict(base)
            del params[name]
            with self.subTest(parameter=name, missing=True):
                with self.assertRaisesRegex(ValueError, name):
                    function(json.dumps(params))
            for invalid in [None, 0, -1, 1.5, "2", True, [], {}]:
                with self.subTest(parameter=name, value=invalid):
                    params = dict(base)
                    params[name] = invalid
                    with self.assertRaisesRegex(ValueError, name):
                        function(json.dumps(params))

    def test_undefined_nmi_returns_json_null(self):
        for function, base, names in self.cases():
            with self.subTest(function=function.__name__):
                params = dict(base, similarity_measure="normalized_mutual_information")
                for name in names:
                    params[name] = [2.0, 2.0, 2.0]
                result = json.loads(function(json.dumps(params)))
                self.assertIn("normalized_mutual_information", result)
                self.assertIsNone(result["normalized_mutual_information"])
                if "sample_size" in params:
                    self.assertIn("stdev_normalized_mutual_information", result)
                    self.assertIsNone(result["stdev_normalized_mutual_information"])


if __name__ == "__main__":
    unittest.main()
