import unittest

import numpy as np

import hnswlib


class ProfileBindingsTestCase(unittest.TestCase):
    def test_profiled_query_matches_knn_query(self):
        dim = 16
        num_elements = 256
        num_queries = 8
        k = 5

        rng = np.random.default_rng(47)
        data = rng.random((num_elements, dim), dtype=np.float32)
        queries = rng.random((num_queries, dim), dtype=np.float32)

        index = hnswlib.Index(space='ip', dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=100, M=16)
        index.set_ef(32)
        index.add_items(data)

        labels, distances = index.knn_query(queries, k=k, num_threads=2)
        profiled_labels, profiled_distances, profile = index.knn_query_profiled(queries, k=k, num_threads=2)

        np.testing.assert_array_equal(labels, profiled_labels)
        np.testing.assert_allclose(distances, profiled_distances)

        self.assertIn('total_distance_computations', profile)
        self.assertIn('total_base_layer_expansions', profile)
        self.assertIn('candidate_count_histogram', profile)
        self.assertIn('contiguous_run_length_histogram', profile)
        self.assertIn('per_query_distance_computations', profile)
        self.assertIn('per_query_base_layer_expansions', profile)

        self.assertEqual(len(profile['per_query_distance_computations']), num_queries)
        self.assertEqual(len(profile['per_query_base_layer_expansions']), num_queries)
        self.assertGreater(profile['total_distance_computations'], 0)
        self.assertGreater(profile['total_base_layer_expansions'], 0)

        self.assertEqual(
            sum(profile['per_query_distance_computations']),
            profile['total_distance_computations'])
        self.assertEqual(
            sum(profile['per_query_base_layer_expansions']),
            profile['total_base_layer_expansions'])
        self.assertEqual(
            sum(profile['candidate_count_histogram']),
            profile['total_base_layer_expansions'])
        self.assertGreater(sum(profile['contiguous_run_length_histogram']), 0)

    def test_bf16_rowmajor_batch_distance_ip(self):
        dim = 32
        num_elements = 128

        rng = np.random.default_rng(47)
        data = rng.random((num_elements, dim), dtype=np.float32)
        data /= np.linalg.norm(data, axis=1, keepdims=True)

        index = hnswlib.Index(space='ip', dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=100, M=16)
        index.set_ef(num_elements)
        index.add_items(data)

        baseline_labels, baseline_distances = index.knn_query(data, k=1, num_threads=1)
        self.assertTrue(np.array_equal(baseline_labels.reshape(-1), np.arange(num_elements)))

        index.set_bf16_rowmajor_batch_distance(True)
        self.assertTrue(index.get_bf16_rowmajor_batch_distance())

        labels, distances = index.knn_query(data, k=1, num_threads=1)
        np.testing.assert_array_equal(labels, baseline_labels)
        np.testing.assert_allclose(distances, baseline_distances, atol=0.2)


if __name__ == '__main__':
    unittest.main()