#include "../../hnswlib/hnswlib.h"

#include <assert.h>

#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {

void test_profiled_search_matches_default() {
    const int dim = 16;
    const size_t num_elements = 256;
    const size_t num_queries = 16;
    const size_t k = 10;

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

    std::vector<float> data(num_elements * dim);
    std::vector<float> queries(num_queries * dim);

    for (size_t i = 0; i < data.size(); i++) {
        data[i] = distrib(rng);
    }
    for (size_t i = 0; i < queries.size(); i++) {
        queries[i] = distrib(rng);
    }

    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, num_elements * 2, 16, 200);

    for (size_t i = 0; i < num_elements; i++) {
        index.addPoint(data.data() + i * dim, i);
    }

    for (size_t qi = 0; qi < num_queries; qi++) {
        const void* query = queries.data() + qi * dim;
        auto baseline = index.searchKnn(query, k);

        hnswlib::SearchProfileStats stats;
        auto profiled = index.searchKnnProfiled(query, k, stats);

        assert(baseline.size() == profiled.size());
        while (!baseline.empty()) {
            assert(baseline.top() == profiled.top());
            baseline.pop();
            profiled.pop();
        }

        assert(stats.total_distance_computations > 0);
        assert(stats.base_layer_expansion_count > 0);

        const size_t expansion_histogram_sum = std::accumulate(
            stats.candidate_count_histogram.begin(),
            stats.candidate_count_histogram.end(),
            static_cast<size_t>(0));
        assert(expansion_histogram_sum == stats.base_layer_expansion_count);

        const size_t run_histogram_sum = std::accumulate(
            stats.contiguous_run_length_histogram.begin(),
            stats.contiguous_run_length_histogram.end(),
            static_cast<size_t>(0));
        assert(run_histogram_sum > 0);
    }
}

}  // namespace

int main() {
    std::cout << "Testing ..." << std::endl;
    test_profiled_search_matches_default();
    std::cout << "Test ok" << std::endl;
    return 0;
}