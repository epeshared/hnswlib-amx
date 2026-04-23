#include "../../hnswlib/hnswlib.h"

#include <assert.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {

void test_bf16_rowmajor_batch_distance_ip() {
    const int dim = 32;
    const size_t num_elements = 128;

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

    std::vector<float> data(num_elements * dim);
    for (size_t i = 0; i < num_elements; i++) {
        float norm = 0.0f;
        for (int d = 0; d < dim; d++) {
            float value = distrib(rng);
            data[i * dim + d] = value;
            norm += value * value;
        }
        norm = std::sqrt(norm);
        for (int d = 0; d < dim; d++) {
            data[i * dim + d] /= norm;
        }
    }

    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, num_elements, 16, 200);
    index.setEf(num_elements);

    for (size_t i = 0; i < num_elements; i++) {
        index.addPoint(data.data() + i * dim, i);
    }

    std::vector<hnswlib::labeltype> baseline_labels(num_elements);
    std::vector<float> baseline_distances(num_elements);
    for (size_t i = 0; i < baseline_labels.size(); i++) {
        auto result = index.searchKnn(data.data() + i * dim, 1);
        assert(result.size() == 1);
        baseline_labels[i] = result.top().second;
        baseline_distances[i] = result.top().first;
        assert(baseline_labels[i] == i);
    }

    index.setBf16RowmajorBatchDistance(true);
    assert(index.getBf16RowmajorBatchDistance());

    for (size_t i = 0; i < baseline_labels.size(); i++) {
        auto result = index.searchKnn(data.data() + i * dim, 1);
        assert(result.size() == 1);
        assert(result.top().second == baseline_labels[i]);
        assert(std::fabs(result.top().first - baseline_distances[i]) < 0.2f);
    }
}

}  // namespace

int main() {
    std::cout << "Testing ..." << std::endl;
    test_bf16_rowmajor_batch_distance_ip();
    std::cout << "Test ok" << std::endl;
    return 0;
}