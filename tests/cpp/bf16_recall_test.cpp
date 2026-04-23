// BF16 recall regression test.
// Verifies that BF16 row-major batch distance achieves acceptable recall vs FP32 baseline.

#include "../../hnswlib/hnswlib.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

namespace {

// Compute recall@k: fraction of BF16 top-k results that appear in FP32 top-k ground truth.
double computeRecall(
    const std::vector<std::vector<hnswlib::labeltype>>& gt,
    const std::vector<std::vector<hnswlib::labeltype>>& pred,
    size_t k) {
    size_t nq = gt.size();
    size_t total_hits = 0;
    for (size_t q = 0; q < nq; q++) {
        std::set<hnswlib::labeltype> gt_set(gt[q].begin(), gt[q].begin() + k);
        for (size_t i = 0; i < k && i < pred[q].size(); i++) {
            if (gt_set.count(pred[q][i])) total_hits++;
        }
    }
    return (double)total_hits / (nq * k);
}

// Generate random unit-normalized data.
std::vector<float> generateNormalizedData(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (size_t i = 0; i < n; i++) {
        float norm = 0.0f;
        for (size_t d = 0; d < dim; d++) {
            float v = dist(rng);
            data[i * dim + d] = v;
            norm += v * v;
        }
        norm = std::sqrt(norm);
        for (size_t d = 0; d < dim; d++) {
            data[i * dim + d] /= norm;
        }
    }
    return data;
}

// Run top-k search and return labels for each query.
std::vector<std::vector<hnswlib::labeltype>> searchAll(
    hnswlib::HierarchicalNSW<float>& index,
    const float* queries, size_t nq, size_t dim, size_t k) {
    std::vector<std::vector<hnswlib::labeltype>> results(nq);
    for (size_t q = 0; q < nq; q++) {
        auto pq = index.searchKnn(queries + q * dim, k);
        results[q].resize(pq.size());
        for (int i = (int)pq.size() - 1; i >= 0; i--) {
            results[q][i] = pq.top().second;
            pq.pop();
        }
    }
    return results;
}

struct TestConfig {
    size_t dim;
    size_t nb;
    size_t nq;
    size_t k;
    size_t M;
    size_t ef_construction;
    size_t ef_search;
    double min_recall;
};

void runRecallTest(const TestConfig& cfg) {
    printf("dim=%zu nb=%zu nq=%zu k=%zu ef=%zu min_recall=%.3f ... ",
           cfg.dim, cfg.nb, cfg.nq, cfg.k, cfg.ef_search, cfg.min_recall);
    fflush(stdout);

    auto db = generateNormalizedData(cfg.nb, cfg.dim, 42);
    auto queries = generateNormalizedData(cfg.nq, cfg.dim, 123);

    hnswlib::InnerProductSpace space(cfg.dim);
    hnswlib::HierarchicalNSW<float> index(&space, cfg.nb, cfg.M, cfg.ef_construction);

    for (size_t i = 0; i < cfg.nb; i++) {
        index.addPoint(db.data() + i * cfg.dim, i);
    }
    index.setEf(cfg.ef_search);

    // FP32 baseline
    auto gt = searchAll(index, queries.data(), cfg.nq, cfg.dim, cfg.k);

    // Enable BF16
    index.setBf16RowmajorBatchDistance(true);
    auto bf16_results = searchAll(index, queries.data(), cfg.nq, cfg.dim, cfg.k);

    double recall = computeRecall(gt, bf16_results, cfg.k);
    printf("recall@%zu = %.4f ", cfg.k, recall);
    if (recall >= cfg.min_recall) {
        printf("OK\n");
    } else {
        printf("FAIL (below %.3f)\n", cfg.min_recall);
        assert(false && "BF16 recall below threshold");
    }

    // Disable BF16 and verify original results still match
    index.setBf16RowmajorBatchDistance(false);
    auto fp32_again = searchAll(index, queries.data(), cfg.nq, cfg.dim, cfg.k);
    double sanity_recall = computeRecall(gt, fp32_again, cfg.k);
    assert(sanity_recall == 1.0 && "FP32 results changed after BF16 toggle");
}

}  // namespace

int main() {
    printf("=== BF16 Recall Regression Test ===\n");

    // dim=32, small index
    runRecallTest({32, 5000, 100, 10, 16, 200, 200, 0.95});

    // dim=128, medium index
    runRecallTest({128, 10000, 100, 10, 16, 200, 200, 0.95});

    // dim=256, larger vectors
    runRecallTest({256, 10000, 100, 10, 16, 200, 200, 0.95});

    // dim=256, top-1 (stricter)
    runRecallTest({256, 10000, 100, 1, 16, 200, 200, 0.97});

    // dim=1024, high-dimensional
    runRecallTest({1024, 5000, 50, 10, 16, 200, 200, 0.95});

    printf("=== All recall tests passed ===\n");
    return 0;
}
