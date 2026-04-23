// AMX BF16 recall test.
// Verifies that AMX BF16 batch distance achieves identical recall to AVX512-BF16.

#include "../../hnswlib/hnswlib.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

namespace {

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

void runAmxRecallTest(const TestConfig& cfg) {
    printf("dim=%zu nb=%zu nq=%zu k=%zu ef=%zu ... ",
           cfg.dim, cfg.nb, cfg.nq, cfg.k, cfg.ef_search);
    fflush(stdout);

    auto db = generateNormalizedData(cfg.nb, cfg.dim, 42);
    auto queries = generateNormalizedData(cfg.nq, cfg.dim, 123);

    hnswlib::InnerProductSpace space(cfg.dim);
    hnswlib::HierarchicalNSW<float> index(&space, cfg.nb, cfg.M, cfg.ef_construction);

    for (size_t i = 0; i < cfg.nb; i++) {
        index.addPoint(db.data() + i * cfg.dim, i);
    }
    index.setEf(cfg.ef_search);

    // FP32 ground truth
    auto gt = searchAll(index, queries.data(), cfg.nq, cfg.dim, cfg.k);

    // AVX512-BF16 baseline
    index.setBf16RowmajorBatchDistance(true);
    auto avx512_results = searchAll(index, queries.data(), cfg.nq, cfg.dim, cfg.k);
    double avx512_recall = computeRecall(gt, avx512_results, cfg.k);

    // AMX-BF16
    index.setAmxBf16(true);
    auto amx_results = searchAll(index, queries.data(), cfg.nq, cfg.dim, cfg.k);
    double amx_recall = computeRecall(gt, amx_results, cfg.k);

    // AMX vs AVX512 cross-recall (should be very close or identical)
    double cross_recall = computeRecall(avx512_results, amx_results, cfg.k);

    printf("avx512_recall=%.4f amx_recall=%.4f cross=%.4f ",
           avx512_recall, amx_recall, cross_recall);

    bool ok = (amx_recall >= cfg.min_recall) && (cross_recall >= 0.99);
    printf("%s\n", ok ? "OK" : "FAIL");

    if (!ok) {
        fprintf(stderr, "FAIL: AMX recall %.4f (min %.3f), cross %.4f (min 0.99)\n",
                amx_recall, cfg.min_recall, cross_recall);
        assert(false);
    }

    // Disable AMX, verify AVX512 still works
    index.setAmxBf16(false);
    auto avx512_again = searchAll(index, queries.data(), cfg.nq, cfg.dim, cfg.k);
    double sanity = computeRecall(avx512_results, avx512_again, cfg.k);
    assert(sanity == 1.0 && "AVX512 results changed after AMX toggle");
}

}  // namespace

int main() {
    printf("=== AMX BF16 Recall Test ===\n");

    runAmxRecallTest({32, 5000, 100, 10, 16, 200, 200, 0.95});
    runAmxRecallTest({128, 10000, 100, 10, 16, 200, 200, 0.95});
    runAmxRecallTest({256, 10000, 100, 10, 16, 200, 200, 0.95});
    runAmxRecallTest({256, 10000, 100, 1, 16, 200, 200, 0.97});
    runAmxRecallTest({1024, 5000, 50, 10, 16, 200, 200, 0.95});

    printf("=== All AMX recall tests passed ===\n");
    return 0;
}
