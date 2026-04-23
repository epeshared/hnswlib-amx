// Benchmark: FP32 baseline vs BF16 + AVX512-BF16 HNSW search.
// Prints QPS and recall for each mode across multiple dimensions.

#include "../../hnswlib/hnswlib.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

namespace {

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

struct BenchResult {
    double qps;
    double recall;
};

BenchResult runBenchmark(
    hnswlib::HierarchicalNSW<float>& index,
    const float* queries, size_t nq, size_t dim, size_t k,
    const std::vector<std::vector<hnswlib::labeltype>>* gt = nullptr) {

    // Warmup
    for (size_t q = 0; q < std::min(nq, (size_t)10); q++) {
        auto pq = index.searchKnn(queries + q * dim, k);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<hnswlib::labeltype>> results(nq);
    for (size_t q = 0; q < nq; q++) {
        auto pq = index.searchKnn(queries + q * dim, k);
        results[q].resize(pq.size());
        for (int i = (int)pq.size() - 1; i >= 0; i--) {
            results[q][i] = pq.top().second;
            pq.pop();
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double qps = nq / elapsed;

    double recall = 1.0;
    if (gt) {
        size_t total_hits = 0;
        for (size_t q = 0; q < nq; q++) {
            std::set<hnswlib::labeltype> gt_set((*gt)[q].begin(), (*gt)[q].begin() + k);
            for (size_t i = 0; i < k && i < results[q].size(); i++) {
                if (gt_set.count(results[q][i])) total_hits++;
            }
        }
        recall = (double)total_hits / (nq * k);
    }

    return {qps, recall};
}

struct TestConfig {
    size_t dim;
    size_t nb;
    size_t nq;
    size_t k;
    size_t M;
    size_t ef_construction;
    size_t ef_search;
};

void runBenchSuite(const TestConfig& cfg) {
    printf("\n--- dim=%zu nb=%zu nq=%zu k=%zu M=%zu ef=%zu ---\n",
           cfg.dim, cfg.nb, cfg.nq, cfg.k, cfg.M, cfg.ef_search);

    auto db = generateNormalizedData(cfg.nb, cfg.dim, 42);
    auto queries = generateNormalizedData(cfg.nq, cfg.dim, 123);

    hnswlib::InnerProductSpace space(cfg.dim);
    hnswlib::HierarchicalNSW<float> index(&space, cfg.nb, cfg.M, cfg.ef_construction);

    for (size_t i = 0; i < cfg.nb; i++) {
        index.addPoint(db.data() + i * cfg.dim, i);
    }
    index.setEf(cfg.ef_search);

    // FP32 baseline: run, collect ground truth
    std::vector<std::vector<hnswlib::labeltype>> gt(cfg.nq);
    {
        for (size_t q = 0; q < cfg.nq; q++) {
            auto pq = index.searchKnn(queries.data() + q * cfg.dim, cfg.k);
            gt[q].resize(pq.size());
            for (int i = (int)pq.size() - 1; i >= 0; i--) {
                gt[q][i] = pq.top().second;
                pq.pop();
            }
        }
    }

    auto fp32_res = runBenchmark(index, queries.data(), cfg.nq, cfg.dim, cfg.k, nullptr);
    printf("FP32 baseline:     QPS=%10.0f  recall@%zu=1.0000\n", fp32_res.qps, cfg.k);

    // BF16 + AVX512-BF16
    index.setBf16RowmajorBatchDistance(true);
    auto bf16_res = runBenchmark(index, queries.data(), cfg.nq, cfg.dim, cfg.k, &gt);
    printf("BF16 AVX512-BF16:  QPS=%10.0f  recall@%zu=%.4f  speedup=%.2fx\n",
           bf16_res.qps, cfg.k, bf16_res.recall, bf16_res.qps / fp32_res.qps);

    index.setBf16RowmajorBatchDistance(false);
}

}  // namespace

int main() {
    printf("=== FP32 vs BF16-AVX512 HNSW Search Benchmark ===\n");
#if defined(__AVX512BF16__)
    printf("AVX512-BF16 (vdpbf16ps) path: ACTIVE\n");
#else
    printf("AVX512-BF16 path: NOT AVAILABLE (scalar fallback)\n");
#endif

    // Various dimensions and sizes
    runBenchSuite({128,  100000, 1000, 10, 32, 200, 200});
    runBenchSuite({256,  100000, 1000, 10, 32, 200, 200});
    runBenchSuite({512,  50000,  1000, 10, 32, 200, 200});
    runBenchSuite({1024, 10000,  200,  10, 32, 200, 200});

    printf("\n=== Benchmark complete ===\n");
    return 0;
}
