// AMX BF16 vs AVX512-BF16 HNSW search benchmark.
// Measures QPS for FP32, BF16-AVX512, and BF16-AMX on the same index.

#include "../../hnswlib/hnswlib.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
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

double benchSearch(
    hnswlib::HierarchicalNSW<float>& index,
    const float* queries, size_t nq, size_t dim, size_t k,
    int warmup_rounds, int bench_rounds) {
    // Warmup
    for (int r = 0; r < warmup_rounds; r++) {
        for (size_t q = 0; q < nq; q++) {
            auto pq = index.searchKnn(queries + q * dim, k);
        }
    }

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < bench_rounds; r++) {
        for (size_t q = 0; q < nq; q++) {
            auto pq = index.searchKnn(queries + q * dim, k);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    return (nq * bench_rounds) / sec;
}

struct BenchConfig {
    size_t dim;
    size_t nb;
    size_t nq;
    size_t k;
    size_t M;
    size_t ef_construction;
    size_t ef_search;
    int warmup;
    int rounds;
};

void runBenchmark(const BenchConfig& cfg) {
    printf("\n--- dim=%zu nb=%zu nq=%zu k=%zu ef=%zu M=%zu ---\n",
           cfg.dim, cfg.nb, cfg.nq, cfg.k, cfg.ef_search, cfg.M);

    auto db = generateNormalizedData(cfg.nb, cfg.dim, 42);
    auto queries = generateNormalizedData(cfg.nq, cfg.dim, 123);

    hnswlib::InnerProductSpace space(cfg.dim);
    hnswlib::HierarchicalNSW<float> index(&space, cfg.nb, cfg.M, cfg.ef_construction);

    for (size_t i = 0; i < cfg.nb; i++) {
        index.addPoint(db.data() + i * cfg.dim, i);
    }
    index.setEf(cfg.ef_search);

    // FP32 baseline
    double qps_fp32 = benchSearch(index, queries.data(), cfg.nq, cfg.dim, cfg.k,
                                   cfg.warmup, cfg.rounds);
    printf("  FP32:       %10.0f QPS\n", qps_fp32);

    // BF16-AVX512
    index.setBf16RowmajorBatchDistance(true);
    double qps_avx512 = benchSearch(index, queries.data(), cfg.nq, cfg.dim, cfg.k,
                                     cfg.warmup, cfg.rounds);
    printf("  BF16-AVX512: %9.0f QPS (%.2fx vs FP32)\n", qps_avx512, qps_avx512 / qps_fp32);

    // BF16-AMX
    index.setAmxBf16(true);
    double qps_amx = benchSearch(index, queries.data(), cfg.nq, cfg.dim, cfg.k,
                                  cfg.warmup, cfg.rounds);
    printf("  BF16-AMX:    %9.0f QPS (%.2fx vs FP32, %.2fx vs AVX512)\n",
           qps_amx, qps_amx / qps_fp32, qps_amx / qps_avx512);

    // Disable AMX, re-check AVX512 to ensure consistency
    index.setAmxBf16(false);
    double qps_avx512_2 = benchSearch(index, queries.data(), cfg.nq, cfg.dim, cfg.k,
                                       cfg.warmup, cfg.rounds);
    printf("  BF16-AVX512 (recheck): %9.0f QPS\n", qps_avx512_2);
}

}  // namespace

int main() {
    printf("=== AMX BF16 vs AVX512-BF16 HNSW Benchmark ===\n");

    // dim=256, 100K vectors
    runBenchmark({256, 100000, 1000, 10, 32, 200, 200, 2, 5});

    // dim=1024, 50K vectors
    runBenchmark({1024, 50000, 500, 10, 32, 200, 200, 2, 5});

    // dim=128, 100K vectors
    runBenchmark({128, 100000, 1000, 10, 32, 200, 200, 2, 5});

    printf("\n=== Benchmark complete ===\n");
    return 0;
}
