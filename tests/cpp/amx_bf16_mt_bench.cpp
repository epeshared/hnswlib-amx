// Multi-threaded AMX BF16 vs AVX512-BF16 HNSW search benchmark.
// Uses hnswlib's built-in ParallelFor to dispatch queries across threads.
// Measures QPS for FP32, BF16-AVX512, and BF16-AMX at various thread counts.

#include "../../hnswlib/hnswlib.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// ParallelFor  (copied from python_bindings/bindings.cpp)
// ---------------------------------------------------------------------------
template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }
    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);
                    if (id >= end) break;
                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lock(lastExceptMutex);
                        lastException = std::current_exception();
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto& t : threads) t.join();
        if (lastException) std::rethrow_exception(lastException);
    }
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Multi-threaded benchmark: dispatch nq queries across numThreads threads
// ---------------------------------------------------------------------------
double benchSearchMT(
    hnswlib::HierarchicalNSW<float>& index,
    const float* queries, size_t nq, size_t dim, size_t k,
    size_t numThreads, int warmup_rounds, int bench_rounds) {

    // Warmup
    for (int r = 0; r < warmup_rounds; r++) {
        ParallelFor(0, nq, numThreads, [&](size_t q, size_t /*tid*/) {
            auto pq = index.searchKnn(queries + q * dim, k);
        });
    }

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < bench_rounds; r++) {
        ParallelFor(0, nq, numThreads, [&](size_t q, size_t /*tid*/) {
            auto pq = index.searchKnn(queries + q * dim, k);
        });
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    return (nq * bench_rounds) / sec;
}

// ---------------------------------------------------------------------------
// Benchmark configuration
// ---------------------------------------------------------------------------
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

void runBenchmark(const BenchConfig& cfg, const std::vector<size_t>& threadCounts) {
    printf("\n====================================================================\n");
    printf("  dim=%zu  nb=%zu  nq=%zu  k=%zu  ef=%zu  M=%zu\n",
           cfg.dim, cfg.nb, cfg.nq, cfg.k, cfg.ef_search, cfg.M);
    printf("====================================================================\n");

    auto db = generateNormalizedData(cfg.nb, cfg.dim, 42);
    auto queries = generateNormalizedData(cfg.nq, cfg.dim, 123);

    // Build index (single-threaded for reproducibility)
    hnswlib::InnerProductSpace space(cfg.dim);
    hnswlib::HierarchicalNSW<float> index(&space, cfg.nb, cfg.M, cfg.ef_construction);
    printf("  Building index ... ");
    fflush(stdout);
    for (size_t i = 0; i < cfg.nb; i++) {
        index.addPoint(db.data() + i * cfg.dim, i);
    }
    index.setEf(cfg.ef_search);
    printf("done\n");

    // Header
    printf("\n  %-10s  %12s  %12s  %12s  %10s\n",
           "Threads", "FP32 QPS", "BF16-AVX512", "BF16-AMX", "AMX/AVX512");
    printf("  ----------  ------------  ------------  ------------  ----------\n");

    for (size_t nt : threadCounts) {
        // FP32
        index.setBf16RowmajorBatchDistance(false);
        index.setAmxBf16(false);
        double qps_fp32 = benchSearchMT(index, queries.data(), cfg.nq, cfg.dim, cfg.k,
                                         nt, cfg.warmup, cfg.rounds);

        // BF16-AVX512
        index.setBf16RowmajorBatchDistance(true);
        index.setAmxBf16(false);
        double qps_avx512 = benchSearchMT(index, queries.data(), cfg.nq, cfg.dim, cfg.k,
                                            nt, cfg.warmup, cfg.rounds);

        // BF16-AMX
        index.setAmxBf16(true);
        double qps_amx = benchSearchMT(index, queries.data(), cfg.nq, cfg.dim, cfg.k,
                                         nt, cfg.warmup, cfg.rounds);

        printf("  %-10zu  %12.0f  %12.0f  %12.0f  %9.2fx\n",
               nt, qps_fp32, qps_avx512, qps_amx, qps_amx / qps_avx512);
    }
}

}  // namespace

int main(int argc, char** argv) {
    printf("=== Multi-threaded AMX BF16 vs AVX512-BF16 HNSW Benchmark ===\n");
    printf("Hardware concurrency: %u\n", std::thread::hardware_concurrency());

    // Default thread counts to sweep
    std::vector<size_t> threadCounts = {1, 2, 4, 8, 16, 32};

    // Allow overriding from command line: amx_bf16_mt_bench [t1 t2 t3 ...]
    if (argc > 1) {
        threadCounts.clear();
        for (int i = 1; i < argc; i++) {
            int t = std::atoi(argv[i]);
            if (t > 0) threadCounts.push_back(static_cast<size_t>(t));
        }
    }

    printf("Thread counts:");
    for (size_t t : threadCounts) printf(" %zu", t);
    printf("\n");

    // dim=256, 100K vectors, 10000 queries
    runBenchmark({256, 100000, 10000, 10, 32, 200, 200, 1, 3}, threadCounts);

    // dim=1024, 50K vectors, 5000 queries
    runBenchmark({1024, 50000, 5000, 10, 32, 200, 200, 1, 3}, threadCounts);

    // dim=128, 100K vectors, 10000 queries
    runBenchmark({128, 100000, 10000, 10, 32, 200, 200, 1, 3}, threadCounts);

    printf("\n=== Benchmark complete ===\n");
    return 0;
}
