/**
 * Test BF16-only mode: enable BF16, release FP32 storage, verify search still works.
 */
#include "../../hnswlib/hnswlib.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void generate_random_data(float* data, size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n * dim; i++) {
        data[i] = dist(rng);
    }
    // normalize for IP
    for (size_t i = 0; i < n; i++) {
        float norm = 0;
        for (size_t d = 0; d < dim; d++) norm += data[i * dim + d] * data[i * dim + d];
        norm = std::sqrt(norm);
        for (size_t d = 0; d < dim; d++) data[i * dim + d] /= norm;
    }
}

static float compute_recall(
    const std::vector<hnswlib::labeltype>& results,
    const std::vector<hnswlib::labeltype>& gt,
    int k) {
    int hit = 0;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (results[i] == gt[j]) { hit++; break; }
        }
    }
    return (float)hit / k;
}

int main() {
    const int dim = 256;
    const int nb = 10000;
    const int nq = 100;
    const int k = 10;
    const int M = 16;
    const int ef_construction = 200;
    const int ef_search = 200;

    printf("=== BF16-Only Mode (Release FP32) Test ===\n");
    printf("dim=%d nb=%d nq=%d k=%d\n", dim, nb, nq, k);

    // Generate data
    std::vector<float> db(nb * dim), queries(nq * dim);
    generate_random_data(db.data(), nb, dim, 42);
    generate_random_data(queries.data(), nq, dim, 123);

    // Build index
    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, nb, M, ef_construction);
    for (int i = 0; i < nb; i++) {
        index.addPoint(db.data() + i * dim, i);
    }
    index.setEf(ef_search);

    // --- Step 1: FP32 baseline search ---
    std::vector<std::vector<hnswlib::labeltype>> gt(nq);
    for (int q = 0; q < nq; q++) {
        auto result = index.searchKnn(queries.data() + q * dim, k);
        gt[q].resize(k);
        for (int i = k - 1; i >= 0; i--) {
            gt[q][i] = result.top().second;
            result.pop();
        }
    }
    printf("FP32 baseline search done.\n");

    // --- Step 2: Enable BF16, search before release ---
    index.setBf16RowmajorBatchDistance(true);
    printf("BF16 enabled.\n");

    float recall_bf16_before = 0;
    for (int q = 0; q < nq; q++) {
        auto result = index.searchKnn(queries.data() + q * dim, k);
        std::vector<hnswlib::labeltype> res(k);
        for (int i = k - 1; i >= 0; i--) {
            res[i] = result.top().second;
            result.pop();
        }
        recall_bf16_before += compute_recall(res, gt[q], k);
    }
    recall_bf16_before /= nq;
    printf("BF16 before release: recall@%d = %.4f\n", k, recall_bf16_before);
    assert(recall_bf16_before >= 0.95f);

    // --- Step 3: Release FP32 storage ---
    index.releaseFp32VectorStorage();
    assert(index.isFp32VectorsReleased());
    printf("FP32 vectors released.\n");

    // --- Step 4: Search after release ---
    float recall_bf16_after = 0;
    for (int q = 0; q < nq; q++) {
        auto result = index.searchKnn(queries.data() + q * dim, k);
        std::vector<hnswlib::labeltype> res(k);
        for (int i = k - 1; i >= 0; i--) {
            res[i] = result.top().second;
            result.pop();
        }
        recall_bf16_after += compute_recall(res, gt[q], k);
    }
    recall_bf16_after /= nq;
    printf("BF16 after FP32 release: recall@%d = %.4f\n", k, recall_bf16_after);
    assert(recall_bf16_after >= 0.95f);

    // Recall should be identical before and after release
    assert(std::abs(recall_bf16_before - recall_bf16_after) < 1e-6f);
    printf("Recall matches before/after release: OK\n");

    // --- Step 5: Verify guards ---
    bool caught = false;

    // addPoint should throw
    caught = false;
    try { index.addPoint(queries.data(), nb + 1); } catch (std::runtime_error&) { caught = true; }
    assert(caught);
    printf("addPoint guard: OK\n");

    // getDataByLabel should throw
    caught = false;
    try { index.getDataByLabel<float>(0); } catch (std::runtime_error&) { caught = true; }
    assert(caught);
    printf("getDataByLabel guard: OK\n");

    // disable BF16 should throw
    caught = false;
    try { index.setBf16RowmajorBatchDistance(false); } catch (std::runtime_error&) { caught = true; }
    assert(caught);
    printf("setBf16(false) guard: OK\n");

    printf("=== All BF16-only mode tests passed ===\n");
    return 0;
}
