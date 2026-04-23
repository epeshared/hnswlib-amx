/**
 * AMX BF16 tile dot-product kernel test.
 *
 * AMX tile layout for HNSW batch distance:
 *   - 1 query  vs  up to 16 candidates  at a time
 *   - dimension processed in chunks of 32 BF16 elements (64 bytes per tile row)
 *
 * Tile A (query):   1 row  × 64 bytes (32 BF16)  — replicated to K rows
 * Tile B (cands):   K rows × 64 bytes (32 BF16)  — each row is one candidate's 32-dim slice
 * Tile C (accum):   1 row  × 64 bytes (16 FP32)  — partial dot products for 16 candidates
 *
 * tdpbf16ps:  C[1×16] += A[1×32bf16] · B[K×32bf16]^T
 *   but actually the hardware does: C[M×N] += A[M×K_bf16] · B[K_bf16×N]
 *   where A rows=M, A cols=K*2 bytes, B rows=K, B cols=N*4 bytes
 *
 * For our case:
 *   Tile A: rows=1, cols_bytes=64 (32 BF16 = 64 bytes)
 *   Tile B: rows=32 (K_bf16 pairs, since each BF16 pair = 1 "K row" in AMX),
 *           cols_bytes=64 (16 FP32 outputs × 4 = 64 bytes)
 *           BUT Tile B layout is column-interleaved: pairs of BF16 from each candidate
 *           packed into "vnni" format.
 *   Tile C: rows=1, cols_bytes=64 (16 FP32)
 *
 * Actually, let me re-derive the AMX BF16 layout carefully.
 *
 * AMX tdpbf16ps semantics (from Intel ISA ref):
 *   FOR m = 0 TO dst.rows - 1
 *     FOR k = 0 TO (src1.colsb / 4) - 1       // colsb in bytes, each iteration processes 4 bytes = 2 BF16
 *       FOR n = 0 TO (dst.colsb / 4) - 1       // colsb of dst in bytes, each iteration = 1 FP32
 *         dst[m][n] += src1[m][2*k+0] * src2[k][2*n+0]   // BF16 multiply
 *                    + src1[m][2*k+1] * src2[k][2*n+1]   // BF16 multiply
 *
 * So src1 (tile A) is M × (K*2) BF16 values, laid out row-major in memory.
 * src2 (tile B) is K × (N*2) BF16 values, also row-major, but the N dimension
 * has pairs of BF16 interleaved (vnni format): for each "k" row and "n" column,
 * there are 2 BF16 values at src2[k][2*n] and src2[k][2*n+1].
 *
 * For 1 query × 16 candidates × 32 BF16 per chunk:
 *   - M = 1 (one query)
 *   - N = 16 (sixteen candidates → 16 FP32 outputs)
 *   - K = 16 (because each K iteration consumes 2 BF16 from src1, so 32 BF16 / 2 = 16 K iterations)
 *
 *   Tile A (src1): rows=1, colsb=64 (32 BF16 = 64 bytes)
 *     A[0][0..31] = query_bf16[0..31]  (32 BF16 values, row-major)
 *
 *   Tile B (src2): rows=16, colsb=64 (16 candidates × 2 BF16 = 64 bytes per row)
 *     B[k][2*n+0] = candidates[n].bf16[2*k+0]
 *     B[k][2*n+1] = candidates[n].bf16[2*k+1]
 *     i.e. for K-row k, we store pairs from each of the 16 candidates
 *
 *   Tile C (dst): rows=1, colsb=64 (16 FP32 = 64 bytes)
 *     C[0][n] = dot(query, candidates[n]) accumulated over dim chunks
 */

#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <cassert>
#include <sys/syscall.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILEDATA      18

static bool request_amx_permission() {
    long ret = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (ret != 0) {
        perror("ARCH_REQ_XCOMP_PERM for XTILEDATA");
        return false;
    }
    return true;
}

// Convert float to BF16 (truncation)
static inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

static inline float bf16_to_float(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Scalar reference: inner product distance = 1 - dot(a, b)
static float scalar_bf16_ip_distance(const uint16_t* a_bf16, const uint16_t* b_bf16, size_t dim) {
    float dot = 0.0f;
    for (size_t d = 0; d < dim; d++) {
        dot += bf16_to_float(a_bf16[d]) * bf16_to_float(b_bf16[d]);
    }
    return 1.0f - dot;
}

#if defined(__AMX_BF16__)

// Tile config structure — must be 64-byte aligned
struct __attribute__((aligned(64))) TileConfig {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];   // columns in bytes for each tile
    uint8_t rows[16];      // rows for each tile
};

/**
 * AMX BF16 batch inner-product distance.
 *
 * Computes distances from 1 query to `count` candidates (count <= 16).
 * query_bf16: BF16 query vector, dim elements.
 * candidate_ptrs: array of pointers to BF16 candidate vectors (each dim elements).
 * distances_out: output array, count elements.
 * dim: vector dimension, must be even (and ideally multiple of 32).
 *
 * Internally processes in chunks of 32 BF16 elements using AMX tiles.
 */
static void amx_bf16_batch_ip_distance(
    const uint16_t* query_bf16,
    const uint16_t* const* candidate_ptrs,
    float* distances_out,
    size_t count,
    size_t dim)
{
    assert(count <= 16);
    assert(dim % 2 == 0);

    // --- Configure tiles ---
    // Tile 0 (src1/A): query — 1 row × 64 bytes (32 BF16)
    // Tile 1 (src2/B): candidates — 16 rows × 64 bytes (vnni-packed)
    // Tile 2 (dst/C):  accumulator — 1 row × 64 bytes (16 FP32)
    TileConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette_id = 1;

    cfg.rows[0] = 1;      // tile A: 1 query row
    cfg.colsb[0] = 64;    // 32 BF16 = 64 bytes

    cfg.rows[1] = 16;     // tile B: 16 K-rows (for 32 BF16 / 2 = 16 pairs)
    cfg.colsb[1] = 64;    // 16 candidates × 2 BF16 × 2 bytes = 64 bytes

    cfg.rows[2] = 1;      // tile C: 1 row
    cfg.colsb[2] = 64;    // 16 FP32 × 4 bytes = 64 bytes

    _tile_loadconfig(&cfg);
    _tile_zero(2);         // zero accumulator

    // Scratch buffers for tile B (vnni-packed)
    // B layout: B[k][2*n+0] = cand[n][2*k+0], B[k][2*n+1] = cand[n][2*k+1]
    // 16 rows × 32 uint16_t per row = 16 × 64 bytes
    alignas(64) uint16_t tile_b[16 * 32];

    // Process dim in chunks of 32
    for (size_t d = 0; d + 31 < dim; d += 32) {
        // Load tile A: just the query's 32 BF16 values
        // _tile_loadd expects: tile_id, base_ptr, stride_in_bytes
        // For a 1-row tile, stride doesn't matter much, but must be >= colsb
        _tile_loadd(0, query_bf16 + d, 64);

        // Pack tile B in vnni format
        // For k in [0..15], n in [0..count-1]:
        //   tile_b[k * 32 + 2*n + 0] = candidate_ptrs[n][d + 2*k + 0]
        //   tile_b[k * 32 + 2*n + 1] = candidate_ptrs[n][d + 2*k + 1]
        // Zero-fill for n >= count
        memset(tile_b, 0, sizeof(tile_b));
        for (size_t n = 0; n < count; n++) {
            const uint16_t* cand = candidate_ptrs[n] + d;
            for (size_t k = 0; k < 16; k++) {
                tile_b[k * 32 + 2 * n + 0] = cand[2 * k + 0];
                tile_b[k * 32 + 2 * n + 1] = cand[2 * k + 1];
            }
        }
        _tile_loadd(1, tile_b, 64);  // stride = 64 bytes per row

        // C += A · B
        _tile_dpbf16ps(2, 0, 1);
    }

    // Handle remaining dimensions (dim % 32 != 0) with scalar
    size_t dim32 = dim & ~31ULL;
    float tail_dots[16] = {0};
    if (dim32 < dim) {
        for (size_t n = 0; n < count; n++) {
            for (size_t d = dim32; d < dim; d++) {
                tail_dots[n] += bf16_to_float(query_bf16[d]) * bf16_to_float(candidate_ptrs[n][d]);
            }
        }
    }

    // Extract results from tile C
    alignas(64) float tile_c[16];  // 1 row × 16 FP32
    _tile_stored(2, tile_c, 64);   // stride = 64 bytes

    _tile_release();

    for (size_t n = 0; n < count; n++) {
        distances_out[n] = 1.0f - (tile_c[n] + tail_dots[n]);
    }
}

#endif // __AMX_BF16__

int main() {
#if !defined(__AMX_BF16__)
    printf("AMX BF16 not available, skipping test.\n");
    return 0;
#else
    printf("=== AMX BF16 Batch IP Distance Test ===\n");

    if (!request_amx_permission()) {
        printf("Failed to get AMX permission from kernel.\n");
        return 1;
    }
    printf("AMX permission granted.\n");

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Test multiple dimensions
    for (size_t dim : {32, 64, 128, 256, 512, 1024}) {
        for (size_t count : {1, 4, 8, 15, 16}) {
            // Generate random query
            std::vector<float> query_f32(dim);
            for (auto& v : query_f32) v = dist(rng);
            // Normalize
            float norm = 0;
            for (auto v : query_f32) norm += v * v;
            norm = std::sqrt(norm);
            for (auto& v : query_f32) v /= norm;

            // Convert to BF16
            std::vector<uint16_t> query_bf16(dim);
            for (size_t i = 0; i < dim; i++) query_bf16[i] = float_to_bf16(query_f32[i]);

            // Generate random candidates
            std::vector<std::vector<uint16_t>> cands_bf16(count);
            std::vector<const uint16_t*> cand_ptrs(count);
            for (size_t c = 0; c < count; c++) {
                std::vector<float> cf(dim);
                for (auto& v : cf) v = dist(rng);
                float cn = 0;
                for (auto v : cf) cn += v * v;
                cn = std::sqrt(cn);
                for (auto& v : cf) v /= cn;
                cands_bf16[c].resize(dim);
                for (size_t i = 0; i < dim; i++) cands_bf16[c][i] = float_to_bf16(cf[i]);
                cand_ptrs[c] = cands_bf16[c].data();
            }

            // AMX batch distance
            std::vector<float> amx_dists(count);
            amx_bf16_batch_ip_distance(
                query_bf16.data(), cand_ptrs.data(),
                amx_dists.data(), count, dim);

            // Scalar reference
            float max_err = 0;
            for (size_t c = 0; c < count; c++) {
                float ref = scalar_bf16_ip_distance(query_bf16.data(), cands_bf16[c].data(), dim);
                float err = std::fabs(amx_dists[c] - ref);
                if (err > max_err) max_err = err;
            }

            // BF16 accumulation can have small numerical differences due to
            // different reduction order, but should be very close
            bool ok = max_err < 1e-3f;
            printf("  dim=%4zu count=%2zu max_err=%.6f %s\n",
                   dim, count, max_err, ok ? "OK" : "FAIL");
            if (!ok) {
                printf("    FAILED! Dumping first mismatch:\n");
                for (size_t c = 0; c < count; c++) {
                    float ref = scalar_bf16_ip_distance(query_bf16.data(), cands_bf16[c].data(), dim);
                    printf("    cand[%zu] amx=%.6f ref=%.6f diff=%.6f\n",
                           c, amx_dists[c], ref, amx_dists[c] - ref);
                }
                return 1;
            }
        }
    }

    printf("=== All AMX BF16 tests passed ===\n");
    return 0;
#endif
}
