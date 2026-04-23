#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>

#if defined(__AMX_BF16__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

struct SearchProfileStats {
    size_t total_distance_computations{0};
    size_t base_layer_expansion_count{0};
    std::vector<size_t> candidate_count_histogram;
    std::vector<size_t> contiguous_run_length_histogram;

    void merge(const SearchProfileStats& other) {
        total_distance_computations += other.total_distance_computations;
        base_layer_expansion_count += other.base_layer_expansion_count;
        mergeHistogram(candidate_count_histogram, other.candidate_count_histogram);
        mergeHistogram(contiguous_run_length_histogram, other.contiguous_run_length_histogram);
    }

    void recordDistanceComputation(size_t count = 1) {
        total_distance_computations += count;
    }

    void recordBaseLayerExpansion(const std::vector<tableint>& candidate_ids) {
        base_layer_expansion_count++;
        growHistogram(candidate_count_histogram, candidate_ids.size());
        candidate_count_histogram[candidate_ids.size()]++;

        if (candidate_ids.empty()) {
            return;
        }

        std::vector<tableint> sorted_ids(candidate_ids);
        std::sort(sorted_ids.begin(), sorted_ids.end());

        size_t run_length = 1;
        for (size_t i = 1; i < sorted_ids.size(); i++) {
            if (sorted_ids[i] == sorted_ids[i - 1]) {
                continue;
            }
            if (sorted_ids[i] == sorted_ids[i - 1] + 1) {
                run_length++;
                continue;
            }
            recordContiguousRun(run_length);
            run_length = 1;
        }
        recordContiguousRun(run_length);
    }

 private:
    static void growHistogram(std::vector<size_t>& histogram, size_t bucket) {
        if (histogram.size() <= bucket) {
            histogram.resize(bucket + 1, 0);
        }
    }

    static void mergeHistogram(std::vector<size_t>& target, const std::vector<size_t>& source) {
        if (target.size() < source.size()) {
            target.resize(source.size(), 0);
        }
        for (size_t i = 0; i < source.size(); i++) {
            target[i] += source[i];
        }
    }

    void recordContiguousRun(size_t run_length) {
        growHistogram(contiguous_run_length_histogram, run_length);
        contiguous_run_length_histogram[run_length]++;
    }
};

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool use_bf16_rowmajor_batch_distance_ = false;
    bool fp32_vectors_released_ = false;
    bool use_amx_bf16_ = false;
    std::vector<uint16_t> bf16_rowmajor_data_;

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements


    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
        bf16_rowmajor_data_.clear();
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    void setEf(size_t ef) {
        ef_ = ef;
    }


    void setBf16RowmajorBatchDistance(bool enabled) {
        if (!enabled) {
            if (fp32_vectors_released_) {
                throw std::runtime_error(
                    "Cannot disable BF16 mode after FP32 vector storage has been released");
            }
            use_bf16_rowmajor_batch_distance_ = false;
            bf16_rowmajor_data_.clear();
            return;
        }

        size_t dim = getDistanceDim();
        if (data_size_ != dim * sizeof(float)) {
            throw std::runtime_error("BF16 row-major batch distance requires float vector storage");
        }

        use_bf16_rowmajor_batch_distance_ = true;
        rebuildBf16RowmajorData();
    }


    bool getBf16RowmajorBatchDistance() const {
        return use_bf16_rowmajor_batch_distance_;
    }


    void setAmxBf16(bool enabled) {
#if defined(__AMX_BF16__)
        if (enabled) {
            if (!use_bf16_rowmajor_batch_distance_) {
                throw std::runtime_error("AMX BF16 requires BF16 row-major batch distance to be enabled first");
            }
            if (!requestAmxPermission()) {
                throw std::runtime_error("Failed to obtain AMX permission from the kernel");
            }
        }
        use_amx_bf16_ = enabled;
#else
        if (enabled) {
            throw std::runtime_error("AMX BF16 is not available (compiled without __AMX_BF16__ support)");
        }
#endif
    }


    bool getAmxBf16() const {
        return use_amx_bf16_;
    }


    bool isFp32VectorsReleased() const {
        return fp32_vectors_released_;
    }


    // Release FP32 vector storage to save memory.
    // Requires BF16 mode to be enabled. After this call:
    //   - Search uses BF16 data only (already the case when BF16 is enabled)
    //   - addPoint / updatePoint / getData will throw
    //   - Index can still be saved (links + labels preserved, vectors zeroed)
    void releaseFp32VectorStorage() {
        if (!use_bf16_rowmajor_batch_distance_ || bf16_rowmajor_data_.empty()) {
            throw std::runtime_error(
                "Cannot release FP32 vectors: BF16 row-major batch distance must be enabled first");
        }
        // Zero out vector data in data_level0_memory_, keep links and labels.
        for (tableint i = 0; i < cur_element_count; i++) {
            memset(getDataByInternalId(i), 0, data_size_);
        }
        fp32_vectors_released_ = true;
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }


    inline size_t getDistanceDim() const {
        return *((size_t *)dist_func_param_);
    }


    // Single-pair distance: dispatches to BF16 or FP32 depending on mode.
    // Used for entry point and upper-level greedy traversal during search.
    inline dist_t computeDistanceSingle(
        const void* query_data, tableint internal_id) const {
        if (use_bf16_rowmajor_batch_distance_ && !bf16_rowmajor_data_.empty()) {
            const size_t dim = getDistanceDim();
            // Convert query on the fly (upper-level calls are rare)
            std::vector<uint16_t> q_bf16(dim);
            convertFloatVectorToBf16(
                reinterpret_cast<const float*>(query_data), q_bf16.data(), dim);
            const uint16_t* cand = bf16_rowmajor_data_.data() + internal_id * dim;
#if defined(USE_AVX512) && defined(__AVX512BF16__)
            return static_cast<dist_t>(
                bf16InnerProductDistanceAVX512BF16(q_bf16.data(), cand, dim));
#else
            return static_cast<dist_t>(
                bf16InnerProductDistanceScalar(q_bf16.data(), cand, dim));
#endif
        }
        return fstdistfunc_(query_data, getDataByInternalId(internal_id), dist_func_param_);
    }


    // Version with pre-converted BF16 query (avoids repeated conversion in loops).
    inline dist_t computeDistanceSingleBf16Query(
        const uint16_t* query_bf16, tableint internal_id) const {
        const size_t dim = getDistanceDim();
        const uint16_t* cand = bf16_rowmajor_data_.data() + internal_id * dim;
#if defined(USE_AVX512) && defined(__AVX512BF16__)
        return static_cast<dist_t>(
            bf16InnerProductDistanceAVX512BF16(query_bf16, cand, dim));
#else
        return static_cast<dist_t>(
            bf16InnerProductDistanceScalar(query_bf16, cand, dim));
#endif
    }


    static uint16_t floatToBf16(float value) {
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        return static_cast<uint16_t>(bits >> 16);
    }


    static float bf16ToFloat(uint16_t value) {
        uint32_t bits = static_cast<uint32_t>(value) << 16;
        float result;
        memcpy(&result, &bits, sizeof(result));
        return result;
    }


    void convertFloatVectorToBf16(const float* src, uint16_t* dst, size_t dim) const {
        for (size_t i = 0; i < dim; i++) {
            dst[i] = floatToBf16(src[i]);
        }
    }


    void refreshBf16RowmajorElement(tableint internal_id) {
        if (!use_bf16_rowmajor_batch_distance_) {
            return;
        }

        size_t dim = getDistanceDim();
        const float* src = reinterpret_cast<const float*>(getDataByInternalId(internal_id));
        uint16_t* dst = bf16_rowmajor_data_.data() + internal_id * dim;
        convertFloatVectorToBf16(src, dst, dim);
    }


    void rebuildBf16RowmajorData() {
        if (!use_bf16_rowmajor_batch_distance_) {
            return;
        }

        size_t dim = getDistanceDim();
        bf16_rowmajor_data_.assign(max_elements_ * dim, 0);
        for (tableint i = 0; i < cur_element_count; i++) {
            refreshBf16RowmajorElement(i);
        }
    }


    // Compute BF16 inner-product distance for a single pair.
    // query_bf16 and candidate_bf16 are both uint16_t[dim] in BF16 format.
    static float bf16InnerProductDistanceScalar(
        const uint16_t* query_bf16,
        const uint16_t* candidate_bf16,
        size_t dim) {
        float dot = 0.0f;
        for (size_t d = 0; d < dim; d++) {
            dot += bf16ToFloat(query_bf16[d]) * bf16ToFloat(candidate_bf16[d]);
        }
        return 1.0f - dot;
    }

#if defined(USE_AVX512) && defined(__AVX512BF16__)
    // AVX512-BF16 accelerated inner-product distance using vdpbf16ps.
    // Requires dim >= 32. Handles residuals for any dim.
    static float bf16InnerProductDistanceAVX512BF16(
        const uint16_t* query_bf16,
        const uint16_t* candidate_bf16,
        size_t dim) {
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();

        size_t d = 0;
        // Main loop: process 64 BF16 elements (2x32) per iteration
        size_t dim64 = dim & ~63ULL;
        for (; d < dim64; d += 64) {
            __m512bh q0 = (__m512bh)_mm512_loadu_si512(query_bf16 + d);
            __m512bh c0 = (__m512bh)_mm512_loadu_si512(candidate_bf16 + d);
            acc0 = _mm512_dpbf16_ps(acc0, q0, c0);

            __m512bh q1 = (__m512bh)_mm512_loadu_si512(query_bf16 + d + 32);
            __m512bh c1 = (__m512bh)_mm512_loadu_si512(candidate_bf16 + d + 32);
            acc1 = _mm512_dpbf16_ps(acc1, q1, c1);
        }
        // Handle remaining 32-element chunk
        if (d + 32 <= dim) {
            __m512bh q0 = (__m512bh)_mm512_loadu_si512(query_bf16 + d);
            __m512bh c0 = (__m512bh)_mm512_loadu_si512(candidate_bf16 + d);
            acc0 = _mm512_dpbf16_ps(acc0, q0, c0);
            d += 32;
        }
        acc0 = _mm512_add_ps(acc0, acc1);
        float dot = _mm512_reduce_add_ps(acc0);

        // Scalar tail for remaining elements
        for (; d < dim; d++) {
            dot += bf16ToFloat(query_bf16[d]) * bf16ToFloat(candidate_bf16[d]);
        }
        return 1.0f - dot;
    }
#endif

#if defined(__AMX_BF16__)
    // Request AMX tile permission from the Linux kernel.
    // Must be called once per process before using AMX tile instructions.
    static bool requestAmxPermission() {
        static bool amx_requested = false;
        static bool amx_granted = false;
        if (amx_requested) return amx_granted;
        amx_requested = true;
        const unsigned long ARCH_REQ_XCOMP_PERM = 0x1023;
        const unsigned long XFEATURE_XTILEDATA = 18;
        long ret = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
        amx_granted = (ret == 0);
        return amx_granted;
    }

    // AMX tile config structure — must be 64-byte aligned.
    struct __attribute__((aligned(64))) AmxTileConfig {
        uint8_t palette_id;
        uint8_t start_row;
        uint8_t reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];
    };

    // AMX BF16 batch inner-product distance: 1 query × count candidates (count <= 16).
    //
    // Swapped-tile layout to avoid expensive vnni repacking:
    //   Tile A (src1): M=16 rows × 64 bytes = candidates' 32 BF16 elements (row-major)
    //   Tile B (src2): 16 rows × 4 bytes = query's 32 BF16 as 16 pairs
    //   Tile C (dst):  M=16 rows × 4 bytes = FP32 dot-product accumulators
    //
    // IMPORTANT: Caller must call _tile_loadconfig with the correct AmxTileConfig
    // before calling this function, and _tile_release after all batches are done.
    //
    // candidate_ptrs: array of `count` pointers to BF16 vectors (each dim elements).
    // distances_out: output array of `count` distances (IP distance = 1 - dot).
    static void bf16InnerProductBatchAMXCore(
        const uint16_t* query_bf16,
        const uint16_t* const* candidate_ptrs,
        float* distances_out,
        size_t count,
        size_t dim)
    {
        _tile_zero(2);

        // Contiguous tile A buffer: 16 × 64 bytes = 1 KB, fits in L1 cache.
        alignas(64) uint8_t tile_a[16 * 64];
        // Zero-fill unused rows once
        if (count < 16) {
            memset(tile_a + count * 64, 0, (16 - count) * 64);
        }

        // Process dim in chunks of 32 BF16 elements
        for (size_t d = 0; d + 31 < dim; d += 32) {
            // Pack tile A: copy each candidate's 64-byte chunk (1 cache line each)
            for (size_t n = 0; n < count; n++) {
                memcpy(tile_a + n * 64, candidate_ptrs[n] + d, 64);
            }
            _tile_loadd(0, tile_a, 64);

            // Tile B: query's 32 BF16 as 16 rows of 4 bytes (stride=4, contiguous)
            _tile_loadd(1, query_bf16 + d, 4);

            _tile_dpbf16ps(2, 0, 1);
        }

        // Handle tail (dim % 32) with scalar
        size_t dim32 = dim & ~31ULL;
        float tail_dots[16] = {0};
        if (dim32 < dim) {
            for (size_t n = 0; n < count; n++) {
                for (size_t d = dim32; d < dim; d++) {
                    tail_dots[n] += bf16ToFloat(query_bf16[d]) * bf16ToFloat(candidate_ptrs[n][d]);
                }
            }
        }

        // Extract results: 16 rows × 1 FP32, stride=4 (contiguous)
        alignas(64) float tile_c[16];
        _tile_stored(2, tile_c, 4);

        for (size_t n = 0; n < count; n++) {
            distances_out[n] = 1.0f - (tile_c[n] + tail_dots[n]);
        }
    }
#endif // __AMX_BF16__

    // Batch distance computation into pre-allocated output buffer.
    // query_bf16: pre-converted BF16 query vector (dim elements).
    // candidate_ids: array of candidate internal IDs (count elements).
    // distances_out: pre-allocated output buffer (count elements).
    void computeBf16BatchDistancesInto(
        const uint16_t* query_bf16,
        const tableint* candidate_ids,
        dist_t* distances_out,
        size_t count) const {
        const size_t dim = getDistanceDim();
        const uint16_t* bf16_base = bf16_rowmajor_data_.data();

#if defined(__AMX_BF16__)
        if (use_amx_bf16_) {
            // Configure AMX tiles once for all batches in this call
            AmxTileConfig cfg;
            memset(&cfg, 0, sizeof(cfg));
            cfg.palette_id = 1;
            cfg.rows[0] = 16;     cfg.colsb[0] = 64;  // tile A: candidates
            cfg.rows[1] = 16;     cfg.colsb[1] = 4;   // tile B: query pairs
            cfg.rows[2] = 16;     cfg.colsb[2] = 4;   // tile C: results
            _tile_loadconfig(&cfg);

            const uint16_t* cand_ptrs[16];
            size_t i = 0;
            for (; i + 16 <= count; i += 16) {
                for (size_t j = 0; j < 16; j++) {
                    cand_ptrs[j] = bf16_base + candidate_ids[i + j] * dim;
                }
                float amx_dists[16];
                bf16InnerProductBatchAMXCore(query_bf16, cand_ptrs, amx_dists, 16, dim);
                for (size_t j = 0; j < 16; j++) {
                    distances_out[i + j] = static_cast<dist_t>(amx_dists[j]);
                }
            }
            if (i < count) {
                size_t remain = count - i;
                for (size_t j = 0; j < remain; j++) {
                    cand_ptrs[j] = bf16_base + candidate_ids[i + j] * dim;
                }
                float amx_dists[16];
                bf16InnerProductBatchAMXCore(query_bf16, cand_ptrs, amx_dists, remain, dim);
                for (size_t j = 0; j < remain; j++) {
                    distances_out[i + j] = static_cast<dist_t>(amx_dists[j]);
                }
            }

            _tile_release();
            return;
        }
#endif

        for (size_t i = 0; i < count; i++) {
            const uint16_t* candidate_bf16 = bf16_base + candidate_ids[i] * dim;
#ifdef USE_SSE
            if (i + 1 < count) {
                _mm_prefetch((const char*)(bf16_base + candidate_ids[i + 1] * dim), _MM_HINT_T0);
            }
#endif
#if defined(USE_AVX512) && defined(__AVX512BF16__)
            distances_out[i] = static_cast<dist_t>(
                bf16InnerProductDistanceAVX512BF16(query_bf16, candidate_bf16, dim));
#else
            distances_out[i] = static_cast<dist_t>(
                bf16InnerProductDistanceScalar(query_bf16, candidate_bf16, dim));
#endif
        }
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr,
        SearchProfileStats* search_profile = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            dist_t dist = computeDistanceSingle(data_point, ep_id);
            if (search_profile) {
                search_profile->recordDistanceComputation();
            }
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), getDataByInternalId(ep_id), dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        // Pre-allocate reusable buffers for the BFS loop (avoid per-expansion heap allocs).
        const bool use_bf16_batch = use_bf16_rowmajor_batch_distance_ && !bf16_rowmajor_data_.empty();
        const size_t max_neighbors = maxM0_;  // max neighbors at level 0
        std::vector<tableint> pending_ids_buf(max_neighbors);
        std::vector<dist_t> pending_dists_buf(max_neighbors);
        std::vector<tableint> profiled_ids_buf;
        if (search_profile) {
            profiled_ids_buf.resize(max_neighbors);
        }

        // Cache query BF16 conversion once for the entire search.
        const size_t dim = use_bf16_batch ? getDistanceDim() : 0;
        std::vector<uint16_t> query_bf16_buf;
        if (use_bf16_batch) {
            query_bf16_buf.resize(dim);
            convertFloatVectorToBf16(
                reinterpret_cast<const float*>(data_point),
                query_bf16_buf.data(), dim);
        }

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

            // Collect unvisited candidates into pre-allocated buffer (no heap alloc).
            size_t pending_count = 0;
            size_t profiled_count = 0;

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;
                    pending_ids_buf[pending_count++] = candidate_id;
                    if (search_profile) {
                        profiled_ids_buf[profiled_count++] = candidate_id;
                    }
                }
            }

            // Batch BF16 distance computation using cached query and pre-allocated buffers.
            if (use_bf16_batch && pending_count > 0) {
                computeBf16BatchDistancesInto(
                    query_bf16_buf.data(),
                    pending_ids_buf.data(),
                    pending_dists_buf.data(),
                    pending_count);
                if (search_profile) {
                    search_profile->recordDistanceComputation(pending_count);
                }
            }

            for (size_t i = 0; i < pending_count; i++) {
                tableint candidate_id = pending_ids_buf[i];
                char *currObj1 = (getDataByInternalId(candidate_id));
                dist_t dist;
                if (use_bf16_batch) {
                    dist = pending_dists_buf[i];
                } else {
                    dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (search_profile) {
                        search_profile->recordDistanceComputation();
                    }
                }

                bool flag_consider_candidate;
                if (!bare_bone_search && stop_condition) {
                    flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                } else {
                    flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                }

                if (flag_consider_candidate) {
                    candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                    offsetLevel0_,
                                    _MM_HINT_T0);
#endif

                    if (bare_bone_search ||
                        (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                        top_candidates.emplace(dist, candidate_id);
                        if (!bare_bone_search && stop_condition) {
                            stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                        }
                    }

                    bool flag_remove_extra = false;
                    if (!bare_bone_search && stop_condition) {
                        flag_remove_extra = stop_condition->should_remove_extra();
                    } else {
                        flag_remove_extra = top_candidates.size() > ef;
                    }
                    while (flag_remove_extra) {
                        tableint id = top_candidates.top().second;
                        top_candidates.pop();
                        if (!bare_bone_search && stop_condition) {
                            stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                    }

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }

            if (search_profile) {
                search_profile->recordBaseLayerExpansion(
                    std::vector<tableint>(profiled_ids_buf.begin(),
                                          profiled_ids_buf.begin() + profiled_count));
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
        if (use_bf16_rowmajor_batch_distance_) {
            rebuildBf16RowmajorData();
        }
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        if (use_bf16_rowmajor_batch_distance_) {
            rebuildBf16RowmajorData();
        }

        input.close();

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        if (fp32_vectors_released_) {
            throw std::runtime_error("Cannot retrieve FP32 vectors after storage has been released");
        }
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if (fp32_vectors_released_) {
            throw std::runtime_error("Cannot add points after FP32 vector storage has been released");
        }
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        if (fp32_vectors_released_) {
            throw std::runtime_error("Cannot update points after FP32 vector storage has been released");
        }
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);
        refreshBf16RowmajorElement(internalId);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);
        refreshBf16RowmajorElement(cur_c);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }


 private:
    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnnInternal(
        const void *query_data,
        size_t k,
        BaseFilterFunctor* isIdAllowed,
        SearchProfileStats* search_profile) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = computeDistanceSingle(query_data, enterpoint_node_);
        if (search_profile) {
            search_profile->recordDistanceComputation();
        }

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = computeDistanceSingle(query_data, cand);
                    if (search_profile) {
                        search_profile->recordDistanceComputation();
                    }

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed, nullptr, search_profile);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed, nullptr, search_profile);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

 public:
    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        return searchKnnInternal(query_data, k, isIdAllowed, nullptr);
    }

    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnnProfiled(
        const void *query_data,
        size_t k,
        SearchProfileStats& search_profile,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        search_profile = SearchProfileStats();
        return searchKnnInternal(query_data, k, isIdAllowed, &search_profile);
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = computeDistanceSingle(query_data, enterpoint_node_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = computeDistanceSingle(query_data, cand);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }
};
}  // namespace hnswlib
