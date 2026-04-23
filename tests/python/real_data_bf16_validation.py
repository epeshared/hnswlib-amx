#!/usr/bin/env python3
"""
Real data validation: FP32 vs BF16-AVX512 on d256/1M and d1024/1M datasets.
Measures: recall@10, QPS (single-thread), memory savings after FP32 release.
"""
import sys, os, time
import numpy as np

# ensure hnswlib from workspace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import hnswlib

DATA_DIR = "/mnt/nvme2/xtang/vdb-workspace/data"

CONFIGS = [
    {
        "name": "d256_1M",
        "db": os.path.join(DATA_DIR, "db_d256_nb1000000_seed1235.npy"),
        "queries": os.path.join(DATA_DIR, "queries_d256_nq128_seed1234.npy"),
        "dim": 256, "M": 32, "ef_construction": 80, "ef_search": [64, 128, 256, 512],
        "k": 10,
    },
    {
        "name": "d1024_1M",
        "db": os.path.join(DATA_DIR, "db_d1024_nb1000000_seed1235.npy"),
        "queries": os.path.join(DATA_DIR, "queries_d1024_nq10000_seed1234.npy"),
        "dim": 1024, "M": 32, "ef_construction": 80, "ef_search": [64, 128, 256, 512],
        "k": 10,
    },
]

def load_npy(path):
    print(f"  Loading {os.path.basename(path)} ...", end=" ", flush=True)
    arr = np.load(path).astype(np.float32)
    print(f"shape={arr.shape}")
    return arr

def normalize(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

def compute_gt_ip(db, queries, k):
    """Brute-force ground truth for inner product (top-k largest IP)."""
    nq = queries.shape[0]
    gt = np.empty((nq, k), dtype=np.int64)
    batch = 1000  # process queries in batches to limit memory
    for i in range(0, nq, batch):
        qb = queries[i:i+batch]
        scores = qb @ db.T  # (batch, nb)
        gt[i:i+batch] = np.argpartition(-scores, k, axis=1)[:, :k]
        # sort within top-k
        for j in range(qb.shape[0]):
            idx = gt[i+j]
            order = np.argsort(-scores[j, idx])
            gt[i+j] = idx[order]
    return gt

def recall_at_k(results, gt, k):
    """results and gt: (nq, k)"""
    nq = results.shape[0]
    hits = 0
    for i in range(nq):
        hits += len(set(results[i, :k].tolist()) & set(gt[i, :k].tolist()))
    return hits / (nq * k)

def bench_qps(index, queries, k, num_threads=1, warmup=1, repeats=3):
    """Measure QPS: single-thread by default."""
    index.set_num_threads(num_threads)
    nq = queries.shape[0]
    # warmup
    for _ in range(warmup):
        index.knn_query(queries, k=k)
    # timed
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        labels, dists = index.knn_query(queries, k=k)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    best = min(times)
    qps = nq / best
    return labels, dists, qps

def run_config(cfg):
    print(f"\n{'='*60}")
    print(f"  Config: {cfg['name']}")
    print(f"{'='*60}")

    db = load_npy(cfg["db"])
    queries = load_npy(cfg["queries"])
    dim = cfg["dim"]
    nb = db.shape[0]
    nq = queries.shape[0]
    k = cfg["k"]

    assert db.shape[1] == dim
    assert queries.shape[1] == dim

    # Normalize for IP
    print("  Normalizing vectors ...", flush=True)
    db = normalize(db)
    queries = normalize(queries)

    # Brute-force ground truth
    print(f"  Computing brute-force GT (nq={nq}, nb={nb}, k={k}) ...", flush=True)
    t0 = time.perf_counter()
    gt = compute_gt_ip(db, queries, k)
    print(f"  GT done in {time.perf_counter()-t0:.1f}s")

    # Build HNSW index
    print(f"  Building HNSW index (M={cfg['M']}, ef_c={cfg['ef_construction']}) ...", flush=True)
    index = hnswlib.Index(space='ip', dim=dim)
    index.init_index(max_elements=nb, M=cfg["M"], ef_construction=cfg["ef_construction"])
    index.set_num_threads(os.cpu_count())
    t0 = time.perf_counter()
    index.add_items(db, np.arange(nb))
    build_time = time.perf_counter() - t0
    print(f"  Index built in {build_time:.1f}s")

    results = []

    for ef in cfg["ef_search"]:
        index.set_ef(ef)
        print(f"\n  --- ef_search={ef} ---")

        # FP32 baseline
        labels_fp32, _, qps_fp32 = bench_qps(index, queries, k)
        r_fp32 = recall_at_k(labels_fp32, gt, k)
        print(f"  FP32:         recall@{k}={r_fp32:.4f}  QPS={qps_fp32:,.0f}")

        # BF16 (with FP32 still present)
        index.set_bf16_rowmajor_batch_distance(True)
        labels_bf16, _, qps_bf16 = bench_qps(index, queries, k)
        r_bf16 = recall_at_k(labels_bf16, gt, k)
        speedup = qps_bf16 / qps_fp32
        print(f"  BF16+FP32:    recall@{k}={r_bf16:.4f}  QPS={qps_bf16:,.0f}  speedup={speedup:.2f}x")

        # BF16-only (FP32 released) — only do this at the last ef to avoid breaking other iterations
        if ef == cfg["ef_search"][-1]:
            index.release_fp32_vector_storage()
            assert index.is_fp32_vectors_released()
            labels_rel, _, qps_rel = bench_qps(index, queries, k)
            r_rel = recall_at_k(labels_rel, gt, k)
            speedup_rel = qps_rel / qps_fp32
            print(f"  BF16-only:    recall@{k}={r_rel:.4f}  QPS={qps_rel:,.0f}  speedup={speedup_rel:.2f}x")
            results.append((ef, r_fp32, qps_fp32, r_bf16, qps_bf16, r_rel, qps_rel))
        else:
            index.set_bf16_rowmajor_batch_distance(False)
            results.append((ef, r_fp32, qps_fp32, r_bf16, qps_bf16, None, None))

    # Summary table
    print(f"\n  {'='*70}")
    print(f"  Summary: {cfg['name']} (nb={nb}, nq={nq}, k={k})")
    print(f"  {'ef':>6} | {'FP32 R@k':>9} {'FP32 QPS':>10} | {'BF16 R@k':>9} {'BF16 QPS':>10} {'speedup':>8} | {'BF16-only R@k':>14} {'BF16-only QPS':>14}")
    print(f"  {'-'*6}-+-{'-'*9}-{'-'*10}-+-{'-'*9}-{'-'*10}-{'-'*8}-+-{'-'*14}-{'-'*14}")
    for row in results:
        ef, r32, q32, r16, q16, r_rel, q_rel = row
        su = q16/q32
        rel_str = f"{r_rel:.4f}" if r_rel is not None else "-"
        qrel_str = f"{q_rel:,.0f}" if q_rel is not None else "-"
        print(f"  {ef:>6} | {r32:>9.4f} {q32:>10,.0f} | {r16:>9.4f} {q16:>10,.0f} {su:>7.2f}x | {rel_str:>14} {qrel_str:>14}")
    print()

    return results

if __name__ == "__main__":
    all_results = {}
    for cfg in CONFIGS:
        all_results[cfg["name"]] = run_config(cfg)
    print("\n=== All real data validations complete ===")
