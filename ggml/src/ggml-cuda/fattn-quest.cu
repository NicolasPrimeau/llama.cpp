#include "fattn-common.cuh"
#include "fattn-quest.cuh"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

// Quest (Tang et al., 2024, arXiv:2406.10774) for single-token decode. Each page of P keys keeps an element-wise min
// and max per KV head; sum_i max(q_i*min_i, q_i*max_i) bounds q.k for every key in the page. Per step:
//   1. quest_score:  refresh stale page summaries and write each query head's bound for every page
//   2. quest_select: per selection group (query head, or KV head when shared), the sink page, the top pages and the
//                    two most recent pages, in ascending order
//   3. quest_gather: copy the selected K/V rows and their mask values into a dense buffer per query head
//   4. the vector kernel attends over the gathered cells, with query heads as sequences so each gets its own mask
// Configured through the same GGML_QUEST_* environment variables as the CPU backend.

struct quest_config {
    int  budget;
    int  page;
    int  skip;
    bool per_head;
};

static const quest_config & quest_get_config() {
    static const quest_config cfg = [] {
        auto env = [](const char * name, int def) {
            const char * s = getenv(name);
            return s ? atoi(s) : def;
        };
        return quest_config{ env("GGML_QUEST_BUDGET", 0), std::max(16, env("GGML_QUEST_PAGE", 16)), env("GGML_QUEST_SKIP", 2), env("GGML_QUEST_PER_HEAD", 1) != 0 };
    }();
    return cfg;
}

// Page summaries persist across decode steps, keyed by the K cache buffer and sized for the whole buffer, so they are
// allocated once, before CUDA graph capture starts. A page is resummarized when its fingerprint changes.
struct quest_pages {
    half     * kmin    = nullptr; // [page][kv head][DK]
    half     * kmax    = nullptr;
    uint64_t * fp      = nullptr; // [page][kv head]
    int64_t    n_pages = 0;
    int64_t    n_head  = 0;
    int64_t    dk      = 0;
};

static std::mutex quest_mutex;
static std::unordered_map<const void *, quest_pages> quest_cache[GGML_CUDA_MAX_DEVICES];

static int quest_layer(const ggml_tensor * k) {
    const char * name = k->view_src ? k->view_src->name : k->name;
    const char * p    = strstr(name, "k_l");
    return p ? atoi(p + 3) : -1;
}

template <ggml_type type>
static __device__ __forceinline__ float quest_load_k(const char * row, const int i) {
    if constexpr (type == GGML_TYPE_F16) {
        return __half2float(((const half *) row)[i]);
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        const block_q8_0 * b = (const block_q8_0 *) row + i/QK8_0;
        return __half2float(b->d) * b->qs[i % QK8_0];
    } else {
        static_assert(type == GGML_TYPE_Q4_0, "unsupported K type");
        const block_q4_0 * b = (const block_q4_0 *) row + i/QK4_0;
        const int          j = i % QK4_0;
        const int          q = j < QK4_0/2 ? b->qs[j] & 0x0F : b->qs[j - QK4_0/2] >> 4;
        return __half2float(b->d) * (q - 8);
    }
}

// first 8 bytes of a row, read as 16-bit words since rows of every supported type are only 2-byte aligned
static __device__ __forceinline__ uint64_t quest_row_prefix(const char * row) {
    const uint16_t * p = (const uint16_t *) row;
    return uint64_t(p[0]) | uint64_t(p[1]) << 16 | uint64_t(p[2]) << 32 | uint64_t(p[3]) << 48;
}

static __device__ __forceinline__ float quest_block_sum(float x, float * buf) {
    x = warp_reduce_sum(x);
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        buf[warp] = x;
    }
    __syncthreads();
    float sum = 0.0f;
    for (int w = 0; w < (int) (blockDim.x / WARP_SIZE); w++) {
        sum += buf[w];
    }
    __syncthreads();
    return sum;
}

// grid: (pages, kv heads). The summarized rows run from the first cell of the page to its last visible one.
template <ggml_type type>
__launch_bounds__(128, 1)
static __global__ void quest_score(
        const char * __restrict__ K, const half * __restrict__ mask, const float * __restrict__ Q,
        half * __restrict__ kmin_all, half * __restrict__ kmax_all, uint64_t * __restrict__ fp_all,
        float * __restrict__ scores, int * __restrict__ n_used,
        const int DK, const int P, const int nek1, const int nek2, const int rk2, const int n_pages,
        const int64_t nbk1, const int64_t nbk2, const int64_t nbq2, const float scale) {
    const int pg  = blockIdx.x;
    const int g   = blockIdx.y;
    const int tid = threadIdx.x;
    const int c0  = pg*P;
    const int c1  = min(c0 + P, nek1);

    __shared__ int   last;
    __shared__ float buf[128/WARP_SIZE];

    if (tid == 0) {
        last = -1;
    }
    __syncthreads();
    for (int c = c0 + tid; c < c1; c += blockDim.x) {
        if (isfinite(__half2float(mask[c]))) {
            atomicMax(&last, c);
        }
    }
    __syncthreads();

    if (last < 0) {
        for (int j = tid; j < rk2; j += blockDim.x) {
            scores[int64_t(g*rk2 + j)*n_pages + pg] = -INFINITY;
        }
        return;
    }
    if (tid == 0 && g == 0) {
        atomicMax(n_used, last + 1);
    }

    const char * k0 = K + int64_t(c0)*nbk1 + g*nbk2;
    const int64_t  idx = int64_t(pg)*nek2 + g;
    const uint64_t a   = quest_row_prefix(k0);
    const uint64_t b   = quest_row_prefix(K + int64_t(last)*nbk1 + g*nbk2);
    const uint64_t f   = (a ^ (b << 1 | b >> 63)) | 1; // never 0, so a zeroed entry is always stale

    half * kmin = kmin_all + idx*DK;
    half * kmax = kmax_all + idx*DK;

    if (fp_all[idx] != f) {
        // rounding to F16 is monotonic, so the min/max of the rounded keys is the rounded min/max
        for (int i = tid; i < DK; i += blockDim.x) {
            float lo = INFINITY;
            float hi = -INFINITY;
            for (int c = c0; c <= last; c++) {
                const float x = quest_load_k<type>(k0 + int64_t(c - c0)*nbk1, i);
                lo = fminf(lo, x);
                hi = fmaxf(hi, x);
            }
            kmin[i] = __float2half(lo);
            kmax[i] = __float2half(hi);
        }
        __syncthreads();
        if (tid == 0) {
            fp_all[idx] = f;
        }
    }

    for (int j = 0; j < rk2; j++) {
        const int     h = g*rk2 + j;
        const float * q = (const float *) ((const char *) Q + h*nbq2);
        float bound = 0.0f;
        for (int i = tid; i < DK; i += blockDim.x) {
            const float qi = q[i]*scale;
            bound += qi > 0.0f ? qi*__half2float(kmax[i]) : qi*__half2float(kmin[i]);
        }
        bound = quest_block_sum(bound, buf);
        if (tid == 0) {
            scores[int64_t(h)*n_pages + pg] = bound;
        }
    }
}

// order-preserving map of a score to an unsigned key, with -inf (invisible page) mapped to 0
static __device__ __forceinline__ uint32_t quest_key(const float s) {
    if (s == -INFINITY) {
        return 0;
    }
    const uint32_t u = __float_as_uint(s);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// appends the indices in [i0, i1) for which pred holds to out[n], in order; every thread must call it
template <typename F>
static __device__ __forceinline__ void quest_compact(int i0, int i1, F pred, int32_t * out, int & n, int * warp_offs) {
    const int warp   = threadIdx.x / WARP_SIZE;
    const int lane   = threadIdx.x % WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;
    for (int base = i0; base < i1; base += blockDim.x) {
        const int      i    = base + threadIdx.x;
        const bool     take = i < i1 && pred(i);
        const uint32_t bits = __ballot_sync(0xFFFFFFFF, take);
        if (lane == 0) {
            warp_offs[warp] = __popc(bits);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int offs = 0;
            for (int w = 0; w < nwarps; w++) {
                const int cnt = warp_offs[w];
                warp_offs[w] = offs;
                offs += cnt;
            }
            warp_offs[nwarps] = offs;
        }
        __syncthreads();
        if (take) {
            out[n + warp_offs[warp] + __popc(bits & ((1u << lane) - 1))] = i;
        }
        n += warp_offs[nwarps];
        __syncthreads();
    }
}

// grid: selection groups, block: 1024 threads. Writes n_keep page indices per group, padded with -1.
__launch_bounds__(1024, 1)
static __global__ void quest_select(
        const float * __restrict__ scores, const int * __restrict__ n_used_ptr, int32_t * __restrict__ sel,
        const int n_pages, const int P, const int n_keep, const int rk_sel) {
    const int grp = blockIdx.x;
    __shared__ int warp_offs[1024/WARP_SIZE + 1];

    const float * sc = scores + int64_t(grp*rk_sel)*n_pages;
    auto key = [&](const int pg) {
        uint32_t k = 0;
        for (int j = 0; j < rk_sel; j++) {
            k = max(k, quest_key(sc[int64_t(j)*n_pages + pg]));
        }
        return k;
    };

    const int n_vis    = min((*n_used_ptr + P - 1)/P, n_pages);
    const int n_recent = min(n_vis, 2);
    const int c1       = n_vis - n_recent; // candidates are pages [1, c1)
    const int n_top    = max(0, n_keep - 1 - n_recent);

    int n_cand = 0;
    for (int base = 1; base < c1; base += blockDim.x) {
        const int pg = base + threadIdx.x;
        n_cand += __syncthreads_count(pg < c1 && key(pg) != 0);
    }

    // the n_top-th largest key, found bit by bit from the top: thr is the largest value with at least n_top keys >= it
    uint32_t thr = 1;
    if (n_cand > n_top) {
        thr = 0;
        for (int bit = 31; bit >= 0; bit--) {
            const uint32_t t = thr | (1u << bit);
            int cnt = 0;
            for (int base = 1; base < c1; base += blockDim.x) {
                const int pg = base + threadIdx.x;
                cnt += __syncthreads_count(pg < c1 && key(pg) >= t);
            }
            if (cnt >= n_top) {
                thr = t;
            }
        }
    }

    // keys above the threshold are all taken, ties at it fill the remaining slots in page order up to tie_last
    int n_above = 0;
    for (int base = 1; base < c1; base += blockDim.x) {
        const int pg = base + threadIdx.x;
        n_above += __syncthreads_count(pg < c1 && key(pg) > thr);
    }
    int tie_last = c1; // the smallest page index with n_top - n_above ties at or before it
    if (n_cand > n_top) {
        int lo = 1;
        while (lo < tie_last) {
            const int mid = (lo + tie_last)/2;
            int cnt = 0;
            for (int base = 1; base <= mid; base += blockDim.x) {
                const int pg = base + threadIdx.x;
                cnt += __syncthreads_count(pg <= mid && key(pg) == thr);
            }
            if (cnt >= n_top - n_above) {
                tie_last = mid;
            } else {
                lo = mid + 1;
            }
        }
    }

    int32_t * out = sel + int64_t(grp)*n_keep;
    int       n   = 0;
    if (c1 > 0) {
        if (threadIdx.x == 0) {
            out[0] = 0;
        }
        n = 1;
    }
    if (n_cand > n_top) {
        quest_compact(1, c1, [&](int pg) { const uint32_t k = key(pg); return k > thr || (k == thr && pg <= tie_last); }, out, n, warp_offs);
    } else {
        quest_compact(1, c1, [&](int pg) { return key(pg) != 0; }, out, n, warp_offs);
    }
    quest_compact(c1, n_vis, [](int) { return true; }, out, n, warp_offs);
    for (int i = n + threadIdx.x; i < n_keep; i += blockDim.x) {
        out[i] = -1;
    }
}

// grid: (selected pages, query heads). Copies whole rows as 16-bit words; cells of padding pages are zeroed and
// masked out, so they add exp(-inf) = 0 and no NaN.
__launch_bounds__(128, 1)
static __global__ void quest_gather(
        const char * __restrict__ K, const char * __restrict__ V, const half * __restrict__ mask,
        const int32_t * __restrict__ sel, char * __restrict__ Kg, char * __restrict__ Vg, half * __restrict__ mask_g,
        const int P, const int nek1, const int rk2, const int sel_div, const int n_keep,
        const int64_t nbk1, const int64_t nbk2, const int64_t nbv1, const int64_t nbv2, const int row_k, const int row_v) {
    const int j  = blockIdx.x;
    const int h  = blockIdx.y;
    const int g  = h/rk2;
    const int pg = sel[int64_t(h/sel_div)*n_keep + j];

    const int64_t n_cells = int64_t(n_keep)*P;
    const int64_t o0      = int64_t(h)*n_cells + int64_t(j)*P;

    for (int t = threadIdx.x; t < P; t += blockDim.x) {
        const int c = pg*P + t;
        mask_g[o0 + t] = pg >= 0 && c < nek1 ? mask[c] : __float2half(-INFINITY);
    }

    auto copy = [&](const char * src, char * dst, const int64_t nb1, const int64_t nb2, const int row) {
        const int words = row/2;
        for (int i = threadIdx.x; i < P*words; i += blockDim.x) {
            const int t = i / words;
            const int w = i % words;
            const int c = pg*P + t;
            uint16_t * d = (uint16_t *) (dst + (o0 + t)*row) + w;
            *d = pg >= 0 && c < nek1 ? ((const uint16_t *) (src + int64_t(c)*nb1 + g*nb2))[w] : 0;
        }
    };
    copy(K, Kg, nbk1, nbk2, row_k);
    copy(V, Vg, nbv1, nbv2, row_v);
}

static quest_pages * quest_get_pages(const int device, const ggml_tensor * K, cudaStream_t stream) {
    const ggml_tensor * base = K->view_src ? K->view_src : K;
    const int64_t DK      = K->ne[0];
    const int64_t nek2    = K->ne[2];
    const int64_t n_cells = ggml_nbytes(base)/K->nb[1];
    const int64_t P       = quest_get_config().page;
    const int64_t n_pages = (n_cells + P - 1)/P;

    std::lock_guard<std::mutex> lock(quest_mutex);
    quest_pages & qp = quest_cache[device][K->data];
    if (qp.n_pages >= n_pages && qp.n_head == nek2 && qp.dk == DK) {
        return &qp;
    }

    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        return nullptr; // allocating would not be replayed by the graph
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(qp.kmin));
    CUDA_CHECK(cudaFree(qp.kmax));
    CUDA_CHECK(cudaFree(qp.fp));
    CUDA_CHECK(cudaMalloc(&qp.kmin, n_pages*nek2*DK*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&qp.kmax, n_pages*nek2*DK*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&qp.fp,   n_pages*nek2*sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(qp.fp, 0, n_pages*nek2*sizeof(uint64_t)));
    qp.n_pages = n_pages;
    qp.n_head  = nek2;
    qp.dk      = DK;
    return &qp;
}

bool ggml_cuda_flash_attn_ext_quest(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_fattn_launch_t attend) {
    const quest_config & cfg = quest_get_config();
    if (cfg.budget <= 0) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int64_t DK   = K->ne[0];
    const int64_t DV   = V->ne[0];
    const int64_t nek1 = K->ne[1];
    const int64_t nek2 = K->ne[2];
    const int64_t neq2 = Q->ne[2];
    const int64_t P    = cfg.page;

    // sinks are indexed by head, which the gathered layout moves to the sequence dimension
    const bool k_type_ok = K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_Q4_0;
    if (Q->ne[1] != 1 || Q->ne[3] != 1 || Q->type != GGML_TYPE_F32 || !mask || mask->ne[0] < K->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1 ||
            sinks || max_bias != 0.0f || logit_softcap != 0.0f || !k_type_ok || DK != DV || DK % 64 != 0 || DK > 256 ||
            V->ne[2] != nek2 || neq2 % nek2 != 0 || K->ne[3] != 1 || V->ne[3] != 1 ||
            K->nb[0] != ggml_type_size(K->type) || V->nb[0] != ggml_type_size(V->type)) {
        return false;
    }
    const int il = quest_layer(K);
    if (il >= 0 && il < cfg.skip) {
        return false;
    }

    // whole pages up to the budget, rounded so the gathered length is a multiple of the kernels' KV stride
    const int64_t n_pages   = (nek1 + P - 1)/P;
    const int64_t page_mult = std::max<int64_t>(1, FATTN_KQ_STRIDE/P);
    const int64_t n_keep    = GGML_PAD(std::max<int64_t>(cfg.budget/P, 3), page_mult);
    if (n_pages <= n_keep + 1 || (n_keep*P) % FATTN_KQ_STRIDE != 0) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    quest_pages * qp = quest_get_pages(ctx.device, K, stream);
    if (!qp) {
        return false;
    }

    const int64_t rk2     = neq2/nek2;
    const int64_t n_grp   = cfg.per_head ? neq2 : nek2;
    const int64_t n_cells = n_keep*P;
    const int64_t row_k   = ggml_row_size(K->type, DK);
    const int64_t row_v   = ggml_row_size(V->type, DV);

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float>   scores(pool, neq2*n_pages);
    ggml_cuda_pool_alloc<int32_t> sel(pool, n_grp*n_keep + 1);
    ggml_cuda_pool_alloc<char>    Kg(pool, neq2*n_cells*row_k);
    ggml_cuda_pool_alloc<char>    Vg(pool, neq2*n_cells*row_v);
    ggml_cuda_pool_alloc<half>    mask_g(pool, neq2*n_cells);
    int32_t * n_used = sel.get() + n_grp*n_keep;

    CUDA_CHECK(cudaMemsetAsync(n_used, 0, sizeof(int32_t), stream));

    const dim3 score_grid(n_pages, nek2);
    auto score = K->type == GGML_TYPE_F16 ? quest_score<GGML_TYPE_F16> :
                 K->type == GGML_TYPE_Q8_0 ? quest_score<GGML_TYPE_Q8_0> : quest_score<GGML_TYPE_Q4_0>;
    score<<<score_grid, 128, 0, stream>>>(
        (const char *) K->data, (const half *) mask->data, (const float *) Q->data,
        qp->kmin, qp->kmax, qp->fp, scores.get(), n_used,
        DK, P, nek1, nek2, rk2, n_pages, K->nb[1], K->nb[2], Q->nb[2], scale);
    CUDA_CHECK(cudaGetLastError());

    quest_select<<<n_grp, 1024, 0, stream>>>(scores.get(), n_used, sel.get(), n_pages, P, n_keep, cfg.per_head ? 1 : rk2);
    CUDA_CHECK(cudaGetLastError());

    const dim3 gather_grid(n_keep, neq2);
    quest_gather<<<gather_grid, 128, 0, stream>>>(
        (const char *) K->data, (const char *) V->data, (const half *) mask->data, sel.get(),
        Kg.get(), Vg.get(), mask_g.get(), P, nek1, rk2, cfg.per_head ? 1 : rk2, n_keep,
        K->nb[1], K->nb[2], V->nb[1], V->nb[2], row_k, row_v);
    CUDA_CHECK(cudaGetLastError());

    // the gathered problem: one query per sequence, one sequence per query head, each with its own K/V/mask
    ggml_tensor q = *Q;
    q.ne[1] = 1; q.ne[2] = 1; q.ne[3] = neq2;
    q.nb[3] = Q->nb[2];

    auto dense = [&](const ggml_tensor * t, void * data, const int64_t row) {
        ggml_tensor r = *t;
        r.data     = data;
        r.view_src = nullptr;
        r.ne[1] = n_cells; r.ne[2] = 1; r.ne[3] = neq2;
        r.nb[1] = row; r.nb[2] = row*n_cells; r.nb[3] = row*n_cells;
        return r;
    };
    ggml_tensor k = dense(K, Kg.get(), row_k);
    ggml_tensor v = dense(V, Vg.get(), row_v);

    ggml_tensor m = *mask;
    m.data     = mask_g.get();
    m.view_src = nullptr;
    m.ne[0] = n_cells; m.ne[1] = 1; m.ne[2] = 1; m.ne[3] = neq2;
    m.nb[1] = n_cells*sizeof(half); m.nb[2] = m.nb[1]; m.nb[3] = m.nb[1];

    ggml_tensor d = *dst;
    d.ne[1] = 1; d.ne[2] = 1; d.ne[3] = neq2;
    d.nb[1] = d.nb[0]*DV; d.nb[2] = d.nb[1]; d.nb[3] = d.nb[1];
    d.src[0] = &q;
    d.src[1] = &k;
    d.src[2] = &v;
    d.src[3] = &m;
    d.src[4] = nullptr;

    attend(ctx, &d);
    return true;
}
