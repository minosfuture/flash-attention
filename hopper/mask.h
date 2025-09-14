/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;

template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, bool SwapAB=false>
struct Mask {

    static_assert(!(PackGQA && SwapAB), "Cannot be both PackGQA and SwapAB");

    int const thread_idx;
    int const seqlen_q, seqlen_k;
    int const window_size_left, window_size_right, sink_token_length;
    cutlass::FastDivmod const qhead_per_khead_divmod;
    int const cp_world_size, cp_rank;

    CUTLASS_DEVICE
    Mask(const int thread_idx, const int seqlen_q, const int seqlen_k,
         const int window_size_left, const int window_size_right, const int sink_token_length,
         cutlass::FastDivmod const &qhead_per_khead_divmod,
         const int cp_world_size = 1, const int cp_rank = 0)
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , sink_token_length(sink_token_length)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
        , cp_world_size(cp_world_size)
        , cp_rank(cp_rank)
    {
        printf("%3d: Mask: seqlen_q=%d, seqlen_k=%d, "
               "window_size_left=%d, window_size_right=%d, sink_token_length=%d, "
               "qhead_per_khead_divmod=%d, cp_world_size=%d, cp_rank=%d, threadIdx(%d,%d)\n", thread_idx,
               seqlen_q, seqlen_k, window_size_left, window_size_right,
               sink_token_length, qhead_per_khead_divmod, cp_world_size, cp_rank,
               threadIdx.x, threadIdx.y);
    };

    template <bool Seqlenk_mask=false, bool Causal_mask=false, bool Local_mask=false,
        typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
        static_assert(!(Causal_mask && Local_mask), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        if (!Seqlenk_mask && !Causal_mask && !Local_mask) {
            printf("%3d: Mask::apply EARLY RETURN: No masks enabled\n", thread_idx);
            return;
        }

        printf("%3d: Mask::apply START: thread_idx=%d, m_block=%d, n_block=%d, Seqlenk_mask=%d, Causal_mask=%d, Local_mask=%d\n",
               thread_idx, m_block, n_block, (int)Seqlenk_mask, (int)Causal_mask, (int)Local_mask);

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});


        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;
        printf("%3d: Mask::apply: SwapAB=%d, Row=%d, Col=%d\n", thread_idx, (int)SwapAB, Row, Col);

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));
        //if (thread_idx == 1) {
        //  cute::print(thread_mma);
        //  cute::print(thread0_mma);
        //  cute::print(tScS_rowcol);
        //  cute::print(tSrS_rowcol);
        //  cute::print(t0ScS_rowcol);
        //}

        //printf("%d: Mask::apply: tSrS_rowcol size=[%d,%d], tScS_rowcol size=[%d,%d]\n",
        //       (int)size<0>(tSrS_rowcol), (int)size<1>(tSrS_rowcol),
        //       (int)size<0>(tScS_rowcol), (int)size<1>(tScS_rowcol));

        // We want to use the col indices of thread0 to compare, since that is known at compile time.
        // So we subtract the limit by the first col index of this thread (get<Col>(tScS_rowcol(_0{}, _0{})))
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
        printf("%3d: Mask::apply: thread_col_offset=%d, seqlenk_col_limit=%d\n", thread_idx, thread_col_offset, seqlenk_col_limit);
        if constexpr (!Causal_mask && !Local_mask) {
            if constexpr (Seqlenk_mask) {  // Just masking based on col
                printf("%d: Mask::apply: Seqlenk masking only\n", thread_idx);
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    int col_idx = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                    printf("%d: Seqlenk mask: n=%d, col_idx=%d, seqlenk_col_limit=%d, mask=%s\n",
                           thread_idx, n, col_idx, seqlenk_col_limit, (col_idx >= seqlenk_col_limit) ? "TRUE" : "FALSE");
                    if (col_idx >= seqlenk_col_limit) {
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        } else {  // mask based on both row and col
            if constexpr (!SwapAB) {
                // If PackGQA, we split the work of compute divmod among threads in the same row
                static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
                static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
                static_assert(!PackGQA || CUTE_STATIC_V(size<0>(tSrS_rowcol)) <= kMmaThreadsPerRow);
                int mma_m_idx;
                // Might get OOB but it's ok since we'll check it later
                if constexpr (PackGQA) {
                    mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
                }
                int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
                if constexpr (Causal_mask) {
                    printf("%d: Mask::apply: Starting causal masking loop, causal_row_offset=%d\n", thread_idx, causal_row_offset);
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int const col_limit_right = !Seqlenk_mask
                            ? row_idx + causal_row_offset
                            : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int col_idx = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                            int local_k_idx = int(get<Col>(t0ScS_rowcol(_0{}, n))) + get<Col>(tScS_rowcol(_0{}, _0{})) + n_block * kBlockN;
                            int abs_k_idx = local_k_idx * cp_world_size + cp_rank;
                            int abs_q_idx = row_idx + cp_world_size * seqlen_k - seqlen_q;
                            bool masked = false;
                            if (cp_world_size > 1) {
                                // For DCP: compute absolute K position from local K index
                                if (abs_k_idx > abs_q_idx || abs_k_idx > cp_world_size * seqlen_k) {
                                    tSrS_rowcol(m, n) = -INFINITY;
                                    masked = true;
                                }
                            } else {
                                // Original non-DCP logic
                // seqlen_k - n_block * kBlockN - thread_col_offset
                                //int col_idx = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                                // col_idx >= row_idx + causal_row_offset
                // col_idx >= get<Row>(tScS_rowcol) + 1 + seqlen_k - seqlen_q - thread_col_offset
                // col_idx >= get<Row>(tScS_rowcol(m, _0{})) + 1 + seqlen_k - seqlen_q - get<Col>(tScS_rowcol(_0{}, _0{}))
                // int(get<Col>(t0ScS_rowcol(_0{}, n))) >= get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM + 1 + seqlen_k - n_block * kBlockM - seqlen_q - get<Col>(tScS_rowcol(_0{}, _0{}))
                // int(get<Col>(t0ScS_rowcol(_0{}, n))) + n_block * kBlockM >= get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM + 1 + seqlen_k - seqlen_q - get<Col>(tScS_rowcol(_0{}, _0{}))
                // int(get<Col>(t0ScS_rowcol(_0{}, n))) + get<Col>(tScS_rowcol(_0{}, _0{})) + n_block * kBlockM >= get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM + 1 + seqlen_k - seqlen_q
                                if (col_idx >= col_limit_right) {
                                    tSrS_rowcol(m, n) = -INFINITY;
                                    masked = true;
                                }
                            }
                            printf("%3d: mask:%s "
                                    "m_block=%d, n_block=%d, m=%d, n=%d, col=%d, limit=%d, tco=%d, abs_k=%d, abs_q=%d, ridx=%d, cro=%d, scl=%d, "
                                    "local_k=%d, dcp_abs_k=%d, dcp_abs_q=%d\n",
                                    thread_idx, masked ? "T" : "F",
                                    m_block, n_block, m, n, col_idx, col_limit_right, thread_col_offset, int(get<Col>(t0ScS_rowcol(_0{}, n))) + get<Col>(tScS_rowcol(_0{}, _0{})) + n_block * kBlockM, get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM + 1 + seqlen_k - seqlen_q, row_idx, causal_row_offset, seqlenk_col_limit,
                                    local_k_idx, abs_k_idx, abs_q_idx);
                        }
                    }
                } else {
                    int const local_row_offset_right = causal_row_offset + window_size_right;
                    int const local_row_offset_left = causal_row_offset - 1 - window_size_left;
                    int const col_limit_sink = sink_token_length - n_block * kBlockN;
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int const col_limit_right = !Seqlenk_mask
                            ? row_idx + local_row_offset_right
                            : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
                        int const col_limit_left = row_idx + local_row_offset_left;
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int const col_idx = int(get<Col>(t0ScS_rowcol(m, n)));
                            if (col_idx >= col_limit_right || (col_idx < col_limit_left && col_idx >= col_limit_sink)) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            } else {
                int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
                int const causal_row_offset = seqlenk_col_limit - seqlen_q + m_block * kBlockM + thread_row_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // If col0 is beyond the column limit, we want to mask out the entire column, by setting
                        // row limit to be kBlockM.
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_top) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const col_limit_sink = sink_token_length - n_block * kBlockN - thread_col_offset;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // If col0 is beyond the column limit, we want to mask out the entire column, by setting
                        // row limit to be kBlockM.
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset - window_size_right;
                        int const row_limit_bot = col0 < col_limit_sink ? kBlockM : col0 - causal_row_offset + window_size_left;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            int const row_idx = int(get<Row>(t0ScS_rowcol(m, _0{})));
                            if (row_idx < row_limit_top || row_idx > row_limit_bot) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            }
        }
    };

};

} // namespace flash
