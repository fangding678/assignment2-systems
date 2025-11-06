import torch
from torch import Tensor
from torch.utils import benchmark
from einops import einsum, rearrange
from jaxtyping import Float, Bool, Int
import einx
import timeit
import math
import triton
import triton.language as tl
from triton.runtime import driver
from cs336_basics.model import BasicsTransformerLM, softmax
from cs336_basics.model import scaled_dot_product_attention
# from cs336_system.torch_util import get_device, precision_context


# DEVICE = triton.runtime.driver.active.get_active_torch_device()


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def naive_forward(Q, K, V, is_causal=False):
        d_model = Q.shape[-1]
        S = einsum(Q, K, '... s d, ... s d -> ... s s') / math.sqrt(d_model)
        if is_causal:
            mask = torch.triu(torch.ones(Q.shape[-2], K.shape[-2], dtype=torch.bool), diagonal=1)
            if S.ndim > mask.ndim:
                mask = mask.view((1, ) * (S.ndim - mask.ndim) + mask.shape)
            S.masked_fill(mask, float('-INF'))

        L = softmax(S)
        O = einsum(L, V, '... s s, ... s d -> ... s d')
        return O

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, naive=False, bq=16, bk=16, eps=1e-6):
        if naive:
            res = FlashAttentionPytorch.naive_forward(Q, K, V, is_causal)
        bs = Q.shape[0]
        Q = Q.view(-1, Q.shape[-2], Q.shape[-1])
        K = K.view(-1, K.shape[-2], K.shape[-1])
        V = V.view(-1, V.shape[-2], V.shape[-1])
        # print('\nQ.shape', Q.shape)
        # print('K.shape', K.shape)
        # print('V.shape', V.shape)
        # Q = einx.rearrange('... s d -> (...) s d', Q)
        # K = einx.rearrange('... s d -> (...) s d', K)
        # V = einx.rearrange('... s d -> (...) s d', V)
        bs_head, n_query, d_model = Q.shape
        n_key = K.shape[1]
        tq = (n_query + bq - 1) // bq
        tk = (n_key + bk - 1) // bk
        O = torch.empty_like(Q, device=Q.device)
        L = torch.empty((bs_head, n_query), device=Q.device)
        mask = torch.triu(torch.ones((1, n_query, n_key), dtype=torch.bool, device=Q.device), diagonal=1)
        for i in range(tq):
            start_i, end_i = i*bq, min((i+1)*bq, n_query)
            bq_i = end_i - start_i
            q_i = Q[:, start_i: end_i]
            o_i = torch.zeros((bs_head, bq_i, d_model), device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros((bs_head, bq_i, 1), device=Q.device, dtype=Q.dtype)
            m_i = torch.full((bs_head, bq_i, 1), float('-inf'), device=Q.device, dtype=Q.dtype)
            for j in range(tk):
                start_j, end_j = j*bk, min((j+1)*bk, n_key)
                k_j = K[:, start_j: end_j]
                v_j = V[:, start_j: end_j]
                s_j = einx.dot('... bq d, ... bk d -> ... bq bk', q_i, k_j) / math.sqrt(d_model)
                if is_causal:
                    mask_ij = mask[:, start_i: end_i, start_j: end_j]
                    s_j = s_j.masked_fill(mask_ij, float('-inf'))
                m_new = torch.max(s_j.max(dim=-1, keepdim=True).values, m_i)
                p_i = torch.exp(s_j - m_new)
                m_exp_diff = torch.exp(m_i - m_new)
                l_i = m_exp_diff * l_i + torch.sum(p_i, dim=-1, keepdim=True)
                o_i = m_exp_diff * o_i + einsum(p_i, v_j, '... bq bk, ... bk d -> ... bq d')
                m_i = m_new
            o_i = o_i / (l_i + eps)
            O[:, start_i: end_i] = o_i
            L[:, start_i: end_i] = m_i[:,:,0] + torch.log(l_i[:,:,0])
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        D = torch.sum(O * dO, dim=-1, keepdim=True)
        d_scale = 1 / math.sqrt(Q.shape[-1])
        dQ, dK, dV = _flash_backward(dO, Q, K, V, L, O, is_causal, d_scale)
        return dQ, dK, dV, None


@torch.compile
def _flash_backward(dO, Q, K, V, L, O, is_causal, d_scale):
    S = einsum(Q, K, '... s_q d, ... s_k d -> ... s_q s_k') * d_scale
    if is_causal:
        mask = torch.ones_like(S, dtype=torch.bool, device=Q.device).triu_(diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
    P = torch.exp(S - torch.unsqueeze(L, -1))
    if is_causal:
        P = P.masked_fill(mask, 0.0)
    dV = einsum(P, dO, '... s_q s_k, ... s_q d -> ... s_k d')
    dP = einsum(dO, V, '... s_q d, ... s_k d -> ... s_q s_k')
    if is_causal:
        dP = dP.masked_fill(mask, 0.0)
    D = einsum(P, dP, '... s_q d, ... s_q d -> ... s_q').unsqueeze(-1)
    dS = P * (dP - D)
    dQ = einsum(dS, K, '... s_q s_k, ... s_k d -> ... s_q d') * d_scale
    dK = einsum(dS, Q, '... s_q s_k, ... s_q d -> ... s_k d') * d_scale
    return dQ, dK, dV


@triton.jit
def flash_fwd_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, 
                     stride_qb, stride_qq, stride_qd, stride_kb, stride_kk, stride_kd, stride_vb, stride_vk, stride_vd, 
                     stride_ob, stride_oq, stride_od, stride_lb, stride_lq, N_QUERIES, N_KEYS, d_scale, IS_CAUSAL, eps,
                     D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # tl.device_print('\ntl.program_id(0)', query_tile_index)
    # tl.device_print('tl.program_id(1)', batch_index)
    Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb,
                                    shape=(N_QUERIES, D),
                                    strides=(stride_qq, stride_qd),
                                    offsets=(query_tile_index * Q_TILE_SIZE, 0),
                                    block_shape=(Q_TILE_SIZE, D),
                                    order=(1, 0),)
    K_block_ptr = tl.make_block_ptr(K_ptr + batch_index * stride_kb,
                                    shape=(D, N_KEYS),
                                    strides=(stride_kd, stride_kk),
                                    offsets=(0, 0),
                                    block_shape=(D, K_TILE_SIZE),
                                    order=(0, 1),)
    V_block_ptr = tl.make_block_ptr(V_ptr + batch_index * stride_vb,
                                    shape=(N_KEYS, D),
                                    strides=(stride_vk, stride_vd),
                                    offsets=(0, 0),
                                    block_shape=(K_TILE_SIZE, D),
                                    order=(1, 0),)
    O_block_ptr = tl.make_block_ptr(O_ptr + batch_index * stride_ob,
                                    shape=(N_QUERIES, D),
                                    strides=(stride_oq, stride_od),
                                    offsets=(query_tile_index * Q_TILE_SIZE, 0),
                                    block_shape=(Q_TILE_SIZE, D),
                                    order=(1, 0),)
    l_ptr = L_ptr + batch_index * stride_lb + (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE))
    q_i = tl.load(Q_block_ptr, boundary_check=(0, 1))
    d_scale = tl.cast(d_scale, q_i.dtype)
    q_i = q_i * d_scale
    l_i = tl.zeros((Q_TILE_SIZE, 1), dtype=q_i.dtype)
    m_i = tl.full((Q_TILE_SIZE, 1), float('-inf'), dtype=q_i.dtype)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=q_i.dtype)
    tk = (N_KEYS + K_TILE_SIZE - 1) // K_TILE_SIZE

    rows = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    rows = rows[:, None]
    # tl.static_print(o_i)
    # tl.static_print(m_i)
    # if query_tile_index == 2 and batch_index == 1:
    #     tl.device_print('q_i', q_i / d_scale)
    #     k1_j = tl.load(K_block_ptr, boundary_check=(0, 1))
    #     v1_j = tl.load(V_block_ptr, boundary_check=(0, 1))
    #     tl.device_print('k1_j', k1_j)
    #     tl.device_print('v1_j', v1_j)
    for j in range(tk):
        k_j = tl.load(K_block_ptr, boundary_check=(0, 1))
        v_j = tl.load(V_block_ptr, boundary_check=(0, 1))
        # tl.device_print('k_j.dtype', k_j.dtype)
        # tl.device_print('v_j.dtype', v_j.dtype)
        # tl.static_print(q_i)
        # tl.static_print(k_j)
        s_j = tl.dot(q_i, k_j).to(q_i.dtype)
        if IS_CAUSAL:
            cols = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            cols = cols[None, :]
            mask = tl.where(cols > rows, float('-inf'), 0).to(q_i.dtype)
            s_j = tl.add(s_j, mask)

        m_new = tl.maximum(tl.max(s_j, axis=-1, keep_dims=True), m_i)
        p_i = tl.exp(s_j - m_new).to(q_i.dtype)
        m_diff = tl.exp(m_i - m_new).to(q_i.dtype)
        l_i = tl.cast(m_diff * l_i + tl.sum(p_i, axis=-1, keep_dims=True), q_i.dtype)
        o_i = tl.cast(m_diff * o_i + tl.dot(p_i, v_j), q_i.dtype)
        m_i = tl.cast(m_new, q_i.dtype)

        K_block_ptr = tl.advance(K_block_ptr, offsets=(0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, offsets=(K_TILE_SIZE, 0))

    o_i = o_i / l_i
    l_i = tl.reshape(m_i + tl.log(l_i + eps), (Q_TILE_SIZE,))

    # tl.static_print(l_ptr)
    # tl.static_print(l_i)

    tl.store(O_block_ptr, o_i.to(O_ptr.dtype.element_ty))
    tl.store(l_ptr, l_i, mask=(query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)) < N_QUERIES)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, bq=16, bk=16, eps=1e-6):
        bs = Q.shape[0]
        bs_head, n_query, d_model = Q.shape
        n_key = K.shape[-2]
        d_scale = 1 / math.sqrt(d_model)
        tq = (n_query + bq - 1) // bq
        grid = (tq, bs)

        O = torch.empty_like(Q, device=Q.device, dtype=Q.dtype)
        L = torch.empty((bs_head, n_query), device=Q.device, dtype=Q.dtype)
        # print('\n\ngrid', grid)
        # print('Q.stride()', Q.stride())
        # print('K.stride()', K.stride())
        # print('V.stride()', V.stride())
        # print('O.stride()', O.stride())
        # print('L.stride()', L.stride())
        # print('Q.dtype', Q.dtype)

        flash_fwd_kernel[grid](
            Q_ptr=Q, K_ptr=K, V_ptr=V, O_ptr=O, L_ptr=L, \
            stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2), \
            stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2), \
            stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2), \
            stride_ob=O.stride(0), stride_oq=O.stride(1), stride_od=O.stride(2), \
            stride_lb=L.stride(0), stride_lq=L.stride(1), \
            N_QUERIES=n_query, N_KEYS=n_key, d_scale=d_scale, IS_CAUSAL=is_causal, eps=1e-6, \
            D=d_model, Q_TILE_SIZE=bq, K_TILE_SIZE=bk
        )
        ctx.save_for_backward(Q, K, V, L, O)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        d_scale = 1 / math.sqrt(Q.shape[-1])
        dQ, dK, dV = _flash_backward(dO, Q, K, V, L, O, is_causal, d_scale)
        return dQ, dK, dV, None


if __name__ == '__main__':
    pass


