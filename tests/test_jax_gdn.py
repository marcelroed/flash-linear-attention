import os

from fla.modules.convolution import ShortConvolution
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import torch
import jax
import jax.numpy as jnp
from einops import rearrange, einsum
from jax import Array
import numpy as np
from torch import nn
import equinox as eqx

torch.set_float32_matmul_precision('highest')

match_dict = {}
def should_match(name, value):
    value = jax.tree.map(np.array, value)
    if name not in match_dict:
        match_dict[name] = value
        return
    jax.tree.map(
        lambda v1, v2: np.testing.assert_allclose(
            v1, v2,
            rtol=1e-3, atol=1e-3,
        ),
        match_dict[name], value
    )
    

def chunk_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    BT: int, #chunk size
):
    # alias
    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    should_match('qvk_beta', (q, v, k_beta))
    assert l % chunk_size == 0
    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    print(mask)
    q, k, v, k_beta, decay = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c = chunk_size), [q, k, v, k_beta, decay.unsqueeze(-1)])
    decay = decay.squeeze(-1).cumsum(-1)
    L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).exp()
    should_match('L_mask', (decay, L_mask, k_beta, k, mask))
    should_match('k_beta@k', einsum(k_beta, k, '... i k, ... o k -> ...'))
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    should_match('attn0', attn)
    for i in range(1, chunk_size):
      attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    should_match('attn1', attn)
    attn = attn
    k_cumsum = attn @ v
    attn = -((k_beta @ k.transpose(-1, -2))).masked_fill(mask, 0)
    for i in range(1, chunk_size):
      attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    w = k_cumdecay = attn @ k_beta
    u = v = k_cumsum
    S = k.new_zeros(b, h, d_k, d_v)
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    print('2:', mask)
    for i in range(0, l // chunk_size):
      q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
      attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
      v_prime = (k_cumdecay[:, :, i] * decay[:, :, i, :, None].exp()) @ S
      v_new = v_i - v_prime
      o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
      o[:, :, i] = o_inter + attn @ v_new
      S = S * decay[:, :, i, -1, None, None].exp() + (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    return rearrange(o, 'b h n c d -> b h (n c) d'),  rearrange(w, 'b h n c d -> b h (n c) d'),  rearrange(u, 'b h n c d -> b h (n c) d')


def tree_rearrange(tree, pattern):
    return jax.tree.map(
        lambda x: rearrange(x, pattern),
        tree
    )

def chunk_gated_delta_rule(
    q: Array,
    k: Array,
    v: Array,
    beta: Array,
    g: Array,
    BT: int, #chunk size
):
    # alias
    q, k, v, beta, g = map(lambda x: x.astype(jnp.float32), [q, k, v, beta, g])
    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * (d_k ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert l % chunk_size == 0
    mask = jnp.arange(chunk_size)[:, None] <= jnp.arange(chunk_size)[None, :]
    q, k, v, k_beta, decay = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c = chunk_size), [q, k, v, k_beta, decay[..., None]])
    decay = decay.squeeze(-1).cumsum(-1)
    L_mask = jnp.tril(jnp.exp(jnp.tril(decay[..., None] - decay[..., None, :])))
    attn = -jnp.where(mask[None, None, None, :, :], 0, (einsum(k_beta, k, '... i k, ... o k -> ... i o') * L_mask))

    print(attn.shape, mask.shape, chunk_size)

    from jax.scipy.linalg import solve_triangular
    @jax.vmap
    @jax.vmap
    @jax.vmap
    def lower_solve(attn):
        lower = jnp.tril(attn, k=-1)
        eye = jnp.eye(chunk_size, dtype=jnp.float32)
        inv = solve_triangular(eye - lower, eye, lower=True)
        update = jnp.tril(inv - eye, k=-1)

        return update
        # attn = attn.at[jnp.tril_indices(attn.shape[0], k=-1)].set(
        #     update[jnp.tril_indices(attn.shape[0], k=-1)]
        # )
        # return attn
    attn = lower_solve(attn)

    # def step_attn(i, attn):
    #     # updated_attn = attn.at[..., i, :i].add((attn[..., i, :i, None] * attn[..., :i, :i]).sum(-2))
    #     in_range = jnp.arange(chunk_size) < i
    #     attn_mask = in_range[None, None, None, :, None] & in_range[None, None, None, None, :]
    #     updated_attn = attn.at[..., i, :].add((attn[..., i, :, None] * attn[..., :, :] * attn_mask).sum(-2))
    #     return updated_attn
    # attn = jax.lax.fori_loop(1, chunk_size, step_attn, attn)
    # for i in range(1, chunk_size):
    #   attn = attn.at[..., i, :i].add((attn[..., i, :i, None] * attn[..., :i, :i]).sum(-2))

    attn = attn + jnp.eye(chunk_size, dtype=jnp.float32)
    attn = attn
    k_cumsum = attn @ v
    attn = jnp.where(mask, 0, -((k_beta @ jnp.matrix_transpose(k))))
    # def step_attn2(i, attn):
    #     # updated_attn = attn.at[..., i, :i].add((attn[..., i, :i, None] * attn[..., :i, :i]).sum(-2))
    #     in_range = jnp.arange(chunk_size) < i
    #     attn_mask = in_range[None, None, None, :, None] & in_range[None, None, None, None, :]
    #     updated_attn = attn.at[..., i, :].add((attn[..., i, :, None] * attn[..., :, :] * attn_mask).sum(-2))
    #     return updated_attn
    # attn = jax.lax.fori_loop(1, chunk_size, step_attn2, attn)
    attn = lower_solve(attn)
    # for i in range(1, chunk_size):
    #   attn = attn.at[..., i, :i].add((attn[..., i, :i, None] * attn[..., :i, :i]).sum(-2))
    # attn = attn + 1

    attn = attn + jnp.eye(chunk_size, dtype=jnp.float32)
    attn = attn
    w = k_cumdecay = attn @ k_beta
    u = k_cumsum
    v = k_cumsum
    S = jnp.zeros((b, h, d_k, d_v), dtype=jnp.float32)
    o = jnp.zeros_like(v)
    mask = jnp.arange(chunk_size)[:, None] < jnp.arange(chunk_size)[None, :]
    print(decay.shape, k_cumdecay.shape, q.shape, k.shape, v.shape, L_mask.shape, o.shape)
    def step_o(carry, X):
        S = carry
        decay_i, k_cumdecay_i, q_i, k_i, v_i, L_mask_i, o_i = X
        attn = (q_i @ jnp.matrix_transpose(k_i) * L_mask_i)
        attn = jnp.where(mask, 0, attn)
        v_prime = (k_cumdecay_i * jnp.exp(decay_i[:, :, :, None])) @ S
        v_new = v_i - v_prime
        o_inter = (q_i * jnp.exp(decay_i[:, :, :, None])) @ S
        o_i = o_inter + attn @ v_new
        S = S * jnp.exp(decay_i[:, :, -1, None, None]) + jnp.matrix_transpose((k_i * jnp.exp(decay_i[:, :, -1, None] - decay_i)[..., None])) @ v_new
        return S, o_i

    def compute_o(S, decay, k_cumdecay, q, k, v, L_mask, o):
        pass

    
    S, o_misordered = jax.lax.scan(step_o, init=S, xs=(
            rearrange(decay, 'a b i c -> i a b c'),
            *tree_rearrange((k_cumdecay, q, k, v, L_mask, o), 'a b i ... -> i a b ...'),
        )
    )
    o = rearrange(o_misordered, 'i a b ... -> a b i ...')

    # for i in range(0, l // chunk_size):
    #     q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
    #     attn = (q_i @ jnp.matrix_transpose(k_i) * L_mask[:, :, i])
    #     attn = jnp.where(mask, 0, attn)
    #     v_prime = (k_cumdecay[:, :, i] * jnp.exp(decay[:, :, i, :, None])) @ S
    #     v_new = v_i - v_prime
    #     o_inter = (q_i * jnp.exp(decay[:, :, i, :, None])) @ S
    #     o = o.at[:, :, i].set(o_inter + attn @ v_new)
    #     S = S * jnp.exp(decay[:, :, i, -1, None, None]) + jnp.matrix_transpose(k_i * jnp.exp(decay[:, :, i, -1, None] - decay[:, :, i])[..., None]) @ v_new
    return rearrange(o, 'b h n c d -> b h (n c) d'),  rearrange(w, 'b h n c d -> b h (n c) d'),  rearrange(u, 'b h n c d -> b h (n c) d')



def test_chunk_gated_delta_rule(dtype='f32'):
    dtype_jax = jnp.float32 if dtype == 'f32' else jnp.bfloat16
    dtype_torch = torch.float32 if dtype == 'f32' else torch.bfloat16

    with jax.default_matmul_precision('F32_F32_F32'):
        b, h, l, d_k = 10, 8, 32, 64
        d_v = 64
        torch.random.manual_seed(0)
        q = torch.randn(b, h, l, d_k, dtype=dtype_torch).clip(-2, 2)
        k = torch.randn(b, h, l, d_k, dtype=dtype_torch).clip(-2, 2)
        v = torch.randn(b, h, l, d_v, dtype=dtype_torch).clip(-2, 2)
        g = torch.randn(b, h, l, dtype=dtype_torch).clip(-2, 2)
        beta = torch.randn(b, h, l, dtype=dtype_torch).clip(-2, 2)
        print(q, k, v, g, beta)
        BT = 8
        result = chunk_gated_delta_rule_ref(q, k, v, beta, g, BT)
        # print(result)
        q = jnp.array(q)
        k = jnp.array(k)
        v = jnp.array(v)
        g = jnp.array(g)
        beta = jnp.array(beta)
        result_jax = chunk_gated_delta_rule(q, k, v, beta, g, BT)

        result = list(map(lambda t: np.array(t), result))
        result_jax = list(map(lambda t: np.array(t), result_jax))
        print('result nans:', np.isnan(result[0]).any(), np.isnan(result[1]).any(), np.isnan(result[2]).any())
        print('result_jax nans:', np.isnan(result_jax[0]).any(), np.isnan(result_jax[1]).any(), np.isnan(result_jax[2]).any())
        np.testing.assert_allclose(result_jax[2], result[2], atol=0.5, rtol=1e-2)
        np.testing.assert_allclose(result_jax[1], result[1], atol=0.5, rtol=1e-2)
        np.testing.assert_allclose(result_jax[0], result[0], atol=0.5, rtol=1e-2)


def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
):
    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k ** -0.5)
    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        S = S.clone() * g[:, :, i].exp()[..., None, None]
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i[..., None]
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    return o


def recurrent_gated_delta_rule(
    q: Array,
    k: Array,
    v: Array,
    beta: Array,
    g: Array,
):
    q, k, v, beta, g = map(lambda x: x.astype(jnp.float32), [q, k, v, beta, g])
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    o = jnp.zeros_like(v)
    S = jnp.zeros((b, h, d_k, d_v), dtype=jnp.float32)
    q = q * (d_k ** -0.5)
    def step_S_o(carry, X):
        S = carry
        _q_i, k_i, v_i, g_i, beta_i = X
        S = S * jnp.exp(g_i[..., None, None])
        _v_i = v_i - (S * k_i[..., None]).sum(-2)
        _v_i = _v_i * beta_i[..., None]
        S = S + jnp.expand_dims(k_i, -1) * jnp.expand_dims(_v_i, -2)
        o_i = jnp.einsum('bhd,bhdm->bhm', _q_i, S)
        return S, o_i
    S, o_misordered = jax.lax.scan(step_S_o, init=S, xs=(
            rearrange(q, 'a b i c -> i a b c'),
            *tree_rearrange((k, v, g, beta), 'a b i -> i a b'),
        )
    )
    o = rearrange(o_misordered, 'i a b -> a b i')
    return o

    for i in range(l):
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].copy()
        S = S * jnp.exp(g[:, :, i][..., None, None])
        beta_i = beta[:, :, i]
        _v = _v - (S * _k[..., None]).sum(-2)
        _v = _v * beta_i[..., None]
        S = S + jnp.expand_dims(_k, -1) * jnp.expand_dims(_v, -2)
        o = o.at[:, :, i].set(jnp.einsum('bhd,bhdm->bhm', _q, S))
    return o