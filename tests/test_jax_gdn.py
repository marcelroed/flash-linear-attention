import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import torch
import jax
import jax.numpy as jnp
from einops import rearrange, einsum
from jax import Array
import numpy as np

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
    should_match('qvk_beta', (q, v, k_beta))
    assert l % chunk_size == 0
    # note that diagonal is masked.
    mask = jnp.arange(chunk_size)[:, None] <= jnp.arange(chunk_size)[None, :]
    q, k, v, k_beta, decay = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c = chunk_size), [q, k, v, k_beta, decay[..., None]])
    decay = decay.squeeze(-1).cumsum(-1)
    L_mask = jnp.exp(decay[..., None] - decay[..., None, :])
    # attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    should_match('L_mask', (decay, L_mask, k_beta, k, mask))
    should_match('k_beta@k', einsum(k_beta, k, '... i k, ... o k -> ...'))
    # attn = -((k_beta @ jnp.matrix_transpose(k)) * L_mask).at[..., mask].set(0)
    attn = -jnp.where(mask[None, None, None, :, :], 0, (einsum(k_beta, k, '... i k, ... o k -> ... i o') * L_mask))
    should_match('attn0', attn)
    for i in range(1, chunk_size):
      attn = attn.at[..., i, :i].add((attn[..., i, :i, None] * attn[..., :i, :i]).sum(-2))
    attn = attn + jnp.eye(chunk_size, dtype=jnp.float32)
    should_match('attn1', attn)
    attn = attn
    k_cumsum = attn @ v
    attn = jnp.where(mask, 0, -((k_beta @ jnp.matrix_transpose(k))))
    for i in range(1, chunk_size):
      attn = attn.at[..., i, :i].add((attn[..., i, :i, None] * attn[..., :i, :i]).sum(-2))
    attn = attn + jnp.eye(chunk_size, dtype=jnp.float32)
    attn = attn
    w = k_cumdecay = attn @ k_beta
    u = k_cumsum
    v = k_cumsum
    S = jnp.zeros((b, h, d_k, d_v), dtype=jnp.float32)
    o = jnp.zeros_like(v)
    mask = jnp.arange(chunk_size)[:, None] < jnp.arange(chunk_size)[None, :]
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ jnp.matrix_transpose(k_i) * L_mask[:, :, i])
        attn = jnp.where(mask, 0, attn)
        v_prime = (k_cumdecay[:, :, i] * jnp.exp(decay[:, :, i, :, None])) @ S
        v_new = v_i - v_prime
        o_inter = (q_i * jnp.exp(decay[:, :, i, :, None])) @ S
        o = o.at[:, :, i].set(o_inter + attn @ v_new)
        S = S * jnp.exp(decay[:, :, i, -1, None, None]) + jnp.matrix_transpose(k_i * jnp.exp(decay[:, :, i, -1, None] - decay[:, :, i])[..., None]) @ v_new
    return rearrange(o, 'b h n c d -> b h (n c) d'),  rearrange(w, 'b h n c d -> b h (n c) d'),  rearrange(u, 'b h n c d -> b h (n c) d')



def test_chunk_gated_delta_rule():
    with jax.default_matmul_precision('F32_F32_F32'):
        b, h, l, d_k = 10, 8, 32, 64
        d_v = 64
        torch.random.manual_seed(0)
        q = torch.randn(b, h, l, d_k).clip(-2, 2) * 0.1
        k = torch.randn(b, h, l, d_k).clip(-2, 2) * 0.1
        v = torch.randn(b, h, l, d_v).clip(-2, 2) * 0.1
        g = torch.randn(b, h, l).clip(-2, 2) * 0.1
        beta = torch.randn(b, h, l).clip(-2, 2) * 0.1
        BT = 4
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
        np.testing.assert_allclose(result[2], result_jax[2], atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result[1], result_jax[1], atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(result[0], result_jax[0], atol=1e-3, rtol=1e-3)

