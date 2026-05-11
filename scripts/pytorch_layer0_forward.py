#!/usr/bin/env python3
"""
PyTorch manual Layer 0 forward for Qwen3-30B.
Generates ground truth for C reference validation.

Output: scripts/layer0_ground_truth.pt (dict of intermediate tensors)
"""

import torch
import torch.nn.functional as F
import math

# Model constants (30B)
N_EMBD = 2048
N_HEAD = 32
N_HEAD_KV = 4
HEAD_DIM = 128
N_EXPERT = 128
N_EXPERT_USED = 8
N_FF_EXP = 768
NORM_EPS = 1e-6
ROPE_THETA = 1e6


def rms_norm(x, weight, eps):
    """x, weight: same shape [n]"""
    ss = (x ** 2).mean()
    return x / math.sqrt(ss + eps) * weight


def apply_rope(q, k, seq_pos):
    """In-place RoPE on q and k.
    q: [N_HEAD, HEAD_DIM]
    k: [N_HEAD_KV, HEAD_DIM]
    """
    for h in range(q.shape[0]):
        for d in range(0, HEAD_DIM, 2):
            freq = ROPE_THETA ** (-d / HEAD_DIM)
            angle = seq_pos * freq
            c, s = math.cos(angle), math.sin(angle)
            v0, v1 = q[h, d], q[h, d + 1]
            q[h, d] = v0 * c - v1 * s
            q[h, d + 1] = v0 * s + v1 * c

    for h in range(k.shape[0]):
        for d in range(0, HEAD_DIM, 2):
            freq = ROPE_THETA ** (-d / HEAD_DIM)
            angle = seq_pos * freq
            c, s = math.cos(angle), math.sin(angle)
            v0, v1 = k[h, d], k[h, d + 1]
            k[h, d] = v0 * c - v1 * s
            k[h, d + 1] = v0 * s + v1 * c


def gqa_attention(q, k, v, w_o, kv_k, kv_v, seq_pos):
    """GQA attention for a single token at seq_pos.
    q: [N_HEAD, HEAD_DIM]
    k: [N_HEAD_KV, HEAD_DIM]
    v: [N_HEAD_KV, HEAD_DIM]
    w_o: [N_EMBD, N_HEAD*HEAD_DIM] = [2048, 4096]
    kv_k, kv_v: [max_seq_len, N_HEAD_KV, HEAD_DIM]
    """
    n_head = q.shape[0]
    n_kv_head = k.shape[0]
    head_per_kv = n_head // n_kv_head
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # Write to KV cache
    kv_k[seq_pos] = k
    kv_v[seq_pos] = v
    seq_len = seq_pos + 1

    # Compute attention scores for each Q head against all cached positions
    scores = torch.zeros(n_head, seq_len)
    for h in range(n_head):
        kvh = h // head_per_kv
        for pos in range(seq_len):
            scores[h, pos] = (q[h] * kv_k[pos, kvh]).sum() * scale

    # Softmax over positions
    scores = F.softmax(scores, dim=-1)

    # Weighted sum of V
    attn_out = torch.zeros(n_head, HEAD_DIM)
    for h in range(n_head):
        kvh = h // head_per_kv
        for pos in range(seq_len):
            attn_out[h] += scores[h, pos] * kv_v[pos, kvh]

    # O projection
    attn_out_flat = attn_out.reshape(-1)  # [4096]
    return attn_out_flat @ w_o.T  # [2048]


def moe_ffn(x, w_gate_inp, w_gate_exps, w_up_exps, w_down_exps):
    """MoE FFN forward for a single token.
    x: [N_EMBD]
    """
    # Router
    logits = x @ w_gate_inp.T  # [N_EXPERT]
    probs = F.softmax(logits, dim=-1)

    # Top-k
    topk_values, topk_indices = torch.topk(probs, N_EXPERT_USED)
    # Renormalize top-k probs (DS3_NORM_TOPK_PROB)
    topk_values = topk_values / topk_values.sum()

    # Routed experts (SwiGLU)
    out = torch.zeros(N_EMBD)
    for i in range(N_EXPERT_USED):
        eid = topk_indices[i].item()
        w_g = w_gate_exps[eid]  # [N_FF_EXP, N_EMBD]
        w_u = w_up_exps[eid]
        w_d = w_down_exps[eid]  # [N_EMBD, N_FF_EXP]

        gate = x @ w_g.T  # [N_FF_EXP]
        up = x @ w_u.T
        silu_gate = gate * torch.sigmoid(gate)
        ffn = (silu_gate * up) @ w_d.T  # [N_EMBD]
        out += topk_values[i] * ffn

    return out


def layer0_forward(weights, input_vec, seq_pos=0):
    """Single-layer forward for Qwen3-30B Layer 0.
    Returns final output and a dict of intermediate activations.
    """
    x = input_vec
    intermediates = {}

    # --- Attention branch ---
    xb = rms_norm(x, weights['blk.0.attn_norm.weight'], NORM_EPS)
    intermediates['attn_norm_out'] = xb.clone()

    # Q/K/V projections
    q = xb @ weights['blk.0.attn_q.weight'].T  # [4096]
    k = xb @ weights['blk.0.attn_k.weight'].T  # [512]
    v = xb @ weights['blk.0.attn_v.weight'].T  # [512]
    intermediates['q_proj'] = q.clone()
    intermediates['k_proj'] = k.clone()
    intermediates['v_proj'] = v.clone()

    # Reshape to heads
    q = q.reshape(N_HEAD, HEAD_DIM)
    k = k.reshape(N_HEAD_KV, HEAD_DIM)
    v = v.reshape(N_HEAD_KV, HEAD_DIM)

    # QK Norm
    for h in range(N_HEAD):
        q[h] = rms_norm(q[h], weights['blk.0.attn_q_norm.weight'], NORM_EPS)
    for h in range(N_HEAD_KV):
        k[h] = rms_norm(k[h], weights['blk.0.attn_k_norm.weight'], NORM_EPS)
    intermediates['q_after_norm'] = q.clone()
    intermediates['k_after_norm'] = k.clone()

    # RoPE
    apply_rope(q, k, seq_pos)
    intermediates['q_after_rope'] = q.clone()
    intermediates['k_after_rope'] = k.clone()

    # GQA Attention (with KV cache)
    kv_cache_k = torch.zeros(seq_pos + 1, N_HEAD_KV, HEAD_DIM)
    kv_cache_v = torch.zeros(seq_pos + 1, N_HEAD_KV, HEAD_DIM)
    attn_out = gqa_attention(
        q, k, v,
        weights['blk.0.attn_output.weight'],
        kv_cache_k, kv_cache_v, seq_pos)
    intermediates['attn_out'] = attn_out.clone()

    # Residual
    attn_out = x + attn_out
    intermediates['attn_residual'] = attn_out.clone()

    # --- FFN branch ---
    xb = rms_norm(attn_out, weights['blk.0.ffn_norm.weight'], NORM_EPS)
    intermediates['ffn_norm_out'] = xb.clone()

    # MoE
    ffn_out = moe_ffn(
        xb,
        weights['blk.0.ffn_gate_inp.weight'],
        weights['blk.0.ffn_gate_exps.weight'],
        weights['blk.0.ffn_up_exps.weight'],
        weights['blk.0.ffn_down_exps.weight'])
    intermediates['ffn_out'] = ffn_out.clone()

    # Residual
    output = attn_out + ffn_out
    intermediates['final_output'] = output.clone()

    return output, intermediates


def main():
    weights_path = "scripts/layer0_q8_weights.pt"
    output_path = "scripts/layer0_ground_truth.pt"

    print(f"Loading weights from {weights_path}...")
    weights = torch.load(weights_path, weights_only=True)

    # Synthetic input: all ones (simplest for debugging)
    input_vec = torch.ones(N_EMBD, dtype=torch.float32)
    print(f"Input: torch.ones({N_EMBD})")

    print("Running Layer 0 forward...")
    output, intermediates = layer0_forward(weights, input_vec, seq_pos=0)

    # Save ground truth
    torch.save(intermediates, output_path)
    print(f"\nSaved ground truth to {output_path}")
    print(f"  final_output: {output.shape}, mean={output.mean():.6f}, std={output.std():.6f}")
    print(f"  intermediates: {list(intermediates.keys())}")


if __name__ == "__main__":
    main()
