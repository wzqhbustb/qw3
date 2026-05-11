#!/usr/bin/env python3
"""
Compare C reference output against Python ground truth.

Usage:
    python3 scripts/compare_c_vs_python.py [--q4km]
"""

import argparse
import torch
import numpy as np
import struct


def load_bin(path: str):
    with open(path, "rb") as f:
        data = f.read()
    n_floats = len(data) // 4
    return np.array(struct.unpack(f"{n_floats}f", data), dtype=np.float32)


def compare(name: str, c_arr: np.ndarray, py_arr: torch.Tensor, tol: float = 1e-3,
            rel_tol: float = None, rel_std_tol: float = None):
    py = py_arr.detach().cpu().numpy().astype(np.float32)
    if c_arr.shape != py.shape:
        print(f"  [FAIL] {name:25s} SHAPE MISMATCH: {c_arr.shape} vs {py.shape}")
        return False
    diff = np.abs(c_arr - py)
    max_diff = diff.max()
    mean_diff = diff.mean()
    py_max = np.abs(py).max()
    py_std = py.std()
    rel_diff = max_diff / (py_max + 1e-8)
    rel_to_std = max_diff / (py_std + 1e-8)

    ok = True
    if tol is not None and max_diff > tol:
        ok = False
    if rel_tol is not None and rel_diff > rel_tol:
        ok = False
    if rel_std_tol is not None and rel_to_std > rel_std_tol:
        ok = False

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:25s} max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  "
          f"rel={rel_diff:.6e}  rel_to_std={rel_to_std:.6e}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Compare C vs Python ground truth")
    parser.add_argument("--q4km", action="store_true",
                        help="Use Q4_K_M tolerant thresholds (vs Q8_0 ground truth)")
    args = parser.parse_args()

    gt = torch.load("scripts/layer0_ground_truth.pt", weights_only=True)

    if args.q4km:
        # Q4_K_M vs Q8_0: quantization error is expected to be ~1-2 std for
        # small-magnitude intermediates, and ~1-5% relative for final output.
        checkpoints = [
            ("attn_norm_out",  "layer0_c_attn_norm.bin",  "attn_norm_out",  1e-5, None, None),
            ("q_proj",         "layer0_c_q_proj.bin",     "q_proj",         None, None, 2.0),
            ("k_proj",         "layer0_c_k_proj.bin",     "k_proj",         None, None, 2.0),
            ("v_proj",         "layer0_c_v_proj.bin",     "v_proj",         None, None, 2.0),
            ("attn_out",       "layer0_c_attn_out.bin",   "attn_out",       None, None, 2.0),
            ("ffn_norm_out",   "layer0_c_ffn_norm.bin",   "ffn_norm_out",   None, 0.05, None),
            ("ffn_out",        "layer0_c_ffn_out.bin",    "ffn_out",        None, 0.05, None),
            ("final_output",   "layer0_c_output.bin",     "final_output",   None, 0.05, None),
        ]
        title = "=== Layer 0: Q4_K_M C vs Q8_0 Python Ground Truth ==="
    else:
        # Q8_0 vs Q8_0: should match almost exactly
        checkpoints = [
            ("attn_norm_out",  "layer0_c_attn_norm.bin",  "attn_norm_out",  1e-3, None, None),
            ("q_proj",         "layer0_c_q_proj.bin",     "q_proj",         1e-3, None, None),
            ("k_proj",         "layer0_c_k_proj.bin",     "k_proj",         1e-3, None, None),
            ("v_proj",         "layer0_c_v_proj.bin",     "v_proj",         1e-3, None, None),
            ("attn_out",       "layer0_c_attn_out.bin",   "attn_out",       1e-3, None, None),
            ("ffn_norm_out",   "layer0_c_ffn_norm.bin",   "ffn_norm_out",   1e-3, None, None),
            ("ffn_out",        "layer0_c_ffn_out.bin",    "ffn_out",        1e-3, None, None),
            ("final_output",   "layer0_c_output.bin",     "final_output",   1e-5, None, None),
        ]
        title = "=== Layer 0: C vs Python Ground Truth ==="

    all_pass = True
    print(title + "\n")

    for item in checkpoints:
        if len(item) == 4:
            label, c_path, py_key, tol = item
            rel_tol = None
            rel_std_tol = None
        else:
            label, c_path, py_key, tol, rel_tol, rel_std_tol = item
        c_arr = load_bin(c_path)
        py_arr = gt[py_key]
        ok = compare(label, c_arr, py_arr, tol=tol, rel_tol=rel_tol, rel_std_tol=rel_std_tol)
        all_pass = all_pass and ok

    print("\n" + ("✅ All checks passed!" if all_pass else "❌ Some checks failed!"))


if __name__ == "__main__":
    main()
