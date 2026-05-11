#!/usr/bin/env python3
"""
Extract Layer 0 weights from Q8_0 GGUF and dequantize to FP32.

Outputs: scripts/layer0_q8_weights.pt (PyTorch dict)

NOTE: gguf Python library reverses dimensions from GGUF ne[] order.
      ti.data is already in NumPy C-order (last axis fastest).
      For Q8_0 we dequantize flat then reshape to reversed(ti.shape).
"""

import sys
assert sys.byteorder == 'little', "Big-endian not supported for FP16 decode"

import numpy as np
import torch
import gguf

Q8_0_BLOCK_SIZE = 32
Q8_0_TYPE_SIZE = 34   # 2 bytes d + 32 bytes qs


def dequantize_q8_0(data_u8: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q8_0 data from uint8 array to float32 array."""
    n_blocks = n_elements // Q8_0_BLOCK_SIZE
    assert n_blocks * Q8_0_TYPE_SIZE == len(data_u8), \
        f"size mismatch: {n_blocks}*{Q8_0_TYPE_SIZE}={n_blocks*Q8_0_TYPE_SIZE} vs {len(data_u8)}"

    out = np.zeros(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        off = i * Q8_0_TYPE_SIZE
        d = np.frombuffer(data_u8[off:off + 2].tobytes(), dtype=np.float16)[0]
        qs = data_u8[off + 2:off + Q8_0_TYPE_SIZE].astype(np.int8).astype(np.float32)
        out[i * Q8_0_BLOCK_SIZE:(i + 1) * Q8_0_BLOCK_SIZE] = qs * d

    return out


def extract_layer0(gguf_path: str, output_path: str):
    reader = gguf.GGUFReader(gguf_path)

    weights = {}

    for ti in reader.tensors:
        name = str(ti.name)
        if not name.startswith("blk.0."):
            continue

        shape_raw = list(ti.shape)      # GGUF ne[] order (n0 fastest)
        shape_rev = shape_raw[::-1]     # NumPy C-order (n0 fastest = last axis)
        n_elements = 1
        for dim in shape_raw:
            n_elements *= dim

        if ti.tensor_type == gguf.GGMLQuantizationType.Q8_0:
            # ti.data is already in reversed shape; flatten in C-order
            data = np.array(ti.data).reshape(-1)
            arr = dequantize_q8_0(data, n_elements)
            # Reshape to NumPy C-order: reversed GGUF dimensions
            arr = arr.reshape(shape_rev)
            print(f"[Q8_0] {name:40s} raw_ne={shape_raw}  numpy_shape={list(arr.shape)}  OK")
        elif ti.tensor_type == gguf.GGMLQuantizationType.F32:
            # ti.data is already float32 in reversed shape
            arr = np.array(ti.data)
            print(f"[F32 ] {name:40s} raw_ne={shape_raw}  numpy_shape={list(arr.shape)}  OK")
        else:
            raise ValueError(f"{name}: unsupported type {ti.tensor_type}")

        weights[name] = torch.from_numpy(arr.copy())

    torch.save(weights, output_path)
    print(f"\nSaved {len(weights)} tensors to {output_path}")


if __name__ == "__main__":
    gguf_path = "/Volumes/ExtremeSSD/qwen3-engine/models/Qwen3-30B-A3B-Q8_0.gguf"
    output_path = "scripts/layer0_q8_weights.pt"
    extract_layer0(gguf_path, output_path)
