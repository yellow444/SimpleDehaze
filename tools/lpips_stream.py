"""Length-prefixed PNG stream evaluator for genuine LPIPS (AlexNet, v0.1).

Protocol: stdin repeats <uint32 result_bytes><uint32 gt_bytes><result PNG><gt PNG>.
The process writes one invariant-culture float per request. Two zero lengths terminate it.
"""

from __future__ import annotations

import io
import os
import struct
import sys

import numpy as np
from PIL import Image
import torch
import lpips


def read_exact(size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sys.stdin.buffer.read(remaining)
        if not chunk:
            raise EOFError(f"unexpected EOF; wanted {remaining} more bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_png(payload: bytes, device: torch.device) -> torch.Tensor:
    with Image.open(io.BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return (tensor.to(device=device) / 127.5) - 1.0


def main() -> int:
    device_name = os.environ.get("SIMPLEDEHAZE_LPIPS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    torch.set_grad_enabled(False)
    model = lpips.LPIPS(net="alex", version="0.1", spatial=False).eval().to(device)
    print(f"READY\t{device}", flush=True)
    while True:
        header = sys.stdin.buffer.read(8)
        if not header:
            break
        if len(header) != 8:
            raise EOFError("truncated LPIPS request header")
        result_size, truth_size = struct.unpack("<II", header)
        if result_size == 0 and truth_size == 0:
            break
        if result_size == 0 or truth_size == 0 or result_size > 100_000_000 or truth_size > 100_000_000:
            raise ValueError(f"invalid LPIPS payload sizes: {result_size}, {truth_size}")
        result = decode_png(read_exact(result_size), device)
        truth = decode_png(read_exact(truth_size), device)
        if result.shape != truth.shape:
            raise ValueError(f"shape mismatch: {tuple(result.shape)} != {tuple(truth.shape)}")
        value = float(model(result, truth).reshape(-1)[0].item())
        print(format(value, ".9g"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
