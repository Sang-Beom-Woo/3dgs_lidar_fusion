"""INRIA 3DGS PLY 에서 needle / spike 아티팩트 가우시안을 post-hoc 제거.

기준:
  aspect ratio = max(scale) / min(scale) > threshold

scale 은 PLY 의 scale_0/1/2 (log-space) 이므로 exp 후 비율 계산.

사용:
  python3 tools/prune_needles.py map_splat.ply \
      --aspect-thr 8 \
      --out map_splat_clean.ply
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def parse_ply_header(data: bytes) -> tuple[list[str], int, int]:
    header_end = data.find(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii", errors="ignore")
    lines = header.splitlines()
    n = int(re.search(r"element vertex (\d+)", header).group(1))
    return lines, n, header_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path)
    ap.add_argument("--aspect-thr", type=float, default=8.0,
                    help="max/min scale 비율 이 값 초과면 제거 (기본 8:1)")
    ap.add_argument("--max-scale", type=float, default=None,
                    help="max scale 이 이 값 (m) 초과면 제거 (선택)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    raw = Path(args.ply).read_bytes()
    lines, n, hdr_end = parse_ply_header(raw)
    props = [ln.split()[-1] for ln in lines if ln.startswith("property ")]
    body = np.frombuffer(raw[hdr_end:], dtype=np.float32).reshape(n, len(props)).copy()

    idx = {p: i for i, p in enumerate(props)}
    scales_log = body[:, [idx["scale_0"], idx["scale_1"], idx["scale_2"]]]
    scales_lin = np.exp(scales_log)                  # [N, 3], meters

    s_max = scales_lin.max(axis=1)
    s_min = np.maximum(scales_lin.min(axis=1), 1e-8)
    aspect = s_max / s_min

    keep = aspect <= args.aspect_thr
    removed_aspect = int((~keep).sum())

    if args.max_scale is not None:
        keep_size = s_max <= args.max_scale
        removed_size = int((~keep_size).sum())
        keep = keep & keep_size
    else:
        removed_size = 0

    keep_idx = np.where(keep)[0]
    n_out = len(keep_idx)

    print(f"[input]  {n:,} gaussians")
    print(f"[aspect] p50={np.median(aspect):.2f}  p95={np.percentile(aspect, 95):.2f}  max={aspect.max():.2f}")
    print(f"[removed] aspect > {args.aspect_thr}: {removed_aspect:,}")
    if args.max_scale is not None:
        print(f"[removed] max_scale > {args.max_scale} m: {removed_size:,}")
    print(f"[output] {n_out:,} gaussians  (kept {100*n_out/n:.1f}%)")

    # 헤더의 vertex 수 치환
    new_header_text = "\n".join(lines[:-1]) + "\n"   # end_header 제외 후 재구성
    new_header_text = re.sub(
        r"element vertex \d+",
        f"element vertex {n_out}",
        new_header_text,
    )
    new_header_text = new_header_text + "end_header\n"

    body_out = body[keep_idx]
    with open(args.out, "wb") as f:
        f.write(new_header_text.encode("ascii"))
        f.write(body_out.tobytes())
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
