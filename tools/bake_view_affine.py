"""INRIA 3DGS PLY 에 평균 view-affine 을 post-hoc 로 bake.

재학습 없이 기존 `map_splat.ply` 의 색을 보정:
  rgb' = rgb × mean_gain + mean_bias

SH DC 와 rest 계수에 대응하는 변환을 적용해 저장.  평균 gain/bias 값은
학습 로그 (viewaffine_train.log) 의 `[view_affine]` 줄에서 복사해 인자로 전달.

사용:
  python3 tools/bake_view_affine.py map_splat.ply \
      --gain 1.050 0.973 1.030  --bias 0.042 0.005 0.037 \
      --out map_splat_baked.ply
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

SH_C0 = 0.28209479177387814


def parse_ply_header(data: bytes) -> tuple[list[str], int, int]:
    """헤더 줄 리스트 + vertex 수 + 헤더 바이트 길이 반환."""
    header_end = data.find(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii", errors="ignore")
    lines = header.splitlines()
    n = int(re.search(r"element vertex (\d+)", header).group(1))
    return lines, n, header_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path)
    ap.add_argument("--gain", type=float, nargs=3, required=True,
                    help="평균 gain R G B (학습 로그 값)")
    ap.add_argument("--bias", type=float, nargs=3, required=True,
                    help="평균 bias R G B (학습 로그 값)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    gain = np.asarray(args.gain, dtype=np.float32)   # [3]
    bias = np.asarray(args.bias, dtype=np.float32)   # [3]

    raw = Path(args.ply).read_bytes()
    lines, n, hdr_end = parse_ply_header(raw)
    props = [ln.split()[-1] for ln in lines if ln.startswith("property ")]
    body = np.frombuffer(raw[hdr_end:], dtype=np.float32).reshape(n, len(props)).copy()

    idx = {p: i for i, p in enumerate(props)}

    # f_dc_*, f_rest_* 위치 찾기
    dc_cols = [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]
    rest_cols = sorted(
        [idx[p] for p in props if p.startswith("f_rest_")],
        key=lambda i: int(re.search(r"\d+", props[i]).group(0))
    )
    n_rest = len(rest_cols)
    K_minus_1 = n_rest // 3

    # DC bake: f_dc'[c] = f_dc[c]·g[c] + (0.5·(g[c]-1) + b[c]) / SH_C0
    for c in range(3):
        col = dc_cols[c]
        body[:, col] = body[:, col] * gain[c] + (0.5 * (gain[c] - 1) + bias[c]) / SH_C0

    # Rest bake: f_rest'[c] = f_rest[c] · g[c]
    # INRIA 레이아웃: [R_sh1, R_sh2, ..., R_sh_K, G_sh1, ..., G_sh_K, B_sh1, ..., B_sh_K]
    # 즉 channel-major.  channel c 의 rest 계수는 [c*K, (c+1)*K) 범위.
    if n_rest > 0:
        for c in range(3):
            start = c * K_minus_1
            end = start + K_minus_1
            body[:, rest_cols[start:end]] *= gain[c]

    # 저장
    with open(args.out, "wb") as f:
        f.write(raw[:hdr_end])
        f.write(body.tobytes())

    print(f"[bake] gain={gain.tolist()}  bias={bias.tolist()}")
    print(f"[bake] SH degree coeff per channel: 1 (DC) + {K_minus_1} (rest)")
    print(f"[saved] {args.out}  ({n:,} gaussians)")


if __name__ == "__main__":
    main()
