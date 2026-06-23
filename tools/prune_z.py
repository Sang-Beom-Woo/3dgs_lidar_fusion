"""INRIA 3DGS PLY 에서 월드 Z 범위 밖 가우시안을 post-hoc 제거.

PLY 의 x/y/z 는 INRIA 컨벤션 상 world 좌표 (3DGS 모델 학습 시 사용한 frame
= run_chapter2.py 의 경우 SLAM world frame, ROS 관용 → Z up).

용도:
  * 야외 학습 결과에서 하늘 / 건물 위 / 가로등 floater 제거.
  * 학습은 한 번만 돌리고 임계값만 바꿔가며 시각 확인.

사용:
  python3 tools/prune_z.py map_splat_baked.ply --z-max 4 --out map_splat_z4.ply
  python3 tools/prune_z.py map_splat_baked.ply --z-min -1 --z-max 6 --out map_splat_z_band.ply
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
    ap.add_argument("--z-min", type=float, default=None,
                    help="world Z 가 이 값 미만이면 제거 (m). 미지정 = 하한 없음.")
    ap.add_argument("--z-max", type=float, default=None,
                    help="world Z 가 이 값 초과면 제거 (m). 미지정 = 상한 없음.")
    ap.add_argument("--axis", type=str, default="z", choices=["x", "y", "z"],
                    help="잘라낼 축.  기본 z (ROS world Z-up).  데이터셋이 "
                         "Z-up 아니면 변경.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.z_min is None and args.z_max is None:
        ap.error("--z-min 또는 --z-max 중 하나는 지정해야 함")

    raw = Path(args.ply).read_bytes()
    lines, n, hdr_end = parse_ply_header(raw)
    props = [ln.split()[-1] for ln in lines if ln.startswith("property ")]
    body = np.frombuffer(raw[hdr_end:], dtype=np.float32).reshape(n, len(props)).copy()

    idx = {p: i for i, p in enumerate(props)}
    axis_col = idx[args.axis]
    coord = body[:, axis_col]

    keep = np.ones(n, dtype=bool)
    if args.z_min is not None:
        keep &= (coord >= args.z_min)
    if args.z_max is not None:
        keep &= (coord <= args.z_max)
    n_out = int(keep.sum())

    print(f"[input]  {n:,} gaussians  ({args.axis}: "
          f"min={coord.min():.2f}  p50={np.median(coord):.2f}  max={coord.max():.2f})")
    print(f"[filter] {args.axis} ∈ "
          f"[{args.z_min if args.z_min is not None else '-inf'}, "
          f"{args.z_max if args.z_max is not None else '+inf'}]")
    print(f"[output] {n_out:,} gaussians  (kept {100*n_out/n:.1f}%)")

    # 헤더의 vertex 수 치환 (prune_needles.py 와 동일 패턴)
    new_header_text = "\n".join(lines[:-1]) + "\n"   # end_header 제외 후 재구성
    new_header_text = re.sub(
        r"element vertex \d+",
        f"element vertex {n_out}",
        new_header_text,
    )
    new_header_text = new_header_text + "end_header\n"

    body_out = body[keep]
    with open(args.out, "wb") as f:
        f.write(new_header_text.encode("ascii"))
        f.write(body_out.tobytes())
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
