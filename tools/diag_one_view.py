"""특정 (kf, cam) 한 뷰만 진단 — GT vs 렌더 vs diff 3 패널.

용도: 목표 객체 (유모차 + 마네킹 등) 가 보이는 한 view 의 sharpness 직접 비교.

사용:
  python3 tools/diag_one_view.py map_splat.ply --kf 89 --cam front \\
      --bg-color "0.0,0.0,0.0" --out diag_kf89_front.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from base_data_parser import BaseDataDataset, build_chapter2_inputs
from tools.diagnose_render import load_splat_ply, render_view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path)
    ap.add_argument("--root", type=Path, default=Path("base_data"))
    ap.add_argument("--kf", type=int, required=True, help="kf 폴더 번호")
    ap.add_argument("--cam", type=str, default="front",
                    choices=["front", "left", "right"])
    ap.add_argument("--bg-color", type=str, default=None,
                    help="학습 시 bg-color 와 동일하게 (예: '0.5,0.5,0.55').  "
                         "random bg 학습이면 검정 '0.0,0.0,0.0' 추천.")
    ap.add_argument("--out", type=Path, default=Path("diag_one.png"))
    args = ap.parse_args()

    bg_color = None
    if args.bg_color is not None:
        bg_color = [float(x) for x in args.bg_color.split(",")]
        if len(bg_color) != 3:
            raise ValueError("--bg-color 는 R,G,B 3개")

    # splat 로드
    means, sh, opacity, scales, quats, sh_degree = load_splat_ply(str(args.ply))
    splat = dict(means=means, sh=sh, opacity=opacity,
                 scales=scales, quats=quats, sh_degree=sh_degree)
    print(f"[ply] N={len(means):,}  sh_degree={sh_degree}")

    # 대상 kf 의 view 입력
    ds = BaseDataDataset(args.root, require_image=True,
                         cam_downscale={"left": 4, "right": 4})
    # ds.indices 에서 args.kf 위치 찾기
    if args.kf not in ds.indices:
        raise ValueError(f"kf={args.kf} dataset 에 없음")
    pos = ds.indices.index(args.kf)
    kf = ds[pos]
    if args.cam not in kf.images:
        raise ValueError(f"kf={args.kf} 에 {args.cam} 카메라 없음")

    s = build_chapter2_inputs(kf, args.cam, ds.calib)
    gt_rgb = (s["gt_image"] * 255).astype(np.uint8)
    gt_bgr = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
    H, W = gt_bgr.shape[:2]

    rendered = render_view(splat, s["viewmat"], s["K"], W, H, bg_color=bg_color)

    diff = cv2.absdiff(gt_bgr, rendered)
    diff_mean = float(diff.mean())
    print(f"[render] {W}×{H}  diff_mean={diff_mean:.1f}")

    # 3 패널 라벨
    panel = np.concatenate([gt_bgr, rendered, diff], axis=1)
    labels = [(10, f"GT (kf={args.kf} {args.cam})"),
              (W + 10, f"Render (bg={args.bg_color})"),
              (2 * W + 10, f"|diff| mean={diff_mean:.1f}")]
    for x, lab in labels:
        cv2.putText(panel, lab, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), panel)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
