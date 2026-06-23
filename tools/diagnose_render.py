"""학습된 PLY 와 학습 뷰를 비교해 모델이 실제 GT 를 재현하는지 진단.

학습 뷰 N 개를 랜덤 샘플 → 각각 (GT, rendered, diff) 3-패널 PNG 저장.
splat 뷰어가 보여주는 게 이상해도 학습 뷰 재현이 좋으면 → 시점 문제.
학습 뷰도 깨졌으면 → 모델/학습 자체 문제.

사용:
  python3 tools/diagnose_render.py map_splat.ply --n 6 --out diag/
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from gsplat import rasterization

from base_data_parser import BaseDataDataset, build_chapter2_inputs

SH_C0 = 0.28209479177387814


def load_splat_ply(path: str):
    """INRIA 3DGS PLY 로더 (SH 차수 자동 추출)."""
    with open(path, "rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            header_lines.append(line.decode("ascii", errors="ignore").rstrip("\n"))
            if line.strip() == b"end_header":
                break
        header = "\n".join(header_lines)
        n = int(re.search(r"element vertex (\d+)", header).group(1))
        props = [ln.split()[-1] for ln in header_lines if ln.startswith("property ")]
        raw = f.read(n * len(props) * 4)

    arr = np.frombuffer(raw, dtype=np.float32).reshape(n, len(props)).copy()
    idx = {name: i for i, name in enumerate(props)}

    means = arr[:, [idx["x"], idx["y"], idx["z"]]]
    f_dc = arr[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]]    # [N, 3]
    opacity = arr[:, idx["opacity"]]                                # logit
    scales = arr[:, [idx["scale_0"], idx["scale_1"], idx["scale_2"]]]   # log
    quats = arr[:, [idx["rot_0"], idx["rot_1"], idx["rot_2"], idx["rot_3"]]]

    # f_rest 자동 검출 → SH 차수 계산
    rest_idx = sorted(int(re.search(r"\d+", p).group(0))
                      for p in props if p.startswith("f_rest_"))
    n_rest = len(rest_idx)
    K = 1 + n_rest // 3                                # K = 1 + (3·(K-1))/3
    sh_degree = int(round((K ** 0.5) - 1))

    if n_rest > 0:
        rest_cols = [idx[f"f_rest_{i}"] for i in rest_idx]
        f_rest_inria = arr[:, rest_cols].reshape(n, 3, K - 1)        # [N, 3, K-1]
        f_rest = np.transpose(f_rest_inria, (0, 2, 1))               # [N, K-1, 3]
        sh = np.concatenate([f_dc[:, None, :], f_rest], axis=1)      # [N, K, 3]
    else:
        sh = f_dc[:, None, :]                                        # [N, 1, 3]
    return means, sh, opacity, scales, quats, sh_degree


@torch.no_grad()
def render_view(splat: dict, viewmat: np.ndarray, K: np.ndarray,
                W: int, H: int, device: str = "cuda",
                bg_color: list[float] | None = None) -> np.ndarray:
    """SH 까지 평가해 한 뷰를 렌더 → [H, W, 3] uint8 BGR.

    bg_color: RGB 0~1 리스트 — 학습 시 사용한 배경색과 맞추면 동일 렌더.
              미지정 시 검정 (가우시안 영역만 보임).
    """
    means    = torch.from_numpy(splat["means"]).to(device)
    sh       = torch.from_numpy(splat["sh"]).to(device)
    opacity  = torch.sigmoid(torch.from_numpy(splat["opacity"]).to(device))
    scales   = torch.exp(torch.from_numpy(splat["scales"]).to(device))
    quats    = torch.from_numpy(splat["quats"]).to(device)

    vm = torch.from_numpy(viewmat).to(device).unsqueeze(0)
    Kt = torch.from_numpy(K).to(device).unsqueeze(0)
    bg = (torch.tensor([bg_color], dtype=torch.float32, device=device)
          if bg_color is not None else None)
    renders, _, _ = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacity, colors=sh,
        viewmats=vm, Ks=Kt,
        width=W, height=H,
        packed=False,
        render_mode="RGB",
        sh_degree=splat["sh_degree"],
        backgrounds=bg,
    )
    img = renders[0].clamp(0.0, 1.0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path)
    ap.add_argument("--root", type=Path, default=Path("base_data"))
    ap.add_argument("--out", type=Path, default=Path("diag"))
    ap.add_argument("--n", type=int, default=6, help="비교할 뷰 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg-color", type=str, default=None,
                    help="학습 시 쓴 배경색 R,G,B (0~1).  "
                         "예: '0.8,0.85,0.9'.  미지정 = 검정 배경.")
    args = ap.parse_args()
    bg_color = None
    if args.bg_color is not None:
        bg_color = [float(x) for x in args.bg_color.split(",")]
        if len(bg_color) != 3:
            raise ValueError("--bg-color 는 R,G,B 3개")

    args.out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    # 1) splat 로드
    means, sh, opacity, scales, quats, sh_degree = load_splat_ply(str(args.ply))
    splat = dict(means=means, sh=sh, opacity=opacity,
                 scales=scales, quats=quats, sh_degree=sh_degree)
    print(f"[ply] N={len(means):,}  K={sh.shape[1]}  sh_degree={sh_degree}")

    # α 분포 quick stats
    alpha = 1.0 / (1.0 + np.exp(-opacity))
    q = np.percentile(alpha, [5, 25, 50, 75, 95])
    print(f"[α] p5={q[0]:.4f} p50={q[2]:.4f} p95={q[4]:.4f} max={alpha.max():.4f}")
    # scale 분포
    s_lin = np.exp(scales)
    print(f"[scale] min={s_lin.min():.4f} p50={np.median(s_lin):.4f} max={s_lin.max():.4f}")
    # aspect ratio (needle 검출)
    ar = s_lin.max(axis=1) / np.clip(s_lin.min(axis=1), 1e-8, None)
    print(f"[aspect] p50={np.median(ar):.2f} p95={np.percentile(ar,95):.2f} max={ar.max():.2f}")
    print(f"[aspect] >5:1 ratio: {(ar>5).sum():,} ({100*(ar>5).mean():.1f}%)")

    # 2) 데이터셋 로드 + 랜덤 뷰 N 개
    ds = BaseDataDataset(args.root, require_image=True)
    kf_indices = np.random.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    cams = ["front", "left", "right"]

    for i, kf_idx in enumerate(kf_indices):
        kf = ds[int(kf_idx)]
        cam = cams[i % 3]
        if cam not in kf.images:
            cam = next(iter(kf.images.keys()))
        s = build_chapter2_inputs(kf, cam, ds.calib)

        gt_rgb = (s["gt_image"] * 255).astype(np.uint8)
        gt_bgr = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
        H, W = gt_bgr.shape[:2]
        rendered = render_view(splat, s["viewmat"], s["K"], W, H,
                               bg_color=bg_color)
        # diff
        diff = cv2.absdiff(gt_bgr, rendered)
        # 3-패널 가로 결합 + 라벨
        panel = np.concatenate([gt_bgr, rendered, diff], axis=1)
        for x, label in [(10, "GT"), (W + 10, "Render"), (2 * W + 10, "|diff|")]:
            cv2.putText(panel, label, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
        path = args.out / f"view_{i:02d}_kf{kf.idx}_{cam}.png"
        cv2.imwrite(str(path), panel)
        print(f"  [{i}] kf={kf.idx} {cam} → {path}  diff_mean={diff.mean():.1f}")

    print(f"\n[done] {len(kf_indices)} comparison panels at {args.out}/")


if __name__ == "__main__":
    main()
