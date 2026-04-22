"""특정 3D 영역의 seed 밀도 / 색 / α 분포 진단.

사용 예 (신호등이 world 좌표 (x, y, z) = (10, 3, 1.5) 근처에 있을 때):
  python3 check_traffic_light.py --center 10 3 1.5 --radius 0.3
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import open3d as o3d

SH_C0 = 0.28209479177387814


def load_splat_ply(path: str):
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
    idx = {p: i for i, p in enumerate(props)}
    return {
        "means":   arr[:, [idx["x"], idx["y"], idx["z"]]],
        "f_dc":    arr[:, [idx["f_dc_0"], idx["f_dc_1"], idx["f_dc_2"]]],
        "opacity": arr[:, idx["opacity"]],
        "scales":  arr[:, [idx["scale_0"], idx["scale_1"], idx["scale_2"]]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", type=Path, default=Path("map_splat.ply"))
    ap.add_argument("--seed-ply", type=Path, default=Path("seed.ply"),
                    help="학습 전 seed (optional — 없어도 됨)")
    ap.add_argument("--center", type=float, nargs=3, required=True,
                    help="3D 점 (x y z) — 신호등 center")
    ap.add_argument("--radius", type=float, default=0.3,
                    help="검사 반경 (m)")
    args = ap.parse_args()

    center = np.array(args.center)

    # --- splat PLY 분석 -----------------------------------------------
    s = load_splat_ply(str(args.ply))
    means, f_dc = s["means"], s["f_dc"]
    opacity = s["opacity"]
    scales = s["scales"]

    dist = np.linalg.norm(means - center, axis=1)
    in_sphere = dist < args.radius
    n_in = int(in_sphere.sum())

    print(f"[검사 영역] center={center}  radius={args.radius} m")
    print(f"[총 가우시안 수] {len(means):,}")
    print(f"[영역 내 가우시안] {n_in:,}  ({100*n_in/len(means):.4f}%)")
    if n_in == 0:
        print("  ❌ 영역 내 가우시안 없음! seed 에서 이 영역이 비어 있다는 뜻.")
        return

    # DC → RGB 복원
    rgb = np.clip(0.5 + f_dc[in_sphere] * SH_C0, 0, 1)
    alpha = 1.0 / (1.0 + np.exp(-opacity[in_sphere]))
    scale_m = np.exp(scales[in_sphere]).max(axis=1)

    print(f"\n[색 분포 (in region)]")
    print(f"  RGB mean = ({rgb[:,0].mean():.3f}, {rgb[:,1].mean():.3f}, {rgb[:,2].mean():.3f})")
    mx, mn = rgb.max(axis=1), rgb.min(axis=1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    print(f"  채도 (max-min/max): mean={sat.mean():.3f}  p95={np.percentile(sat,95):.3f}  max={sat.max():.3f}")
    # 어두운 것 비율
    brightness = rgb.mean(axis=1)
    print(f"  밝기 (V): mean={brightness.mean():.3f}  "
          f"어두운 점 (V<0.2): {(brightness<0.2).sum()} ({100*(brightness<0.2).mean():.1f}%)  "
          f"밝은 점 (V>0.8): {(brightness>0.8).sum()} ({100*(brightness>0.8).mean():.1f}%)")

    print(f"\n[α 분포 (in region)]")
    print(f"  min={alpha.min():.4f}  median={np.median(alpha):.4f}  "
          f"p95={np.percentile(alpha,95):.4f}  max={alpha.max():.4f}")

    print(f"\n[크기 분포 (max axis, m)]")
    print(f"  min={scale_m.min():.4f}  median={np.median(scale_m):.4f}  "
          f"p95={np.percentile(scale_m,95):.4f}  max={scale_m.max():.4f}")

    # --- seed 와 비교 (있으면) ------------------------------------------
    if args.seed_ply.exists():
        print(f"\n[seed 비교]  {args.seed_ply}")
        pc_seed = o3d.io.read_point_cloud(str(args.seed_ply))
        pts_seed = np.asarray(pc_seed.points)
        cols_seed = np.asarray(pc_seed.colors)
        dist_seed = np.linalg.norm(pts_seed - center, axis=1)
        in_seed = dist_seed < args.radius
        n_seed = int(in_seed.sum())
        print(f"  seed 내 점: {n_seed:,}  (학습 후 {n_in:,} 대비 {n_seed/max(n_in,1):.1f}×)")
        if n_seed > 0:
            rgb_s = cols_seed[in_seed]
            print(f"  seed RGB mean = ({rgb_s[:,0].mean():.3f}, "
                  f"{rgb_s[:,1].mean():.3f}, {rgb_s[:,2].mean():.3f})")
            sat_s = (rgb_s.max(axis=1) - rgb_s.min(axis=1)) / np.maximum(rgb_s.max(axis=1), 1e-6)
            print(f"  seed 채도 mean = {sat_s.mean():.3f}")
            br_s = rgb_s.mean(axis=1)
            print(f"  seed 어두운 점 (V<0.2): {(br_s<0.2).sum()} ({100*(br_s<0.2).mean():.1f}%)")

    # --- 결론 -----------------------------------------------------------
    print(f"\n=== 진단 ===")
    if n_in < 10:
        print(f"⚠️  가우시안 부족 — 이 영역에 {n_in} 개만 있어 디테일 표현 불가")
    if sat.mean() < 0.1:
        print(f"⚠️  평균 채도 {sat.mean():.3f} — 학습 후 회색끼로 수렴")
    if (brightness < 0.2).mean() < 0.1:
        print(f"⚠️  어두운 점 비율 {100*(brightness<0.2).mean():.1f}% — "
              f"검정색 housing 표현 안 됨")
    if alpha.mean() < 0.1:
        print(f"⚠️  평균 α {alpha.mean():.3f} — 가우시안들이 너무 투명 → 뒤 배경이 비쳐 회색")


if __name__ == "__main__":
    main()
