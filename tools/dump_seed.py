"""학습 전 시드 포인트클라우드를 PLY 로 덤프.

map.ply (학습 후) 와 seed.ply (학습 전) 를 비교하면:
  * 둘 다 가운데 비어 있음 → LiDAR 시드 자체가 sparse, 학습 잘못 아님
  * seed 는 가득, map.ply 만 비어 있음 → 학습이 과도하게 prune

사용:
  python3 tools/dump_seed.py --init-voxel 0.1 --out seed.ply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# tools/ 하위에 있으므로 부모 디렉터리를 import path 에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import open3d as o3d

from base_data_parser import BaseDataDataset, colorize_keyframe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("base_data"))
    ap.add_argument("--out", type=Path, default=Path("seed.ply"))
    ap.add_argument("--init-voxel", type=float, default=0.1)
    ap.add_argument("--max-kf", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--keep-unseen", action="store_true",
                    help="카메라 FOV 밖 LiDAR 점도 포함 (회색)")
    args = ap.parse_args()

    ds = BaseDataDataset(args.root, require_image=True)
    print(f"[dataset] {len(ds)} kf")

    pts_all, col_all = [], []
    used = 0
    for k, kf in enumerate(ds):
        if k % args.stride != 0:
            continue
        if args.max_kf is not None and used >= args.max_kf:
            break
        p, c = colorize_keyframe(kf, ds.calib, drop_unseen=not args.keep_unseen)
        pts_all.append(p)
        col_all.append(c)
        used += 1
    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(col_all, axis=0)
    print(f"[seed] {used} kf → raw {len(pts):,} pts")

    if args.init_voxel > 0:
        # run_chapter2 와 동일한 voxel downsample 로직 사용 (색 평균 대신 max-sat)
        from run_chapter2 import _voxel_downsample_max_saturation
        pts, cols = _voxel_downsample_max_saturation(pts.astype(np.float32),
                                                     cols.astype(np.float32),
                                                     args.init_voxel)
        print(f"[seed] voxel={args.init_voxel} → {len(pts):,} pts (max-saturation)")
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    else:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

    o3d.io.write_point_cloud(str(args.out), pc)
    print(f"[save] {args.out}  ({len(pts):,} pts)")

    # 공간 분포 통계
    print(f"\n[bbox] min={pts.min(axis=0).round(2)}  max={pts.max(axis=0).round(2)}")
    print(f"[bbox] extent={(pts.max(axis=0) - pts.min(axis=0)).round(2)} m")


if __name__ == "__main__":
    main()
