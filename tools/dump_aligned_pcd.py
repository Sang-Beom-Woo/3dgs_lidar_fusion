"""키프레임 LiDAR 점들을 지정한 포즈 파일로 통합 → 색상 입힌 PLY 저장.

ICP refined 와 SLAM 원본 포즈 정합 정확도를 시각 비교하기 위한 유틸.
같은 벽/모서리가:
  * SLAM 쪽: 흐릿하게 겹침 / 이중 벽 → 정렬 오차 큼
  * ICP 쪽:  단일 평면으로 떨어짐    → 정렬 양호

사용:
  # ICP refined (현재 keyframe_pose.txt)
  python3 tools/dump_aligned_pcd.py --pose-name keyframe_pose.txt \
      --out icp_map.ply --voxel 0.05

  # SLAM 원본 (백업)
  python3 tools/dump_aligned_pcd.py --pose-name keyframe_pose.txt.bak \
      --out slam_map.ply --voxel 0.05

  # CloudCompare 또는 MeshLab 에서 두 PLY 띄워 비교.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from base_data_parser import (
    BaseDataDataset, load_keyframe, _parse_pose_txt,
    project_lidar_to_camera, _undistort_rgb,
)


def write_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """ASCII 헤더 + binary little-endian body PLY."""
    n = len(xyz)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    body = np.empty(n, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    body["x"] = xyz[:, 0].astype(np.float32)
    body["y"] = xyz[:, 1].astype(np.float32)
    body["z"] = xyz[:, 2].astype(np.float32)
    body["r"] = rgb[:, 0]
    body["g"] = rgb[:, 1]
    body["b"] = rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(body.tobytes())


def voxel_downsample_mean(pts: np.ndarray, cols: np.ndarray, voxel: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """간단한 voxel mean downsample (XYZ 평균, RGB 평균)."""
    if voxel <= 0:
        return pts, cols
    origin = pts.min(axis=0)
    coords = np.floor((pts - origin) / voxel).astype(np.int64)
    keys = (coords[:, 0] * 73856093
            ^ coords[:, 1] * 19349663
            ^ coords[:, 2] * 83492791)
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]; pts_s = pts[order]; cols_s = cols[order]
    bounds = np.concatenate(
        ([0], np.where(np.diff(keys_s) != 0)[0] + 1, [len(keys_s)])
    )
    n_vox = len(bounds) - 1
    out_pts = np.empty((n_vox, 3), dtype=np.float32)
    out_cols = np.empty((n_vox, 3), dtype=np.uint8)
    for i in range(n_vox):
        s, e = bounds[i], bounds[i + 1]
        out_pts[i] = pts_s[s:e].mean(axis=0)
        out_cols[i] = cols_s[s:e].mean(axis=0).astype(np.uint8)
    return out_pts, out_cols


def colorize_one_kf(kf, calib) -> tuple[np.ndarray, np.ndarray]:
    """한 kf 의 LiDAR 포인트 + 색.  front/left/right 차례로 시도, 안 보이면
    회색 (128) 으로 패딩.

    LiDAR 포인트는 base 프레임 그대로 (호출자가 T_world_base 적용해야 함).
    """
    N = kf.points_base.shape[0]
    colors = np.full((N, 3), 128, dtype=np.uint8)
    seen = np.zeros(N, dtype=bool)
    for cam_name in ("front", "left", "right"):
        if cam_name not in kf.images or seen.all():
            continue
        cam = calib.intrinsics[cam_name]
        img_rgb = _undistort_rgb(kf.images[cam_name], cam)         # float [0,1] RGB
        img_u8 = (img_rgb * 255).astype(np.uint8)
        uv, _, mask = project_lidar_to_camera(kf.points_base, cam_name, calib)
        if uv.size == 0:
            continue
        ui = np.clip(uv[:, 0].astype(np.int32), 0, cam.width - 1)
        vi = np.clip(uv[:, 1].astype(np.int32), 0, cam.height - 1)
        idx_in_full = np.where(mask)[0]
        # 이미 색 입은 점은 건너뜀 — 첫 카메라 (front) 우선
        new_mask = ~seen[idx_in_full]
        chosen = idx_in_full[new_mask]
        colors[chosen] = img_u8[vi[new_mask], ui[new_mask]]
        seen[chosen] = True
    return kf.points_base, colors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("base_data"))
    ap.add_argument("--pose-name", type=str, default="keyframe_pose.txt",
                    help="각 kf 폴더 내 포즈 파일명.  비교용: "
                         "keyframe_pose.txt (refined) / keyframe_pose.txt.bak (SLAM 원본)")
    ap.add_argument("--out", type=Path, required=True, help="출력 PLY 경로")
    ap.add_argument("--voxel", type=float, default=0.05,
                    help="voxel downsample (m).  0 이면 비활성 (raw 합치기). "
                         "기본 0.05 (5cm) — 시각 비교에 적당")
    ap.add_argument("--max-kf", type=int, default=None,
                    help="처음 N kf 만 (디버그)")
    args = ap.parse_args()

    ds = BaseDataDataset(args.root, require_image=False)
    ds.calib.build_undistort_maps()
    n = len(ds) if args.max_kf is None else min(args.max_kf, len(ds))
    print(f"[dump] root={args.root}  pose={args.pose_name}  kfs={n}  voxel={args.voxel}m")

    t0 = time.time()
    all_pts_world: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    n_skipped = 0
    for i in range(n):
        kf_idx = ds.indices[i]
        kf_dir = args.root / str(kf_idx)
        pose_path = kf_dir / args.pose_name
        if not pose_path.exists():
            n_skipped += 1
            continue
        # 포즈 직접 로드 (parser 의 자동 keyframe_pose.txt 우회)
        _ts, T_world_base = _parse_pose_txt(pose_path)
        kf = load_keyframe(args.root, kf_idx)
        if kf is None or kf.points_base.shape[0] < 100:
            n_skipped += 1
            continue
        pts_base, cols = colorize_one_kf(kf, ds.calib)
        # world 변환
        pts_world = (T_world_base @ np.hstack(
            [pts_base, np.ones((pts_base.shape[0], 1))]).T).T[:, :3].astype(np.float32)
        all_pts_world.append(pts_world)
        all_cols.append(cols)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n}]  accumulated {sum(len(p) for p in all_pts_world):,} pts")

    if not all_pts_world:
        print("[dump] 누적된 점 없음")
        return

    pts = np.concatenate(all_pts_world, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    print(f"[dump] raw 누적: {len(pts):,} pts  (skip {n_skipped} kf)")

    pts, cols = voxel_downsample_mean(pts, cols, args.voxel)
    print(f"[dump] voxel={args.voxel}: {len(pts):,} pts")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_ply_xyzrgb(args.out, pts, cols)
    dt = time.time() - t0
    print(f"[dump] saved {args.out}  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
