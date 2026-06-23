"""키프레임 포즈를 LiDAR 점군 ICP 로 후처리 정렬.

배경
----
SLAM 이 출력한 `keyframe_pose.txt` 는 보통 좋지만 실내 회랑 같은 환경에서는
누적 drift 가 cm ~ 수십 cm 수준으로 남는다.  특히 우리 학습 (LiDAR + camera
3DGS) 처럼 mm-level 일관성이 필요하면 학습 중에 cam-refine / kf-refine 으로
보정해야 하는데, **사전 정렬** 로 한 단계 제거하면:
  * 학습 시 pose-refine 변동폭이 줄어 안정성 ↑
  * pose-refine freeze (300k 붕괴 방지) 후의 잔여 mismatch 도 줄어듦
  * LiDAR seed 가 sharper → 가우시안 초기 위치 정확

방법
----
**SLAM 포즈를 무시하는 pure ICP odometry** (point-to-plane):

  kf[0]:  SLAM 포즈를 anchor 로 그대로 사용 (절대 좌표 정의).
  kf[i] (i>0):
      source = kf[i].points_base          (base 프레임, 미변환)
      target = ∪ kf[j<i].points_world     (refined 포즈로 world 변환)
      init   = refined_poses[i-1]         (이전 kf 포즈를 초기 추정)
      ICP result → refined_poses[i]       (SLAM 포즈 의존성 0)

이 방식은 SLAM 의 drift / outlier 를 완전히 우회.  단점은 LiDAR 만의 정보로
연쇄 정렬하므로 한 kf 의 ICP 실패가 이후 모두에 전파됨 — 그래서 fitness 가
임계 아래면 SLAM 포즈로 폴백.

target = 이전 W 개 kf 의 점군 union (`--target-window`, 기본 3).  멀티-프레임
target 이 occlusion / 좁은 overlap 에서 견고함.

출력
----
기본 (sidecar):  base_data/<kf>/keyframe_pose_refined.txt  새 파일
`--replace`  :  원본을 keyframe_pose.txt.bak 으로 백업 후 덮어쓰기

요약 통계:  보정 translation (m) / rotation (°) 의 평균 / p95 / max.

사용
----
  python3 tools/align_pose_icp.py --root base_data
  python3 tools/align_pose_icp.py --root base_data --target-window 5 --voxel 0.1
  python3 tools/align_pose_icp.py --root base_data --replace      # 덮어쓰기

학습에서 활용
----
  sidecar 모드면 BaseDataDataset(root, prefer_refined_pose=True) 로 자동
  선택.  (parser 변경은 별도 — 이 도구는 파일만 생성.)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import open3d as o3d

from base_data_parser import (
    BaseDataDataset, load_keyframe, _parse_pose_txt, _quat_to_R,
)


# -----------------------------------------------------------------------------
# 헬퍼
# -----------------------------------------------------------------------------
def _R_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """3×3 회전 → quaternion (qx, qy, qz, qw) — ROS 관용.

    Shepperd 알고리즘 (수치 안정).
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


def _make_o3d_pcd(pts: np.ndarray, voxel: float = 0.0) -> o3d.geometry.PointCloud:
    """numpy [N,3] → Open3D PointCloud + voxel downsample + normal 추정."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    # normal: point-to-plane ICP 에 필요
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel * 3, 0.3), max_nn=30)
    )
    return pcd


def _icp_align(source: o3d.geometry.PointCloud,
               target: o3d.geometry.PointCloud,
               init: np.ndarray, threshold: float, max_iter: int
               ) -> tuple[np.ndarray, float, float]:
    """point-to-plane ICP.  반환: (T_correction [4,4], fitness, inlier_rmse)

    T_correction 은 source 를 추가로 변환해 target 에 맞추는 보정 행렬.
    init 은 SLAM 포즈로 이미 변환했으므로 보통 단위행렬 근처.
    """
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    return np.asarray(result.transformation), result.fitness, result.inlier_rmse


def _read_pose_tokens(path: Path) -> list[str]:
    """전체 토큰 보존 (포맷: id frame ts tx ty tz qx qy qz qw + optional 추가 열)."""
    return path.read_text().split()


def _write_pose(path: Path, tokens: list[str], T: np.ndarray) -> None:
    """T_world_base [4,4] 를 받아 토큰 3..9 (tx ty tz qx qy qz qw) 만 갱신해 저장.

    나머지 열 (id, frame, ts, 추가 메타) 는 원본 보존.
    """
    tx, ty, tz = T[:3, 3]
    qx, qy, qz, qw = _R_to_quat_xyzw(T[:3, :3])
    new_toks = list(tokens)
    new_toks[3] = f"{tx:.17f}"
    new_toks[4] = f"{ty:.17f}"
    new_toks[5] = f"{tz:.17f}"
    new_toks[6] = f"{qx:.17f}"
    new_toks[7] = f"{qy:.17f}"
    new_toks[8] = f"{qz:.17f}"
    new_toks[9] = f"{qw:.17f}"
    path.write_text(" ".join(new_toks) + "\n")


def _T_diff_norm(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """두 SE3 의 translation (m) / rotation (deg) 차이."""
    dt = np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])
    dR = T_a[:3, :3] @ T_b[:3, :3].T
    cos_t = (np.trace(dR) - 1) / 2
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    deg = np.degrees(np.arccos(cos_t))
    return dt, deg


# -----------------------------------------------------------------------------
# 메인
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("base_data"),
                    help="base_data 루트 (기본 ./base_data)")
    ap.add_argument("--voxel", type=float, default=0.1,
                    help="ICP voxel downsample (m). 작을수록 정확/느림 (기본 0.1)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="ICP correspondence 거리 임계 (m).  잘못 매칭 차단 "
                         "(기본 0.5 — SLAM drift 이내).")
    ap.add_argument("--max-iter", type=int, default=50,
                    help="ICP 최대 반복 (기본 50)")
    ap.add_argument("--target-window", type=int, default=3,
                    help="target 으로 쓸 이전 kf 수 (기본 3).  "
                         "1 = pairwise.  많을수록 안정하지만 overlap 적은 "
                         "코너 회전에선 노이즈 가능")
    ap.add_argument("--fitness-min", type=float, default=0.3,
                    help="이 값 미만의 ICP fitness 면 SLAM 포즈로 폴백 "
                         "(기본 0.3)")
    ap.add_argument("--replace", action="store_true",
                    help="원본 keyframe_pose.txt 덮어쓰기 (백업: .bak).  "
                         "기본은 keyframe_pose_refined.txt 로 sidecar 저장.")
    ap.add_argument("--max-kf", type=int, default=None,
                    help="처음 N kf 만 처리 (디버그용)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="sidecar 이미 있으면 skip")
    args = ap.parse_args()

    print(f"[icp] root={args.root}  voxel={args.voxel}  thr={args.threshold}  "
          f"window={args.target_window}  mode=pure-odometry (SLAM 무시)")

    ds = BaseDataDataset(args.root, require_image=False)
    n = len(ds) if args.max_kf is None else min(args.max_kf, len(ds))
    print(f"[icp] {n} keyframes 처리 시작")

    # SLAM 원본 포즈 모두 미리 로드 — 비교 통계 + 폴백용
    slam_poses: list[np.ndarray] = []
    pose_tokens: list[list[str]] = []
    for i in range(n):
        kf_idx = ds.indices[i]
        ppath = args.root / str(kf_idx) / "keyframe_pose.txt"
        toks = _read_pose_tokens(ppath)
        _ts, T = _parse_pose_txt(ppath)
        slam_poses.append(T)
        pose_tokens.append(toks)

    # kf[0] 은 anchor — SLAM 포즈 그대로.  이후는 pure ICP odometry.
    refined_poses: list[np.ndarray] = [slam_poses[0].copy()]

    # 점군 캐시 (downsample + normal 추정 비용 큰 작업 — 재사용)
    pcd_world_cache: dict[int, o3d.geometry.PointCloud] = {}

    def _kf_pcd_in_world(idx_in_list: int, T: np.ndarray
                         ) -> o3d.geometry.PointCloud | None:
        """idx_in_list 번째 kf 의 점군을 world 프레임 PCD 로 반환 (캐시).

        T 는 그 kf 의 T_world_base.  refined 포즈가 바뀌면 캐시 무효지만
        이 도구에선 한 번 정해진 refined 포즈는 그대로 쓰므로 OK.
        """
        if idx_in_list in pcd_world_cache:
            return pcd_world_cache[idx_in_list]
        kf = load_keyframe(args.root, ds.indices[idx_in_list])
        if kf is None or kf.points_base.shape[0] < 100:
            return None
        pts_w = (T @ np.hstack(
            [kf.points_base, np.ones((kf.points_base.shape[0], 1))]).T).T[:, :3]
        pcd = _make_o3d_pcd(pts_w.astype(np.float32), args.voxel)
        pcd_world_cache[idx_in_list] = pcd
        return pcd

    # kf[0] 점군 캐시 (world 변환은 자기 자신 anchor 포즈로)
    _kf_pcd_in_world(0, refined_poses[0])

    # 통계
    translation_corrections: list[float] = []
    rotation_corrections: list[float] = []
    fitness_log: list[float] = []
    rmse_log: list[float] = []
    failed: list[int] = []

    t0 = time.time()
    for i in range(1, n):
        kf_idx = ds.indices[i]
        kf = load_keyframe(args.root, kf_idx)
        if kf is None or kf.points_base.shape[0] < 100:
            print(f"  kf={kf_idx}: 점군 부족, 이전 포즈 그대로 사용")
            refined_poses.append(refined_poses[-1].copy())
            continue

        # source: 현 kf 의 점군 (BASE 프레임, 미변환).  ICP 의 init 이
        # 이 점들을 world 로 보낸다.
        source = _make_o3d_pcd(kf.points_base.astype(np.float32), args.voxel)

        # target: 이전 W kfs 의 점군 union — refined 포즈로 world 변환.
        tgt_pts_list: list[np.ndarray] = []
        for j in range(max(0, i - args.target_window), i):
            prev_pcd = _kf_pcd_in_world(j, refined_poses[j])
            if prev_pcd is None:
                continue
            tgt_pts_list.append(np.asarray(prev_pcd.points))
        if not tgt_pts_list:
            refined_poses.append(refined_poses[-1].copy())
            continue
        tgt_pts = np.concatenate(tgt_pts_list, axis=0)
        target = _make_o3d_pcd(tgt_pts.astype(np.float32), args.voxel)

        # init = 이전 kf 의 refined 포즈.  SLAM 정보는 전혀 안 씀.
        # ICP 결과 행렬 자체가 이 kf 의 새 T_world_base.
        init = refined_poses[-1]
        T_new, fit, rmse = _icp_align(
            source, target, init, args.threshold, args.max_iter)

        if fit < args.fitness_min:
            # ICP 신뢰 불가 → SLAM 포즈로 폴백 (drift 누적 차단 시도).
            failed.append(kf_idx)
            T_new = slam_poses[i].copy()

        refined_poses.append(T_new)

        dt_m, dr_deg = _T_diff_norm(T_new, slam_poses[i])
        translation_corrections.append(dt_m)
        rotation_corrections.append(dr_deg)
        fitness_log.append(fit)
        rmse_log.append(rmse)

        if i % 10 == 0 or i < 5:
            print(f"  kf={kf_idx}  fit={fit:.3f} rmse={rmse:.3f}m  "
                  f"vs-SLAM: Δt={dt_m*1000:.1f}mm  Δrot={dr_deg:.3f}°")

    dt_total = time.time() - t0
    print(f"\n[icp] done  ({dt_total:.1f}s)")

    # 통계
    if translation_corrections:
        ts_arr = np.asarray(translation_corrections) * 1000  # mm
        rs_arr = np.asarray(rotation_corrections)
        fits = np.asarray(fitness_log)
        print(f"[stats] ICP refined vs SLAM 차이 — translation (mm):  "
              f"mean={ts_arr.mean():.1f}  p50={np.median(ts_arr):.1f}  "
              f"p95={np.percentile(ts_arr,95):.1f}  max={ts_arr.max():.1f}")
        print(f"[stats] ICP refined vs SLAM 차이 — rotation (deg):  "
              f"mean={rs_arr.mean():.3f}  p50={np.median(rs_arr):.3f}  "
              f"p95={np.percentile(rs_arr,95):.3f}  max={rs_arr.max():.3f}")
        print(f"[stats] ICP fitness:  "
              f"mean={fits.mean():.3f}  p5={np.percentile(fits,5):.3f}  "
              f"min={fits.min():.3f}")
        if failed:
            print(f"[stats] ICP 실패 (fitness<{args.fitness_min}, SLAM 폴백): "
                  f"{len(failed)}/{n-1} kf — id={failed[:10]}{'...' if len(failed)>10 else ''}")

    # 저장
    n_written = 0
    for i in range(n):
        kf_idx = ds.indices[i]
        out_dir = args.root / str(kf_idx)
        if args.replace:
            orig = out_dir / "keyframe_pose.txt"
            bak = out_dir / "keyframe_pose.txt.bak"
            if not bak.exists():
                shutil.copyfile(orig, bak)
            _write_pose(orig, pose_tokens[i], refined_poses[i])
        else:
            out = out_dir / "keyframe_pose_refined.txt"
            if args.skip_existing and out.exists():
                continue
            _write_pose(out, pose_tokens[i], refined_poses[i])
        n_written += 1
    print(f"[write] {n_written} 파일 저장 "
          f"({'replace' if args.replace else 'sidecar'} 모드)")
    if not args.replace:
        print(f"        예: {args.root}/{ds.indices[0]}/keyframe_pose_refined.txt")
        print("\n학습에서 활용하려면 parser 측에 prefer_refined_pose 옵션 추가 필요.")


if __name__ == "__main__":
    main()
