"""3DGS PLY 의 origin 을 LiDAR (base_link) → 카메라 광학 (optical) 프레임으로 이동.

원래 학습은 SLAM world frame 으로 진행 — 첫 kf 의 base_link 가 origin.
뷰어에서 보기 불편 (씬이 멀리 떨어져 있거나 회전이 어색) 할 때 카메라를
origin 으로 옮기면 자연스러움.

처리:
  T_world_cam = T_world_base · T_base_cam[cam]   (kf 의 카메라 pose, world 기준)
  T_cam_world = inv(T_world_cam)                  (역행 — 새 world 변환)

  모든 가우시안에 대해 T_cam_world 적용:
    1) 위치        : p_new = R · p_old + t
    2) Quaternion  : q_new = q_R · q_old  (Hamilton 곱)
    3) SH degree 1 : world 회전에 따라 view-dep 계수 회전 (DC 는 invariant)

기본 모드: 카메라 광학 frame 으로 변환 (z forward, y down — OpenCV/COLMAP 컨벤션).
  뷰어가 Y-up 가정하면 결과가 뒤집어져 보일 수 있어서 그 뒤 reorient.py 로
  Y-up 변환 권장.

`--translate-only` 옵션: 위치만 이동 (world 축 그대로 유지).  Z-up 그대로
보고 싶고 그저 origin 만 가깝게 옮기고 싶을 때.

사용:
  python3 tools/center_on_camera.py map_splat_30000.ply \\
      --root base_data --cam front --kf 0 \\
      --out map_splat_cam.ply

  # 그 뒤 Y-up 으로 (뷰어용)
  python3 tools/reorient.py map_splat_cam.ply --out map_splat_cam_yup.ply

  # 또는 위치만 이동
  python3 tools/center_on_camera.py map_splat_30000.ply \\
      --root base_data --cam front --kf 0 --translate-only \\
      --out map_splat_translated.ply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from base_data_parser import _parse_pose_txt, _parse_sensor_tf
# reorient.py 의 헬퍼 재사용 — 같은 컨벤션 보장
from tools.reorient import (
    parse_ply_header, infer_sh_degree, rot_to_quat_wxyz,
    quat_mul_batch, rotate_sh_degree1,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("base_data"),
                    help="base_data 루트 (kf pose + sensor_tf 읽기)")
    ap.add_argument("--cam", type=str, default="front",
                    choices=["front", "left", "right"],
                    help="origin 으로 만들 카메라 (기본 front)")
    ap.add_argument("--kf", type=int, default=0,
                    help="기준 kf 인덱스 (기본 0)")
    ap.add_argument("--translate-only", action="store_true",
                    help="회전 없이 위치만 이동 (world Z-up 유지).  뷰어가 "
                         "Z-up 이고 가우시안만 카메라 근처로 끌어오고 싶을 때.")
    args = ap.parse_args()

    # 1) 기준 kf 의 T_world_base 로드
    pose_path = args.root / str(args.kf) / "keyframe_pose.txt"
    if not pose_path.exists():
        raise FileNotFoundError(f"pose 파일 없음: {pose_path}")
    _ts, T_world_base = _parse_pose_txt(pose_path)
    print(f"[anchor kf={args.kf}] T_world_base xyz="
          f"{T_world_base[:3, 3].round(3).tolist()}")

    # 2) sensor_tf 에서 T_base_cam[cam] 로드
    T_base_cam_dict, _T_base_lidar = _parse_sensor_tf(
        args.root / "calibration" / "sensor_tf.txt")
    if args.cam not in T_base_cam_dict:
        raise KeyError(f"{args.cam} not in sensor_tf: {list(T_base_cam_dict.keys())}")
    T_base_cam = T_base_cam_dict[args.cam]
    print(f"[anchor cam={args.cam}] T_base_cam xyz="
          f"{T_base_cam[:3, 3].round(3).tolist()}")

    # 3) T_world_cam = T_world_base · T_base_cam
    T_world_cam = T_world_base @ T_base_cam
    # 카메라 광학 위치 (world)
    cam_world_pos = T_world_cam[:3, 3]
    print(f"[anchor cam world pos] {cam_world_pos.round(3).tolist()}")

    # 4) 변환 행렬 결정
    if args.translate_only:
        # 위치만 이동 — R = I, t = -cam_world_pos
        R = np.eye(3, dtype=np.float64)
        t = -cam_world_pos.astype(np.float64)
        mode_label = "translate-only"
    else:
        # 풀 SE3 역행 — T_cam_world = inv(T_world_cam)
        T_cam_world = np.linalg.inv(T_world_cam)
        R = T_cam_world[:3, :3].astype(np.float64)
        t = T_cam_world[:3, 3].astype(np.float64)
        mode_label = "full SE3 (camera optical frame becomes new world)"
    print(f"[mode] {mode_label}")
    print(f"[R]\n{R}")
    print(f"[t] {t.tolist()}")

    # 5) Quaternion 형태 (가우시안 회전용)
    q_R = rot_to_quat_wxyz(R)
    print(f"[q_R] w={q_R[0]:.4f}  xyz=({q_R[1]:.4f},{q_R[2]:.4f},{q_R[3]:.4f})")

    # 6) PLY 로드
    raw = Path(args.ply).read_bytes()
    lines, n, hdr_end = parse_ply_header(raw)
    props = [ln.split()[-1] for ln in lines if ln.startswith("property ")]
    body = np.frombuffer(raw[hdr_end:], dtype=np.float32).reshape(n, len(props)).copy()
    idx = {p: i for i, p in enumerate(props)}
    sh_deg = infer_sh_degree(props)
    print(f"[input] {n:,} gaussians  sh_degree={sh_deg}")

    # 7) 위치 변환:  p_new = R · p_old + t
    xyz = body[:, [idx["x"], idx["y"], idx["z"]]]
    xyz_new = (xyz @ R.T.astype(np.float32)) + t.astype(np.float32)
    body[:, idx["x"]] = xyz_new[:, 0]
    body[:, idx["y"]] = xyz_new[:, 1]
    body[:, idx["z"]] = xyz_new[:, 2]

    # 8) Normal (있으면) 회전만 (translation 영향 없음)
    if all(p in idx for p in ["nx", "ny", "nz"]):
        nxyz = body[:, [idx["nx"], idx["ny"], idx["nz"]]]
        nxyz_new = nxyz @ R.T.astype(np.float32)
        body[:, idx["nx"]] = nxyz_new[:, 0]
        body[:, idx["ny"]] = nxyz_new[:, 1]
        body[:, idx["nz"]] = nxyz_new[:, 2]

    # 9) Quaternion 회전:  q_new = q_R · q_old (Hamilton 곱)
    quats = body[:, [idx["rot_0"], idx["rot_1"], idx["rot_2"], idx["rot_3"]]]
    quats_new = quat_mul_batch(q_R, quats)
    norm = np.linalg.norm(quats_new, axis=1, keepdims=True)
    quats_new = quats_new / np.clip(norm, 1e-8, None)
    body[:, idx["rot_0"]] = quats_new[:, 0]
    body[:, idx["rot_1"]] = quats_new[:, 1]
    body[:, idx["rot_2"]] = quats_new[:, 2]
    body[:, idx["rot_3"]] = quats_new[:, 3]

    # 10) SH rest 회전 (DC 는 변경 없음)
    if sh_deg == 0:
        print("[sh] degree 0 — DC only, invariant")
    elif sh_deg == 1:
        rest = np.stack(
            [body[:, idx[f"f_rest_{i}"]] for i in range(9)], axis=1
        ).reshape(n, 3, 3)
        rest_new = rotate_sh_degree1(rest, R)
        for i in range(9):
            ch, sh_i = divmod(i, 3)
            body[:, idx[f"f_rest_{i}"]] = rest_new[:, ch, sh_i]
        print("[sh] degree 1 — rest 9 coefs 회전 완료")
    else:
        print(f"[sh] WARNING: degree {sh_deg} ≥ 2 회전 미구현 (Wigner D-matrix 필요).  "
              "view-dependent 색 미세 어긋남 가능.")

    # 11) 저장
    with open(args.out, "wb") as f:
        f.write(raw[:hdr_end])
        f.write(body.tobytes())
    print(f"[saved] {args.out}  ({n:,} gaussians)")

    # 12) 검증 — 변환 후 위치 범위 + 카메라가 정말 origin 인지
    pos_min = body[:, [idx["x"], idx["y"], idx["z"]]].min(axis=0)
    pos_max = body[:, [idx["x"], idx["y"], idx["z"]]].max(axis=0)
    print(f"[verify] 변환 후 가우시안 위치 범위:")
    print(f"  x: {pos_min[0]:+.2f} ~ {pos_max[0]:+.2f}")
    print(f"  y: {pos_min[1]:+.2f} ~ {pos_max[1]:+.2f}")
    print(f"  z: {pos_min[2]:+.2f} ~ {pos_max[2]:+.2f}")
    # 카메라 origin 위치 (변환 후) 는 (R · cam_world_pos + t).  완전히 0 이어야.
    cam_after = (R @ cam_world_pos) + t
    print(f"[verify] 카메라 origin (변환 후) = {cam_after.round(6).tolist()}  (모두 0 이어야 함)")


if __name__ == "__main__":
    main()
