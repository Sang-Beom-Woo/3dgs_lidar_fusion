"""INRIA 3DGS PLY 축 재정렬 (post-hoc).

웹 기반 3DGS 뷰어 (SuperSplat, splatviz, antimatter15 등) 는 보통 **Y-up**
컨벤션 (Y 가 위, -Z 가 카메라 정면).  ROS world frame 으로 학습된 splat 은
**Z-up** (Z 가 위, X 가 정면) 이라 뷰어에서 옆으로 눕거나 거꾸로 보임.

기본 모드 "zup-to-yup": X 축 -90° 회전.
    x_new = x_old
    y_new = z_old      (ROS 의 위 → 뷰어의 위)
    z_new = -y_old     (ROS 의 왼쪽 → 뷰어의 뒤)

변환되는 속성:
  1. 위치 (x, y, z)
  2. 가우시안 회전 quaternion (rot_0..3 = w, x, y, z)
  3. SH degree 1 rest coefs (view-dependent 색)
  4. f_dc, opacity, scale 등은 rotation-invariant 라 변경 없음
  5. sh_degree ≥ 2 면 SH rest 회전 미지원 (Wigner D-matrix 필요) → 경고 후 그대로 유지

사용:
  python3 tools/reorient.py map_splat.ply --out map_splat_yup.ply

  # 임의 축/각도 회전
  python3 tools/reorient.py map_splat.ply --axis x --deg -90 --out map_splat_rot.ply

  # SuperSplat 같은 일부 뷰어는 Z forward / -Y up 등 변종 → 시행착오 필요
  python3 tools/reorient.py map_splat.ply --axis x --deg 90 --out map_splat_flip.ply
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


# -----------------------------------------------------------------------------
# 회전 헬퍼
# -----------------------------------------------------------------------------
def rot_axis_deg(axis: str, deg: float) -> np.ndarray:
    """단일 축 둘레 회전 행렬 [3, 3]."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"unknown axis: {axis}")


def rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3×3 회전행렬 → quaternion (w, x, y, z).  Shepperd 알고리즘 (수치 안정)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def quat_mul_batch(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """quaternion 곱:  q_left * q_right.  shapes:
       q_left  [4]            (회전 행렬에 대응하는 단일 quat)
       q_right [N, 4]         (per-gaussian quats)
       반환     [N, 4]         (Hamilton 곱, w,x,y,z 순서)

    가우시안의 원래 orientation 위에 R 을 좌측에서 곱해 world 회전을 흡수.
    """
    w1, x1, y1, z1 = q_left
    w2, x2, y2, z2 = q_right[:, 0], q_right[:, 1], q_right[:, 2], q_right[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=1).astype(np.float32)


def rotate_sh_degree1(coefs: np.ndarray, R: np.ndarray) -> np.ndarray:
    """SH degree 1 rest coefs 를 world 회전 R 에 맞춰 변환.

    INRIA 3DGS SH 컨벤션 (실수 SH, 정규화 상수 SH_C1 = sqrt(3/4π)/2):
        basis_0(d) = SH_C1 · (-d.y)
        basis_1(d) = SH_C1 · ( d.z)
        basis_2(d) = SH_C1 · (-d.x)

    coefs [N, 3, 3] = (가우시안, 채널 R/G/B, sh_idx 0/1/2).

    한 채널 색 기여 = c0·(-y) + c1·(z) + c2·(-x) = m · d,  m = (-c2, -c0, c1).
    회전 후 invariance: m_new = R · m_old.
    역변환:  c2_new = -m_new.x,  c0_new = -m_new.y,  c1_new = m_new.z.
    """
    # m = (-c2, -c0, c1).  shape [N, 3, 3] → reshape per channel.
    c0, c1, c2 = coefs[..., 0], coefs[..., 1], coefs[..., 2]    # [N, 3]
    mx = -c2
    my = -c0
    mz = c1
    m = np.stack([mx, my, mz], axis=-1)                          # [N, 3, 3] (가우시안, 채널, xyz)
    # R @ m^T  per gaussian, per channel — einsum 으로 일괄 적용
    # m shape: [N, 3, 3]  (N=가우시안, 3=채널, 3=xyz)
    # 우리는 마지막 차원만 R 로 회전 → m_new = m @ R^T
    m_new = m @ R.T.astype(np.float32)                           # [N, 3, 3]
    c0_new = -m_new[..., 1]
    c1_new = m_new[..., 2]
    c2_new = -m_new[..., 0]
    return np.stack([c0_new, c1_new, c2_new], axis=-1).astype(np.float32)


# -----------------------------------------------------------------------------
# PLY I/O
# -----------------------------------------------------------------------------
def parse_ply_header(data: bytes) -> tuple[list[str], int, int]:
    header_end = data.find(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii", errors="ignore")
    lines = header.splitlines()
    n = int(re.search(r"element vertex (\d+)", header).group(1))
    return lines, n, header_end


def infer_sh_degree(props: list[str]) -> int:
    """f_rest 갯수에서 SH degree 역추정.  K = (degree+1)², rest = (K-1)·3."""
    n_rest = sum(1 for p in props if p.startswith("f_rest_"))
    rest_per_channel = n_rest // 3                # 채널 3 (R, G, B)
    K = rest_per_channel + 1                      # DC 포함 총 SH 개수
    deg = int(np.sqrt(K)) - 1
    return deg


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", type=str, default="zup-to-yup",
                    choices=["zup-to-yup", "custom"],
                    help="zup-to-yup: 기본 ROS Z-up → 뷰어 Y-up "
                         "(X 축 -90° 회전).  custom: --axis/--deg 직접 지정.")
    ap.add_argument("--axis", type=str, default="x", choices=["x", "y", "z"])
    ap.add_argument("--deg", type=float, default=-90.0)
    args = ap.parse_args()

    if args.mode == "zup-to-yup":
        # ROS world (Z up, X forward) → viewer (Y up, -Z forward)
        # X 축 둘레 -90°:  (x, y, z) → (x, z, -y)
        R = rot_axis_deg("x", -90.0)
        print(f"[mode] zup-to-yup  (R = rot_x(-90°))")
    else:
        R = rot_axis_deg(args.axis, args.deg)
        print(f"[mode] custom  axis={args.axis}  deg={args.deg}")
    q_R = rot_to_quat_wxyz(R)                                  # 회전을 quat 으로
    print(f"[R]\n{R}")
    print(f"[q_R] w={q_R[0]:.4f}  x={q_R[1]:.4f}  y={q_R[2]:.4f}  z={q_R[3]:.4f}")

    raw = Path(args.ply).read_bytes()
    lines, n, hdr_end = parse_ply_header(raw)
    props = [ln.split()[-1] for ln in lines if ln.startswith("property ")]
    body = np.frombuffer(raw[hdr_end:], dtype=np.float32).reshape(n, len(props)).copy()
    idx = {p: i for i, p in enumerate(props)}

    sh_deg = infer_sh_degree(props)
    print(f"[input] {n:,} gaussians  sh_degree={sh_deg}")

    # 1) 위치 회전
    xyz = body[:, [idx["x"], idx["y"], idx["z"]]]
    xyz_new = xyz @ R.T.astype(np.float32)                     # (N, 3) · R^T  =  R · xyz^T
    body[:, idx["x"]] = xyz_new[:, 0]
    body[:, idx["y"]] = xyz_new[:, 1]
    body[:, idx["z"]] = xyz_new[:, 2]

    # 2) Normal — 보통 0 으로 저장되지만 만약 값 있으면 같이 회전
    if all(p in idx for p in ["nx", "ny", "nz"]):
        nxyz = body[:, [idx["nx"], idx["ny"], idx["nz"]]]
        nxyz_new = nxyz @ R.T.astype(np.float32)
        body[:, idx["nx"]] = nxyz_new[:, 0]
        body[:, idx["ny"]] = nxyz_new[:, 1]
        body[:, idx["nz"]] = nxyz_new[:, 2]

    # 3) Quaternion 회전 — q_new = q_R · q_old.  PLY 는 (w, x, y, z) 순서.
    quats = body[:, [idx["rot_0"], idx["rot_1"], idx["rot_2"], idx["rot_3"]]]
    quats_new = quat_mul_batch(q_R, quats)
    # 정규화 (수치 안전)
    quats_new = quats_new / np.linalg.norm(quats_new, axis=1, keepdims=True)
    body[:, idx["rot_0"]] = quats_new[:, 0]
    body[:, idx["rot_1"]] = quats_new[:, 1]
    body[:, idx["rot_2"]] = quats_new[:, 2]
    body[:, idx["rot_3"]] = quats_new[:, 3]

    # 4) SH rest 회전
    n_rest = sum(1 for p in props if p.startswith("f_rest_"))
    if sh_deg == 0:
        print("[sh] degree 0 — DC only, rotation-invariant.  변경 없음.")
    elif sh_deg == 1:
        # rest 9개 = 채널 R/G/B 각각 sh_idx 0/1/2.
        # INRIA 컨벤션: 채널-major (R_sh0,R_sh1,R_sh2,G_sh0,G_sh1,G_sh2,B_sh0,B_sh1,B_sh2)
        rest = np.stack(
            [body[:, idx[f"f_rest_{i}"]] for i in range(9)], axis=1
        ).reshape(n, 3, 3)                                     # [N, ch, sh_idx]
        rest_new = rotate_sh_degree1(rest, R)
        for i in range(9):
            ch, sh_i = divmod(i, 3)
            body[:, idx[f"f_rest_{i}"]] = rest_new[:, ch, sh_i]
        print("[sh] degree 1 — rest 9 coefs 회전 완료.")
    else:
        print(f"[sh] WARNING: degree {sh_deg} ≥ 2 SH 회전 미구현 "
              f"(Wigner D-matrix 필요).  view-dependent 색이 회전 후 어긋날 수 있음. "
              f"DC 색은 정상.")

    # 저장 — 헤더 그대로 (vertex 수 변동 없음)
    with open(args.out, "wb") as f:
        f.write(raw[:hdr_end])                                 # 원본 헤더 그대로
        f.write(body.tobytes())
    print(f"[saved] {args.out}  ({n:,} gaussians)")


if __name__ == "__main__":
    main()
