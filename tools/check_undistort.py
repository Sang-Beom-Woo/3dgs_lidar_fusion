"""Fisheye 언디스토션 적합성 진단.

체크 항목:
  1) 캘리브레이션 값 자체의 sanity (f, principal point, D 계수 범위)
  2) new_K (undistort 후 intrinsic) 가 합리적인지
  3) remap 후 결은 line 이 직선으로 복원되는지 — 원본/언디스토션 나란히 시각화
  4) FOV 손실 여부 (balance 파라미터 영향)
  5) 언디스토션 후 실제로 LiDAR 투영이 잘 맞는지 (reprojection error)

사용:
  python3 tools/check_undistort.py --kf 100 --cam front --out calib_check/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from base_data_parser import BaseDataDataset, project_lidar_to_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("base_data"))
    ap.add_argument("--kf", type=int, default=100, help="검사할 keyframe id")
    ap.add_argument("--cam", default="front", choices=["front", "left", "right"])
    ap.add_argument("--out", type=Path, default=Path("calib_check"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ds = BaseDataDataset(args.root, require_image=True)
    calib = ds.calib
    cam_intr = calib.intrinsics[args.cam]

    # --- 1) 파라미터 sanity 체크 ----------------------------------------
    K = cam_intr.K
    D = cam_intr.D
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    W, H = cam_intr.width, cam_intr.height
    print("=" * 60)
    print(f"카메라: {args.cam}  ({W} × {H})")
    print("=" * 60)
    print(f"[원본 K]")
    print(f"  fx = {fx:.3f}   fy = {fy:.3f}   aspect = {fy/fx:.4f}")
    print(f"  cx = {cx:.3f}  ({100*cx/W:.1f}% of width)")
    print(f"  cy = {cy:.3f}  ({100*cy/H:.1f}% of height)")
    print(f"  HFOV ≈ {np.rad2deg(2*np.arctan(W/(2*fx))):.1f}°  "
          f"VFOV ≈ {np.rad2deg(2*np.arctan(H/(2*fy))):.1f}°")
    print(f"  (fisheye equidistant 모델이라 위 HFOV 는 '핀홀 가정' 근사, 실제는 더 넓음)")
    print(f"\n[왜곡 계수 (equidistant k1..k4)]")
    print(f"  D = {D}")
    k_mag = np.abs(D).max()
    if k_mag > 1.0:
        print(f"  ⚠️  max |k| = {k_mag:.3f} — 매우 큰 왜곡. 값 정상인지 의심")
    elif k_mag > 0.5:
        print(f"  ⚠️  max |k| = {k_mag:.3f} — 왜곡 큼 (매우 wide fisheye 일 수 있음)")
    else:
        print(f"  ✅  max |k| = {k_mag:.3f} — fisheye 로서 합리적 범위")

    # --- 2) new_K 분석 --------------------------------------------------
    nK = cam_intr.new_K
    nfx, nfy = nK[0, 0], nK[1, 1]
    ncx, ncy = nK[0, 2], nK[1, 2]
    print(f"\n[언디스토션 new_K]")
    print(f"  fx = {nfx:.3f}  fy = {nfy:.3f}")
    print(f"  cx = {ncx:.3f}  cy = {ncy:.3f}")
    print(f"  원본 대비 fx 비율: {nfx/fx:.3f}  (1.0 유사 = FOV 유지)")
    print(f"  언디스토션 후 HFOV ≈ {np.rad2deg(2*np.arctan(W/(2*nfx))):.1f}°  "
          f"VFOV ≈ {np.rad2deg(2*np.arctan(H/(2*nfy))):.1f}°")

    # --- 3) 이미지 원본/언디스토션 비교 -------------------------------
    kf = None
    for k in ds:
        if k.idx == args.kf:
            kf = k
            break
    if kf is None or args.cam not in kf.images:
        # fallback: 첫 가능한 것
        kf = next(iter(ds))
        args.cam = next(iter(kf.images.keys()))
        print(f"⚠️  요청한 kf#{args.kf}/{args.cam} 없음. kf#{kf.idx}/{args.cam} 로 대체")

    img_bgr = kf.images[args.cam]

    # 직접 cv2.fisheye.undistortImage 로 전체 영역 보존 언디스토션
    # balance=1.0 은 모든 원본 픽셀 보존 (검은 가장자리 생김)
    full_map1, full_map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3),
        cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (W, H), np.eye(3), balance=1.0),
        (W, H), cv2.CV_16SC2
    )
    img_full_und = cv2.remap(img_bgr, full_map1, full_map2, cv2.INTER_LINEAR)

    # 우리 파이프라인 그대로 (balance=0, 모든 픽셀 유효)
    img_our_und = cv2.remap(img_bgr, cam_intr.map1, cam_intr.map2, cv2.INTER_LINEAR)

    # 가로로 붙여 비교
    gap = np.zeros((H, 20, 3), dtype=np.uint8) + 50
    panel = np.concatenate([img_bgr, gap, img_our_und, gap, img_full_und], axis=1)
    for x, label in [(10, "Original (distorted)"),
                     (W + 30, "Our undistort (balance=0, zoom in)"),
                     (2*W + 50, "Full undistort (balance=1, black border)")]:
        cv2.putText(panel, label, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
    out_panel = args.out / f"{args.cam}_kf{kf.idx}_compare.png"
    cv2.imwrite(str(out_panel), panel)
    print(f"\n[compare panel] → {out_panel}")

    # --- 4) LiDAR 재투영 정확도 확인 ---------------------------------
    # LiDAR 점들을 언디스토션 이미지에 점찍기 → 구조 선과 일치하는지
    pts_base = kf.points_in("base", calib)
    uv, depth, mask = project_lidar_to_camera(pts_base, args.cam, calib)
    if uv.shape[0] > 0:
        # depth 로 색 입히기 (가까울수록 빨강, 멀수록 파랑)
        d_norm = np.clip((depth - depth.min()) / max(depth.max() - depth.min(), 1e-6), 0, 1)
        img_overlay = img_our_und.copy()
        for i in range(uv.shape[0]):
            u, v = int(uv[i, 0]), int(uv[i, 1])
            c = (int(255 * (1 - d_norm[i])), 0, int(255 * d_norm[i]))   # BGR
            cv2.circle(img_overlay, (u, v), 1, c, -1)
        cv2.putText(img_overlay, f"LiDAR reprojection ({uv.shape[0]} pts)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        overlay_path = args.out / f"{args.cam}_kf{kf.idx}_reproj.png"
        cv2.imwrite(str(overlay_path), img_overlay)
        print(f"[reproj overlay] → {overlay_path}")
        print(f"  LiDAR 점 {uv.shape[0]:,} 개 재투영, depth range {depth.min():.2f}~{depth.max():.2f} m")
        print(f"  → 점들이 이미지의 실제 구조물 (바닥/벽/기둥) 윤곽에 맞아야 정상.")

    # --- 5) 직선 검증용 — Canny edge 직선성 ---------------------------
    # 언디스토션된 이미지에서 Hough line detection → 직선이 얼마나 잘 잡히는지
    gray_our = cv2.cvtColor(img_our_und, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_our, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                             minLineLength=50, maxLineGap=10)
    n_lines = 0 if lines is None else len(lines)
    print(f"\n[직선 검증] HoughLinesP 에서 검출된 선분: {n_lines}개")
    print(f"  왜곡이 잘 보정되어 있으면 씬의 벽/천장 경계가 직선으로 잡혀야 함.")
    print(f"  (이 숫자는 씬 복잡도에 따라 다르지만 수십~수백 개가 일반적)")

    if lines is not None:
        img_lines = img_our_und.copy()
        for line in lines[:200]:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(img_lines, f"Hough lines: {n_lines}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        lines_path = args.out / f"{args.cam}_kf{kf.idx}_lines.png"
        cv2.imwrite(str(lines_path), img_lines)
        print(f"[hough lines] → {lines_path}")


if __name__ == "__main__":
    main()
