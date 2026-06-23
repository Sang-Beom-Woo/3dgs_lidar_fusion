"""학습 품질을 떨어뜨리는 outlier kf 자동 감지.

기준 (sliding-window 평균 대비 임계 초과):
  1. 밝기 outlier   — gray mean 이 이웃 평균보다 +X 이상 (과노출 / 햇빛 정면)
  2. 회전 outlier   — 인접 kf 간 회전 변화가 임계 이상 (rolling-scan 왜곡 위험)
  3. 포화 픽셀 비율 — 8-bit 클리핑된 픽셀 비율 임계 이상

각 카메라별로 독립 평가.  최종 출력은
  exclude_kf.json  →  {"front": [184, ...], "left": [...], "right": [...]}
형태로 저장 → run_chapter2.py 의 --exclude-kf-file 로 학습 시 skip.

사용:
  python3 tools/detect_outlier_kf.py --root base_data \
      --bright-delta 12 --rot-deg 30 --saturation-pct 13 \
      --out exclude_kf.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from base_data_parser import BaseDataDataset, _parse_pose_txt


CAM_NAMES = ("front", "left", "right")


def _image_stats(img_path: Path) -> tuple[float, float] | None:
    """이미지에서 (gray_mean, sat_pct) 반환.  파일 없으면 None."""
    if not img_path.exists():
        return None
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float((gray >= 250).mean() * 100)


def _rot_delta_deg(T_a: np.ndarray, T_b: np.ndarray) -> float:
    """두 SE3 간 회전 차이 (deg)."""
    dR = T_b[:3, :3] @ T_a[:3, :3].T
    cos_t = float(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_t)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("base_data"))
    ap.add_argument("--out", type=Path, default=Path("exclude_kf.json"))
    ap.add_argument("--window", type=int, default=5,
                    help="밝기 sliding window 크기 (인접 kf 수, 기본 5)")
    ap.add_argument("--bright-delta", type=float, default=12.0,
                    help="이웃 평균 대비 gray-mean 차이 임계 (기본 12). "
                         "이 값 이상이면 밝기 outlier — 보통 햇빛 정면.")
    ap.add_argument("--saturation-pct", type=float, default=13.0,
                    help="포화 픽셀 (>=250) 비율 임계 % (기본 13). "
                         "이 값 이상이면 과노출 outlier.")
    ap.add_argument("--rot-deg", type=float, default=30.0,
                    help="인접 kf 회전 임계 deg (기본 30).  "
                         "이 값 이상이면 빠른 회전 outlier — LiDAR rolling-scan 왜곡.")
    ap.add_argument("--rot-applies-to", type=str, default="front",
                    choices=["front", "all"],
                    help="회전 outlier 를 front 만 제외할지 (기본), 3 카메라 모두 제외할지.")
    args = ap.parse_args()

    ds = BaseDataDataset(args.root, require_image=False)
    indices = ds.indices
    n = len(indices)
    print(f"[outlier] {n} keyframes 분석 시작")

    # 1) 카메라별 밝기 / 포화도
    cam_stats: dict[str, list[tuple[int, float, float]]] = {c: [] for c in CAM_NAMES}
    for kf_idx in indices:
        for cam in CAM_NAMES:
            stats = _image_stats(args.root / str(kf_idx) / f"{cam}_color.bmp")
            if stats is None:
                continue
            cam_stats[cam].append((kf_idx, stats[0], stats[1]))

    # 2) 회전 변화 (front 기준)
    poses: dict[int, np.ndarray] = {}
    for kf_idx in indices:
        _ts, T = _parse_pose_txt(args.root / str(kf_idx) / "keyframe_pose.txt")
        poses[kf_idx] = T
    rot_outlier_kfs: list[int] = []
    for i in range(1, n):
        a, b = indices[i - 1], indices[i]
        deg = _rot_delta_deg(poses[a], poses[b])
        if deg >= args.rot_deg:
            rot_outlier_kfs.append(b)
    print(f"[outlier] 회전 outlier ({args.rot_deg}°+): {len(rot_outlier_kfs)} kf — "
          f"{rot_outlier_kfs[:10]}{'...' if len(rot_outlier_kfs)>10 else ''}")

    # 3) 카메라별 sliding-window 밝기 차이 + 포화도
    exclude: dict[str, set[int]] = {c: set() for c in CAM_NAMES}
    for cam, stats_list in cam_stats.items():
        stats_list.sort(key=lambda x: x[0])
        idx_arr = np.asarray([s[0] for s in stats_list])
        bright = np.asarray([s[1] for s in stats_list])
        sat = np.asarray([s[2] for s in stats_list])
        # sliding window mean (자기 자신 제외)
        bright_neighbor_mean = np.zeros_like(bright)
        for i in range(len(bright)):
            lo = max(0, i - args.window)
            hi = min(len(bright), i + args.window + 1)
            mask = np.ones(hi - lo, dtype=bool)
            mask[i - lo] = False
            bright_neighbor_mean[i] = bright[lo:hi][mask].mean()
        bright_delta = bright - bright_neighbor_mean

        n_bright = int((bright_delta >= args.bright_delta).sum())
        n_sat = int((sat >= args.saturation_pct).sum())
        for i in range(len(bright)):
            if bright_delta[i] >= args.bright_delta:
                exclude[cam].add(int(idx_arr[i]))
            if sat[i] >= args.saturation_pct:
                exclude[cam].add(int(idx_arr[i]))
        print(f"[outlier] {cam}: bright Δ>={args.bright_delta} → {n_bright} kf, "
              f"포화>={args.saturation_pct}% → {n_sat} kf  "
              f"(중복 제거 후 총 {len(exclude[cam])} kf)")

    # 4) 회전 outlier 추가
    if args.rot_applies_to == "all":
        for cam in CAM_NAMES:
            exclude[cam].update(rot_outlier_kfs)
    else:
        exclude["front"].update(rot_outlier_kfs)

    # 5) 저장
    out_dict = {cam: sorted(list(s)) for cam, s in exclude.items()}
    total = sum(len(v) for v in out_dict.values())
    print(f"\n[outlier] 최종 제외 카운트 (cam 별):")
    for cam, ids in out_dict.items():
        print(f"  {cam}: {len(ids)} kf  → {ids[:15]}{'...' if len(ids)>15 else ''}")
    print(f"[outlier] 총 {total} (cam, kf) 페어 제외 / {n*3} 가능")
    args.out.write_text(json.dumps(out_dict, indent=2))
    print(f"[outlier] saved {args.out}")


if __name__ == "__main__":
    main()
