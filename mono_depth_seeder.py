"""Depth Anything v2 기반 mono depth 시드 생성.

목적:  LiDAR 가 못 본 영역 (천장, 유리, 먼 벽) 을 mono depth 추정으로 메워
가우시안 시드를 densify.

원리:
  1) Depth Anything v2 가 이미지에서 disparity-like 추정 (값 클수록 가까움).
  2) 같은 뷰의 LiDAR 점들로 metric 스케일 캘리브레이트:
         metric_depth = a / mono_pred + b
     (robust linear regression on disparity space)
  3) 캘리브 후 dense metric depth → 픽셀 backproject → world 좌표 3D 점
  4) stride 로 subsample (모든 픽셀 → 메모리 폭증 방지)
  5) 컬러는 이미지에서 직접 샘플

LiDAR 시드와 union → voxel downsample → 가우시안 초기화.

사용 (run_chapter2.py 의 --mono-depth 플래그로 자동 호출).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from base_data_parser import (
    CAM_NAMES, BaseDataDataset, project_lidar_to_camera,
)

_DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"


# =============================================================================
# 모델 로드 (한 번만)
# =============================================================================

def load_depth_model(device: str = "cuda"):
    """Depth Anything v2 Small (~100MB) 로드.  반환: (processor, model)."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(_DEPTH_MODEL_NAME, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(_DEPTH_MODEL_NAME).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[mono] Depth Anything v2 Small loaded  ({n_params:.1f}M params)")
    return proc, model


@torch.no_grad()
def predict_disparity(model, proc, img_rgb_u8: np.ndarray, device: str) -> np.ndarray:
    """이미지 [H, W, 3] uint8 RGB → disparity-like map [H, W] float32.

    값이 클수록 가까운 표면.  스케일은 임의 (캘리브레이트로 메트릭화 필요).
    """
    H, W = img_rgb_u8.shape[:2]
    inputs = proc(images=Image.fromarray(img_rgb_u8), return_tensors="pt").to(device)
    out = model(**inputs).predicted_depth                            # [1, h', w']
    out = torch.nn.functional.interpolate(
        out.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    return out.astype(np.float32)


# =============================================================================
# 캘리브레이션 — LiDAR 점으로 메트릭 스케일 추정
# =============================================================================

def calibrate_to_lidar(mono_pred: np.ndarray,
                       uv: np.ndarray,
                       depth_lidar: np.ndarray,
                       min_inliers: int = 50,
                       max_iters: int = 3,
                       sigma_thr: float = 3.0,
                       ) -> tuple[float, float, float] | None:
    """metric_depth = a / mono + b 로 fit (LiDAR 점이 GT).

    Returns (a, b, rmse) 또는 inlier 부족 시 None.
    """
    if uv.shape[0] < min_inliers:
        return None

    H, W = mono_pred.shape
    ui = np.clip(uv[:, 0].astype(int), 0, W - 1)
    vi = np.clip(uv[:, 1].astype(int), 0, H - 1)
    mono_at = mono_pred[vi, ui]

    # 1/(mono + eps) 로 변환된 design matrix
    eps = 1e-6
    x = 1.0 / (mono_at + eps)
    A = np.stack([x, np.ones_like(x)], axis=1)
    y = depth_lidar.astype(np.float64)

    # 초기 fit
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = sol

    # σ-clipping 반복 (간이 RANSAC)
    keep = np.ones(len(y), dtype=bool)
    for _ in range(max_iters):
        res = y[keep] - (A[keep] @ np.array([a, b]))
        sigma = np.std(res)
        if sigma < 1e-3:
            break
        new_keep = np.abs(y - (A @ np.array([a, b]))) < sigma_thr * sigma
        if new_keep.sum() < min_inliers or new_keep.sum() == keep.sum():
            break
        keep = new_keep
        sol, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        a, b = sol

    res_all = y - (A @ np.array([a, b]))
    rmse = float(np.sqrt((res_all[keep] ** 2).mean()))
    return float(a), float(b), rmse


# =============================================================================
# Backprojection
# =============================================================================

def backproject_pixels(depth: np.ndarray, K: np.ndarray,
                       T_world_cam: np.ndarray, img_rgb_u8: np.ndarray,
                       stride: int = 8,
                       min_depth: float = 0.3,
                       max_depth: float = 30.0,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """stride 픽셀마다 depth 로 3D 백프로젝트 → (world XYZ, RGB 0~1)."""
    H, W = depth.shape
    vs, us = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    us = us.ravel(); vs = vs.ravel()
    d = depth[vs, us]

    # range filter + NaN/Inf 가드
    valid = np.isfinite(d) & (d > min_depth) & (d < max_depth)
    us = us[valid]; vs = vs[valid]; d = d[valid]
    if len(d) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (us - cx) / fx * d
    y_cam = (vs - cy) / fy * d
    pts_cam = np.stack([x_cam, y_cam, d, np.ones_like(d)], axis=1)
    pts_world = (T_world_cam @ pts_cam.T).T[:, :3].astype(np.float32)
    cols = (img_rgb_u8[vs, us].astype(np.float32) / 255.0)
    return pts_world, cols


# =============================================================================
# 메인 API — 데이터셋 전체 처리
# =============================================================================

def generate_mono_seeds(ds: BaseDataDataset,
                        max_kf: int | None = None,
                        stride: int = 1,
                        pixel_stride: int = 8,
                        min_depth: float = 0.3,
                        max_depth: float = 30.0,
                        device: str = "cuda",
                        verbose: bool = True,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """전체 학습 뷰에 Depth Anything 적용 → 모든 backproject 점 모음.

    Args:
        max_kf:        이 수만큼 kf 만 처리 (None=전체).
        stride:        keyframe stride (1=전체, 2=절반).
        pixel_stride:  뷰당 backproject 픽셀 stride.  8 이면 한 뷰당 ~3.6k 점.
    Returns:
        pts  [M, 3]  world XYZ float32
        cols [M, 3]  RGB 0~1 float32
    """
    proc, model = load_depth_model(device)
    pts_all, col_all = [], []
    used = 0
    failed_calib = 0

    for k, kf in enumerate(ds):
        if k % stride != 0:
            continue
        if max_kf is not None and used >= max_kf:
            break
        used += 1

        for cam_name in CAM_NAMES:
            if cam_name not in kf.images:
                continue
            cam_intr = ds.calib.intrinsics[cam_name]

            # 1) 언디스토션된 RGB
            img_bgr = cv2.remap(kf.images[cam_name], cam_intr.map1, cam_intr.map2,
                                cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 2) Mono disparity 추정
            mono_pred = predict_disparity(model, proc, img_rgb, device)

            # 3) LiDAR 로 캘리브
            pts_base = kf.points_in("base", ds.calib)
            uv, depth_lidar, _ = project_lidar_to_camera(pts_base, cam_name, ds.calib)
            calib_res = calibrate_to_lidar(mono_pred, uv, depth_lidar)
            if calib_res is None:
                failed_calib += 1
                continue
            a, b, rmse = calib_res

            # 4) 메트릭 depth → 백프로젝트
            depth_metric = a / (mono_pred + 1e-6) + b
            T_world_cam = kf.T_world_base @ ds.calib.T_base_cam[cam_name]
            pts, cols = backproject_pixels(depth_metric, cam_intr.new_K,
                                            T_world_cam, img_rgb,
                                            stride=pixel_stride,
                                            min_depth=min_depth,
                                            max_depth=max_depth)
            if pts.shape[0] > 0:
                pts_all.append(pts)
                col_all.append(cols)

    if verbose:
        n_total = sum(p.shape[0] for p in pts_all)
        print(f"[mono] {used} kf 처리, {failed_calib} 캘리브 실패, "
              f"총 {n_total:,} 점 생성 (pixel_stride={pixel_stride})")

    if not pts_all:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32))
    return np.concatenate(pts_all, axis=0), np.concatenate(col_all, axis=0)
