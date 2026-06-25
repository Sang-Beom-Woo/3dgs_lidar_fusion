"""chapter2.LidarVisualGS 학습 드라이버 (CLI).

================================================================================
 파이프라인
================================================================================

    BaseDataDataset
         │
         ├─ accumulate_seed()          # LiDAR + 이미지 투영 → seed (world)
         │                             # max-saturation voxel downsample
         │
         ├─ build_samples()            # (kf × cam) 별 학습 sample 풀
         │
         └─ LidarVisualGS + train()    # for iter in 1..N:
                                       #   랜덤 뷰 → train_step → ADC (optional)
                                       # → prune_floaters → save

================================================================================
 기본 구성 (검증 완료된 recipe)
================================================================================

  SH degree 1 · LPIPS ON · cam_refine OFF · kf_refine OFF
  ADC OFF · opacity reset OFF · regularizer OFF
  voxel 0.1 · init_scale 0.05 · iters 30000

LiDAR-seeded 데이터에선 ADC 가 표면 가우시안을 과잉 pruning 하는 문제가 확인돼
기본 OFF.  sparse SfM 시드로 시작하면 재활성 (`--densify-from-iter 500`).

================================================================================
 로그 읽기 팁
================================================================================

매 iter SGD 라 단일 뷰 loss 는 편차 크다.  수렴은 EMA (β=0.98) 값으로 본다.
ADC 사용 시 densify 직후 loss 가 잠시 튀는 것은 정상.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from base_data_parser import BaseDataDataset, build_chapter2_inputs, colorize_keyframe
from chapter2 import LidarVisualGS


# =============================================================================
# 1. Seed 포인트 누적 — LiDAR + 이미지 투영 + voxel downsample
# =============================================================================

def accumulate_seed(ds: BaseDataDataset, max_kf: int | None, stride: int,
                    voxel: float, keep_unseen: bool = False,
                    mono_depth: bool = False, mono_pixel_stride: int = 8,
                    mono_min_depth: float = 0.3, mono_max_depth: float = 30.0,
                    device: str = "cuda"
                    ) -> tuple[np.ndarray, np.ndarray]:
    """모든 (또는 선택된) 키프레임을 돌며 world 좌표 + 컬러 누적.

    mono_depth=True 면 Depth Anything v2 로 LiDAR 미커버 영역 densify.
    """
    pts_all, col_all = [], []
    used = 0
    for k, kf in enumerate(ds):
        if k % stride != 0:
            continue
        if max_kf is not None and used >= max_kf:
            break
        p, c = colorize_keyframe(kf, ds.calib, drop_unseen=not keep_unseen)
        pts_all.append(p)
        col_all.append(c)
        used += 1
    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(col_all, axis=0)
    print(f"[seed:lidar] {used} kf → raw {len(pts):,} pts")

    if mono_depth:
        # 별도 import: 패키지 없을 때 mono_depth=False 면 영향 없음
        from mono_depth_seeder import generate_mono_seeds
        mono_pts, mono_cols = generate_mono_seeds(
            ds, max_kf=max_kf, stride=stride,
            pixel_stride=mono_pixel_stride, device=device,
            min_depth=mono_min_depth, max_depth=mono_max_depth,
        )
        if mono_pts.shape[0] > 0:
            pts = np.concatenate([pts, mono_pts], axis=0)
            cols = np.concatenate([cols, mono_cols], axis=0)
            print(f"[seed:combined] LiDAR + mono = {len(pts):,} pts")

    if voxel > 0.0:
        pts, cols = _voxel_downsample_max_saturation(pts, cols, voxel)
        print(f"[seed] voxel={voxel} → {len(pts):,} pts (max-saturation representative)")
    return pts, cols


def _voxel_downsample_max_saturation(pts: np.ndarray, cols: np.ndarray,
                                     voxel: float
                                     ) -> tuple[np.ndarray, np.ndarray]:
    """Voxel downsample, 각 voxel 대표색 = 채도가 가장 높은 점의 색.

    표준 산술평균 색은 다중 뷰 섞임 시 회색으로 수렴해 신호등 같은 소물체
    색이 죽음.  max-saturation 은 voxel 내에서 가장 "색깔 있는" 점을 유지.
    """
    origin = pts.min(axis=0)
    coords = np.floor((pts - origin) / voxel).astype(np.int64)
    # 3D 정수 좌표 → 1D hash (대형 소수 조합).  순서 보존 위해 stable sort.
    keys = (coords[:, 0] * 73856093
            ^ coords[:, 1] * 19349663
            ^ coords[:, 2] * 83492791)
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]; pts_s = pts[order]; cols_s = cols[order]

    # voxel 경계
    bounds = np.concatenate(
        ([0], np.where(np.diff(keys_s) != 0)[0] + 1, [len(keys_s)])
    )
    n_vox = len(bounds) - 1

    # 채도 미리 계산 (max - min) / max
    col_max = cols_s.max(axis=1)
    sat = (col_max - cols_s.min(axis=1)) / np.maximum(col_max, 1e-6)

    out_pts = np.empty((n_vox, 3), dtype=np.float32)
    out_cols = np.empty((n_vox, 3), dtype=np.float32)
    for i in range(n_vox):
        s, e = bounds[i], bounds[i + 1]
        out_pts[i] = pts_s[s:e].mean(axis=0)
        out_cols[i] = cols_s[s + int(np.argmax(sat[s:e]))]
    return out_pts, out_cols


# =============================================================================
# 2. 학습 샘플 풀
# =============================================================================

def build_samples(ds: BaseDataDataset, cameras: list[str],
                  max_kf: int | None, stride: int,
                  exclude_kf: dict[str, set[int]] | None = None,
                  exposure_clip_pct: float = 0.0) -> list[dict]:
    """(kf, cam) 쌍별 학습 입력 dict 리스트.

    RAM 절약을 위해 두 가지 후처리:
      1) gt_image 를 float32 [0,1] → uint8 [0,255] 로 저장 (×4 작음).
         학습 시 GPU 업로드 직후 float32/255 로 복원.
      2) lidar_points_world / lidar_colors 는 train_step 이 미사용 (모델 init
         은 accumulate_seed 가 합친 통합 시드로 따로 처리됨) → 드롭.

    추가 옵션:
      exclude_kf       : {cam → set(kf_idx)}.  outlier 자동 제외 (햇빛/회전 등).
                         tools/detect_outlier_kf.py 결과 JSON 사용.
      exposure_clip_pct: 0~100, p99-pixel 이 240+ 일 때 그 픽셀들을 그 뷰
                         mean 으로 클램프 (3번 옵션).  0 = 비활성.
    """
    exclude_kf = exclude_kf or {}
    samples = []
    used = 0
    n_excluded = 0
    n_exposure_clipped = 0
    for k, kf in enumerate(ds):
        if k % stride != 0:
            continue
        if max_kf is not None and used >= max_kf:
            break
        for cam in cameras:
            if cam not in kf.images:
                continue
            if kf.idx in exclude_kf.get(cam, set()):
                n_excluded += 1
                continue
            s = build_chapter2_inputs(kf, cam, ds.calib)
            # 학습에 안 쓰는 큰 텐서 드롭 (~600 MB 절약)
            s.pop("lidar_points_world", None)
            s.pop("lidar_colors", None)
            # exposure spike 클리핑 (3번 옵션) — 과노출 픽셀을 뷰 mean 으로
            # 끌어내림.  loss 가 학습 가능 영역에 집중하게 함.
            if exposure_clip_pct > 0.0:
                img = s["gt_image"]                    # float [0,1]
                # gray 통계 — RGB 평균으로 빠른 근사
                gray = img.mean(axis=-1)
                thr = exposure_clip_pct / 100.0
                mask = gray >= thr
                if mask.any():
                    view_mean_rgb = img[~mask].mean(axis=0) if (~mask).any() \
                        else np.array([thr, thr, thr], dtype=np.float32)
                    img[mask] = view_mean_rgb
                    s["gt_image"] = img
                    n_exposure_clipped += 1
            # uint8 quantize (~11 GB 절약).  view-affine 정보 보존엔 8bit 충분.
            s["gt_image"] = (np.clip(s["gt_image"], 0.0, 1.0) * 255.0
                             ).astype(np.uint8)
            samples.append(s)
        used += 1
    print(f"[samples] {used} kf × {cameras} → {len(samples)} views  "
          f"(excluded {n_excluded} outlier views, clipped {n_exposure_clipped} bright views)")
    return samples


# =============================================================================
# 3. 학습 루프
# =============================================================================

def train(gs: LidarVisualGS, samples: list[dict], iters: int,
          log_every: int, device: str, *,
          opacity_reset_every: int = 0,
          opacity_reset_alpha: float = 0.01,
          opacity_reset_warmup: int = 500,
          densify_from_iter: int = 999_999,
          densify_until_iter: int = 15_000,
          densify_interval: int = 100,
          densify_grad_threshold: float = 2e-4,
          prune_min_opacity: float = 1e-3,
          prune_max_screen_size: float = 20.0,
          prune_max_world_scale: float = 0.1,
          scene_extent: float = 10.0,
          max_gaussians: int | None = 2_000_000,
          checkpoint_every: int = 0,
          checkpoint_callback=None,
          refine_freeze_iter: int = 0) -> None:
    """
    매 iter 랜덤 뷰 학습 + 스케줄된 ADC / opacity reset 호출.

    checkpoint_every > 0 이면 매 N iter 마다 checkpoint_callback(iter, gs) 호출.
    refine_freeze_iter > 0 이면 그 iter 직후 pose refine (cam/kf/view-affine)
        파라미터의 LR 을 0 으로 만들어 freeze — 긴 학습에서 refine 폭주 방지.
    """
    t0 = time.time()
    ema: float | None = None
    ema_beta = 0.98

    refine_frozen = False
    refine_param_names = {"cam_delta", "kf_pose_delta", "view_affine"}

    for it in range(1, iters + 1):
        loss = _one_iter_forward_backward(gs, samples, device)
        ema = loss if ema is None else ema_beta * ema + (1 - ema_beta) * loss

        _maybe_densify(gs, it, iters, densify_from_iter, densify_until_iter,
                       densify_interval, densify_grad_threshold,
                       prune_min_opacity, prune_max_screen_size,
                       prune_max_world_scale, scene_extent, max_gaussians)
        _maybe_opacity_reset(gs, it, iters, opacity_reset_every,
                             opacity_reset_alpha, opacity_reset_warmup,
                             densify_until_iter)

        # pose-refine freeze — 한 번만 실행.  warmup 동안 학습한 보정값은 유지
        # 하고 이후 LR=0 으로 고정.  cam/kf 폭주 (300k 실험에서 cam 89° 회전,
        # 1.5m 이동 등) 를 원천 차단.
        if (not refine_frozen and refine_freeze_iter > 0
                and it >= refine_freeze_iter):
            frozen = []
            for g in gs.optimizer.param_groups:
                name = g.get("name")
                if name in refine_param_names and g["lr"] > 0:
                    g["lr"] = 0.0
                    frozen.append(name)
            if frozen:
                print(f"  [refine-freeze iter {it}] LR=0 for: "
                      f"{', '.join(frozen)}")
            refine_frozen = True

        if it % log_every == 0 or it == 1:
            dt = time.time() - t0
            print(f"  iter {it:6d}/{iters}  loss={loss:.5f}  ema={ema:.5f}  "
                  f"N={gs.N:,}  ({it/dt:.1f} it/s)")

        if (checkpoint_every > 0 and checkpoint_callback is not None
                and it % checkpoint_every == 0 and it != iters):
            # 마지막 iter 는 main 의 정식 save 가 처리하므로 콜백에서 제외.
            checkpoint_callback(it, gs)


def _one_iter_forward_backward(gs: LidarVisualGS, samples: list[dict],
                               device: str) -> float:
    """랜덤 뷰 하나 업로드 → train_step.  loss float 반환."""
    s = random.choice(samples)
    # gt_image 는 build_samples 에서 uint8 [0,255] 로 압축됨 — GPU 업로드 직후
    # float32 [0,1] 로 복원.  업로드 자체도 uint8 (4×작음) 라 PCIe 대역폭에 유리.
    gt_image = torch.from_numpy(s["gt_image"]).to(device).float() / 255.0
    gt_depth = torch.from_numpy(s["gt_depth"]).to(device)
    K       = torch.from_numpy(s["K"]).to(device)
    viewmat = torch.from_numpy(s["viewmat"]).to(device)
    cam_index = int(s.get("cam_index", 0))
    kf_index = int(s.get("kf_index", 0))
    # view_index: (kf, cam) 쌍의 선형 인덱스.  per-view affine embedding 용.
    view_index = kf_index * 3 + cam_index
    return gs.train_step(gt_image, gt_depth, K, viewmat,
                         cam_index=cam_index, kf_index=kf_index,
                         view_index=view_index)


def _maybe_densify(gs: LidarVisualGS, it: int, iters: int,
                   from_iter: int, until_iter: int, interval: int,
                   grad_thr: float, min_opa: float, max_screen: float,
                   max_world: float, scene_extent: float,
                   max_gaussians: int | None) -> None:
    """[from, until) 구간에서 interval 배수마다 densify_and_prune."""
    if not (from_iter <= it < until_iter and it % interval == 0):
        return
    stats = gs.densify_and_prune(
        max_grad=grad_thr,
        min_opacity=min_opa,
        max_screen_size=max_screen,
        max_world_scale=max_world,
        scene_extent=scene_extent,
        max_gaussians=max_gaussians,
    )
    print(f"  iter {it:5d}/{iters}  [densify] "
          f"+{stats['clone']}c +{stats['split']}s -{stats['prune']}p "
          f"→ N={stats['N']:,}")


def _maybe_opacity_reset(gs: LidarVisualGS, it: int, iters: int,
                         every: int, target_alpha: float, warmup: int,
                         until_iter: int) -> None:
    """ADC 활성 구간 내에서만 주기적 reset."""
    active = (every > 0 and it >= warmup and it < until_iter and it < iters)
    if active and it % every == 0:
        gs.reset_opacity(target_alpha=target_alpha)
        print(f"  iter {it:5d}/{iters}  [opacity reset → α≤{target_alpha}]")


# =============================================================================
# 4. CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LiDAR-Visual 3DGS trainer")

    # ----- 데이터 / 출력 -----
    g_io = ap.add_argument_group("I/O")
    g_io.add_argument("--root", type=Path, default=Path("base_data"))
    g_io.add_argument("--out", type=Path, default=Path("map.ply"),
                      help="포인트클라우드 PLY (MeshLab/CloudCompare 용)")
    g_io.add_argument("--splat-out", type=Path, default=None,
                      help="INRIA 3DGS PLY (splatviz/SuperSplat 용)")
    g_io.add_argument("--cameras", nargs="+", default=["front", "left", "right"])
    g_io.add_argument("--max-kf", type=int, default=None)
    g_io.add_argument("--stride", type=int, default=1)
    g_io.add_argument("--cam-downscale", type=str, default="",
                      help='카메라별 출력 해상도 1/s 다운스케일. 예: '
                           '"left=4,right=4". 큰 카메라 (4032×3036) OOM 방지용. '
                           '빈 문자열이면 적용 없음.')

    # ----- 초기화 (seed) -----
    g_init = ap.add_argument_group("Initialization")
    g_init.add_argument("--init-voxel", type=float, default=0.1,
                        help="seed voxel (m). 0 이면 비활성")
    g_init.add_argument("--init-scale", type=float, default=None,
                        help="가우시안 초기 반경 (m). 미지정 시 init-voxel*0.5")
    g_init.add_argument("--keep-unseen", action="store_true",
                        help="카메라 FOV 밖 LiDAR 점도 회색으로 seed (기본 drop)")
    g_init.add_argument("--mono-depth", action="store_true",
                        help="Depth Anything v2 로 mono depth 시드 추가 densify")
    g_init.add_argument("--mono-pixel-stride", type=int, default=8,
                        help="mono depth 백프로젝트 픽셀 stride (작을수록 dense, 메모리 ↑)")
    g_init.add_argument("--mono-min-depth", type=float, default=0.3,
                        help="mono backproject 시 최소 depth (m)")
    g_init.add_argument("--mono-max-depth", type=float, default=30.0,
                        help="mono backproject 시 최대 depth (m). 야외면 80~100 권장")
    g_init.add_argument("--sh-degree", type=int, default=1,
                        help="SH 차수. 0~3.  1=권장 (indoor), 3=원 3DGS 표준")
    # 월드 Z 컷오프 — 야외에서 하늘/높은 건물 제거.  z up 가정 (ROS world).
    g_init.add_argument("--prune-z-max", type=float, default=None,
                        help="seed 단계에서 world Z > 이 값 인 점 제거 (m). "
                             "야외 4 = 사람 키 위 차단.  미지정 시 비활성.")
    g_init.add_argument("--prune-z-min", type=float, default=None,
                        help="seed 단계에서 world Z < 이 값 인 점 제거 (m). "
                             "지면 아래 LiDAR 노이즈 차단용.  미지정 시 비활성.")

    # ----- 학습 -----
    g_train = ap.add_argument_group("Training")
    g_train.add_argument("--iters", type=int, default=30000)
    g_train.add_argument("--log-every", type=int, default=50)
    g_train.add_argument("--seed", type=int, default=0)
    g_train.add_argument("--checkpoint-every", type=int, default=0,
                         help="N iter 마다 중간 PLY 저장.  0 = 끔.  "
                              "파일 이름: {out_stem}_{iter}.ply / {splat_stem}_{iter}.ply.  "
                              "마지막 iter 는 정식 저장 경로로 따로 저장됨.")
    g_train.add_argument("--exclude-kf-file", type=Path, default=None,
                         help="tools/detect_outlier_kf.py 결과 JSON 경로. "
                              "outlier (kf, cam) 쌍을 학습 sample 풀에서 제외.")
    g_train.add_argument("--exposure-clip-pct", type=float, default=0.0,
                         help="과노출 픽셀 클리핑 임계 (0~100, 0=비활성). "
                              "예: 95 → gray>=0.95 픽셀을 그 뷰의 (non-saturated) "
                              "mean 으로 끌어내림.  과노출 영역의 학습 압력 제거.")

    # ----- Pose refinement -----
    g_pose = ap.add_argument_group("Pose refinement (자동 calibration)")
    g_pose.add_argument("--cam-refine", action="store_true",
                        help="카메라 extrinsic drift 학습 (3 × 6DoF)")
    g_pose.add_argument("--kf-refine", action="store_true",
                        help="per-keyframe SE3 delta 학습 (BA-lite)")
    g_pose.add_argument("--view-affine", action="store_true",
                        help="per-view RGB gain+bias 학습 (NeRF-W 스타일 appearance embedding)")
    g_pose.add_argument("--refine-freeze-iter", type=int, default=0,
                        help="이 iter 이후 cam/kf/view-affine 의 LR=0 (freeze). "
                             "0 = 끝까지 학습 (기본).  긴 학습 (100k+) 에선 "
                             "30000 권장 — 300k 실험에서 pose-refine 폭주가 "
                             "모델 붕괴의 주된 원인이었음.")

    # ----- 저장 필터 / floater prune -----
    g_save = ap.add_argument_group("Save / cleanup")
    g_save.add_argument("--opacity-threshold", type=float, default=1e-3,
                        help="저장 시 α 필터")
    g_save.add_argument("--floater-prune", action="store_true",
                        help="학습 후 α 낮은 + outlier 가우시안 정리")
    g_save.add_argument("--floater-opacity", type=float, default=5e-3)
    g_save.add_argument("--floater-neighbors", type=int, default=20)
    g_save.add_argument("--floater-std", type=float, default=2.0)

    # ----- Regularizer (기본 OFF.  실험 시에만 활성) -----
    g_reg = ap.add_argument_group("Regularizers (기본 OFF)")
    g_reg.add_argument("--opacity-reg-lambda", type=float, default=None,
                       help="hinge regularizer.  미지정 시 chapter2.py 기본값")
    g_reg.add_argument("--opacity-bimodal-lambda", type=float, default=None,
                       help="bimodal regularizer.  미지정 시 chapter2.py 기본값")
    g_reg.add_argument("--aniso-lambda", type=float, default=None,
                       help="anisotropy penalty 가중치 — needle (max/min scale "
                            "비율 > max_ratio) 가우시안 길이 제한.  "
                            "0=OFF (기본).  0.01 권장 — 야외 sparse seed 학습에서 "
                            "needle 폭증 차단.")
    g_reg.add_argument("--aniso-max-ratio", type=float, default=None,
                       help="aniso 임계 — 이 비율 초과 시 penalty.  기본 5.0")

    # ----- Photometric loss weights -----
    g_loss = ap.add_argument_group("Photometric loss")
    g_loss.add_argument("--lpips-lambda", type=float, default=None,
                        help="LPIPS 가중치 (default 0.2). 0 이면 LPIPS 끔 — "
                             "고해상도+많은 가우시안일 때 GPU 메모리 확보용. "
                             "끄면 sharp 함이 약간 줄어듦.")
    g_loss.add_argument("--ssim-lambda", type=float, default=None,
                        help="SSIM 가중치 (default 0.2). 0 이면 L1 만 사용.")
    g_loss.add_argument("--lpips-downscale", type=int, default=None,
                        help="LPIPS 만 1/s 해상도로 평가 (default 1).  "
                             "2: 4× GPU 절약 (1280×1024 → 640×512), "
                             "4: 16× 절약.  6GB GPU 에 LPIPS 켤 때 권장.")
    g_loss.add_argument("--bg-color", type=str, default=None,
                        help="배경색 R,G,B (0~1).  가우시안이 안 닿는 픽셀이 "
                             "검정 대신 이 색으로 채워짐.  야외 학습 시 "
                             "하늘 자리에 거대 floater 가 생기는 부작용 차단.  "
                             "예: '0.8,0.85,0.9' (회청색 — 흐린 하늘), "
                             "'0.5,0.5,0.5' (중성 회색).  미지정 = 검정 (이전 동작).")
    g_loss.add_argument("--bg-random", action="store_true",
                        help="INRIA 3DGS 표준 — 매 iter 무작위 bg [0,1]^3.  "
                             "가우시안이 어떤 배경에서도 잘 보이도록 강제 → "
                             "opacity 자연 학습 + ADC 정상 작동.  bg-color 와 "
                             "동시 지정 시 random 우선.")

    # ----- ADC (기본 OFF — LiDAR-seeded 에선 해로움) -----
    g_adc = ap.add_argument_group("Adaptive Density Control (기본 OFF)")
    g_adc.add_argument("--densify-from-iter", type=int, default=999_999,
                       help="densify 시작 iter.  999999 = OFF (기본).  "
                            "ADC 쓸 거면 500 권장")
    g_adc.add_argument("--densify-until-iter", type=int, default=15000)
    g_adc.add_argument("--densify-interval", type=int, default=100)
    g_adc.add_argument("--densify-grad-threshold", type=float, default=2e-4)
    g_adc.add_argument("--prune-min-opacity", type=float, default=1e-3)
    g_adc.add_argument("--prune-max-screen-size", type=float, default=20.0)
    g_adc.add_argument("--prune-max-world-scale", type=float, default=None,
                       help="미지정 시 scene_extent*0.1 자동")
    g_adc.add_argument("--aspect-prune-thr", type=float, default=None,
                       help="ADC prune 의 aspect ratio 임계 (default 8). "
                            "needle 살리려면 20+ 권장.  aniso-lambda 와 함께 조정.")
    g_adc.add_argument("--scene-extent", type=float, default=None,
                       help="미지정 시 seed 범위에서 자동 계산")
    g_adc.add_argument("--max-gaussians", type=int, default=2_000_000)
    g_adc.add_argument("--opacity-reset-every", type=int, default=0,
                       help="N iter 마다 opacity reset. 0=OFF (기본)")
    g_adc.add_argument("--opacity-reset-alpha", type=float, default=0.01)
    g_adc.add_argument("--opacity-reset-warmup", type=int, default=500)

    return ap


# =============================================================================
# 5. main
# =============================================================================

def main() -> None:
    args = build_argparser().parse_args()

    _set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # 데이터셋 — --cam-downscale "left=4,right=4" 같은 문자열을 dict 로 파싱
    cam_downscale: dict[str, int] = {}
    if args.cam_downscale.strip():
        for item in args.cam_downscale.split(","):
            k, _, v = item.strip().partition("=")
            if not v:
                raise ValueError(f"--cam-downscale 항목 형식 오류: {item!r} (예: left=4)")
            cam_downscale[k.strip()] = int(v)
    ds = BaseDataDataset(args.root, require_image=True, cam_downscale=cam_downscale)
    print(f"[dataset] {len(ds)} kf")
    for cam_name, ci in ds.calib.intrinsics.items():
        tag = f" (downscale ×{ci.downscale})" if ci.downscale != 1 else ""
        print(f"[cam] {cam_name}: {ci.raw_width}×{ci.raw_height} → "
              f"{ci.width}×{ci.height}{tag}")
    pts, cols = accumulate_seed(ds, args.max_kf, args.stride, args.init_voxel,
                                keep_unseen=args.keep_unseen,
                                mono_depth=args.mono_depth,
                                mono_pixel_stride=args.mono_pixel_stride,
                                mono_min_depth=args.mono_min_depth,
                                mono_max_depth=args.mono_max_depth,
                                device=device)
    # 월드 Z 컷 — 야외에서 하늘 / 건물 위쪽 가우시안이 무거운 floater 가 되는
    # 것을 막는다.  학습 시 메모리도 직접 절약.  axis 가정: world Z up (ROS).
    if args.prune_z_max is not None or args.prune_z_min is not None:
        z = pts[:, 2]
        mask = np.ones(len(pts), dtype=bool)
        if args.prune_z_max is not None:
            mask &= (z <= args.prune_z_max)
        if args.prune_z_min is not None:
            mask &= (z >= args.prune_z_min)
        n_before = len(pts)
        pts, cols = pts[mask], cols[mask]
        print(f"[prune-z] {n_before:,} → {len(pts):,} pts  "
              f"(min={args.prune_z_min}, max={args.prune_z_max})")
    # outlier kf 제외 목록 로드 (옵션)
    exclude_kf: dict[str, set[int]] = {}
    if args.exclude_kf_file is not None:
        import json
        raw = json.loads(args.exclude_kf_file.read_text())
        exclude_kf = {cam: set(ids) for cam, ids in raw.items()}
        total = sum(len(s) for s in exclude_kf.values())
        print(f"[exclude] {args.exclude_kf_file}: {total} (cam, kf) 페어 제외 예정")
    samples = build_samples(ds, args.cameras, args.max_kf, args.stride,
                            exclude_kf=exclude_kf,
                            exposure_clip_pct=args.exposure_clip_pct)

    # 모델
    init_scale = (args.init_scale if args.init_scale is not None
                  else max(args.init_voxel * 0.5, 0.01))
    n_keyframes = max(ds.indices) + 1 if ds.indices else 0
    # view_index = kf_index * 3 + cam_index (CAM_NAMES 기준: front=0, left=1, right=2)
    # 전체 kf × 3 슬롯을 만들어두고 실제 학습된 뷰만 움직임.
    n_views = n_keyframes * 3
    gs = LidarVisualGS(
        lidar_points=torch.from_numpy(pts),
        lidar_colors=torch.from_numpy(cols),
        device=device,
        initial_scale=init_scale,
        sh_degree=args.sh_degree,
        n_cameras=len(args.cameras),
        cam_refine=args.cam_refine,
        n_keyframes=n_keyframes,
        kf_refine=args.kf_refine,
        n_views=n_views,
        view_affine=args.view_affine,
    )
    _apply_reg_overrides(gs, args)
    print(f"[init] {gs.N:,} gaussians  initial_scale={init_scale:.3f} m  "
          f"sh_degree={gs.sh_degree}")
    print(f"[reg] opacity_reg_lambda={gs.opacity_reg_lambda}  "
          f"bimodal_lambda={gs.opacity_bimodal_lambda}  "
          f"reset_every={args.opacity_reset_every}")

    # 스케일 기준 자동 계산
    extent, prune_world = _compute_scene_extent(pts, args)
    print(f"[scene] extent≈{extent:.2f} m  prune-world-scale={prune_world:.3f} m")

    # 중간 체크포인트 콜백.  --checkpoint-every 0 이면 None 으로 비활성.
    if args.checkpoint_every > 0:
        out_stem = args.out.with_suffix("")     # map.ply  → map
        splat_stem = (args.splat_out.with_suffix("")
                      if args.splat_out is not None else None)
        out_suffix = args.out.suffix or ".ply"
        splat_suffix = (args.splat_out.suffix or ".ply") if args.splat_out else ".ply"

        def _checkpoint_cb(it: int, gs_obj: LidarVisualGS) -> None:
            ckpt_pcd = Path(f"{out_stem}_{it}{out_suffix}")
            print(f"  [ckpt iter {it}] saving intermediate PLY → {ckpt_pcd.name} ...")
            ckpt_pcd.parent.mkdir(parents=True, exist_ok=True)
            gs_obj.save_ply(str(ckpt_pcd), opacity_threshold=args.opacity_threshold)
            if splat_stem is not None:
                ckpt_splat = Path(f"{splat_stem}_{it}{splat_suffix}")
                gs_obj.save_splat_ply(str(ckpt_splat),
                                       opacity_threshold=args.opacity_threshold)
        checkpoint_cb = _checkpoint_cb
    else:
        checkpoint_cb = None

    train(
        gs, samples, args.iters, args.log_every, device,
        opacity_reset_every=args.opacity_reset_every,
        opacity_reset_alpha=args.opacity_reset_alpha,
        opacity_reset_warmup=args.opacity_reset_warmup,
        densify_from_iter=args.densify_from_iter,
        densify_until_iter=args.densify_until_iter,
        densify_interval=args.densify_interval,
        densify_grad_threshold=args.densify_grad_threshold,
        prune_min_opacity=args.prune_min_opacity,
        prune_max_screen_size=args.prune_max_screen_size,
        prune_max_world_scale=prune_world,
        scene_extent=extent,
        max_gaussians=args.max_gaussians,
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=checkpoint_cb,
        refine_freeze_iter=args.refine_freeze_iter,
    )

    # 학습 후 정리 / 진단 / 저장
    if args.floater_prune:
        gs.prune_floaters(opacity_thr=args.floater_opacity,
                          nb_neighbors=args.floater_neighbors,
                          std_ratio=args.floater_std)
    _report_pose_refine(gs, args)
    _report_alpha_distribution(gs, args.opacity_threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gs.save_ply(str(args.out), opacity_threshold=args.opacity_threshold)
    if args.splat_out is not None:
        args.splat_out.parent.mkdir(parents=True, exist_ok=True)
        gs.save_splat_ply(str(args.splat_out),
                          opacity_threshold=args.opacity_threshold)


# --- main 보조 ----------------------------------------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _apply_reg_overrides(gs: LidarVisualGS, args: argparse.Namespace) -> None:
    """CLI 에서 override 한 regularizer / loss λ 를 모델에 적용."""
    if args.opacity_reg_lambda is not None:
        gs.opacity_reg_lambda = args.opacity_reg_lambda
    if args.opacity_bimodal_lambda is not None:
        gs.opacity_bimodal_lambda = args.opacity_bimodal_lambda
    if args.aniso_lambda is not None:
        gs.aniso_lambda = float(args.aniso_lambda)
    if args.aniso_max_ratio is not None:
        gs.aniso_max_ratio = float(args.aniso_max_ratio)
    if args.aspect_prune_thr is not None:
        gs.aspect_prune_thr = float(args.aspect_prune_thr)
        print(f"[adc] aspect prune threshold = {gs.aspect_prune_thr}")
    # LPIPS λ = 0 으로 끄면 lpips_net 자체도 None 으로 비워 GPU 메모리 해제.
    if args.lpips_lambda is not None:
        gs.lpips_lambda = args.lpips_lambda
        if args.lpips_lambda <= 0.0:
            gs.lpips_net = None
    if args.ssim_lambda is not None:
        gs.ssim_lambda = args.ssim_lambda
    if args.lpips_downscale is not None:
        if args.lpips_downscale < 1:
            raise ValueError(f"--lpips-downscale 는 1 이상 (got {args.lpips_downscale})")
        gs.lpips_downscale = int(args.lpips_downscale)
    if args.bg_color is not None:
        # "0.8,0.85,0.9" → [0.8, 0.85, 0.9] tensor on GPU
        try:
            rgb = [float(x) for x in args.bg_color.split(",")]
        except ValueError as e:
            raise ValueError(f"--bg-color 파싱 실패: {args.bg_color!r}") from e
        if len(rgb) != 3:
            raise ValueError(f"--bg-color 는 R,G,B 3개 (got {len(rgb)})")
        if any(v < 0.0 or v > 1.0 for v in rgb):
            raise ValueError(f"--bg-color 값은 0~1 (got {rgb})")
        gs.bg_color = torch.tensor(rgb, dtype=torch.float32, device=gs.device)
        print(f"[bg] background color = {rgb}")
    if args.bg_random:
        gs.bg_random = True
        print("[bg] random per-iter background (INRIA 3DGS 표준)")


def _compute_scene_extent(pts: np.ndarray,
                          args: argparse.Namespace) -> tuple[float, float]:
    """seed 범위로 scene_extent 자동 계산 + prune-world-scale 결정."""
    if args.scene_extent is None:
        extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)) * 0.5)
        extent = max(extent, 1.0)
    else:
        extent = args.scene_extent
    prune_world = (args.prune_max_world_scale
                   if args.prune_max_world_scale is not None else extent * 0.1)
    return extent, prune_world


def _report_pose_refine(gs: LidarVisualGS, args: argparse.Namespace) -> None:
    """학습된 cam / kf delta 통계 출력 (하드웨어팀 피드백용)."""
    if args.cam_refine:
        with torch.no_grad():
            d = gs.cam_delta.cpu().numpy()
        for i, cam in enumerate(args.cameras):
            rx, ry, rz, tx, ty, tz = d[i]
            theta = np.rad2deg(np.linalg.norm([rx, ry, rz]))
            trans = 1000 * np.linalg.norm([tx, ty, tz])
            print(f"[cam_refine] {cam:6s}  rotation={theta:+.3f}°  "
                  f"translation={trans:+.2f}mm  "
                  f"(rx={rx:+.4f} ry={ry:+.4f} rz={rz:+.4f}  "
                  f"tx={tx:+.4f} ty={ty:+.4f} tz={tz:+.4f})")

    if args.view_affine and gs.view_affine_params is not None:
        with torch.no_grad():
            va = gs.view_affine_params.cpu().numpy()    # [n_views, 6]
        # 학습된 것만 (|delta| > 0)
        used = np.abs(va).sum(axis=1) > 0
        if used.any():
            gains = np.exp(va[used, :3])         # RGB gain (multiplier)
            biases = va[used, 3:]                # RGB bias
            print(f"[view_affine] 학습된 뷰 수: {used.sum():,}/{len(va):,}")
            print(f"  gain  R: mean={gains[:,0].mean():.3f}  p5={np.percentile(gains[:,0],5):.3f}  p95={np.percentile(gains[:,0],95):.3f}")
            print(f"  gain  G: mean={gains[:,1].mean():.3f}  p5={np.percentile(gains[:,1],5):.3f}  p95={np.percentile(gains[:,1],95):.3f}")
            print(f"  gain  B: mean={gains[:,2].mean():.3f}  p5={np.percentile(gains[:,2],5):.3f}  p95={np.percentile(gains[:,2],95):.3f}")
            print(f"  bias  mean={biases.mean(axis=0).round(3).tolist()}  "
                  f"range max={np.abs(biases).max():.3f}")

    if args.kf_refine and gs.kf_pose_delta is not None:
        with torch.no_grad():
            kd = gs.kf_pose_delta.cpu().numpy()
        norms = np.linalg.norm(kd, axis=1)
        used = norms > 0
        if used.any():
            rot_deg = np.rad2deg(np.linalg.norm(kd[used, :3], axis=1))
            trans_mm = 1000 * np.linalg.norm(kd[used, 3:], axis=1)
            print(f"[kf_refine] 학습된 kf 수: {used.sum():,}/{len(kd):,}")
            print(f"  rotation (°)    mean={rot_deg.mean():.3f}  "
                  f"p95={np.percentile(rot_deg,95):.3f}  max={rot_deg.max():.3f}")
            print(f"  translation (mm)  mean={trans_mm.mean():.2f}  "
                  f"p95={np.percentile(trans_mm,95):.2f}  max={trans_mm.max():.2f}")
            top_idx = np.argsort(-trans_mm)[:3]
            kfs_used = np.where(used)[0]
            top_kfs = kfs_used[top_idx]
            print(f"  최대 drift kf: "
                  + ", ".join(f"kf#{k}({trans_mm[i]:.1f}mm)"
                              for i, k in zip(top_idx, top_kfs)))


def _report_alpha_distribution(gs: LidarVisualGS, save_threshold: float) -> None:
    """α percentile 통계 — 학습 건강성 체크."""
    with torch.no_grad():
        alpha = torch.sigmoid(gs.opacities_raw).cpu().numpy()
    q = np.percentile(alpha, [5, 25, 50, 75, 95])
    print(f"[alpha] min={alpha.min():.4f}  p5={q[0]:.4f}  p50={q[2]:.4f}  "
          f"p95={q[4]:.4f}  max={alpha.max():.4f}  "
          f"α>thr({save_threshold}): "
          f"{(alpha > save_threshold).sum():,}/{len(alpha):,}")


if __name__ == "__main__":
    main()
