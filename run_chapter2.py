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
                    voxel: float, keep_unseen: bool = False
                    ) -> tuple[np.ndarray, np.ndarray]:
    """모든 (또는 선택된) 키프레임을 돌며 world 좌표 + 컬러 누적."""
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
    print(f"[seed] {used} kf → raw {len(pts):,} pts")

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
                  max_kf: int | None, stride: int) -> list[dict]:
    """(kf, cam) 쌍별 학습 입력 dict 리스트."""
    samples = []
    used = 0
    for k, kf in enumerate(ds):
        if k % stride != 0:
            continue
        if max_kf is not None and used >= max_kf:
            break
        for cam in cameras:
            if cam not in kf.images:
                continue
            samples.append(build_chapter2_inputs(kf, cam, ds.calib))
        used += 1
    print(f"[samples] {used} kf × {cameras} → {len(samples)} views")
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
          max_gaussians: int | None = 2_000_000) -> None:
    """
    매 iter 랜덤 뷰 학습 + 스케줄된 ADC / opacity reset 호출.
    """
    t0 = time.time()
    ema: float | None = None
    ema_beta = 0.98

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

        if it % log_every == 0 or it == 1:
            dt = time.time() - t0
            print(f"  iter {it:5d}/{iters}  loss={loss:.5f}  ema={ema:.5f}  "
                  f"N={gs.N:,}  ({it/dt:.1f} it/s)")


def _one_iter_forward_backward(gs: LidarVisualGS, samples: list[dict],
                               device: str) -> float:
    """랜덤 뷰 하나 업로드 → train_step.  loss float 반환."""
    s = random.choice(samples)
    gt_image = torch.from_numpy(s["gt_image"]).to(device)
    gt_depth = torch.from_numpy(s["gt_depth"]).to(device)
    K       = torch.from_numpy(s["K"]).to(device)
    viewmat = torch.from_numpy(s["viewmat"]).to(device)
    cam_index = int(s.get("cam_index", 0))
    kf_index = int(s.get("kf_index", 0))
    return gs.train_step(gt_image, gt_depth, K, viewmat,
                         cam_index=cam_index, kf_index=kf_index)


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

    # ----- 초기화 (seed) -----
    g_init = ap.add_argument_group("Initialization")
    g_init.add_argument("--init-voxel", type=float, default=0.1,
                        help="seed voxel (m). 0 이면 비활성")
    g_init.add_argument("--init-scale", type=float, default=None,
                        help="가우시안 초기 반경 (m). 미지정 시 init-voxel*0.5")
    g_init.add_argument("--keep-unseen", action="store_true",
                        help="카메라 FOV 밖 LiDAR 점도 회색으로 seed (기본 drop)")
    g_init.add_argument("--sh-degree", type=int, default=1,
                        help="SH 차수. 0~3.  1=권장 (indoor), 3=원 3DGS 표준")

    # ----- 학습 -----
    g_train = ap.add_argument_group("Training")
    g_train.add_argument("--iters", type=int, default=30000)
    g_train.add_argument("--log-every", type=int, default=50)
    g_train.add_argument("--seed", type=int, default=0)

    # ----- Pose refinement -----
    g_pose = ap.add_argument_group("Pose refinement (자동 calibration)")
    g_pose.add_argument("--cam-refine", action="store_true",
                        help="카메라 extrinsic drift 학습 (3 × 6DoF)")
    g_pose.add_argument("--kf-refine", action="store_true",
                        help="per-keyframe SE3 delta 학습 (BA-lite)")

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

    # 데이터셋
    ds = BaseDataDataset(args.root, require_image=True)
    print(f"[dataset] {len(ds)} kf")
    pts, cols = accumulate_seed(ds, args.max_kf, args.stride, args.init_voxel,
                                keep_unseen=args.keep_unseen)
    samples = build_samples(ds, args.cameras, args.max_kf, args.stride)

    # 모델
    init_scale = (args.init_scale if args.init_scale is not None
                  else max(args.init_voxel * 0.5, 0.01))
    n_keyframes = max(ds.indices) + 1 if ds.indices else 0
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
    """CLI 에서 override 한 regularizer λ 를 모델에 적용."""
    if args.opacity_reg_lambda is not None:
        gs.opacity_reg_lambda = args.opacity_reg_lambda
    if args.opacity_bimodal_lambda is not None:
        gs.opacity_bimodal_lambda = args.opacity_bimodal_lambda


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
