#!/usr/bin/env bash
# =============================================================================
# Smoke A-1 — LPIPS full-res 1280×1024 (메모리 절약 위해 voxel 0.05)
# -----------------------------------------------------------------------------
# Plan A 의 핵심 (LPIPS full res) 를 살리되 voxel 0.05 로 시드 줄여 메모리 fit.
# B 결과 (smoke_B_*) 는 보존, A-1 결과는 smoke_A1_* 로 저장.
# =============================================================================

set -e
set -u

cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -u run_chapter2.py \
  --cam-downscale "left=4,right=4" \
  \
  `# ----- Seed (voxel 0.05 — LPIPS full res 메모리 fit) -----` \
  --init-voxel 0.05 \
  --init-scale 0.025 \
  --sh-degree 1 \
  \
  `# ----- Photometric (LPIPS full res — Plan A 의 핵심) -----` \
  --lpips-lambda 0.15 \
  --lpips-downscale 1 \
  --ssim-lambda 0.25 \
  --aniso-lambda 0.003 \
  \
  `# ----- 배경 (random, 고정 제약) -----` \
  --bg-random \
  \
  `# ----- ADC (고정 제약) -----` \
  --densify-from-iter 500 \
  --densify-until-iter 4000 \
  --densify-interval 200 \
  --densify-grad-threshold 1e-5 \
  --prune-min-opacity 1e-3 \
  --prune-max-screen-size 1000 \
  --prune-max-world-scale 1000 \
  --aspect-prune-thr 8 \
  --max-gaussians 1500000 \
  \
  `# ----- Pose refinement -----` \
  --cam-refine \
  --kf-refine \
  --view-affine \
  \
  `# ----- 학습 -----` \
  --iters 5000 \
  --log-every 500 \
  --opacity-threshold 0 \
  \
  --out smoke_A1_map.ply \
  --splat-out smoke_A1_map_splat.ply 2>&1 | tee smoke_A1.log

echo
echo "=================================================================="
echo "[smoke A-1] 5k iter 완주.  B 와 비교 가능"
echo "=================================================================="

# diag
python3 tools/diag_one_view.py smoke_A1_map_splat.ply --kf 89 --cam front \
  --bg-color "0.0,0.0,0.0" --out diag_A1_kf89.png 2>&1 | tail -3
rm -rf diag_A1_smoke
python3 tools/diagnose_render.py smoke_A1_map_splat.ply --n 9 \
  --out diag_A1_smoke/ --seed 42 2>&1 | tail -15
