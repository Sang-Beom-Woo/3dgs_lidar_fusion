#!/usr/bin/env bash
# =============================================================================
# Smoke — ADC ON (고정) + random bg (고정) + 그 외 조정
# -----------------------------------------------------------------------------
# 사용자 고정 제약:
#   1. ADC ON
#   2. bg-random (INRIA 표준)
#
# 현재 조정:
#   voxel 0.1 → 0.05  (close 시드 4× — 스티로폼 같은 close 영역 보강)
#   init_scale 0.05 → 0.025
#   aniso 0 → 0.003   (needle 발산 차단 복구)
#   aspect 30 → 8     (표준 needle 컷 복구)
#   prune-opa 5e-3 → 1e-3  (close 가우시안 보존 시간 ↑)
#
#   prune cap 1000/1000 유지 (close-view 보호)
#   grad 1e-5 유지 (적극 grow)
# =============================================================================

set -e
set -u

cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -u run_chapter2.py \
  --cam-downscale "left=4,right=4" \
  --cameras front \
  \
  `# ----- Seed (voxel 0.03, init_scale 0.008 — finer edge 표현) -----` \
  --init-voxel 0.03 \
  --init-scale 0.008 \
  --sh-degree 2 \
  \
  `# ----- Photometric Plan S — sharpness + edge -----` \
  --lpips-lambda 0.30 \
  --lpips-downscale 2 \
  --ssim-lambda 0.40 \
  --aniso-lambda 0.003 \
  --opacity-bimodal-lambda 0.01 \
  \
  `# ----- 배경 (random — 고정 제약) -----` \
  --bg-random \
  \
  `# ----- ADC ON (고정 제약) -----` \
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
  \
  --opacity-threshold 0 \
  --out smoke_map.ply \
  --splat-out smoke_map_splat.ply 2>&1 | tee smoke_test.log

echo
echo "===================================================="
echo "[smoke] 5k iter 완주.  ADC + random bg 의 효과 확인."
echo "===================================================="
