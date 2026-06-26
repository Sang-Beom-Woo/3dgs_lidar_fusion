#!/usr/bin/env bash
# =============================================================================
# 실내 3DGS 학습 — Plan S Long (30k iter)
# -----------------------------------------------------------------------------
# Plan S smoke (5k) 검증 결과:
#   - front 평균 diff_mean 22.3 (이전 B 25.0 보다 ↓)
#   - 가우시안 947K, aspect>5 1.9% (안정)
#   - α p95 0.45 (opacity-bimodal 작동)
#
# 사용자 고정 제약:
#   - ADC ON
#   - bg-random
#   - cameras front
#
# Plan S 핵심:
#   - LPIPS λ 0.30  (이전 0.15 의 2×)
#   - SSIM  λ 0.40  (이전 0.25)
#   - opacity-bimodal-lambda 0.01  ✨ edge sharp
#   - init_scale 0.008 (finer)
#
# 체크포인트: 5k 마다 (5k, 10k, 15k, 20k, 25k, 30k)
# 예상 시간: ~100 분 (5 it/s 가정)
# =============================================================================

set -e
set -u

cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_PLY="map_S.ply"
OUT_SPLAT="map_S_splat.ply"
LOG="indoor_S_long_train.log"

python3 -u run_chapter2.py \
  --cam-downscale "left=4,right=4" \
  --cameras front \
  \
  `# ----- Seed (voxel 0.03, init_scale 0.008 — finer edge) -----` \
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
  `# ----- 배경 (random, 고정 제약) -----` \
  --bg-random \
  \
  `# ----- ADC (고정 제약: ON) — 30k 학습에 맞춰 15k 까지 -----` \
  --densify-from-iter 500 \
  --densify-until-iter 15000 \
  --densify-interval 200 \
  --densify-grad-threshold 1e-5 \
  --prune-min-opacity 1e-3 \
  --prune-max-screen-size 1000 \
  --prune-max-world-scale 1000 \
  --aspect-prune-thr 8 \
  --max-gaussians 1500000 \
  \
  `# ----- Pose refinement (30k 안 freeze 안 닿음, 끝까지 학습) -----` \
  --cam-refine \
  --kf-refine \
  --view-affine \
  --refine-freeze-iter 100000 \
  \
  `# ----- 학습 길이 -----` \
  --iters 30000 \
  --log-every 1000 \
  --checkpoint-every 5000 \
  \
  `# ----- 학습 후 정리 -----` \
  --floater-prune \
  --floater-opacity 5e-3 \
  --floater-neighbors 20 \
  --floater-std 2.0 \
  --opacity-threshold 0 \
  \
  --out "$OUT_PLY" \
  --splat-out "$OUT_SPLAT" 2>&1 | tee "$LOG"

echo
echo "===================================================="
echo "[done] Plan S Long 30k 학습 완료"
echo "===================================================="
ls -lh map_S_*.ply map_S.ply map_S_splat.ply 2>/dev/null
