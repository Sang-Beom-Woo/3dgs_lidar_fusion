#!/usr/bin/env bash
# =============================================================================
# 실내 3DGS 학습 — Plan B Long (30k iter)
# -----------------------------------------------------------------------------
# B smoke 검증 결과:
#   - 평균 diff_mean 26.1 (5k)
#   - 가우시안 961K (voxel 0.03, ADC grow)
#   - aspect>5 1.4% (안정)
#   - view-affine bias -0.005 (매우 안정)
#
# 고정 제약:
#   - ADC ON
#   - bg-random
#
# B 핵심 인자:
#   - voxel 0.03 (close 영역 dense)
#   - SH degree 2 (시점 의존 색)
#   - LPIPS down 2 (메모리 안전)
#   - exposure-clip 제거 (형광등 회색 → 흰색)
#
# 체크포인트: 10k, 20k, 30k
# 예상 시간: ~78 분 (B smoke 6.4 it/s 기준)
# =============================================================================

set -e
set -u

cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_PLY="map_B.ply"
OUT_SPLAT="map_B_splat.ply"
LOG="indoor_B_long_train.log"

python3 -u run_chapter2.py \
  --cam-downscale "left=4,right=4" \
  \
  `# ----- 카메라 (front 만 사용 — left/right 이중상 원인 회피) -----` \
  --cameras front \
  \
  `# ----- Seed (voxel 0.03 — close 영역 dense, 글자 stroke 매칭) -----` \
  --init-voxel 0.03 \
  --init-scale 0.012 \
  --sh-degree 2 \
  \
  `# ----- Photometric (LPIPS down 2 + SH 2) -----` \
  --lpips-lambda 0.15 \
  --lpips-downscale 2 \
  --ssim-lambda 0.25 \
  --aniso-lambda 0.003 \
  \
  `# ----- 배경 (고정 제약: random) -----` \
  --bg-random \
  \
  `# ----- ADC (고정 제약: ON) — 100k 학습에 맞춰 ADC 50k 까지 -----` \
  --densify-from-iter 500 \
  --densify-until-iter 50000 \
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
  --refine-freeze-iter 100000 \
  \
  `# ----- 학습 길이 (100k iter, 10k 마다 체크포인트 — 10개) -----` \
  --iters 100000 \
  --log-every 2000 \
  --checkpoint-every 10000 \
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
echo "[done] B long 30k 학습 완료"
echo "===================================================="
ls -lh map_B_10000.ply map_B_splat_10000.ply map_B_20000.ply map_B_splat_20000.ply map_B.ply map_B_splat.ply 2>/dev/null
