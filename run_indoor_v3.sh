#!/usr/bin/env bash
# =============================================================================
# 실내 3DGS 학습 v3 — 30k iter (smoke 검증 후 본격)
# -----------------------------------------------------------------------------
# smoke (5k) 결과:
#   - 평균 diff_mean 25.66 (smoke 2 의 26.3 보다 좋음)
#   - 가우시안 510K, aspect>5 = 2.5% (매우 안정)
#   - EMA loss 0.262 (5k 시점) — 추세 살아있음, 30k 까지 개선 여지
#
# 사용자 고정 제약:
#   - ADC ON
#   - bg-random
#
# 체크포인트: 10k 마다 (10k, 20k, 30k 최종)
# 예상 시간: 30k / ~7.1 it/s ≈ 70 분
# =============================================================================

set -e
set -u

cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_PLY="map.ply"
OUT_SPLAT="map_splat.ply"
LOG="indoor_v3_train.log"

python3 -u run_chapter2.py \
  --cam-downscale "left=4,right=4" \
  \
  `# ----- Seed (voxel 0.05 — close 영역 dense) -----` \
  --init-voxel 0.05 \
  --init-scale 0.025 \
  --sh-degree 1 \
  \
  `# ----- Photometric -----` \
  --lpips-lambda 0.15 \
  --lpips-downscale 2 \
  --ssim-lambda 0.25 \
  --aniso-lambda 0.003 \
  \
  `# ----- 배경 (고정 제약: random) -----` \
  --bg-random \
  \
  `# ----- ADC (고정 제약: ON) -----` \
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
  `# ----- Exposure outlier -----` \
  --exposure-clip-pct 95 \
  \
  `# ----- 학습 길이 -----` \
  --iters 30000 \
  --log-every 1000 \
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
echo "[done] 30k 학습 완료"
echo "===================================================="
ls -lh map_10000.ply map_splat_10000.ply map_20000.ply map_splat_20000.ply map.ply map_splat.ply 2>/dev/null
