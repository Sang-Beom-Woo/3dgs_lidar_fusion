#!/usr/bin/env bash
# =============================================================================
# 실내 3DGS 학습 — ADC ON + 디테일 우선 + Long run (90k iter)
# -----------------------------------------------------------------------------
# 사용자 지시:
#   - ADC 무조건 ON (성능 안 좋아도 다른 인자로 조정)
#   - needle 허용해서라도 디테일 살리는 방향
#   - max-gaussians 1.5M (6GB GPU 안전)
#   - 30k iter 마다 체크포인트
#
# config 요약 (smoke A''' 검증 후 확장):
#   voxel 0.1 (sparse 시드, ADC grow 여지)
#   ADC: grad-threshold 1e-5 (적극), prune cap 모두 풀림, aspect 30 (needle 허용)
#   aniso-lambda 0 (needle 페널티 X)
#   LPIPS 0.15 (sharpness), SSIM 0.25
#   bg-color fixed (실내 회색)
#
# 체크포인트
#   map_30000.ply, map_splat_30000.ply
#   map_60000.ply, map_splat_60000.ply
#   map.ply, map_splat.ply   (= 90000 최종)
#
# 예상 시간
#   가우시안 1.5M + LPIPS = ~4 it/s
#   90k / 4 = ~6.3 시간
# =============================================================================

set -e
set -u

cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_PLY="map.ply"
OUT_SPLAT="map_splat.ply"
LOG="indoor_adc_long_train.log"

python3 -u run_chapter2.py \
  --cam-downscale "left=4,right=4" \
  \
  `# ----- Seed (sparse — ADC grow 여지) -----` \
  --init-voxel 0.1 \
  --init-scale 0.05 \
  --sh-degree 1 \
  \
  `# ----- Photometric (needle 허용, sharpness 우선) -----` \
  --lpips-lambda 0.15 \
  --lpips-downscale 2 \
  --ssim-lambda 0.25 \
  --aniso-lambda 0 \
  \
  `# ----- 배경 (실내 중성 회색) -----` \
  --bg-color "0.5,0.5,0.55" \
  \
  `# ----- ADC (적극 grow, prune 거의 비활성) -----` \
  --densify-from-iter 500 \
  --densify-until-iter 30000 \
  --densify-interval 200 \
  --densify-grad-threshold 1e-5 \
  --prune-min-opacity 5e-3 \
  --prune-max-screen-size 1000 \
  --prune-max-world-scale 1000 \
  --aspect-prune-thr 30 \
  --max-gaussians 1500000 \
  \
  `# ----- Pose refinement: 100k freeze (90k 안 닿음 — 끝까지 학습) -----` \
  --cam-refine \
  --kf-refine \
  --view-affine \
  --refine-freeze-iter 100000 \
  \
  `# ----- Exposure outlier 처리 -----` \
  --exposure-clip-pct 95 \
  \
  `# ----- 학습 길이 -----` \
  --iters 90000 \
  --log-every 1000 \
  --checkpoint-every 30000 \
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
echo "[done] ADC long 학습 완료"
echo "===================================================="
ls -lh map_30000.ply map_splat_30000.ply map_60000.ply map_splat_60000.ply map.ply map_splat.ply 2>/dev/null
