#!/usr/bin/env bash
# =============================================================================
# 실내 3DGS 학습 — Long run (300k iter) + 30k 마다 중간 체크포인트
# -----------------------------------------------------------------------------
# 대상: base_data/ (175 kf)
# 예상 시간:
#   - LPIPS 다운스케일 ×2 + ADC ON 시 약 8~9 it/s
#   - 300k iter / 8.5 it/s ≈ 10 시간
#   - 매 30k 마다 체크포인트 저장 (PLY 쓰기 ~10초)
#
# 체크포인트 파일
#   map_30000.ply, map_60000.ply, ..., map_270000.ply   (point-cloud)
#   map_splat_30000.ply, ..., map_splat_270000.ply       (INRIA splat)
#   map.ply, map_splat.ply                               (최종 = iter 300000)
# =============================================================================

set -e
set -u

cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_PLY="map.ply"
OUT_SPLAT="map_splat.ply"
LOG="indoor_long_train.log"

python3 -u run_chapter2.py \
  --cam-downscale "left=4,right=4" \
  \
  `# ----- Seed -----` \
  --init-voxel 0.08 \
  --init-scale 0.04 \
  --sh-degree 1 \
  --mono-depth \
  --mono-pixel-stride 12 \
  --mono-min-depth 0.3 \
  --mono-max-depth 15 \
  \
  `# ----- Photometric (LPIPS at 640×512 for 6GB GPU) -----` \
  --ssim-lambda 0.2 \
  --lpips-lambda 0.1 \
  --lpips-downscale 2 \
  \
  `# ----- Pose refinement -----` \
  --cam-refine \
  --kf-refine \
  --view-affine \
  \
  `# ----- ADC — densify 기간 길어진 학습에 맞춰 확장 -----` \
  --densify-from-iter 500 \
  --densify-until-iter 150000 \
  --densify-interval 200 \
  --densify-grad-threshold 3e-4 \
  --prune-min-opacity 5e-3 \
  --max-gaussians 2500000 \
  \
  `# ----- 학습 길이 -----` \
  --iters 300000 \
  --log-every 5000 \
  \
  `# ----- 중간 체크포인트 -----` \
  --checkpoint-every 30000 \
  \
  `# ----- 학습 후 정리 -----` \
  --floater-prune \
  --floater-opacity 5e-3 \
  --floater-neighbors 20 \
  --floater-std 2.0 \
  --opacity-threshold 0 \
  \
  `# ----- 출력 -----` \
  --out "$OUT_PLY" \
  --splat-out "$OUT_SPLAT" \
  2>&1 | tee "$LOG"

echo
echo "===================================================="
echo "[done] 300k 학습 완료"
echo "===================================================="
echo
echo "체크포인트:"
ls -la map_*.ply map_splat_*.ply 2>/dev/null | grep -E '_[0-9]+\.ply' | head -20
echo
echo "최종 결과:"
echo "  $OUT_PLY  /  $OUT_SPLAT"
echo
echo "체크포인트별 diff 진단:"
echo "  for i in 30000 60000 90000 120000 150000 180000 210000 240000 270000; do"
echo "    python3 tools/diagnose_render.py map_splat_\${i}.ply \\"
echo "      --n 9 --out diag_iter\${i}/ --seed 42"
echo "  done"
