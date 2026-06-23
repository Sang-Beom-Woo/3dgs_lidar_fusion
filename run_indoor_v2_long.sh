#!/usr/bin/env bash
# =============================================================================
# 실내 3DGS 학습 — V2 검증 설정 + Long run (300k iter) + 체크포인트
# -----------------------------------------------------------------------------
# V2 가 30k 에서 2.65M 가우시안 / diff_mean 24.6 으로 안전하게 수렴한 설정에
# 다음 두 가지만 추가:
#   1. iter 30000 → 300000  (10× 학습 길이)
#   2. --refine-freeze-iter 30000  ← 핵심 안전장치
#      처음 30k 동안만 pose-refine (cam_refine + kf_refine + view-affine) 학습.
#      그 후 LR=0 으로 freeze.  이전 300k 붕괴 (cam 89°, view-affine bias 0.5)
#      의 직접적 원인을 차단.
#
# v3 변경 (GPU hang 대응)
#   - voxel 0.05 → 0.06  (가우시안 2.84M → ~2.0M)
#   - init_scale 0.025 → 0.03  (voxel 매칭)
#   - mono_pixel_stride 12 → 16  (mono 점 36% 로, 메모리 절약)
#   목적: GPU 메모리 안전 마진 +30%.  이전 2.84M 에서 iter 1 직후 hang 한
#         것의 직접적 추정 원인 (peak GPU 메모리 한계 근접) 차단.
#
# 의도적으로 켜지 않은 것 (V2 와 동일)
#   - LPIPS off (GPU 여유 + LPIPS 가 길어진 학습에서 가우시안 흔드는 위험)
#   - ADC off  (densify-from-iter 999999 기본값.  V2 가 ADC 없이 잘 됐음)
#
# 체크포인트 매 30k:
#   map_30000.ply, map_splat_30000.ply
#   map_60000.ply, map_splat_60000.ply
#   ...
#   map_270000.ply, map_splat_270000.ply
#   map.ply, map_splat.ply  (= iter 300000 최종)
#
# 예상 시간: 300k / ~13 it/s ≈ 6.5 시간 (가우시안 줄어 약간 빨라짐)
# =============================================================================

set -e
set -u

cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_PLY="map.ply"
OUT_SPLAT="map_splat.ply"
LOG="indoor_v2_long_train.log"

python3 -u run_chapter2.py \
  --cam-downscale "left=4,right=4" \
  \
  `# ----- Seed (v3: GPU 메모리 안전 마진 +30%) -----` \
  --init-voxel 0.06 \
  --init-scale 0.03 \
  --sh-degree 1 \
  --mono-depth \
  --mono-pixel-stride 16 \
  --mono-min-depth 0.3 \
  --mono-max-depth 15 \
  \
  `# ----- Photometric (LPIPS off — V2 와 동일) -----` \
  --lpips-lambda 0 \
  \
  `# ----- 배경색 (하늘 floater 차단).  실내라 약한 회색.  야외는 0.8,0.85,0.9 권장 -----` \
  --bg-color "0.5,0.5,0.55" \
  \
  `# ----- Pose refinement: 30k 까지만 학습, 그 후 freeze -----` \
  --cam-refine \
  --kf-refine \
  --view-affine \
  --refine-freeze-iter 30000 \
  \
  `# ----- 학습 길이 -----` \
  --iters 300000 \
  --log-every 5000 \
  \
  `# ----- 중간 체크포인트 매 30k -----` \
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
echo "[done] V2 long 학습 완료"
echo "===================================================="
echo
echo "체크포인트:"
ls -la map_*.ply map_splat_*.ply 2>/dev/null | grep -E '_[0-9]+\.ply' | head -20
echo
echo "최종 결과:"
echo "  $OUT_PLY  /  $OUT_SPLAT"
echo
echo "체크포인트별 diff 진단 (loop):"
echo "  for i in 30000 60000 90000 120000 150000 180000 210000 240000 270000; do"
echo "    python3 tools/diagnose_render.py map_splat_\${i}.ply \\"
echo "      --n 9 --out diag_iter\${i}/ --seed 42"
echo "  done"
