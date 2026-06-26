#!/usr/bin/env bash
# =============================================================================
# 야외 3DGS 학습 — V2 long version (300k iter, 30k 체크포인트, refine freeze, bg)
# -----------------------------------------------------------------------------
# 대상: 440 kf 야외 (Z up=87m, 600m extent).
# GPU: RTX 3060 Laptop 6GB
#
# 안전장치
#   - LPIPS off       : GPU 여유 + 긴 학습에서 가우시안 흔드는 위험 차단
#   - ADC off         : 기본값 999999 (V2 검증 OK)
#   - refine-freeze 100k : view-affine 충분히 학습 후 freeze (1번 — 30k 너무 빨랐음)
#   - --bg-color 0.8,0.85,0.9 : 하늘 자리에 거대 floater 생기는 부작용 차단
#   - mono-depth OFF  : LiDAR 만 사용 (사용자 지정)
#   - prune-z-max OFF : 위쪽 텍스쳐 보존 (bg-color 가 floater 차단 역할)
#   - --exclude-kf-file : 보수 임계 (saturation 25%, bright 25) — 1 kf 만 제외
#   - --exposure-clip-pct 95 : 과노출 픽셀 normalize (3번)
#
# 야외 vs 실내 차이
#   * init-voxel  0.06 → 0.5  : 씬 8× 큼 (75m vs 600m) 가우시안 밀도 매칭
#   * init-scale  0.03 → 0.25 : voxel 절반 비율
#   * mono-max    15   → 80   : 야외 원경
#   * bg-color    회색 → 하늘 : 학습 영향
#   * prune-z     없음  → 4m  : 하늘 차단
#
# 체크포인트: map_30000.ply, map_60000.ply, ..., map.ply (=300000 최종)
# 예상 시간: 300k / ~10 it/s ≈ 8 시간
# =============================================================================

set -e
set -u

cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_PLY="map.ply"
OUT_SPLAT="map_splat.ply"
LOG="outdoor_v2_long_train.log"

python3 -u run_chapter2.py \
  --cam-downscale "left=4,right=4" \
  \
  `# ----- Seed (야외, mono off — LiDAR 만, voxel 더 작게 → 디테일 ↑) -----` \
  --init-voxel 0.15 \
  --init-scale 0.07 \
  --sh-degree 1 \
  \
  `# ----- Needle 방어 (이전 run 에서 60k 시점 aspect>5 비율 68.9% 폭증) -----` \
  --aniso-lambda 0.01 \
  --aniso-max-ratio 5.0 \
  \
  `# ----- Photometric (LPIPS perceptual sharpness — 640×512 다운스케일로 GPU 안전) -----` \
  --lpips-lambda 0.05 \
  --lpips-downscale 2 \
  \
  `# ----- 배경색 (회청색 — 흐린 하늘) -----` \
  --bg-color "0.8,0.85,0.9" \
  \
  `# ----- Outlier kf 제외 (3번 옵션) — tools/detect_outlier_kf.py 결과 사용 -----` \
  --exclude-kf-file exclude_kf.json \
  \
  `# ----- 과노출 픽셀 클리핑 (3번 옵션) — gray>=0.95 픽셀 normalize -----` \
  --exposure-clip-pct 95 \
  \
  `# ----- Pose refinement: 100k 후 freeze (1번 옵션 — view-affine 학습 길게) -----` \
  --cam-refine \
  --kf-refine \
  --view-affine \
  --refine-freeze-iter 100000 \
  \
  `# ----- 학습 길이 (30k 짧은 run — 18시 자동 종료 대비) -----` \
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
  `# ----- 출력 -----` \
  --out "$OUT_PLY" \
  --splat-out "$OUT_SPLAT" \
  2>&1 | tee "$LOG"

echo
echo "===================================================="
echo "[done] 야외 V2 long 학습 완료"
echo "===================================================="
echo
echo "체크포인트:"
ls -la map_*.ply map_splat_*.ply 2>/dev/null | grep -E '_[0-9]+\.ply' | head -20
echo
echo "최종:  $OUT_PLY  /  $OUT_SPLAT"
echo
echo "체크포인트별 diff 진단 (bg-color 동일하게):"
echo "  for i in 30000 60000 90000 120000 150000 180000 210000 240000 270000; do"
echo "    python3 tools/diagnose_render.py map_splat_\${i}.ply \\"
echo "      --bg-color '0.8,0.85,0.9' --n 9 --out diag_iter\${i}/ --seed 42"
echo "  done"
