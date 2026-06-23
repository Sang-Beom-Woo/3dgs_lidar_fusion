#!/usr/bin/env bash
# =============================================================================
# 실내 3DGS 학습 — Option B (LPIPS on) + Option C (ADC on) 동시 적용
# -----------------------------------------------------------------------------
# 대상 데이터: base_data/ (175 kf, 실내 야외 둘 다 있지만 평면 ~75m 씬)
# GPU: RTX 3060 Laptop 6GB
#
# 핵심 전략
#   1) voxel 0.08 (이전 0.15 vs 0.05 의 중간) → 초기 ~1M 가우시안.  ADC 여유.
#   2) LPIPS λ=0.1 (default 0.2 의 절반) — sharpness 확보, 메모리 부담 ↓
#   3) ADC 켜기 — 디테일 영역 자동 densify, max 2.5M 캡
#   4) 야외 4032×3036 left/right 는 ×4 downscale (1008×759) 로 GPU 절약
#
# 예상 GPU 사용 (peak)
#   가우시안 2.5M × params/grad/Adam ≈ 1.0 GB
#   렌더 1280×1024 + tile sort       ≈ 1.5 GB
#   LPIPS VGG (λ=0.1, full res)      ≈ 1.8 GB
#   기타 buffer                       ≈ 0.5 GB
#   ───────────────────────────────────────
#   합                                ≈ 4.8 GB   (5.67 GB 중)
# 위험 신호: nvidia-smi 에서 5.3 GB 넘기면 LPIPS λ 더 낮추거나 max-gaussians ↓
#
# 예상 시간: 30k iter / ~9 it/s = ~55분 (ADC 켜면 약간 느려짐, LPIPS 추가)
# =============================================================================

set -e   # 첫 실패에서 멈춤
set -u   # 미정의 변수 사용 시 멈춤

cd "$(dirname "$0")"

# OOM-prone 메모리 단편화 완화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 출력 파일 (기존 결과 보존을 위해 새 이름)
OUT_PLY="map_v2.ply"
OUT_SPLAT="map_v2_splat.ply"
LOG="indoor_v2_train.log"

python3 -u run_chapter2.py \
  --cam-downscale "left=4,right=4" \
  \
  `# ----- 초기화 (Option B: 중간 voxel) -----` \
  --init-voxel 0.08 \
  --init-scale 0.04 \
  --sh-degree 1 \
  \
  `# ----- mono depth seed (실내 ~10m 까지) -----` \
  --mono-depth \
  --mono-pixel-stride 12 \
  --mono-min-depth 0.3 \
  --mono-max-depth 15 \
  \
  `# ----- Photometric loss (Option B: LPIPS 켜되 640×512 다운스케일) -----` \
  --ssim-lambda 0.2 \
  --lpips-lambda 0.1 \
  --lpips-downscale 2 \
  \
  `# ----- Pose refinement (학습 시 자동 calibration) -----` \
  --cam-refine \
  --kf-refine \
  --view-affine \
  \
  `# ----- ADC (Option C: densify 켜기) -----` \
  --densify-from-iter 500 \
  --densify-until-iter 15000 \
  --densify-interval 200 \
  --densify-grad-threshold 3e-4 \
  --prune-min-opacity 5e-3 \
  --max-gaussians 2500000 \
  \
  `# ----- 학습 길이 -----` \
  --iters 30000 \
  --log-every 1000 \
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
echo "[done] 학습 완료"
echo "  PLY     : $OUT_PLY"
echo "  Splat   : $OUT_SPLAT"
echo "  Log     : $LOG"
echo "===================================================="
echo
echo "다음 단계 (선택):"
echo "  1) 진단:"
echo "       python3 tools/diagnose_render.py $OUT_SPLAT --n 9 --out diag_v2/ --seed 42"
echo "  2) 천장 제거 (Z>1.0m):"
echo "       python3 tools/prune_z.py $OUT_SPLAT --z-max 1.0 --out map_v2_no_ceiling.ply"
echo "  3) 웹뷰어용 Y-up:"
echo "       python3 tools/reorient.py $OUT_SPLAT --out map_v2_yup.ply"
