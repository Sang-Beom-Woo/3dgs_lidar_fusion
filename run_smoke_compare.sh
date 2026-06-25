#!/usr/bin/env bash
# =============================================================================
# Smoke 비교 — Plan A (LPIPS full-res) vs Plan B (SH 2)
# -----------------------------------------------------------------------------
# 공통 (사용자 고정 제약 + 디테일 베이스):
#   - ADC ON
#   - bg-random
#   - voxel 0.03 (글자/edge 시드 향상)
#   - init_scale 0.012
#   - exposure-clip-pct 제거 (형광등 회색 문제 해결)
#
# 차이:
#   A: --lpips-downscale 1  --sh-degree 1   (edge sharpness)
#   B: --lpips-downscale 2  --sh-degree 2   (view-dep 색)
#
# 출력:
#   A: smoke_A_map_splat.ply, diag_A_kf89.png, diag_A_smoke/
#   B: smoke_B_map_splat.ply, diag_B_kf89.png, diag_B_smoke/
#
# 예상 시간: A ~15분 + B ~15분 = ~30분
# =============================================================================

set -e
set -u

cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(
  --cam-downscale "left=4,right=4"
  --init-voxel 0.03 --init-scale 0.012
  --bg-random
  --densify-from-iter 500
  --densify-until-iter 4000
  --densify-interval 200
  --densify-grad-threshold 1e-5
  --prune-min-opacity 1e-3
  --prune-max-screen-size 1000
  --prune-max-world-scale 1000
  --aspect-prune-thr 8
  --max-gaussians 1500000
  --aniso-lambda 0.003
  --ssim-lambda 0.25
  --cam-refine --kf-refine --view-affine
  --iters 5000 --log-every 500
  --opacity-threshold 0
)

# ─── A: LPIPS full-res, SH 1 ─────────────────────────────────────────
echo "=================================================================="
echo " SMOKE A — LPIPS full-res (downscale 1), SH 1"
echo "=================================================================="
python3 -u run_chapter2.py \
  "${COMMON_ARGS[@]}" \
  --lpips-lambda 0.15 \
  --lpips-downscale 1 \
  --sh-degree 1 \
  --out smoke_A_map.ply --splat-out smoke_A_map_splat.ply \
  2>&1 | tee smoke_A.log

# A diag — kf 89 (마네킹/유모차) + 9-view 표준
python3 tools/diag_one_view.py smoke_A_map_splat.ply --kf 89 --cam front \
  --bg-color "0.0,0.0,0.0" --out diag_A_kf89.png 2>&1 | tail -3
rm -rf diag_A_smoke
python3 tools/diagnose_render.py smoke_A_map_splat.ply --n 9 \
  --out diag_A_smoke/ --seed 42 2>&1 | tail -12

# ─── B: LPIPS downscale 2, SH 2 ──────────────────────────────────────
echo ""
echo "=================================================================="
echo " SMOKE B — LPIPS downscale 2, SH 2 (view-dep 색)"
echo "=================================================================="
python3 -u run_chapter2.py \
  "${COMMON_ARGS[@]}" \
  --lpips-lambda 0.15 \
  --lpips-downscale 2 \
  --sh-degree 2 \
  --out smoke_B_map.ply --splat-out smoke_B_map_splat.ply \
  2>&1 | tee smoke_B.log

# B diag
python3 tools/diag_one_view.py smoke_B_map_splat.ply --kf 89 --cam front \
  --bg-color "0.0,0.0,0.0" --out diag_B_kf89.png 2>&1 | tail -3
rm -rf diag_B_smoke
python3 tools/diagnose_render.py smoke_B_map_splat.ply --n 9 \
  --out diag_B_smoke/ --seed 42 2>&1 | tail -12

# ─── 정리 ────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo " 두 smoke 완료 — 비교"
echo "=================================================================="
echo "A (LPIPS full):  diag_A_kf89.png, diag_A_smoke/"
echo "B (SH 2):        diag_B_kf89.png, diag_B_smoke/"
ls -lh smoke_A_map_splat.ply smoke_B_map_splat.ply
