#!/usr/bin/env bash
# =============================================================================
# 두 config 30k 순차 학습 + 비교
# -----------------------------------------------------------------------------
# A. baseline   — smoke 2 config (LPIPS 0.15, SH 1, no opacity reset)
# B. guide      — smoke F config (LPIPS 0.15, SH 2, opacity-reset 3000)
#
# 출력 (구분 위해 접두사):
#   A: map_A_*.ply, diag_A_*/
#   B: map_B_*.ply, diag_B_*/
#
# 체크포인트 매 5000 — 셧다운 대비
# 예상 시간:
#   A: ~1.2h, B: ~1.5h = 총 2.7h
# =============================================================================

set -e
set -u

cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── 공통 인자 ────────────────────────────────────────────────────────
COMMON_ARGS=(
  --cam-downscale "left=4,right=4"
  --init-voxel 0.03 --init-scale 0.015
  --lpips-lambda 0.15 --lpips-downscale 2
  --ssim-lambda 0.25
  --aniso-lambda 0.003
  --bg-color "0.5,0.5,0.55"
  --cam-refine --kf-refine --view-affine
  --refine-freeze-iter 100000
  --exposure-clip-pct 95
  --iters 30000 --log-every 1000
  --checkpoint-every 5000
  --floater-prune --opacity-threshold 0
)

# ─── A: baseline (SH 1, no reset) ────────────────────────────────────
echo "=================================================================="
echo " RUN A: baseline (smoke 2 검증된 최고)"
echo "=================================================================="
python3 -u run_chapter2.py \
  "${COMMON_ARGS[@]}" \
  --sh-degree 1 \
  --out map_A.ply --splat-out map_splat_A.ply \
  2>&1 | tee train_A.log

# A 의 체크포인트 이름 변경 (map_5000.ply → map_A_5000.ply 등)
for i in 5000 10000 15000 20000 25000; do
  [ -f map_${i}.ply ] && mv -v map_${i}.ply map_A_${i}.ply
  [ -f map_splat_${i}.ply ] && mv -v map_splat_${i}.ply map_A_splat_${i}.ply
done
# A diag
python3 tools/diagnose_render.py map_splat_A.ply --bg-color "0.5,0.5,0.55" \
  --n 9 --out diag_A_final/ --seed 42 2>&1 | tail -15

# ─── B: guide standard (SH 2, opacity reset 3000) ────────────────────
echo ""
echo "=================================================================="
echo " RUN B: guide standard (SH 2 + opacity reset 3000)"
echo "=================================================================="
python3 -u run_chapter2.py \
  "${COMMON_ARGS[@]}" \
  --sh-degree 2 \
  --opacity-reset-every 3000 \
  --opacity-reset-alpha 0.01 \
  --opacity-reset-warmup 500 \
  --out map_B.ply --splat-out map_splat_B.ply \
  2>&1 | tee train_B.log

# B 의 체크포인트 이름 변경
for i in 5000 10000 15000 20000 25000; do
  [ -f map_${i}.ply ] && mv -v map_${i}.ply map_B_${i}.ply
  [ -f map_splat_${i}.ply ] && mv -v map_splat_${i}.ply map_B_splat_${i}.ply
done
python3 tools/diagnose_render.py map_splat_B.ply --bg-color "0.5,0.5,0.55" \
  --n 9 --out diag_B_final/ --seed 42 2>&1 | tail -15

# ─── 비교 ────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo " 두 run 완료 — 결과 위치:"
echo "=================================================================="
echo "  A: map_splat_A.ply, diag_A_final/"
echo "  B: map_splat_B.ply, diag_B_final/"
ls -lh map_splat_A.ply map_splat_B.ply 2>/dev/null
