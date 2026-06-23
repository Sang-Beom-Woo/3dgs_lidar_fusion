#!/usr/bin/env bash
# =============================================================================
# Smoke test — 본격 학습 전 5k iter 만 돌려 GPU 안정성 + 메모리 확인
# -----------------------------------------------------------------------------
# 용도:
#   * v3 파라미터 (voxel 0.06, mono_stride 16) 가 6GB GPU 에 안전한지 빠르게 검증
#   * 첫 1k iter 통과 = JIT 컴파일 + 메모리 peak 안정 시점
#   * 5k iter 마침 = ~7분 (12 it/s 가정), 본 학습 6.5시간 시작 전 확인
#
# 모니터링:
#   별 터미널에서 같이:
#     watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv'
#
# 결과 해석:
#   * "iter 5000/5000 loss=... (N it/s)" 까지 완주 → 안전, 본 학습 시작 OK
#   * iter 1000 전후에서 hang / OOM → 가우시안 추가로 줄여야 (voxel=0.08)
#   * GPU 메모리 5.0GB 이상 사용 → 위험 마진 부족
# =============================================================================

set -e
set -u

cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -u run_chapter2.py \
  --cam-downscale "left=4,right=4" \
  --init-voxel 0.06 \
  --init-scale 0.03 \
  --sh-degree 1 \
  --mono-depth \
  --mono-pixel-stride 16 \
  --mono-min-depth 0.3 \
  --mono-max-depth 15 \
  --lpips-lambda 0 \
  --cam-refine --kf-refine --view-affine \
  --iters 5000 \
  --log-every 500 \
  --opacity-threshold 0 \
  --out smoke_map.ply \
  --splat-out smoke_map_splat.ply 2>&1 | tee smoke_test.log

echo
echo "===================================================="
echo "[smoke] 5k iter 완주.  안전.  본격 학습 시작 가능:"
echo "  ./run_indoor_v2_long.sh"
echo "===================================================="
