# mvalley 3DGS 학습 연구 노트

> **데이터**: 야외 주행 시퀀스 (440 keyframes, 600m extent, Z=-2~87m, 건물/소나무/도로 포함)
> **장비**: Robot 탑재 front (1280×1024) + left/right (4032×3036) fisheye 카메라 + 회전 LiDAR
> **GPU**: RTX 3060 Laptop 6GB
> **최종 결과**: 1.44M 가우시안 / 평균 diff_mean **32.1** / aspect>5 **17.6%**

---

## 1. 데이터셋 특징

| 항목 | 값 | 학습에 미친 영향 |
|---|---|---|
| Keyframe 수 | 440 | 30k iter 학습에 충분 |
| Scene extent | ~600m | 가우시안 밀도 vs GPU 메모리 trade-off 결정적 |
| Z range | -2 ~ 87m | sky/건물 위 prune 가능 (실제로는 안 좋았음) |
| Front 카메라 | 1280×1024, fisheye equidistant | 그대로 처리 |
| Left/Right | 4032×3036, fisheye equidistant | 4× downscale 필요 (메모리) |
| LiDAR | 회전식, ~17k points/scan | LiDAR-only 시드의 한계 노출 |
| 동적 객체 | 사람, 차량 일부 | kf 184 같은 dynamic outlier 식별됨 |

---

## 2. 시도한 실험 (시간 순)

### 실험 A: 베이스라인 — 모든 옵션 OFF

```
voxel 0.5, init_scale 0.25, ADC OFF, LPIPS OFF, bg OFF, refine OFF, mono OFF
```

| 결과 | 값 |
|---|---|
| 가우시안 N | ~1.5M |
| 평균 diff_mean | ~30 |
| **문제** | 흐릿함 / 하늘에 거대 floater (max_scale 18m!) / 24% needle |

### 실험 B: pose-refine 추가 (cam-refine + kf-refine + view-affine)

| 결과 |
|---|
| 평균 diff_mean 미세 개선 |
| cam_refine: 30-140mm translation drift 학습됨 (calibration 정상 보정) |
| kf_refine: 평균 11mm drift 보정 (SLAM trajectory 정상) |

### 실험 C: 300k iter Long run (refine-freeze 없이)

```
iters 300000, refine 학습 계속 진행
```

**🚨 catastrophic 붕괴**:
- 1.5M 가우시안 → **7 개** (전체 학습 동안 점진적 prune)
- cam_refine 회전: **89°** (정상 0.3°)
- cam_refine 이동: **1.5m** (정상 100mm)
- view-affine bias: **0.5** (정상 0.01)
- EMA loss: 0.27 → **1.83** (학습 발산)

**원인**: pose-refine 가 자유 학습되면 degenerate solution (포즈/affine 으로 GT 흉내) 로 수렴.
**교훈**: **긴 학습엔 `--refine-freeze-iter` 필수** (30k~100k 시점에 freeze).

### 실험 D: ICP 포즈 재정렬 시도

도구 [`tools/align_pose_icp.py`](../tools/align_pose_icp.py) 작성 — sequential pairwise ICP (point-to-plane, target window 3).

| 데이터셋 | 결과 |
|---|---|
| 175 kf 실내 (75m) | ✅ 잘 됨, 평균 100mm drift 보정 (kf 80 에서 17° 정렬 오류 잡음) |
| **440 kf 야외 (600m)** | ❌ **catastrophic** — kf 310 에서 34° rotation jump → 누적 65m drift |

**교훈**: pure ICP odometry 는 짧은 trajectory + SLAM 부정확 케이스에서만 안전. 야외 600m 같은 긴 궤적엔 **SLAM 의 loop closure 정보가 결정적**. → 원본 SLAM 포즈 복구.

### 실험 E: 배경색 (bg-color) 도입

```
--bg-color "0.8,0.85,0.9"   ← 회청색 (흐린 하늘)
```

**효과**:
- 하늘 자리에 만들어지던 거대 floater 가우시안 사라짐 (max_scale 18m → 6m)
- view-affine bias 부담 감소

**알게 된 것**: bg-color 는 학습 시 픽셀 안 닿는 곳 메우는 용. PLY 자체엔 안 들어가서 **외부 뷰어에선 검정으로 나옴** — 뷰어 설정 필요.

### 실험 F: prune-z-max 4m + mono depth

```
--mono-depth --mono-max-depth 80 --prune-z-max 4.0
```

**문제 발견**:
- 위쪽 (건물 윗부분, 나무 위, 하늘) 가우시안 **전부 0** → 렌더 시 회청색 빈칸
- 평균 diff_mean **44** (악화, 위쪽 영역 못 학습 영향)

**교훈**: `--prune-z-max` 는 bg-color 가 이미 차단 역할 하니까 **거의 무의미**. 위쪽 정보를 통째로 버린 셈.

### 실험 G: outlier kf 제외 (--exclude-kf-file)

도구 [`tools/detect_outlier_kf.py`](../tools/detect_outlier_kf.py): 회전 속도 + 과노출 픽셀 비율로 outlier kf 자동 검출.

**공격적 임계 (saturation 15%, bright Δ15)**: 17% 제외 → 학습 sample 크게 줄어듦 → diff 악화
**보수적 임계 (saturation 25%, bright Δ25)**: 1 kf 제외 → 거의 영향 없음

**교훈**: outlier 검출은 **보수적으로**. 의심스러우면 포함하는 게 안전.

### 실험 H: LiDAR-only + voxel 0.25 (mono off)

| 결과 |
|---|
| 가우시안 691K (이전 880K → -21%) |
| 평균 diff_mean **33.5** (개선) |
| max_scale **6.99m** (이전 43m → ✅ floater 거의 사라짐) |
| **aspect>5 = 46.6%** ❌ (LiDAR-only 라 sparse → 가우시안이 옆으로 늘어남) |

iter 30k → 60k 진행 시 **aspect>5 가 46.6% → 68.9% 폭증** = needle 만들기 학습. diff 도 33.5 → 35.5 악화.

### 실험 I: **🎯 핵심 돌파구 — aniso loss 0.01 + voxel 0.15**

```
--aniso-lambda 0.01    # needle penalty
--init-voxel 0.15      # 더 dense seed
--init-scale 0.07
```

| 메트릭 | 이전 (H) | **이번 (I)** | 변화 |
|---|---|---|---|
| 가우시안 N | 691K | 1,470K | +113% |
| **aspect>5 비율** | 46.6% | **15.6%** | ✅ **-67%** |
| **aspect p95** | 21.9 | **6.36** | ✅ **-71%** |
| aspect max | 1243 | 641 | ✅ -48% |
| 평균 diff_mean | 33.5 | 33.1 | ≈ |

**aniso loss 는 단순 ReLU(log_ratio − log(5))** — aspect 5 넘으면 선형 penalty. **거의 영향 없는 비용으로 needle 폭증 완전 차단**.

### 실험 J: + LPIPS 0.05 (downscale 2) — **최종 설정**

```
--lpips-lambda 0.05
--lpips-downscale 2    # 640×512 평가, 6GB GPU 안전
```

| 메트릭 | 이전 (I, no LPIPS) | **최종 (J, + LPIPS)** | 변화 |
|---|---|---|---|
| 가우시안 N | 1,470K | 1,438K (floater prune 후) | ≈ |
| 평균 diff_mean | 33.1 | **32.1** | ✅ -1.0 |
| best 뷰 diff | 17.4 | **17.1** | ✓ |
| aspect>5 | 15.6% | 17.6% | ⚠ +2% (LPIPS 의 sharpness 부담) |
| 학습 속도 | 15 it/s | 6.6 it/s | -56% (LPIPS 비용) |

**LPIPS 효과**: pixel diff 는 -1 정도지만 perceptual sharpness 가 핵심 — edge, texture 또렷.

---

## 3. 퀄리티 상승 기여도 순위

| 순위 | 변경 | 영향 | 영향 메트릭 |
|---|---|---|---|
| 🥇 1 | **aniso loss 0.01** | needle 폭증 차단 | aspect>5: 46.6% → 15.6% |
| 🥈 2 | **bg-color 회청색** | 거대 floater 제거 | max_scale: 18m → 6m |
| 🥉 3 | **refine-freeze 100k** | 300k 붕괴 방지 | 7 가우시안 → 1.44M 유지 |
| 4 | **LPIPS 0.05 + downscale 2** | perceptual sharpness | diff: 33.1 → 32.1, edge ↑↑ |
| 5 | **voxel 0.5 → 0.15** | 디테일 표현 | N: 691K → 1.47M |
| 6 | **cam-downscale (left/right ×4)** | GPU OOM 방지 | 14GB → 3.5GB |
| 7 | **exposure-clip-pct 95** | 과노출 영역 안정 | kf 184 diff: 71 → 36 |
| 8 | **distortion_model 자동 분기** | 캘리브 호환성 | 미래 데이터 호환 |

---

## 4. 최종 설정 ([run_outdoor_v2_long.sh](run_outdoor_v2_long.sh))

```bash
python3 -u run_chapter2.py \
  --cam-downscale "left=4,right=4" \
  --init-voxel 0.15 --init-scale 0.07 \
  --sh-degree 1 \
  --lpips-lambda 0.05 --lpips-downscale 2 \
  --aniso-lambda 0.01 \
  --bg-color "0.8,0.85,0.9" \
  --cam-refine --kf-refine --view-affine \
  --refine-freeze-iter 100000 \
  --exposure-clip-pct 95 \
  --exclude-kf-file exclude_kf.json \
  --iters 30000 --log-every 1000 \
  --checkpoint-every 10000 \
  --floater-prune --floater-opacity 5e-3 \
  --opacity-threshold 0 \
  --out map.ply --splat-out map_splat.ply
```

---

## 5. 최종 결과 (30000 iter)

### per-view diff_mean (낮을수록 좋음, 0~255 단위)

| view | kf | cam | diff_mean | 평가 |
|---|---|---|---|---|
| 5 | 291 | right | **17.1** | 🟢 best |
| 0 | 265 | front | 26.6 | 🟢 |
| 4 | 327 | left | 27.1 | 🟢 |
| 2 | 347 | right | 32.6 | 🟡 |
| 3 | 255 | front | 35.0 | 🟡 |
| 1 | 78 | left | 36.1 | 🟡 |
| 7 | 82 | left | 36.6 | 🟡 |
| 8 | 438 | right | 38.4 | 🟠 |
| 6 | 184 | front | 39.6 | 🟠 (과노출 + 회전 콤보 — 데이터 한계) |
| **평균** | | | **32.1** | |

### 모델 통계
- N = 1,438,519 (floater prune 후)
- α p50 = 0.222, p95 = 0.768
- max_scale = 6.24m (이전 18m 대비 ✅)
- aspect>5 = 17.6%, p95 = 6.73 (이전 46.6%, p95 21.9 대비 ✅)
- view-affine 최종 gain ~0.89, bias ~-0.044 (전체 살짝 어둡게 정규화)
- kf_refine: mean 9.0mm, max 23.7mm drift 보정 (정상 mm-level)

---

## 6. 주요 실패와 교훈

### 실패 1: 300k iter Long run 붕괴
- **원인**: pose-refine LR decay 없이 300k iter 학습 → degenerate solution
- **교훈**: `--refine-freeze-iter` 핵심

### 실패 2: ICP odometry 야외 적용 (kf 310 에서 34° jump)
- **원인**: pure ICP 는 loop closure 없어 누적 drift 발산
- **교훈**: 짧은 trajectory + SLAM 부정확 케이스에서만 사용

### 실패 3: prune-z-max 4 + 공격적 exclude
- **원인**: 위쪽 정보 통째 버림 + 학습 view 17% 제외
- **교훈**: bg-color 가 이미 floater 차단 → prune-z 거의 무의미. exclude 는 보수적으로.

### 실패 4: LPIPS at full resolution → OOM
- **원인**: 1280×1024 VGG features ~1.3GB 필요 (6GB GPU)
- **교훈**: `--lpips-downscale 2` 로 640×512 평가 → 충분

### 실패 5: needle 폭증 (aniso loss 도입 전)
- **원인**: LiDAR-only seed sparse + photometric loss 만 → 가우시안이 빈공간 메우려 늘어남
- **교훈**: `--aniso-lambda 0.01` 이 simple 한 ReLU penalty 로 완전 해결

---

## 7. PLY 후처리 도구

| 도구 | 용도 |
|---|---|
| [`tools/diagnose_render.py`](../tools/diagnose_render.py) | GT vs render 9-view 비교 (bg-color 매칭) |
| [`tools/prune_needles.py`](../tools/prune_needles.py) | aspect ratio 기준 needle 제거 |
| [`tools/prune_z.py`](../tools/prune_z.py) | 높이 기준 cut |
| [`tools/reorient.py`](../tools/reorient.py) | 축 재정렬 (Z-up → Y-up, SH 회전 포함) |
| [`tools/center_on_camera.py`](../tools/center_on_camera.py) | 카메라 위치를 origin 으로 이동 |
| [`tools/bake_view_affine.py`](../tools/bake_view_affine.py) | view-affine 평균을 SH 에 흡수 |
| [`tools/dump_aligned_pcd.py`](../tools/dump_aligned_pcd.py) | LiDAR 통합 PCD 검증용 export |
| [`tools/align_pose_icp.py`](../tools/align_pose_icp.py) | ICP 포즈 재정렬 (주의: 야외 catastrophic 가능) |
| [`tools/detect_outlier_kf.py`](../tools/detect_outlier_kf.py) | outlier kf 자동 검출 |

---

## 8. 알려진 한계

### A. 바닥/하늘 동시 표현 불가
- bg-color 는 단일 색 → 위는 sky 색 OK 면 아래는 부자연
- 항공뷰에서 바닥 가우시안 sparse → bg 가 비집고 보임 ("투명 바닥")
- 해결: mono-depth ON / ADC ON / 더 dense seed 필요

### B. 학습 뷰 방향 편향
- 모든 학습 view = 지상 카메라 + 정면
- 항공뷰 / 측면뷰 = 학습 안 본 시점 → 가우시안 측면 노출 → 품질 ↓
- 해결: 다양한 시점 데이터 필요 (드론 + 지상 결합 등)

### C. kf 184 같은 outlier
- 빠른 회전 (40°/kf) + 과노출 콤보
- 본질적 데이터 한계 — 어떤 모델도 완벽 복원 불가
- 해결: exposure-clip 으로 부분 완화 (71 → 36)

### D. 외부 뷰어에서 bg-color 안 보임
- PLY 포맷 자체엔 bg 정보 없음
- SuperSplat / antimatter15 등은 자체 bg 설정 사용
- 해결: 뷰어에서 수동 설정 또는 sphere 베이크 (별도 구현 필요)

---

## 9. 다음 단계 후보

| 우선순위 | 변경 | 예상 효과 |
|---|---|---|
| 1 | mono-depth back ON + voxel 0.1 | 바닥/원경 가우시안 확보 (sphere 베이크 대안) |
| 2 | ADC ON (densify) | 디테일 영역 자동 보강 |
| 3 | SH degree 1 → 2 | 시점 의존 색 표현 향상 (메모리 +30%) |
| 4 | 30k → 60k iter | LPIPS 수렴 더 진행 (가능성 있음, 자동 종료 주의) |
| 5 | sphere 베이크 (배경) | PLY 단독 공유 가능하게 |

---

*마지막 업데이트: 2026-06-23*
*작성 환경: Claude (sonnet/opus) + 사용자 협업*
