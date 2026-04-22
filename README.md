# 3DGS × LiDAR 퓨전 — 교육용 프로젝트 개요

본 리포지토리는 로봇 SLAM 출력 (LiDAR + 다중 카메라 + keyframe pose) 을
**3D Gaussian Splatting** 기반의 포토리얼 맵으로 변환하는 실험 파이프라인이다.
단순 "돌아가는 코드" 가 아니라, 각 단계의 수식·설계 결정·시행착오를
**SLAM 전공자가 3DGS 를 처음 접할 때** 의 학습 자료로 기능하도록 구성했다.

---

## 목차

1. [배경 — 왜 3DGS 인가](#1-배경--왜-3dgs-인가)
2. [3DGS 수학적 기초](#2-3dgs-수학적-기초)
3. [본 프로젝트 변형 — LidarVisualGS](#3-본-프로젝트-변형--lidarvisualgs)
4. [핵심 알고리즘 상세](#4-핵심-알고리즘-상세)
5. [데이터셋 규약](#5-데이터셋-규약)
6. [코드 구조](#6-코드-구조)
7. [실행 가이드](#7-실행-가이드)
8. [실험 기록 / 배운 점](#8-실험-기록--배운-점)
9. [향후 과제](#9-향후-과제)
10. [참고 문헌](#10-참고-문헌)

---

## 1. 배경 — 왜 3DGS 인가

### 1.1 기존 SLAM 지도의 한계

고전 SLAM 은 **포인트 클라우드 (LiDAR/Depth)** 또는 **voxel/mesh 지도** 를
출력한다. 장점은 빠르고 기하학적으로 정밀하다는 것. 단점은:

- **포토리얼 렌더링 불가**: 포인트 자체는 구멍이 많고, mesh 는 텍스처 baking
  절차가 별도 필요.
- **시점 보간 어려움**: 학습 당시 보지 못한 각도에서는 정보 부족.
- **조명 / 반사 표현 빈약**: 단순 색 attach 로는 금속 반사, 거울, 창유리
  같은 view-dependent 효과 불가.

### 1.2 NeRF → 3DGS 의 흐름

**NeRF** (Mildenhall et al., ECCV 2020) 는 implicit representation
(좌표 → RGB+σ MLP) 으로 포토리얼 렌더를 달성했지만 **학습·렌더 속도가 느려**
(한 장에 수 분) 로봇/실시간 응용엔 부적합.

**3D Gaussian Splatting** (Kerbl et al., SIGGRAPH 2023) 은 explicit
표현 (수백만 개의 3D 가우시안) + tile-based 래스터화로 **학습 30 분 /
렌더 > 100 FPS** 를 달성하며 판도를 바꿨다. 로봇 맵핑·디지털 트윈에 직접
응용 가능한 수준이 됐다.

### 1.3 LiDAR 와 결합하는 이유

원 3DGS 는 **COLMAP SfM** (수만 개 희소 점) 으로 초기화한다. 로봇 시나리오엔
**LiDAR 가 이미 조밀 (수백만 점, metric scale)** 하므로:

- SfM 생략 → 초기화 속도·품질 ↑
- Metric 스케일 자동 확보 (나중에 simulation 에서 바로 사용 가능)
- Depth supervision → 기하 구속 강화 (floater 방지)
- **ADC (adaptive density control) 필요성 감소** — 씨앗이 이미 조밀함

---

## 2. 3DGS 수학적 기초

### 2.1 가우시안 한 개의 정의

각 가우시안 $i$ 는 다음 파라미터를 가진다:

```
μᵢ ∈ ℝ³        : 중심 좌표 (world frame, meters)
Σᵢ ∈ ℝ³ˣ³     : 공분산 행렬 (3D 타원체 형태)
αᵢ ∈ (0, 1)   : 투명도 (opacity)
cᵢ ∈ ℝᴷˣ³     : 색 (SH 계수, K = (d+1)²)
```

공분산은 회전 $R$ 과 축 스케일 $S$ 의 합성으로 분해해 저장:

$$
\Sigma = R\, S\, S^\top R^\top
$$

- $R$: 쿼터니언 $q = (w, x, y, z)$ 에서 생성한 3×3 회전 행렬
- $S = \mathrm{diag}(s_x, s_y, s_z)$: 축별 반경 (log-공간에 저장 → 학습 중 양수 보장)

### 2.2 렌더링 — tile-based α-compositing

카메라 뷰가 주어지면:

**Step 1. 3D → 2D 투영.** EWA (Elliptical Weighted Average) splatting 으로
3D 공분산 $\Sigma$ 를 2D 공분산 $\Sigma'$ 로 사영.

$$
\Sigma' = J\, W\, \Sigma\, W^\top J^\top
$$

- $W$: world → camera 회전
- $J$: affine approximation of perspective projection

**Step 2. Tile sorting.** 각 16×16 픽셀 타일마다 depth 로 정렬 (front-to-back).

**Step 3. Pixel compositing.** 각 픽셀 $u$ 에서:

$$
C(u) = \sum_{i=1}^{N} c_i\, \alpha_i(u)\, \prod_{j < i} (1 - \alpha_j(u))
$$

$\alpha_i(u)$ 는 2D 가우시안 평가값 × 기본 opacity.

**핵심 포인트**: 모든 연산이 미분 가능 → 픽셀 loss 에서 $\mu, \Sigma, \alpha, c$
로 역전파 → Adam 으로 학습.

### 2.3 파라미터화 규약

| 개념 | 저장값 | 변환 |
|---|---|---|
| scale | $s = \log(\mathrm{scale})$ | $\mathrm{scale} = \exp(s)$ |
| opacity | $o = \mathrm{logit}(\alpha)$ | $\alpha = \sigma(o)$ |
| rotation | quaternion $(w, x, y, z)$ | R = Rodrigues |
| color (SH DC) | $f_{dc} = (\mathrm{rgb} - 0.5) / \mathrm{SH}_0$ | $\mathrm{rgb} = 0.5 + f_{dc} \cdot \mathrm{SH}_0$ |

$\mathrm{SH}_0 = 1/(2\sqrt{\pi}) \approx 0.282094$. 이 상수는 구면 조화함수
0 차의 정규화 계수.

### 2.4 Spherical Harmonics — view-dependent 색

같은 3D 위치를 **다른 각도에서 볼 때** 색이 다른 경우 (금속 반사, 안경
유리 등) 를 표현하기 위해 SH 전개를 쓴다.

$$
c(\mathbf{v}) = \sum_{l=0}^{d}\sum_{m=-l}^{l} c_l^m\, Y_l^m(\mathbf{v})
$$

- $\mathbf{v}$: 가우시안에서 카메라로 향하는 단위 벡터
- $Y_l^m$: 구면 조화 기저 함수
- $d=0$: DC 항만 (view-independent)
- $d=3$: 원 3DGS 표준 (16 계수 / 채널)

**디퓨즈 실내 씬은 $d=1$ 로 충분**, 강한 specular 있으면 $d=2$ 이상 필요.

---

## 3. 본 프로젝트 변형 — LidarVisualGS

### 3.1 원 3DGS 대비 차이

| 항목 | 원 3DGS | **본 프로젝트** |
|---|---|---|
| 초기화 | COLMAP SfM (희소 점) | LiDAR + 이미지 투영 (조밀 점) |
| 스케일 | unit-less (SfM) | metric (meter) |
| Supervision | RGB 만 | RGB + LiDAR depth |
| Loss | L1 + SSIM | **L1 + SSIM + LPIPS** |
| 포즈 | 고정 (COLMAP output) | **cam / kf 모두 refine 가능** |
| ADC | 필수 (SfM 희소 보완) | **기본 OFF** (LiDAR 시드 조밀) |

### 3.2 데이터 흐름 한 눈에

```
base_data/                          (입력)
    ↓
BaseDataDataset → 각 kf 당:
    • LiDAR pcd (base 프레임)
    • 3 카메라 BMP
    • T_world_base (SLAM 출력)
    • 캘리브레이션 (K, D, T_base_cam)
    ↓
colorize_keyframe() → 각 LiDAR 점을 3 카메라에 투영해 RGB 샘플
    ↓
accumulate_seed() → 모든 kf world 좌표 누적 + max-saturation voxel 다운샘플
    ↓                                (가우시안 초기 μ, c)
LidarVisualGS(init)
    ↓
for iter in 1..N:
    무작위 뷰 선택
    forward: rasterize → (rendered RGB, depth)
    loss: L1 + SSIM + LPIPS + depth_MSE
    backward + Adam step (per-param LR)
    ↓ (옵션) densify_and_prune, opacity_reset
    ↓
prune_floaters() → α 낮은 + outlier 제거
save_splat_ply() → INRIA 3DGS 포맷 (splatviz 호환)
```

---

## 4. 핵심 알고리즘 상세

### 4.1 SSIM (Structural Similarity Index)

**왜**: L1 은 픽셀별 평균만 보므로 "전체적으로 뿌옇지만 평균 색 맞는" 국소
최적해를 쉽게 허용. SSIM 은 로컬 11×11 윈도우의 평균·분산·공분산을 비교해
**엣지/대비** 가 살아 있는 방향으로 학습을 유도한다.

**수식** (Wang et al. 2004):

$$
\mathrm{SSIM}(x, y) = \frac{(2\mu_1\mu_2 + C_1)(2\sigma_{12} + C_2)}
                          {(\mu_1^2 + \mu_2^2 + C_1)(\sigma_1^2 + \sigma_2^2 + C_2)}
$$

여기서 $\mu, \sigma$ 는 11×11 가우시안 윈도우 $N(\cdot)$ 기반 국소 통계.
$C_1 = (0.01)^2, C_2 = (0.03)^2$ 는 flat 영역 안정화 상수.

**조합**: Photometric loss = $(1 - \lambda_{\text{ssim}}) \cdot L_1 + \lambda_{\text{ssim}} \cdot (1 - \mathrm{SSIM})$,
$\lambda_{\text{ssim}} = 0.2$ (원 3DGS 규약).

구현: [math_utils.py](math_utils.py) / [losses.py](losses.py).

### 4.2 LPIPS — 지각적 유사성

**왜 SSIM 만으로 부족한가**: SSIM 도 결국 로컬 통계 평균이라 **소수
픽셀의 큰 오류** (예: 작은 신호등 색 틀림) 는 잘 못 잡는다. LPIPS (Zhang
et al., CVPR 2018) 는 VGG-16 중간 feature 의 가중 L2 거리를 사용 → 사람
눈이 중요하게 보는 부분을 자동으로 강조.

$$
\mathrm{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} \| w_l \odot (\phi_l(x) - \phi_l(y))\|_2^2
$$

- $\phi_l$: VGG 의 $l$번째 층 feature
- $w_l$: 사람 평가 데이터로 학습된 per-channel 가중치

**대가**: VGG forward 로 학습 속도 ~2× 느림, GPU 메모리 +~500MB.

구현: [losses.py](losses.py) `photometric_loss(...)` 내부.

### 4.3 Adaptive Density Control (ADC)

**개념**: 학습 중 가우시안의 수를 동적으로 조절. 표현력 부족한 곳은 증식,
과잉인 곳은 가지치기.

세 가지 동작:

1. **Clone** — 작은 가우시안 + 높은 2D 중심 gradient → 같은 위치에 복제
   (한 영역을 "두 자식이 서로 다른 방향으로 학습" 하도록 유도).
2. **Split** — 큰 가우시안 + 높은 gradient → 원본 가우시안 분포에서
   샘플링한 위치에 자식 2개 생성 (각 scale 은 1/1.6). 원본은 prune.
3. **Prune** — $\alpha < \epsilon$, screen-space 반경 과대, world-scale 과대,
   aspect-ratio 과대 중 하나라도 해당하면 제거.

**신호**: means2d (2D 투영 중심) 의 grad L2 norm 을 per-gaussian 누적.
grad 가 크다는 것은 pixel loss 를 줄이려면 이 가우시안이 화면에서
움직여야 한다는 뜻 → under-reconstruction.

**왜 본 프로젝트에선 기본 OFF 인가**: LiDAR 시드가 이미 조밀 (수백만 점)
해서 densification 이 불필요하고, **prune 이 오히려 표면 가우시안을
낭비적으로 제거**하는 현상을 관찰. Sparse SfM 시드 시나리오에만 유효.

구현: [chapter2.py](chapter2.py) 의 `densify_and_prune`.

### 4.4 Opacity Reset

**왜**: 학습이 진행될수록 "한 번 불투명해진 가우시안은 잘 안 사라지는"
편향이 생김. 주기적으로 α 를 인위적으로 0.01 같이 낮게 깎으면, 진짜 필요한
가우시안은 다음 몇백 iter 에 그래디언트가 다시 올려놓고, 불필요한 것은
낮은 상태에 머물러 prune 대상이 됨.

**핵심**: ADC 와 짝으로 동작해야 한다. Reset 만 반복하면 가우시안이
단조 감소해 맵이 빈다. 본 프로젝트는 ADC OFF 가 기본이라 reset 도 OFF.

### 4.5 Per-parameter Learning Rate

Adam 에 **파라미터별 다른 LR** 을 주는 것이 원 3DGS 의 핵심 안정화 기법:

| 파라미터 | LR |
|---|---|
| means | $1.6 \times 10^{-4}$ |
| scales | $5 \times 10^{-3}$ |
| quats | $1 \times 10^{-3}$ |
| opacities | $1 \times 10^{-2}$ (원 논문 5e-2, 우리 setup 조정) |
| sh_dc | $2.5 \times 10^{-3}$ |
| sh_rest | $1.25 \times 10^{-4}$ (sh_dc / 20) |

위치는 천천히 움직이도록 (잘못 흔들리면 복구 어려움), 반면 opacity 는
빠르게 움직이게 (reset 후 회복 + ADC 와 동조). sh_rest 는 천천히 (고차
SH 가 너무 빨리 학습되면 색이 들떠 α 가 흔들림).

### 4.6 Pose Refinement

SLAM 출력 포즈에는 언제나 오차가 있다. 두 종류로 나눠 학습 가능 delta 를 둠.

**Cam extrinsic refinement** (`cam_refine`)
  - 파라미터: 3 카메라 × 6DoF = 18
  - 역할: **static** calibration 오차 (LiDAR-camera 배치 부정확)
  - 본 프로젝트에서 실측 발견: front/left/right 모두 z축 +30~50mm drift

**Per-keyframe pose refinement** (`kf_refine`)
  - 파라미터: N_kf × 6DoF
  - 역할: **dynamic** 궤적 드리프트 (SLAM 루프클로저 부족 구간)
  - 실측 최대 drift: 43mm, 평균 12mm

**수학**:

$$
T^{\text{true}}_{\text{cam, world}} = \delta T_{\text{cam}}^{-1} \cdot \delta T_{\text{kf}}^{-1} \cdot T^{\text{nominal}}_{\text{cam, world}}
$$

(코드는 부호 학습에 의존하므로 좌측 곱만 사용.)

Delta 는 **6D tangent (rx, ry, rz, tx, ty, tz)** 로 파라미터화 후 Rodrigues
공식으로 SE(3) 으로 변환. 구현: [math_utils.py](math_utils.py) `delta_to_se3(...)`.

### 4.7 Max-saturation Voxel Downsample

**문제**: Open3D 기본 `voxel_down_sample` 은 같은 voxel 안 점들의 색을
**산술평균**. 신호등 / 차량 램프 같은 소수의 뷰에서만 선명한 색을 갖는
소물체는 평균 시 **회색으로 수렴**한다.

**해법**: 각 voxel 에서 가장 **saturation 이 큰** 점의 색을 대표로 선택.

$$
\text{saturation} = \frac{\max(R, G, B) - \min(R, G, B)}{\max(R, G, B)}
$$

무채색 영역 (바닥·벽) 은 모든 후보가 saturation 낮아서 아무거나 골라도 OK.
고채도 피처는 **선명색 유지**. 실험 결과 평균 채도 +23%, p95 +41% 개선.

구현: [run_chapter2.py](run_chapter2.py) `_voxel_downsample_max_saturation(...)`.

### 4.8 Floater Prune (학습 종료 후 1 회)

ADC 가 OFF 인 경우 학습 중 가우시안 수가 줄지 않으므로 종료 후 한 번 정리.

1. **낮은 α 제거**: $\alpha < \epsilon$ 가우시안 삭제.
2. **Statistical outlier 제거**: 각 가우시안의 이웃 평균거리가 전역
   평균 + $k\sigma$ 초과하면 "고립된 점 = floater" 판정 → 제거. (Open3D
   `remove_statistical_outlier`)

표면 위 정상 가우시안은 이웃이 빽빽해서 outlier 판정 안 됨 → 안전.

---

## 5. 데이터셋 규약

### 5.1 `base_data/` 폴더 구조

```
base_data/
├── calibration/
│   ├── v4l2_intrinsic.txt   # 3 카메라 fisheye intrinsic
│   └── sensor_tf.txt        # T_base_<front/left/right/lidar>
└── <kf_id>/                 # 0..680 (정수 폴더명, 건너뜀 가능)
    ├── point_cloud.pcd      # LiDAR binary (x y z intensity, base 프레임)
    ├── front_color.bmp      # 640×360, 없을 수 있음
    ├── left_color.bmp
    ├── right_color.bmp
    ├── keyframe_pose.txt    # id base_link ts_ns tx ty tz qx qy qz qw
    ├── keyframe_type.txt    # id <char>
    ├── outdoor.txt          # id {0, 1}
    ├── elevator.txt         # id {0, 1}
    └── descriptor.txt       # NetVLAD-류 place descriptor
```

### 5.2 좌표계 규약

| Frame | 의미 |
|---|---|
| `base_link` | 로봇 몸체. LiDAR 점과 keyframe_pose 의 기준 |
| `world` | SLAM 이 정의한 전역 좌표계 |
| `optical` (cam) | OpenCV 규약: x=right, y=down, z=forward |

변환:

$$
T^{A}_{B}: \text{"A frame for B's points"} \Leftrightarrow p_A = T^{A}_{B} \cdot p_B
$$

자주 쓰이는 조합:

- $T^{\text{world}}_{\text{cam}} = T^{\text{world}}_{\text{base}} \cdot T^{\text{base}}_{\text{cam}}$
- $T^{\text{cam}}_{\text{world}} = (T^{\text{world}}_{\text{cam}})^{-1}$ ← gsplat viewmat

### 5.3 Fisheye Undistortion

원본 이미지는 fisheye (equidistant 모델), 우리 파이프라인은 핀홀 전제.
학습 전 **언디스토션 + new_K** 로 변환.

절차 (`Calibration.build_undistort_maps`):
1. `estimateNewCameraMatrixForUndistortRectify(balance=0)` → 모든 픽셀 valid
   유지하는 new_K 결정 (검은 가장자리 최소)
2. `initUndistortRectifyMap` → 픽셀 매핑 테이블 (CV_16SC2, 고속)
3. 런타임 `cv2.remap` 한 번 호출

---

## 6. 코드 구조

```
gsplat_ws/
├── base_data_parser.py    # 데이터셋 로더 (694 줄)
│   ├── Calibration, CameraIntrinsics, KeyFrame (dataclass)
│   ├── load_calibration, load_keyframe
│   ├── BaseDataDataset (iterable)
│   ├── project_lidar_to_camera
│   ├── build_chapter2_inputs  ← 학습 샘플 1개 dict
│   └── colorize_keyframe      ← seed 컬러링
│
├── math_utils.py          # 순수 수학 (151 줄)
│   ├── ssim, _gaussian_1d, _ssim_window
│   ├── quat_to_rot, axis_angle_to_rot
│   ├── delta_to_se3      ← pose refinement 핵심
│   └── quat_xyzw_to_rot_np (ROS 규약)
│
├── losses.py              # Loss 함수 (125 줄)
│   ├── build_lpips(device)  ← VGG 로드
│   ├── photometric_loss (L1 + SSIM + LPIPS)
│   ├── depth_loss
│   └── opacity_hinge / bimodal / anisotropy (regularizer)
│
├── chapter2.py            # 메인 모델 (713 줄)
│   └── class LidarVisualGS
│       ├── __init__: 5 단계 서브루틴 분리
│       │   ├── _init_gaussian_params
│       │   ├── _init_optimizer
│       │   ├── _init_loss_config
│       │   ├── _init_densify_buffers
│       │   └── _init_pose_refine
│       ├── train_step + _apply_pose_refine + _compute_total_loss
│       ├── ADC: _swap_param, _slice_all, _append_all, _clone, _split,
│       │        densify_and_prune, _pad_mask
│       ├── reset_opacity, prune_floaters
│       └── save_splat_ply (INRIA), save_ply (pointcloud)
│
├── run_chapter2.py        # CLI 드라이버 (461 줄)
│   ├── accumulate_seed + _voxel_downsample_max_saturation
│   ├── build_samples
│   ├── train + _one_iter_forward_backward + _maybe_densify / _maybe_opacity_reset
│   └── main + argparse 그룹 (I/O, Init, Training, Pose, Save, Regularizer, ADC)
│
├── chapter1.py            # 입문 예제 (랜덤 1000 가우시안 → 빨간 사각형)
│
└── tools/                 # 진단 / 디버깅 스크립트
    ├── dump_seed.py           # 학습 전 seed PLY 덤프
    ├── check_undistort.py     # 언디스토션 적합성 + LiDAR 재투영
    ├── check_traffic_light.py # 3D 영역의 색·α·밀도 통계
    └── diagnose_render.py     # 학습된 PLY vs GT 이미지 N-view 비교
```

---

## 7. 실행 가이드

### 7.1 준비

```bash
pip install torch torchvision gsplat lpips open3d opencv-python numpy
```

GPU 필수 (RTX 3060 Laptop 6GB 에서 확인됨).

### 7.2 기본 학습

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 run_chapter2.py \
    --init-voxel 0.05 --init-scale 0.03 \
    --iters 30000 --log-every 2000 \
    --cam-refine --kf-refine \
    --floater-prune \
    --out map.ply --splat-out map_splat.ply
```

**주요 플래그 의미:**

| 플래그 | 역할 |
|---|---|
| `--init-voxel` | seed downsample (m). 작을수록 가우시안 ↑, 메모리 ↑ |
| `--init-scale` | 가우시안 초기 반경 (m). voxel × 0.5 권장 |
| `--cam-refine` | 3 카메라 × 6DoF extrinsic 학습 |
| `--kf-refine` | per-kf pose 학습 (BA-lite) |
| `--sh-degree` | 0~3. 1=권장 (indoor), 3=원 3DGS 표준 |
| `--floater-prune` | 학습 종료 후 α<5e-3 + outlier 제거 |
| `--densify-from-iter 500` | ADC 활성화 (기본 999999=OFF) |

### 7.3 메모리 한계 가이드

| `--init-voxel` | 가우시안 수 | 메모리 | 권장 GPU |
|---|---|---|---|
| 0.1 | ~400k | ~1 GB | 4GB 이상 |
| **0.05** | ~1.6M | ~3 GB | **6GB (현재)** |
| 0.03 | ~4M | ~5.5 GB | 8GB 이상 |
| 0.01 | ~9M | OOM | 16GB 이상 |

### 7.4 진단 도구

```bash
# 학습 전 seed PLY (컬러 포인트클라우드)
python3 tools/dump_seed.py --init-voxel 0.05 --out seed.ply

# 학습된 모델을 학습 뷰에서 렌더 → GT 비교
python3 tools/diagnose_render.py map_splat.ply --n 6 --out diag/

# 특정 3D 영역 통계 (신호등 등 작은 물체 분석)
python3 tools/check_traffic_light.py --center 10 3 1.5 --radius 0.3

# 언디스토션 + LiDAR 재투영 시각화
python3 tools/check_undistort.py --kf 100 --cam front --out calib_check/
```

### 7.5 PLY 뷰어

- **map_splat.ply** (INRIA 3DGS 포맷): [splatviz](https://github.com/Florian-Barthel/splatviz),
  [SuperSplat](https://superspl.at/editor), antimatter15 web viewer
- **map.ply** (일반 포인트클라우드): MeshLab, CloudCompare, Open3D

---

## 8. 실험 기록 / 배운 점

본 프로젝트는 단계적 개선을 거쳤고, 각 단계에서 **확인된 사실** 을 아래 기록.
향후 유사 프로젝트의 시행착오 감축용.

### 8.1 시드 구축 단계

- ✅ **LiDAR-seeded 3DGS 는 SfM 대비 빠름** — 초기 수렴 5~10× 빠름.
- ❌ **카메라 FOV 밖 LiDAR 점을 회색으로 seed 하면 안 됨** — 학습되지 않는
  "데드 가우시안" 이 누적. `drop_unseen=True` 가 기본값.
- ✅ **Max-saturation voxel downsample** — 산술평균은 다중 뷰 섞임 시 회색
  수렴. 채도 최대 점 선택이 소물체 색 보존에 결정적.

### 8.2 ADC 단계

- ❌ **LiDAR 시드에서 ADC 활성 시 재앙** — prune 이 표면 가우시안을 제거해
  구멍 뚫린 맵 발생. 기본 OFF 가 정답.
- ❌ **Opacity reset 단독 사용 불가** — ADC 없이 reset 만 돌리면 맵이 점점 빔.
- ✅ **Floater prune (학습 종료 후 1회) 로 충분** — ADC 없이도 outlier
  제거만 하면 깨끗한 결과.

### 8.3 표현력 단계

- ✅ **LPIPS 가 샤프니스 핵심** — L1+SSIM 만으론 "수채화 같은 블러" 에서
  멈춤. LPIPS 추가 후 엣지 확실히 살아남.
- ⚠️ **SSIM λ 를 0.4 로 높이면 needle artifact** — 가성비 좋은 값은 0.2.
- ✅ **SH1 이 indoor 에선 sweet spot** — SH3 는 메모리만 3× 늘고 품질
  거의 같음. specular 많은 야외는 다를 수 있음.
- ✅ **Per-parameter LR 필수** — 전부 1e-3 로 돌리면 opacity 가 너무 느려
  ADC/reset 이 의도대로 작동 못 함.

### 8.4 Pose Refinement 단계

- ✅ **cam_refine 에서 30~50mm drift 학습됨** → 하드웨어팀이 **재확인 권고**
  (체계적 calibration 편향 시사).
- ✅ **kf_refine 로 특정 구간 문제 진단 가능** — kf #386, #655, #368 에서
  >34mm drift → SLAM 루프클로저 부족 구간으로 지목.
- 📝 **Pose refinement 로 loss 는 크게 안 떨어지지만 ghosting 시각적으로
  크게 감소** — 수치 기반 비교 만으로 부족, 시각 검증 필수.

### 8.5 회색 수렴 문제

**이슈**: 신호등처럼 명확히 원색인 물체도 학습 후 회색으로 수렴.

**원인 계층 (기여도 순)**:
1. **카메라 출력 자체가 저채도** (v4l2 raw ISP, HSV 0.113) — 가장 큰 원인
2. **Voxel downsample 산술평균** — 다중 뷰 섞이면 평균은 무채색 ← **max-sat 로 해결**
3. **Loss 에서 소물체 기여 부족** (0.2% 픽셀 = 희생됨)
4. **LiDAR 는 검은 표면 (신호등 housing) 을 잘 못 봄** — seed 부재
5. **Alpha compositing 은 원리적으로 pure black 불가** — 앞 floater α×색 합성
6. **Vignetting** — 모델링 안 하면 가장자리 어두움이 "씬 색" 에 baked

해결은 각 원인별 독립적 조치 필요. 본 프로젝트는 (2) 만 완료. (1), (6) 은
하드웨어/전처리 개선, (3)~(5) 는 salient weighting / image seed 등 향후 과제.

---

## 9. 향후 과제

### 9.1 품질 개선

| 개선 | 난이도 | 효과 |
|---|---|---|
| 이미지 다이나믹 마스킹 (YOLOv8-seg 전처리) | 🟢 낮음 | ⭐⭐⭐ ghosting 제거 |
| Vignetting 학습 (3 cams × 2 coeffs) | 🟢 낮음 | ⭐⭐ 채도 향상 |
| Salient loss weighting (edge/contrast 가중) | 🟡 중간 | ⭐⭐ 소물체 디테일 |
| Mono depth (Depth Anything v2) 보완 | 🟡 중간 | ⭐⭐ LiDAR 사각지대 채움 |
| Image-based densification | 🟡 중간 | ⭐⭐ 검은 표면 seed |
| 이미지 해상도 ↑ (upsampling) | 🔴 높음 | ⭐ 한계 명확 |

### 9.2 시뮬레이터 plug & play

시뮬레이션 자산으로 바로 쓰려면 추가 필요:

1. **다이나믹 객체 완전 제거** (LiDAR ERASOR + image 마스킹 둘 다)
2. **충돌 지오메트리** — Poisson reconstruction → OBJ/STL
3. **좌표 표준화** — 원점=로봇 시작점, Z축=중력 반대
4. **포맷 변환** — Isaac Sim(USD), Unreal(FBX), Unity plugin 등
5. **메타데이터 JSON** — 스케일, bbox, 학습된 calibration delta 포함

### 9.3 다른 SLAM 시스템 연계

현재는 특정 SLAM 출력 포맷 (`keyframe_pose.txt` 등) 전제. 다른 시스템
(LIO-SAM, FAST-LIO2, Kimera) 연계하려면 `BaseDataDataset` 을 어댑터
패턴으로 일반화 필요.

---

## 10. 참고 문헌

### 핵심 논문

- **Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G.** (2023).
  *3D Gaussian Splatting for Real-Time Radiance Field Rendering.*
  ACM Transactions on Graphics (SIGGRAPH).
  [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)

- **Mildenhall, B., Srinivasan, P.P., Tancik, M., et al.** (2020).
  *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.*
  ECCV.

- **Wang, Z., Bovik, A.C., Sheikh, H.R., & Simoncelli, E.P.** (2004).
  *Image Quality Assessment: From Error Visibility to Structural Similarity.*
  IEEE TIP.

- **Zhang, R., Isola, P., Efros, A.A., Shechtman, E., & Wang, O.** (2018).
  *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.*
  CVPR. (LPIPS)

- **Zwicker, M., Pfister, H., van Baar, J., & Gross, M.** (2001).
  *EWA Volume Splatting.*
  IEEE Visualization. (splatting 수학적 기초)

### 확장 / 변종

- **Mip-Splatting** — anti-aliasing 개선 (Yu et al., CVPR 2024)
- **Scaffold-GS** — 계층적 anchor 기반 GS
- **2D-GS** — 평면 기반 GS (표면 재구성 유리)
- **BAD-Gaussians** — camera pose joint optimization
- **Depth-Regularized GS** — depth supervision 공식화

### 구현 참조

- **gsplat** 라이브러리 — [https://github.com/nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)
- **INRIA 3DGS 원본** — [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

### 관련 SLAM 주제

- **Removert**, **ERASOR**, **Dynablox**, **MapMOS** — LiDAR 다이나믹 제거
- **NetVLAD** — place recognition descriptor (`descriptor.txt` 의 출처 추정)

---

## 라이선스 / 크레딧

내부 실험 프로젝트. 외부 의존 라이브러리 (gsplat, lpips, torch, open3d, cv2,
numpy) 는 각 라이선스 따름.
