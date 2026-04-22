# LiDAR-Visual 3DGS 하이브리드 파이프라인 (Gemini 대화 정리)

> 이 문서는 Sparse LiDAR + RGB 이미지 세트를 활용한 3DGS 디지털 트윈 구축 기획안과
> 스켈레톤 코드에 대해 Gemini와 나눈 대화를 정리한 것이다.
> 관점: 2021년 SLAM 전문가가 최신 3DGS(gsplat v1.x)로 옮겨가는 상황.

---

## 1. 기획 목표

> **"Sparse한 LiDAR의 한계를 3DGS의 렌더링 능력으로 극복하여 고밀도(Dense) 지도를 만든다."**

- 입력: Sparse LiDAR (Velodyne / Ouster 16·32·64ch) + RGB 이미지 세트
- 출력: Photorealistic Map, Dense Depth Map, (옵션) 3D Mesh

---

## 2. 파이프라인 4단계

### 2-1. Pre-processing & Global Registration
- **LiDAR Odometry**: LIO-SAM 또는 FAST-LIO2 → 고정밀 궤적 $T_{world}$ 및 통합 포인트 클라우드.
- **Camera-LiDAR Calibration**: 기확보된 외부 파라미터 $T_{calib}$ 로 LiDAR → 이미지 평면 투영 준비.
- **Keyframe Selection**: 이동거리(0.5~1 m) 혹은 회전각 기준으로 프레임 정제.

### 2-2. LiDAR-Informed Gaussian Initialization (핵심)
랜덤 초기화 대신 LiDAR 점군을 **seed** 로 사용.

| 파라미터 | 초기화 방식 |
|---|---|
| Means (μ) | LiDAR XYZ 그대로 |
| Scales (S) | k-NN 거리 기반, Sparse 영역은 조금 더 크게 (빈 공간 선점) |
| Colors (C) | LiDAR 점을 이미지에 투영해 RGB 샘플링, 미투영 점은 주변 보간/회색 |

### 2-3. Hybrid Optimization Loop

$$
L_{total} = \lambda_1 L_{rgb} + \lambda_2 L_{ssim} + \lambda_3 L_{depth}
$$

- $L_{depth}$ : 렌더된 깊이 $\hat{D}$ 와 LiDAR 측정 깊이 $D_{lidar}$ 의 차이 최소화.
- 효과: 가우시안이 시각적으로만 그럴듯한 게 아니라, 실제 물리 거리에 강하게 구속됨.

### 2-4. Geometry Refinement (Densification)
- LiDAR가 놓친 세밀 구조(전신주, 난간 등)는 **이미지 오차가 큰 영역**에서
  gsplat 의 gradient-based splitting 이 자동으로 가우시안을 쪼개 채움.

---

## 3. 기술 스택

| 역할 | 추천 도구 | 이유 |
|---|---|---|
| Data Handling | Open3D / PCL | PLY·PCD 전처리, k-NN 효율 |
| Core Engine | **gsplat v1.x** | 최신 `rasterization` API, 최고 속도 |
| Optimization | PyTorch (LibTorch) | 커스텀 $L_{depth}$ 추가 유연 |
| Visualization | Splatviz / Nerfstudio Viewer | 학습과정 실시간 웹 뷰어 |

---

## 4. 전략 조언 — Adaptive Weighting

- **낮 / 정상 조명**: $L_{rgb}$ 비중 ↑ → 텍스처 디테일 확보.
- **야간 / 역광**: $L_{depth}$ 비중 ↑ → 기하 붕괴 방지.
- 즉, LiDAR 신뢰도 vs. 이미지 신뢰도의 **동적 가중** 이 성패를 가름.

---

## 5. Python 스켈레톤 (gsplat v1.x)

```python
import torch
import torch.nn.functional as F
from gsplat import rasterization
import numpy as np

class LidarVisualGS:
    def __init__(self, lidar_points, lidar_colors, device="cuda"):
        self.device = device
        self.N = lidar_points.shape[0]

        # LiDAR 기반 초기화
        self.means  = torch.nn.Parameter(lidar_points.to(device))
        self.colors = torch.nn.Parameter(lidar_colors.to(device))

        # log-space scales (음수 방지, 넓은 dynamic range)
        self.scales        = torch.nn.Parameter(torch.full((self.N, 3), -5.0, device=device))
        self.quats         = torch.nn.Parameter(
            torch.tensor([1., 0., 0., 0.], device=device).repeat(self.N, 1))
        self.opacities_raw = torch.nn.Parameter(torch.full((self.N,), 0.1, device=device))

        self.optimizer = torch.optim.Adam(
            [self.means, self.scales, self.quats, self.opacities_raw, self.colors],
            lr=1e-3)

    def train_step(self, gt_image, gt_depth, K, viewmat):
        """
        gt_image : [H, W, 3]  카메라 이미지
        gt_depth : [H, W]     LiDAR 투영 Sparse Depth (빈 곳은 0)
        K        : [3, 3]     intrinsic
        viewmat  : [4, 4]     world-to-camera
        """
        self.optimizer.zero_grad()
        H, W = gt_image.shape[:2]

        renders, alphas, meta = rasterization(
            means     = self.means,
            quats     = self.quats,
            scales    = torch.exp(self.scales),
            opacities = torch.sigmoid(self.opacities_raw),
            colors    = self.colors,
            viewmats  = viewmat.unsqueeze(0),
            Ks        = K.unsqueeze(0),
            width     = W,
            height    = H,
            packed    = False,
        )

        rendered_rgb   = renders.squeeze(0)                 # [H, W, 3]
        rendered_depth = meta["expected_depth"].squeeze(0)  # [H, W, 1]

        # Photometric
        loss_rgb = F.l1_loss(rendered_rgb, gt_image)

        # LiDAR depth (유효 픽셀만)
        mask = gt_depth > 0
        loss_depth = F.mse_loss(rendered_depth[mask], gt_depth[mask])

        total_loss = loss_rgb + 0.5 * loss_depth
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()
```

### 코드 포인트
- `meta["expected_depth"]` : 픽셀별 기대 깊이 (α-blended z). LiDAR 실거리와 직접 비교 가능.
- `mask = gt_depth > 0` : Sparse LiDAR 특성상 유효 픽셀만 loss. 나머지는 $L_{rgb}$ 가 densification 유도.
- `torch.exp(scales)` : 스케일 음수 방지 + 넓은 동적 범위.
- **카메라 포즈도 추정하고 싶다면** `viewmat` 을 `nn.Parameter` 로 승격해 옵티마이저에 포함.

---

## 6. 다음 할 일 (Open Items)

- [ ] LiDAR → 이미지 평면 투영 함수 ($P = K[R\,|\,t]$) 작성.
- [ ] Adaptive weighting 스케줄러 ($\lambda_{rgb}, \lambda_{depth}$ 의 동적 조절).
- [ ] Keyframe 선택기 (거리/각도 threshold).
- [ ] Depth rendering 검증 (gsplat 버전별 `expected_depth` 키 존재 여부 확인).
- [ ] Pose refinement 모드(viewmat 학습 가능화) 스위치.
