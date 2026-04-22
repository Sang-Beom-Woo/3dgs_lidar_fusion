import torch
import torch.nn.functional as F
from gsplat import rasterization
import PIL.Image as Image
import numpy as np

# 1. 초기 파라미터 설정 (SLAM의 '지도' 역할)
N = 1000 # 가우시안 개수
device = torch.device("cuda")

# 가우시안 파라미터들: 학습 가능하도록 requires_grad=True 설정
# 2021년의 당신이 최적화하던 State Vector [x, y, z, q, s, a, c] 입니다.
means = torch.randn((N, 3), device=device, requires_grad=True)  # 위치 (mu)
scales = torch.randn((N, 3), device=device, requires_grad=True) # 크기 (S)
quats = torch.randn((N, 4), device=device, requires_grad=True)  # 회전 (R)
opacities_raw = torch.randn(N, device=device, requires_grad=True) # 투명도 (alpha) — sigmoid 전 raw 값
colors = torch.rand((N, 3), device=device, requires_grad=True) # 색상 (C)

# 2. 카메라 파라미터 설정 (내부 파라미터 K 및 포즈 T)
H, W  = 512, 512

K = torch.tensor([[[H, 0, H/2], [0, W, W/2], [0, 0, 1]]], device=device) # [C, 3, 3] 카메라 내부 파라미터
viewmat = torch.eye(4, device=device).unsqueeze(0) # [C, 4, 4] 카메라 포즈

# 3. 최적화 도구 (GTSAM의 Optimizer 대신 Adam 사용)
optimizer = torch.optim.Adam([means, scales, quats, opacities_raw, colors], lr=1e-2)

# 4. Target 설정 (우리가 도달하고 싶은 '실제 사진')
# 여기서는 간단히 중앙에 붉은 원이 있다고 가정하거나 실제 이미지를 로드합니다.
target = torch.zeros((H, W, 3), device=device)
target[H//4:3*H//4, W//4:3*W//4, 0] = 1.0 # 중앙에 빨간 사각형

# --- 최적화 루프 시작 ---
for step in range(1000):
    optimizer.zero_grad() # 옵티마이저 초기화

    # [Forward Pass]
    # v1.x: rasterization()이 projection + rasterization을 한번에 수행
    # 내부에서 3D→2D 투영(Jacobian J 계산)과 타일 기반 Alpha Blending이 모두 처리됨
    rendered, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities_raw),
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        near_plane=0.01,
        backgrounds=torch.zeros((3,), device=device),
    )
    out_img = rendered.squeeze(0)  # [C, H, W, 3] → [H, W, 3]

    # [Loss Calculation]
    loss = F.mse_loss(out_img, target)

    # [Backward Pass]
    loss.backward() # 이 한 줄이 모든 가우시안에 대한 편미분 값을 계산합니다.

    optimizer.step()
