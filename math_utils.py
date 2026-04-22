"""3DGS 파이프라인의 순수 수학 유틸.

두 그룹으로 묶여 있음:
  1. SSIM — photometric loss 에 쓰이는 구조 유사성 지수 (Wang et al. 2004)
  2. 회전 / SE(3) — 쿼터니언·축각 → 행렬, 카메라 pose 미세 보정용

모두 stateless, 외부 의존 (torch 만).  단위 테스트 쉬움.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# SSIM
#
# 참조: Wang et al., "Image Quality Assessment: From Error Visibility to
#       Structural Similarity", IEEE TIP 2004.
#
# 수식 (11×11 가우시안 윈도우 N(·) 기반 국소 통계):
#
#     μ₁ = N(x),       μ₂ = N(y)
#     σ₁² = N(x²) - μ₁²,    σ₂² = N(y²) - μ₂²
#     σ₁₂ = N(x·y) - μ₁μ₂
#
#     SSIM(x,y) = (2 μ₁μ₂ + C₁)(2 σ₁₂ + C₂)
#                 ─────────────────────────
#                 (μ₁² + μ₂² + C₁)(σ₁² + σ₂² + C₂)
#
# C1=(K1·L)², C2=(K2·L)², L=1, K1=0.01, K2=0.03.
# 반환: 스칼라 평균 SSIM ∈ [-1, 1], 1 이면 완전 동일.
# =============================================================================

def gaussian_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """1D 정규화 가우시안 커널.  SSIM 2D 윈도우의 기본 블록 (외적으로 합성)."""
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-coords.pow(2) / (2.0 * sigma * sigma))
    return g / g.sum()


def _ssim_window(size: int, sigma: float, channels: int,
                 device: torch.device) -> torch.Tensor:
    """[C, 1, size, size] 2D 가우시안 윈도우 (depthwise conv 용 레이아웃)."""
    g1 = gaussian_1d(size, sigma, device)
    w2 = g1.view(-1, 1) * g1.view(1, -1)
    return w2.expand(channels, 1, size, size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor,
         window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """평균 SSIM.  입력 [H, W, C] float in [0, 1]."""
    # gsplat 은 [H, W, C] 규약, conv2d 는 [N, C, H, W]
    x = img1.permute(2, 0, 1).unsqueeze(0)
    y = img2.permute(2, 0, 1).unsqueeze(0)
    C = x.shape[1]

    window = _ssim_window(window_size, sigma, C, x.device)
    pad = window_size // 2

    mu1 = F.conv2d(x, window, padding=pad, groups=C)
    mu2 = F.conv2d(y, window, padding=pad, groups=C)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    sigma1_sq = F.conv2d(x * x, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=pad, groups=C) - mu2_sq
    sigma12   = F.conv2d(x * y, window, padding=pad, groups=C) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    num = (2.0 * mu12 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (num / den).mean()


# =============================================================================
# 회전 / SE(3)
#
# 두 가지 규약을 지원:
#   - quat (w, x, y, z)          : 가우시안 자체의 rotation 파라미터
#   - axis-angle / 6DoF delta    : 카메라·키프레임 pose 미세 보정
#
# 모두 torch Tensor 입력 → torch Tensor 출력, autograd 친화.
# =============================================================================

def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """쿼터니언 (w, x, y, z) → 3×3 회전 행렬.  입력 [*, 4], 출력 [*, 3, 3].

    Hamilton 규약, 입력 정규화 후 Rodrigues 공식을 Hamilton 형태로 전개.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x),
            2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def axis_angle_to_rot(omega: torch.Tensor) -> torch.Tensor:
    """축-각 ω ∈ ℝ³ → 3×3 회전 행렬 (Rodrigues 공식).

    θ = |ω|, u = ω/θ.  R = I + sin(θ)·[u]× + (1 - cos(θ))·[u]×²
    θ 작을 때 수치 안정 위해 분모 clamp.
    """
    theta = omega.norm().clamp(min=1e-10)
    axis = omega / theta
    K = torch.zeros(3, 3, device=omega.device, dtype=omega.dtype)
    K[0, 1] = -axis[2]; K[0, 2] =  axis[1]
    K[1, 0] =  axis[2]; K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]; K[2, 1] =  axis[0]
    I3 = torch.eye(3, device=omega.device, dtype=omega.dtype)
    return I3 + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def delta_to_se3(delta: torch.Tensor) -> torch.Tensor:
    """6D tangent (rx, ry, rz, tx, ty, tz) → 4×4 SE(3) 교정 행렬.

    카메라 extrinsic / 키프레임 pose 의 미세 조정용.
    R 은 Rodrigues 로, 병진은 단순 add (small-delta 에선 충분히 정확).
    """
    T = torch.eye(4, device=delta.device, dtype=delta.dtype)
    T[:3, :3] = axis_angle_to_rot(delta[:3])
    T[:3,  3] = delta[3:]
    return T


# =============================================================================
# numpy 버전 (seed 변환, 저장 시 사용)
# =============================================================================

def quat_xyzw_to_rot_np(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """ROS 규약 쿼터니언 (x, y, z, w) → 3×3 회전 행렬. numpy.

    keyframe_pose.txt 가 이 순서.
    """
    x, y, z, w = qx, qy, qz, qw
    n = x * x + y * y + z * z + w * w
    s = 0.0 if n == 0.0 else 2.0 / n
    xx, yy, zz = s * x * x, s * y * y, s * z * z
    xy, xz, yz = s * x * y, s * x * z, s * y * z
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ], dtype=np.float64)
