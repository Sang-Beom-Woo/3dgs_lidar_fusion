"""3DGS 학습 loss 함수들.

구성:
  * PhotometricLoss    — L1 + SSIM + (optional) LPIPS.  메인 감독 신호.
  * depth_loss         — LiDAR 투영 sparse depth 와의 MSE.  기하 구속.
  * opacity_hinge_loss — logit-space α floor (ADC 있을 때만 유용).
  * opacity_bimodal_loss — α(1-α) 중간값 penalty.
  * anisotropy_loss    — max/min scale 비율 제한 (needle 방어).

모두 마스크 없는 기본형.  다이나믹 마스킹을 추가하려면 각 함수에 valid_mask
를 받도록 확장하면 됨.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from math_utils import ssim

# LPIPS 는 선택적.  패키지 없으면 None 반환.
try:
    import lpips as _lpips_module
except ImportError:
    _lpips_module = None


# =============================================================================
# LPIPS 모델 생성 helper
# =============================================================================

def build_lpips(device: torch.device | str, net: str = "vgg") -> "torch.nn.Module | None":
    """LPIPS 모듈 준비.  패키지 없으면 None 반환 (soft-fail)."""
    if _lpips_module is None:
        print("[warn] lpips 패키지 없음 → LPIPS loss 비활성")
        return None
    model = _lpips_module.LPIPS(net=net, verbose=False).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# =============================================================================
# Photometric loss:  (1-λ_ssim)·L1 + λ_ssim·(1-SSIM) + λ_lpips·LPIPS
# =============================================================================

def photometric_loss(
    rendered_rgb: torch.Tensor,
    gt_image: torch.Tensor,
    ssim_lambda: float = 0.2,
    lpips_net: "torch.nn.Module | None" = None,
    lpips_lambda: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """세 항을 합친 photometric loss 반환.

    Args:
        rendered_rgb  [H, W, 3] gsplat 출력.
        gt_image      [H, W, 3] GT (undistorted, 0~1).
        ssim_lambda   SSIM 가중치 (L1 가중치 = 1 - λ).
        lpips_net     build_lpips() 결과 또는 None.
        lpips_lambda  LPIPS 가중치.  0 이거나 net None 이면 미적용.

    Returns:
        total_loss (0차원 Tensor),
        components dict {'l1', 'ssim_term', 'lpips'}
    """
    l1 = F.l1_loss(rendered_rgb, gt_image)
    ssim_term = 1.0 - ssim(rendered_rgb, gt_image)
    photo = (1.0 - ssim_lambda) * l1 + ssim_lambda * ssim_term

    if lpips_net is not None and lpips_lambda > 0.0:
        # LPIPS 는 [N, 3, H, W] in [-1, 1]
        x = rendered_rgb.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0) * 2.0 - 1.0
        y = gt_image.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0) * 2.0 - 1.0
        lpips_val = lpips_net(x, y).mean()
    else:
        lpips_val = torch.zeros((), device=rendered_rgb.device)

    total = photo + lpips_lambda * lpips_val
    return total, {"l1": l1, "ssim_term": ssim_term, "lpips": lpips_val}


# =============================================================================
# Depth loss — LiDAR 가 본 픽셀만
# =============================================================================

def depth_loss(rendered_depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
    """MSE on pixels where gt_depth > 0 (LiDAR valid).

    gt_depth == 0 은 LiDAR 가 그 픽셀에 투영되지 않았다는 뜻.  학습 제외.
    """
    mask = gt_depth > 0
    if mask.sum() == 0:
        return torch.zeros((), device=rendered_depth.device)
    return F.mse_loss(rendered_depth[mask], gt_depth[mask])


# =============================================================================
# Regularizer 들 — 기본값 OFF (λ=0).  ADC 시나리오에서 선택적으로 사용.
# =============================================================================

def opacity_hinge_loss(opacities_raw: torch.Tensor, target_alpha: float = 0.05) -> torch.Tensor:
    """logit 공간 ReLU hinge.  α < target 일 때만 선형 push up.

    α=0.005 같은 바닥값에서도 gradient 가 죽지 않아 "깔려 붙는" 가우시안 방어.
    """
    target_logit = float(np.log(target_alpha / (1.0 - target_alpha)))
    return F.relu(target_logit - opacities_raw).mean()


def opacity_bimodal_loss(opacities_raw: torch.Tensor) -> torch.Tensor:
    """α(1-α) 중간값 penalty.  값을 0 또는 1 양극단으로 유도."""
    a = torch.sigmoid(opacities_raw)
    return (a * (1.0 - a)).mean()


def anisotropy_loss(scales_log: torch.Tensor, max_ratio: float = 5.0) -> torch.Tensor:
    """max/min scale 비율이 max_ratio 초과할 때만 선형 penalty.

    log 공간에서 (max - min) = log(ratio) 이므로 간단히 뺄셈 + ReLU.
    needle/spike artifact 방어용.
    """
    log_ratio = scales_log.max(dim=1).values - scales_log.min(dim=1).values
    log_thr = float(np.log(max_ratio))
    return F.relu(log_ratio - log_thr).mean()
