"""LiDAR-Visual 하이브리드 3D Gaussian Splatting — 메인 모델.

================================================================================
 3DGS 한 줄 요약
================================================================================

씬을 수백만 개의 3D 가우시안 (μ, Σ, α, c) 집합으로 표현하고, 이를 미분 가능한
래스터화로 투영·블렌딩 → 이미지 생성.  loss 에서 역전파해 모든 파라미터를 Adam
으로 직접 학습.  원 논문: Kerbl et al., SIGGRAPH 2023.

================================================================================
 우리 변형 (LidarVisualGS) 의 특징
================================================================================

  * **LiDAR seed**          — 가우시안 μ/c 를 LiDAR 점 + 이미지 투영으로 초기화.
  * **Depth supervision**   — 렌더 expected depth vs LiDAR depth MSE.  기하 제약.
  * **Per-param Adam LR**   — 원 3DGS 규약 (opacity 5e-2, means 1.6e-4 등).
  * **LPIPS perceptual**    — VGG 특징 기반 추가 loss.  샤프함 대폭 ↑.
  * **cam_refine**          — 3 카메라 × 6DoF extrinsic delta 학습 (calibration drift 자동 보정).
  * **kf_refine**           — per-keyframe SE(3) delta (BA-lite).  SLAM 궤적 drift 보정.
  * **SH 0/1/2/3 지원**      — view-dependent 색 표현.
  * **ADC 기본 OFF**         — LiDAR 시드가 이미 조밀해서 densification 오히려 표면 깎음.
                              필요하면 `--densify-from-iter 500` 등으로 활성.
  * **Floater prune**       — 학습 종료 후 α 낮은 + outlier 가우시안 일괄 정리.

================================================================================
 파라미터 규약 (수치적으로 민감한 부분)
================================================================================

  means          ℝ³        — world 좌표 (m)
  scales         ℝ³        — log-space.  실제 반경 = exp(scales)
  quats          ℝ⁴ (w,x,y,z)
  opacities_raw  ℝ         — logit.  실제 α = sigmoid(opacities_raw)
  sh_dc          [N,1,3]   — SH DC (= RGB 를 (rgb - 0.5)/SH_C0 로 저장)
  sh_rest        [N,K-1,3] — higher-order SH (K = (sh_degree+1)²)

log / logit 공간을 쓰는 이유는 Adam 이 음수까지 자유롭게 이동해도 실제 값이
물리적으로 유효 (scale>0, 0<α<1) 을 유지하게 하려는 표준 트릭.

================================================================================
 파일 구조 (리팩토링 후)
================================================================================

  chapter2.py        ← 이 파일.  LidarVisualGS 클래스 + 저장 메서드.
  math_utils.py      ← SSIM, 쿼터니언, SE(3) 변환.
  losses.py          ← photometric / depth / regularizer.
  base_data_parser.py ← 데이터셋 로더.
  run_chapter2.py    ← CLI 드라이버.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization

from math_utils import delta_to_se3
from losses import (
    anisotropy_loss,
    build_lpips,
    depth_loss as depth_loss_fn,
    opacity_bimodal_loss,
    opacity_hinge_loss,
    photometric_loss,
)


# =============================================================================
# 상수
# =============================================================================

# SH 0차 상수 Y₀⁰ = 1/(2√π) ≈ 0.2821.  RGB ↔ DC: f_dc = (rgb - 0.5) / SH_C0.
SH_C0 = 0.28209479177387814


# =============================================================================
# LidarVisualGS
# =============================================================================

class LidarVisualGS:
    """LiDAR-seeded 3D Gaussian Splatting 모델.

    옵티마이저 상태 포함 전체 학습 단위.  train_step 을 반복 호출하면서
    필요에 따라 reset_opacity / densify_and_prune / prune_floaters 를 외부에서
    스케줄링 (run_chapter2.py 가 담당).

    주요 구성 요소 (아래 섹션별):
      1. __init__             : 파라미터 초기화, optimizer 셋업, LPIPS / pose refine
      2. train_step           : 1 iter forward/backward/step
      3. ADC 지원 (선택)       : clone/split/prune + Adam state 수술
      4. Opacity reset (선택)  : 주기적 α 깎기
      5. Floater prune         : 학습 후 cleanup
      6. 저장                  : save_splat_ply (INRIA), save_ply (pointcloud)
    """

    # ADC 유틸이 일괄 처리하는 Parameter 이름.  순서는 param_groups 와 무관.
    _GAUSSIAN_ATTRS = ("means", "scales", "quats", "opacities_raw", "sh_dc", "sh_rest")

    # ---------------------------------------------------------------------
    # 1) 생성자 — 파라미터·옵티마이저·보조 모델
    # ---------------------------------------------------------------------
    def __init__(self,
                 lidar_points: torch.Tensor,
                 lidar_colors: torch.Tensor,
                 device: str = "cuda",
                 initial_scale: float = 0.05,
                 sh_degree: int = 1,
                 n_cameras: int = 3,
                 cam_refine: bool = False,
                 n_keyframes: int = 0,
                 kf_refine: bool = False):
        """
        Args:
            lidar_points  [N, 3]  world 좌표 XYZ.
            lidar_colors  [N, 3]  RGB 0~1 (내부에서 SH DC 로 변환).
            initial_scale         가우시안 초기 반경 (m).  voxel*0.5 권장.
            sh_degree             0=view-independent, 1=권장, 2~3=원 3DGS.
            n_cameras             cam_refine 시 delta 파라미터 수.
            cam_refine            True 면 카메라 extrinsic delta 학습.
            n_keyframes           kf_refine 시 delta 파라미터 수 (max kf.idx+1).
            kf_refine             True 면 per-kf pose delta 학습.
        """
        self.device = device
        self.N = lidar_points.shape[0]
        self.sh_degree = int(sh_degree)

        self._init_gaussian_params(lidar_points, lidar_colors, initial_scale)
        self._init_optimizer()
        self._init_loss_config(device)
        self._init_densify_buffers()
        self._init_pose_refine(n_cameras, cam_refine, n_keyframes, kf_refine)

    # --- __init__ 서브루틴 ----------------------------------------------

    def _init_gaussian_params(self, lidar_points: torch.Tensor,
                              lidar_colors: torch.Tensor,
                              initial_scale: float) -> None:
        """5 개 학습 Parameter 생성: means, scales, quats, opacities_raw, sh_dc, sh_rest."""
        dev = self.device
        K = (self.sh_degree + 1) ** 2

        self.means = torch.nn.Parameter(lidar_points.to(dev))

        # scale: log 공간.  exp(-3) ≈ 0.05m 부터 시작.
        log_s = float(np.log(max(initial_scale, 1e-6)))
        self.scales = torch.nn.Parameter(torch.full((self.N, 3), log_s, device=dev))

        # quat: identity (w=1).  학습 중 |q|=1 는 내부 정규화로 유지.
        self.quats = torch.nn.Parameter(
            torch.tensor([1., 0., 0., 0.], device=dev).repeat(self.N, 1))

        # opacity: logit(0.1) = α≈0.1 에서 시작 (원 3DGS 규약).
        self.opacities_raw = torch.nn.Parameter(
            torch.full((self.N,), float(np.log(0.1 / 0.9)), device=dev))

        # SH DC: RGB → (rgb-0.5)/SH_C0.  sh_rest 는 0 에서 시작.
        rgb = lidar_colors.to(dev)
        dc_init = ((rgb - 0.5) / SH_C0).unsqueeze(1).contiguous()   # [N, 1, 3]
        self.sh_dc = torch.nn.Parameter(dc_init)
        self.sh_rest = torch.nn.Parameter(torch.zeros(self.N, K - 1, 3, device=dev))

    def _init_optimizer(self) -> None:
        """Per-parameter LR Adam.  원 3DGS 표준 값."""
        # 우리 데이터에선 opacity LR 5e-2 가 fine-tune 단계에서 자유낙하 경향.
        # → 1e-2 로 보수적 조정.
        self.optimizer = torch.optim.Adam([
            {"params": [self.means],         "lr": 1.6e-4, "name": "means"},
            {"params": [self.scales],        "lr": 5.0e-3, "name": "scales"},
            {"params": [self.quats],         "lr": 1.0e-3, "name": "quats"},
            {"params": [self.opacities_raw], "lr": 1.0e-2, "name": "opacities"},
            {"params": [self.sh_dc],         "lr": 2.5e-3, "name": "sh_dc"},
            {"params": [self.sh_rest],       "lr": 1.25e-4, "name": "sh_rest"},
        ])

    def _init_loss_config(self, device: str) -> None:
        """Loss 가중치 + LPIPS 모듈."""
        # photometric:  (1-λ_ssim)·L1 + λ_ssim·(1-SSIM) + λ_lpips·LPIPS
        self.ssim_lambda  = 0.2
        self.lpips_lambda = 0.2
        self.lpips_net    = build_lpips(device, net="vgg")

        # LiDAR depth 제약
        self.depth_lambda = 0.1

        # 선택적 regularizer (ADC 병용 시 의미, 기본 OFF)
        self.opacity_reg_lambda     = 0.0
        self.opacity_reg_target     = float(np.log(0.05 / 0.95))
        self.opacity_bimodal_lambda = 0.0
        self.aniso_lambda           = 0.0
        self.aniso_max_ratio        = 5.0

    def _init_densify_buffers(self) -> None:
        """ADC (selective) 용 per-gaussian 통계 버퍼."""
        dev = self.device
        self.xyz_grad_accum = torch.zeros(self.N, device=dev)
        self.xyz_grad_count = torch.zeros(self.N, device=dev)
        self.max_radii2d    = torch.zeros(self.N, device=dev)

    def _init_pose_refine(self, n_cameras: int, cam_refine: bool,
                          n_keyframes: int, kf_refine: bool) -> None:
        """카메라 / 키프레임 pose 의 6DoF delta 학습 파라미터."""
        dev = self.device
        self.cam_refine = bool(cam_refine)
        self.kf_refine  = bool(kf_refine)
        self.n_cameras  = int(n_cameras)
        self.n_keyframes = int(n_keyframes)

        # cam_delta: 3 × 6.  calibration drift 보정.
        self.cam_delta = torch.nn.Parameter(torch.zeros(self.n_cameras, 6, device=dev))
        if self.cam_refine:
            self.optimizer.add_param_group(
                {"params": [self.cam_delta], "lr": 1e-4, "name": "cam_delta"})

        # kf_pose_delta: N_kf × 6.  SLAM 궤적 drift 보정.
        if self.n_keyframes > 0:
            self.kf_pose_delta = torch.nn.Parameter(
                torch.zeros(self.n_keyframes, 6, device=dev))
            if self.kf_refine:
                self.optimizer.add_param_group(
                    {"params": [self.kf_pose_delta], "lr": 5e-5, "name": "kf_pose_delta"})
        else:
            self.kf_pose_delta = None

    # ---------------------------------------------------------------------
    # 2) train_step — forward + loss + backward + step
    # ---------------------------------------------------------------------
    def train_step(self, gt_image: torch.Tensor, gt_depth: torch.Tensor,
                   K: torch.Tensor, viewmat: torch.Tensor,
                   cam_index: int | None = None,
                   kf_index: int | None = None,
                   track_densify: bool = True) -> float:
        """한 뷰에 대해 1 step 학습.  loss 의 Python float 반환."""
        self.optimizer.zero_grad()
        H, W = gt_image.shape[:2]

        viewmat = self._apply_pose_refine(viewmat, cam_index, kf_index)

        # gsplat 은 [N, C, H, W] 아닌 [N, H, W, C] 출력.  render_mode RGB+ED 로
        # 마지막 채널이 α-blended depth.
        sh_combined = torch.cat([self.sh_dc, self.sh_rest], dim=1)   # [N, K, 3]
        renders, _alphas, meta = rasterization(
            means=self.means,
            quats=self.quats,
            scales=torch.exp(self.scales),
            opacities=torch.sigmoid(self.opacities_raw),
            colors=sh_combined,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            packed=False,
            render_mode="RGB+ED",
            sh_degree=self.sh_degree,
        )
        if track_densify:
            # means2d 는 non-leaf 이므로 retain_grad() 없으면 backward 후 .grad 비어 있음.
            meta["means2d"].retain_grad()

        rendered_rgb   = renders[0, ..., :3]
        rendered_depth = renders[0, ..., 3]

        total_loss = self._compute_total_loss(rendered_rgb, rendered_depth,
                                              gt_image, gt_depth)
        total_loss.backward()

        if track_densify:
            self._accumulate_densify_stats(meta)

        self.optimizer.step()
        return total_loss.item()

    # --- train_step 서브루틴 --------------------------------------------

    def _apply_pose_refine(self, viewmat: torch.Tensor,
                           cam_index: int | None,
                           kf_index: int | None) -> torch.Tensor:
        """viewmat 에 cam_delta (cam 프레임) + kf_pose_delta (cam 프레임) 좌측 곱.

        양쪽 delta 는 0 에서 학습 시작.  부호/프레임은 optimizer 가 찾음.
        """
        if self.cam_refine and cam_index is not None:
            viewmat = delta_to_se3(self.cam_delta[cam_index]) @ viewmat
        if (self.kf_refine and kf_index is not None
                and self.kf_pose_delta is not None):
            viewmat = delta_to_se3(self.kf_pose_delta[kf_index]) @ viewmat
        return viewmat

    def _compute_total_loss(self, rendered_rgb: torch.Tensor,
                            rendered_depth: torch.Tensor,
                            gt_image: torch.Tensor,
                            gt_depth: torch.Tensor) -> torch.Tensor:
        """모든 loss 항을 합산.  기본 OFF 항은 0 반환 → 최적화 오버헤드 없음."""
        loss_photo, _ = photometric_loss(
            rendered_rgb, gt_image,
            ssim_lambda=self.ssim_lambda,
            lpips_net=self.lpips_net,
            lpips_lambda=self.lpips_lambda,
        )
        loss_d = depth_loss_fn(rendered_depth, gt_depth)

        zero = torch.zeros((), device=self.device)
        loss_hinge   = opacity_hinge_loss(self.opacities_raw, target_alpha=0.05) \
                       if self.opacity_reg_lambda > 0 else zero
        loss_bimodal = opacity_bimodal_loss(self.opacities_raw) \
                       if self.opacity_bimodal_lambda > 0 else zero
        loss_aniso   = anisotropy_loss(self.scales, max_ratio=self.aniso_max_ratio) \
                       if self.aniso_lambda > 0 else zero

        return (loss_photo
                + self.depth_lambda           * loss_d
                + self.opacity_reg_lambda     * loss_hinge
                + self.opacity_bimodal_lambda * loss_bimodal
                + self.aniso_lambda           * loss_aniso)

    # ---------------------------------------------------------------------
    # 3) ADC 지원 — Adaptive Density Control (선택적, 기본 OFF)
    #
    #    호출 체계:
    #      train_step() → _accumulate_densify_stats() 로 매 iter 통계 누적.
    #      외부 주기적 호출: gs.densify_and_prune(...) → clone/split/prune
    #      통계 리셋은 densify_and_prune 내부에서 자동.
    #
    #    핵심 어려움은 "nn.Parameter 교체 + Adam state (exp_avg) 동기화".
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _accumulate_densify_stats(self, meta: dict) -> None:
        """per-iter ADC 통계 누적.  means2d.grad L2 norm + screen-space max radii."""
        grad = meta["means2d"].grad
        if grad is None:
            return
        g = grad.squeeze(0).norm(dim=-1)
        if g.shape[0] != self.N:    # densify 와 iter 타이밍 race — 건너뜀
            return
        visible = g > 0
        self.xyz_grad_accum[visible] += g[visible]
        self.xyz_grad_count[visible] += 1.0

        radii = meta["radii"].squeeze(0)
        if radii.dim() == 2:
            radii = radii.max(dim=-1).values
        self.max_radii2d = torch.maximum(self.max_radii2d, radii.float())

    @torch.no_grad()
    def _reset_densify_stats(self) -> None:
        self.xyz_grad_accum.zero_()
        self.xyz_grad_count.zero_()
        self.max_radii2d.zero_()

    # --- ADC 파라미터 수술 유틸 (Adam state 포함) ------------------------

    @torch.no_grad()
    def _swap_param(self, attr: str, new_data: torch.Tensor,
                    new_exp_avg: torch.Tensor, new_exp_avg_sq: torch.Tensor) -> None:
        """nn.Parameter 교체 + Adam state 를 새 shape 에 맞춰 이동."""
        old_param = getattr(self, attr)
        new_param = torch.nn.Parameter(new_data)

        for group in self.optimizer.param_groups:
            for i, p in enumerate(group["params"]):
                if p is old_param:
                    group["params"][i] = new_param
                    break

        old_state = self.optimizer.state.pop(old_param, {})
        new_state = {"exp_avg": new_exp_avg, "exp_avg_sq": new_exp_avg_sq}
        if "step" in old_state:
            new_state["step"] = old_state["step"]
        self.optimizer.state[new_param] = new_state

        setattr(self, attr, new_param)

    @torch.no_grad()
    def _slice_all(self, keep_mask: torch.Tensor) -> None:
        """keep_mask 로 선택된 가우시안만 남김."""
        for attr in self._GAUSSIAN_ATTRS:
            p = getattr(self, attr)
            new_data = p.data[keep_mask]
            st = self.optimizer.state.get(p, {})
            new_ea = (st["exp_avg"][keep_mask]
                      if "exp_avg" in st else torch.zeros_like(new_data))
            new_easq = (st["exp_avg_sq"][keep_mask]
                        if "exp_avg_sq" in st else torch.zeros_like(new_data))
            self._swap_param(attr, new_data, new_ea, new_easq)

        self.xyz_grad_accum = self.xyz_grad_accum[keep_mask]
        self.xyz_grad_count = self.xyz_grad_count[keep_mask]
        self.max_radii2d    = self.max_radii2d[keep_mask]
        self.N = int(keep_mask.sum().item())

    @torch.no_grad()
    def _append_all(self, additions: dict) -> None:
        """각 파라미터 뒤에 새 가우시안 append.  Adam 모멘트는 0 패딩."""
        M = additions["means"].shape[0]
        for attr in self._GAUSSIAN_ATTRS:
            p = getattr(self, attr)
            add = additions[attr].to(p.device).to(p.dtype)
            new_data = torch.cat([p.data, add], dim=0)
            st = self.optimizer.state.get(p, {})
            zero_pad = torch.zeros_like(add)
            new_ea = (torch.cat([st["exp_avg"], zero_pad], dim=0)
                      if "exp_avg" in st else torch.zeros_like(new_data))
            new_easq = (torch.cat([st["exp_avg_sq"], zero_pad], dim=0)
                        if "exp_avg_sq" in st else torch.zeros_like(new_data))
            self._swap_param(attr, new_data, new_ea, new_easq)

        zero1 = torch.zeros(M, device=self.device)
        self.xyz_grad_accum = torch.cat([self.xyz_grad_accum, zero1], dim=0)
        self.xyz_grad_count = torch.cat([self.xyz_grad_count, zero1], dim=0)
        self.max_radii2d    = torch.cat([self.max_radii2d,    zero1], dim=0)
        self.N += M

    # --- Clone / Split / Prune -----------------------------------------

    @torch.no_grad()
    def _pad_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """mask 길이가 현재 self.N 보다 짧으면 False 로 패딩.

        _clone 이 self.N 을 늘린 뒤 옛 길이 mask 를 받는 _split 이 여기서 안전."""
        if mask.shape[0] == self.N:
            return mask
        if mask.shape[0] > self.N:
            raise ValueError(f"mask len {mask.shape[0]} > current N {self.N}")
        pad = torch.zeros(self.N - mask.shape[0], dtype=torch.bool, device=mask.device)
        return torch.cat([mask, pad], dim=0)

    @torch.no_grad()
    def _clone(self, mask: torch.Tensor) -> int:
        """mask=True 가우시안을 그대로 복제 (under-reconstructed 작은 것)."""
        mask = self._pad_mask(mask)
        if not mask.any():
            return 0
        additions = {a: getattr(self, a).data[mask].clone() for a in self._GAUSSIAN_ATTRS}
        self._append_all(additions)
        return int(mask.sum().item())

    @torch.no_grad()
    def _split(self, mask: torch.Tensor, n_samples: int = 2,
               scale_div: float = 1.6) -> int:
        """큰 가우시안을 n_samples 자식으로 분할.  원본은 호출자가 prune."""
        from math_utils import quat_to_rot   # local import: 여기서만 필요

        mask = self._pad_mask(mask)
        if not mask.any():
            return 0
        M = int(mask.sum().item())

        src_means      = self.means.data[mask]
        src_scales_log = self.scales.data[mask]
        src_scales_lin = torch.exp(src_scales_log)
        src_quats      = self.quats.data[mask]
        src_opa        = self.opacities_raw.data[mask]
        src_sh_dc      = self.sh_dc.data[mask]
        src_sh_rest    = self.sh_rest.data[mask]

        # 원 가우시안 분포에서 샘플링 후 월드로 회전 → 자식 μ 오프셋
        stds = src_scales_lin.unsqueeze(1).expand(-1, n_samples, -1)
        local = torch.randn_like(stds) * stds
        R = quat_to_rot(src_quats)
        world_off = torch.einsum("mij,mkj->mki", R, local)
        new_means = (src_means.unsqueeze(1) + world_off).reshape(-1, 3)

        shrink = float(np.log(scale_div))
        new_scales = (src_scales_log - shrink).unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, 3)
        new_quats  = src_quats.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, 4)
        new_opa    = src_opa.unsqueeze(1).expand(-1, n_samples).reshape(-1).clone()

        K_rest = src_sh_rest.shape[1]
        new_sh_dc = (src_sh_dc.unsqueeze(1).expand(-1, n_samples, -1, -1)
                     .reshape(-1, 1, 3).clone())
        if K_rest == 0:
            new_sh_rest = torch.zeros(M * n_samples, 0, 3,
                                      device=src_sh_rest.device, dtype=src_sh_rest.dtype)
        else:
            new_sh_rest = (src_sh_rest.unsqueeze(1).expand(-1, n_samples, -1, -1)
                           .reshape(-1, K_rest, 3).clone())

        self._append_all({
            "means":         new_means,
            "scales":        new_scales,
            "quats":         new_quats,
            "opacities_raw": new_opa,
            "sh_dc":         new_sh_dc,
            "sh_rest":       new_sh_rest,
        })
        return M * n_samples

    @torch.no_grad()
    def densify_and_prune(self,
                          max_grad: float = 2e-4,
                          min_opacity: float = 5e-3,
                          max_screen_size: float = 20.0,
                          max_world_scale: float = 0.1,
                          scene_extent: float = 10.0,
                          max_gaussians: int | None = 2_000_000) -> dict:
        """ADC 한 사이클.  외부에서 densify_interval iter 마다 호출."""
        avg_grad = self.xyz_grad_accum / self.xyz_grad_count.clamp(min=1.0)
        grad_big = avg_grad >= max_grad

        scale_world = torch.exp(self.scales.data).max(dim=1).values
        small = scale_world <= scene_extent * 0.01
        big   = scale_world >  scene_extent * 0.01
        clone_mask = grad_big & small
        split_mask = grad_big & big

        if max_gaussians is not None:
            self._cap_densify_budget(avg_grad, clone_mask, split_mask, max_gaussians)

        n_cloned = self._clone(clone_mask)
        split_origin_idx = torch.nonzero(split_mask, as_tuple=False).squeeze(-1)
        n_split = self._split(split_mask)

        # Prune: 최종 self.N 기준 재평가
        alpha = torch.sigmoid(self.opacities_raw.data)
        scale_world_now = torch.exp(self.scales.data).max(dim=1).values
        aspect_ratio = (scale_world_now /
                        torch.exp(self.scales.data).min(dim=1).values.clamp(min=1e-8))
        prune_mask = (
            (alpha < min_opacity)
            | (self.max_radii2d > max_screen_size)
            | (scale_world_now > max_world_scale)
            | (aspect_ratio > 8.0)               # needle 방어 (loose)
        )
        if split_origin_idx.numel() > 0:
            prune_mask[split_origin_idx] = True

        n_pruned = int(prune_mask.sum().item())
        keep = ~prune_mask
        if n_pruned > 0 and keep.sum() > 0:
            self._slice_all(keep)

        self._reset_densify_stats()
        return {"clone": n_cloned, "split": n_split, "prune": n_pruned, "N": self.N}

    @torch.no_grad()
    def _cap_densify_budget(self, avg_grad: torch.Tensor,
                            clone_mask: torch.Tensor, split_mask: torch.Tensor,
                            max_gaussians: int) -> None:
        """예상 증가량이 max_gaussians 초과하면 grad 큰 순서로 top-k 만 유지."""
        want = int(clone_mask.sum().item()) + int(split_mask.sum().item()) * 2
        budget = max_gaussians - self.N
        if want <= budget:
            return
        topk = max(int(max(budget, 0) * 0.6), 0)    # 60% 를 clone/split 에 배분
        both = clone_mask | split_mask
        if topk > 0:
            ranking = torch.where(both, avg_grad, torch.zeros_like(avg_grad))
            idx = torch.topk(ranking, k=min(topk, self.N)).indices
            keep_top = torch.zeros_like(both)
            keep_top[idx] = True
            clone_mask.logical_and_(keep_top)
            split_mask.logical_and_(keep_top)
        else:
            clone_mask.zero_()
            split_mask.zero_()

    # ---------------------------------------------------------------------
    # 4) Opacity reset — 주기적 α 깎기 (ADC 와 짝)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def reset_opacity(self, target_alpha: float = 0.01) -> None:
        """모든 α 를 target 이하로 깎음 (이미 낮은 건 유지).

        ADC 와 짝으로만 의미.  reset 만 단독으로 반복하면 맵이 점점 빈다.
        Adam 의 opacity 모멘트도 0 으로 초기화 (관성 잔재 제거).
        """
        target_logit = float(np.log(target_alpha / (1.0 - target_alpha)))
        new_logit = torch.full_like(self.opacities_raw, target_logit)
        self.opacities_raw.copy_(torch.minimum(self.opacities_raw, new_logit))

        state = self.optimizer.state.get(self.opacities_raw, None)
        if state is not None:
            if "exp_avg" in state:
                state["exp_avg"].zero_()
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].zero_()

    # ---------------------------------------------------------------------
    # 5) Floater prune — 학습 종료 후 1회 cleanup
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def prune_floaters(self, opacity_thr: float = 5e-3,
                       nb_neighbors: int = 20, std_ratio: float = 2.0) -> None:
        """낮은 α + 통계적 outlier 제거.

        단계:
          (1) α < opacity_thr 가우시안 삭제.
          (2) 남은 점 중 이웃 평균거리 + std·σ 초과 점을 outlier 로 제거.
              Open3D 의 remove_statistical_outlier 사용.
        표면 위 정상 가우시안은 이웃이 빽빽해 outlier 판정 안 됨 → 안전.
        """
        import open3d as o3d

        alpha = torch.sigmoid(self.opacities_raw)
        keep1 = alpha > opacity_thr
        n_before = self.N
        n_mid = int(keep1.sum().item())
        if n_mid == 0:
            print("[floater_prune] α 필터로 모두 제거 → skip")
            return

        means_np = self.means.detach()[keep1].cpu().numpy()
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(means_np.astype(np.float64))
        _, idx_in = pc.remove_statistical_outlier(nb_neighbors, std_ratio)

        inlier_mask = torch.zeros(n_mid, dtype=torch.bool, device=self.device)
        inlier_mask[torch.tensor(idx_in, device=self.device, dtype=torch.long)] = True
        keep_full = keep1.clone()
        keep_full[keep1] = inlier_mask

        self._slice_all(keep_full)
        n_low = n_before - n_mid
        n_out = n_mid - len(idx_in)
        print(f"[floater_prune] {n_before:,} → {self.N:,}  "
              f"(α<{opacity_thr}: -{n_low:,}, outlier: -{n_out:,})")

    # ---------------------------------------------------------------------
    # 6) 저장 — INRIA PLY + 간단 포인트클라우드
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def save_splat_ply(self, path: str, opacity_threshold: float = 0.0) -> int:
        """INRIA 3DGS 포맷 (splatviz/SuperSplat/antimatter15 호환).

        Per-gaussian 필드:
            x,y,z · nx,ny,nz (0) · f_dc_0..2 · f_rest_0..{3(K-1)-1} ·
            opacity (logit) · scale_0..2 (log) · rot_0..3 (w,x,y,z)

        shape/opacity 를 raw 공간에 저장 → 뷰어가 sigmoid/exp 재계산.
        """
        means, dc, rest, scales, opacity, quats = self._gather_save_arrays()

        # 쿼터니언 정규화
        qn = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / np.where(qn > 0, qn, 1.0)

        # SH rest 레이아웃:  INRIA 는 [N, 3, K-1] (channel-major) 로 flatten.
        #                   gsplat 내부는 [N, K-1, 3] → transpose 필요.
        f_dc = dc[:, 0, :]                               # [N, 3]
        if rest.shape[1] > 0:
            f_rest = np.transpose(rest, (0, 2, 1)).reshape(rest.shape[0], -1)
        else:
            f_rest = np.zeros((dc.shape[0], 0), dtype=np.float32)

        if opacity_threshold > 0.0:
            keep = (1.0 / (1.0 + np.exp(-opacity))) > opacity_threshold
            means = means[keep]; f_dc = f_dc[keep]; f_rest = f_rest[keep]
            opacity = opacity[keep]; scales = scales[keep]; quats = quats[keep]

        n = means.shape[0]
        normals = np.zeros_like(means)

        rest_props = "".join(f"property float f_rest_{i}\n"
                             for i in range(f_rest.shape[1]))
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
            f"{rest_props}"
            "property float opacity\n"
            "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
            "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
            "end_header\n"
        )
        buf = np.concatenate(
            [means, normals, f_dc, f_rest, opacity[:, None], scales, quats], axis=1
        ).astype(np.float32, copy=False)
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(buf.tobytes())

        print(f"[save_splat_ply] {n:,} gaussians (SH degree {self.sh_degree}) → {path}")
        return n

    @torch.no_grad()
    def save_ply(self, path: str, opacity_threshold: float = 0.0) -> int:
        """가우시안 중심 + DC 색만 저장하는 간이 컬러 포인트클라우드.

        MeshLab / CloudCompare / Open3D 등 일반 뷰어용.  scale/opacity/rot
        정보는 버려지므로 진짜 3DGS 렌더링은 불가능.
        """
        import open3d as o3d

        means = self.means.detach().cpu().numpy()
        dc = self.sh_dc.detach()[:, 0, :].cpu().numpy()
        colors = np.clip(0.5 + dc * SH_C0, 0.0, 1.0)

        if opacity_threshold > 0.0:
            alpha = torch.sigmoid(self.opacities_raw).detach().cpu().numpy()
            keep = alpha > opacity_threshold
            means, colors = means[keep], colors[keep]

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(means.astype(np.float64))
        pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        o3d.io.write_point_cloud(path, pc)
        print(f"[save_ply] {len(means):,} gaussians → {path}")
        return len(means)

    # --- 저장 헬퍼 -------------------------------------------------------

    def _gather_save_arrays(self):
        """학습 파라미터를 numpy 로 추출."""
        return (
            self.means.detach().cpu().numpy().astype(np.float32),
            self.sh_dc.detach().cpu().numpy().astype(np.float32),     # [N, 1, 3]
            self.sh_rest.detach().cpu().numpy().astype(np.float32),   # [N, K-1, 3]
            self.scales.detach().cpu().numpy().astype(np.float32),    # log
            self.opacities_raw.detach().cpu().numpy().astype(np.float32),  # logit
            self.quats.detach().cpu().numpy().astype(np.float32),
        )
