"""base_data 파서 — 로봇 키프레임 데이터셋을 3DGS 파이프라인으로 공급한다.

================================================================================
 데이터셋 구조 (base_data/)
================================================================================

  base_data/
    calibration/
      v4l2_intrinsic.txt   ──  3 카메라의 fisheye (equidistant) 내부 파라미터.
                               640×360, f≈337, 4-계수 equidistant 왜곡.
      sensor_tf.txt        ──  base_link 좌표계 기준의 4×4 변환 행렬:
                                 base_to_front_optical
                                 base_to_left_optical
                                 base_to_right_optical
                                 base_to_lidar_3d
                               (여기선 "base_to_X" = "X 프레임이 base 에서
                               어디에 있는지" 의 pose.  p_base = T · p_X.)

    <kf_id>/               ──  키프레임 폴더 (id = 0 .. 680).  id 는 정수지만
                                연속 안 될 수 있음.  이미지가 없는 kf 도 있다.

      point_cloud.pcd      ──  PCD v0.7 binary.  fields = x y z intensity (f32).
                               좌표계는 이미 **base_link 기준** 으로 저장됨
                               (LiDAR 프레임 아님 → T_base_lidar 추가 적용 불필요).
      front_color.bmp      ──  640×360 24-bit BMP.  카메라 없는 kf 도 있으므로
      left_color.bmp           require_image=True 필터로 걸러야 안전.
      right_color.bmp
      keyframe_pose.txt    ──  한 줄:
                                 <id> <frame> <ts_ns> tx ty tz qx qy qz qw
                               = T_world_base (base_link 의 world 상 pose).
      keyframe_type.txt    ──  <id> <char>.  키프레임 타입 (문자 코드, 예: 'X').
      outdoor.txt          ──  <id> 0|1.  야외 여부.
      elevator.txt         ──  <id> 0|1.  엘리베이터 내부 여부.
      descriptor.txt       ──  NetVLAD 류 place descriptor.  1 줄, 수백 float.

================================================================================
 이 모듈의 역할
================================================================================

chapter2.LidarVisualGS 가 기대하는 입력:
  * 모델 초기화 시 :   lidar_points [N,3] (world),  lidar_colors [N,3] (0~1)
  * 학습 iter 마다 :   gt_image [H,W,3] (0~1),  gt_depth [H,W] (0=invalid, meters),
                       K [3,3] (undistort 후),  viewmat [4,4] (world→cam)

여기서 제공하는 공개 API:
  * Calibration / CameraIntrinsics / KeyFrame  (dataclass)
  * load_calibration(root)
  * BaseDataDataset(root, require_image)
  * project_lidar_to_camera(points_base, cam, calib)
  * build_chapter2_inputs(kf, cam, calib)             ← 학습 sample 한 개
  * colorize_keyframe(kf, calib, drop_unseen)         ← seed 초기화용

================================================================================
 좌표계·변환 규약 (혼동 방지)
================================================================================

  base_link      로봇 몸체. LiDAR 포인트와 키프레임 pose 의 기준.
  world          SLAM 이 정의한 고정 좌표계. keyframe_pose.txt 의 T_world_base 가 이것.
  optical (cam)  OpenCV 카메라 규약: x=right, y=down, z=forward.
  T_A_B          "A frame for B's points" — p_A = T_A_B · p_B.

흔히 필요한 조합:
  T_world_cam   = T_world_base · T_base_cam
  T_cam_world   = (T_world_cam)⁻¹              ← gsplat 의 viewmat 가 요구
  p_cam         = T_cam_base · p_base
  p_world       = T_world_base · p_base

================================================================================
 Fisheye undistort 메모 (v4l2 카메라의 equidistant 모델)
================================================================================

원본 K 는 fisheye (equidistant: θ = atan(r)) 로 찍힌 사진에 맞춰진 intrinsic.
gsplat 과 우리 3DGS 파이프라인은 핀홀 카메라를 전제로 하므로, 학습에 쓰기 전에
이미지를 **언디스토션** 하고 언디스토션된 이미지에 맞는 **new_K** 를 써야 한다.

절차 (`Calibration.build_undistort_maps`):
  1) estimateNewCameraMatrixForUndistortRectify 로 undistort 후의 K (= new_K) 결정.
     balance=0 이면 모든 valid pixel 이 이미지 안에 들어오게 (= 검은 가장자리 최소).
  2) initUndistortRectifyMap 로 픽셀 매핑 (원본 → 언디스토션) 테이블 생성.
     CV_16SC2 포맷은 fixed-point 라 런타임 remap 이 빠르다.
  3) 런타임엔 cv2.remap 만 호출 (O(HW)).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# 글로벌 상수
# -----------------------------------------------------------------------------

# 이 데이터셋은 "front/left/right" 3 카메라 고정 구성.  이 상수는 iterate 순서를
# 결정하고, calibration 파일의 섹션 키와도 일치해야 한다.
CAM_NAMES = ("front", "left", "right")

# 카메라 출력이 거의 무채색 (R ≈ G ≈ B) 이라 학습된 가우시안 색도 회색으로 수렴
# 하는 문제가 있었다.  HSV saturation 채널에 곱해주는 부스트 팩터.
#   1.0 = 무보정         ← 야외 / 채도 충분한 데이터셋 (현재 기본)
#   1.7 = 채도 70% 증가  ← 실내, 카메라가 자연 desaturated 일 때 sweet spot
#   2.0+ = 부자연스러운 tint 발생
# 이미지 밝기(V)는 건드리지 않으므로 noise 증폭 없이 색감만 살아난다.
SAT_BOOST = 1.0


def _undistort_rgb(img_bgr_raw: np.ndarray, cam_intr: "CameraIntrinsics") -> np.ndarray:
    """원본 BGR 이미지 → (undistort) → (HSV saturation boost) → RGB float32 [0, 1].

    학습용 gt_image 와 seed 생성용 샘플링 이미지를 한 곳에서 만들어 **두 곳의
    색 공간이 반드시 일치** 하도록 하는 것이 목적.  한쪽만 부스트하면 loss 가
    원본 색 쪽으로 다시 끌고 가서 부스트 효과가 상쇄된다.

    처리 순서:
      1) cv2.remap 으로 fisheye → pinhole 언디스토션 (pre-computed map1/map2 사용)
      2) BGR → HSV 변환 후 S 채널에 SAT_BOOST 곱하고 255 clamp
      3) HSV → BGR → RGB → float32/255.
    """
    # 1) 언디스토션 (픽셀 매핑 테이블 이미 build_undistort_maps() 에서 준비됨)
    img_bgr = cv2.remap(img_bgr_raw, cam_intr.map1, cam_intr.map2, cv2.INTER_LINEAR)

    # 2) 채도 부스트 (SAT_BOOST == 1.0 이면 건너뜀 — cv2.cvtColor 왕복 비용 절약)
    if SAT_BOOST != 1.0:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        # H(0~179) · S(0~255) · V(0~255).  S 만 건드림 — V 는 노출 그대로 유지.
        hsv[..., 1] = np.clip(hsv[..., 1] * SAT_BOOST, 0, 255)
        img_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3) BGR → RGB (OpenCV → PyTorch 컨벤션) + uint8 → float32 [0, 1]
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# =============================================================================
# 1. Calibration (카메라 내부 + 센서 간 외부)
# =============================================================================
# -----------------------------------------------------------------------------
# 지원하는 distortion model 종류
# -----------------------------------------------------------------------------
# DistortionModel:
#   "equidistant"          → Kannala-Brandt fisheye (k1, k2, k3, k4).
#                            OpenCV cv2.fisheye.* 서브모듈 사용.
#   "plumb_bob"            → ROS / OpenCV 표준 핀홀 + 방사/접선 왜곡
#                            (k1, k2, p1, p2, k3).  cv2.undistort 계열.
#   "rational_polynomial"  → plumb_bob 의 확장형 (k1, k2, p1, p2, k3, k4, k5, k6).
#                            cv2.undistort 가 처리 가능.
#   "none" / "pinhole" / "" / 미지정 → 왜곡 없음 (이미 핀홀).  remap 패스스루.
#
# 이외 (e.g. "thin_prism", "omnidirectional") 는 ValueError 로 알림.
SUPPORTED_DISTORTION_MODELS = (
    "equidistant", "plumb_bob", "rational_polynomial", "none", "pinhole", "",
)
# "왜곡 없음" 으로 처리할 alias 집합 (D 길이 검사 스킵 + identity remap).
PINHOLE_MODEL_ALIASES = ("none", "pinhole", "")


@dataclass
class CameraIntrinsics:
    """한 카메라의 내부 파라미터 + 언디스토션 캐시.

    distortion_model 종류
      equidistant (= Kannala-Brandt fisheye): r = f · θ.  D = 4 계수 (k1..k4).
      plumb_bob (= ROS pinhole + radial-tangential): D = 5 계수 (k1, k2, p1, p2, k3).
      rational_polynomial: D = 8 계수 (k1..k6, p1, p2).
      none / "": 왜곡 없음. D 무시.
    """
    name: str                           # 'front' | 'left' | 'right'
    width: int                          # 출력 (언디스토션 후) width — downscale 반영
    height: int                         # 출력 height — downscale 반영
    K: np.ndarray                       # [3, 3]  원본 intrinsic (원본 해상도 기준)
    D: np.ndarray                       # [N]     왜곡 계수.  N = 모델별 다름
    distortion_model: str = "equidistant"   # SUPPORTED_DISTORTION_MODELS 중 하나
    raw_width: int = 0                  # 디스크 상 원본 BMP 폭 (downscale 적용 전)
    raw_height: int = 0                 # 디스크 상 원본 BMP 높이
    downscale: int = 1                  # 출력 해상도 = raw / downscale
    # 언디스토션 후 값들 — build_undistort_maps() 이 lazy 하게 채운다:
    new_K: Optional[np.ndarray] = None  # [3, 3]  핀홀 K (출력 해상도용, downscale 반영)
    map1: Optional[np.ndarray] = None   # remap 테이블 (CV_16SC2) — 출력 사이즈
    map2: Optional[np.ndarray] = None


@dataclass
class Calibration:
    """데이터셋 전체의 캘리브레이션 묶음.

    intrinsics  :  {cam_name → CameraIntrinsics}.  3 카메라 동일 K (이 데이터셋)
    T_base_cam  :  {cam_name → [4,4]}.  p_base = T · p_cam.
    T_base_lidar:  [4, 4].  PCD 가 이미 base 프레임이면 실제 투영엔 미사용.
    """
    intrinsics: Dict[str, CameraIntrinsics]
    T_base_cam: Dict[str, np.ndarray]
    T_base_lidar: np.ndarray

    def build_undistort_maps(self, balance: float = 0.0) -> None:
        """각 카메라에 대해 언디스토션 remap 테이블을 1 회만 계산해 캐시.

        distortion_model 별로 OpenCV 함수가 다름:
          equidistant         → cv2.fisheye.*
          plumb_bob / rational_polynomial → cv2.* (표준 핀홀+왜곡)
          none / ""           → 왜곡 없음, identity remap

        balance ∈ [0, 1]:
            0  →  new_K 를 "모든 픽셀이 이미지 안에 들어오도록" (zoom in,
                  검은 가장자리 최소) 으로 결정.
            1  →  원본 영역 최대한 보존 (검은 가장자리 허용).
        """
        for cam in self.intrinsics.values():
            if cam.new_K is not None:
                continue                      # 이미 계산됨 — 반복 호출 가드
            model = cam.distortion_model.lower()
            if model not in SUPPORTED_DISTORTION_MODELS:
                raise ValueError(
                    f"[calib] {cam.name}: distortion_model={cam.distortion_model!r} "
                    f"미지원. 지원: {SUPPORTED_DISTORTION_MODELS}")
            raw_size = (cam.raw_width, cam.raw_height)
            new_K = self._compute_new_K(cam, raw_size, balance, model)
            # downscale 적용 — fx, fy, cx, cy 모두 1/s 배.  출력 (w/s, h/s).
            s = cam.downscale
            if s != 1:
                new_K = new_K.copy()
                new_K[0, 0] /= s; new_K[1, 1] /= s
                new_K[0, 2] /= s; new_K[1, 2] /= s
            out_size = (cam.width, cam.height)
            m1, m2 = self._compute_remap(cam, new_K, out_size, model)
            cam.new_K, cam.map1, cam.map2 = new_K, m1, m2

    # --- distortion_model 별 헬퍼 -----------------------------------------
    @staticmethod
    def _compute_new_K(cam: "CameraIntrinsics", raw_size: tuple[int, int],
                       balance: float, model: str) -> np.ndarray:
        """모델별 new_K 계산."""
        if model == "equidistant":
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                cam.K, cam.D, raw_size, np.eye(3), balance=balance
            )
            # 폴백: D ≈ 0 일 때 OpenCV fisheye 가 fx ≈ 0 으로 깨짐.
            # 원본 K 를 그대로 사용 (거의 핀홀이라 손실 미미).
            if (new_K[0, 0] < cam.K[0, 0] * 0.1
                    or new_K[1, 1] < cam.K[1, 1] * 0.1):
                print(f"[calib] {cam.name}: fisheye estimate 깨짐 "
                      f"(fx={new_K[0,0]:.4f}). 원본 K 로 폴백.")
                new_K = cam.K.copy()
            return new_K
        if model in ("plumb_bob", "rational_polynomial"):
            # cv2.getOptimalNewCameraMatrix: alpha=balance.
            #   0 → 모든 픽셀 유효 (zoom in, 가장자리 잘림)
            #   1 → 원본 영역 보존 (검은 가장자리)
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                cam.K, cam.D, raw_size, alpha=balance, newImgSize=raw_size
            )
            return new_K
        # "none" / "pinhole" / "" → 왜곡 없음, K 그대로 핀홀.
        return cam.K.copy()

    @staticmethod
    def _compute_remap(cam: "CameraIntrinsics", new_K: np.ndarray,
                       out_size: tuple[int, int], model: str
                       ) -> tuple[np.ndarray, np.ndarray]:
        """모델별 remap 테이블 계산.

        CV_16SC2: 16-bit 정수 2채널 fixed-point.  메모리/속도 우위, 정확도
        손실 무시 가능.
        """
        if model == "equidistant":
            return cv2.fisheye.initUndistortRectifyMap(
                cam.K, cam.D, np.eye(3), new_K, out_size, cv2.CV_16SC2
            )
        if model in ("plumb_bob", "rational_polynomial"):
            return cv2.initUndistortRectifyMap(
                cam.K, cam.D, np.eye(3), new_K, out_size, cv2.CV_16SC2
            )
        # "none" / "pinhole" / "" → identity remap.  D=0 vector 전달 =
        # 변환 없음 (단 K 와 new_K 차이 만큼 픽셀만 이동).
        return cv2.initUndistortRectifyMap(
            cam.K, np.zeros(5, dtype=np.float64),
            np.eye(3), new_K, out_size, cv2.CV_16SC2
        )


# -----------------------------------------------------------------------------
# calibration 파일 파서 (ad-hoc 텍스트 포맷, 정규식으로 블록 분해)
# -----------------------------------------------------------------------------
def _parse_intrinsics(path: Path) -> Dict[str, CameraIntrinsics]:
    """v4l2_intrinsic.txt 파싱.

    파일 형식:
        [<name>_camera_intrinsics]
        image_width: 640
        image_height: 360
        camera_matrix:
        fx  0 cx
         0 fy cy
         0  0  1
        distortion_model: equidistant
        distortion_coefficients: k1 k2 k3 k4
        -------------------------------------------
        ... (다른 카메라) ...
    """
    text = path.read_text()
    # '---' 라인으로 블록 구분 (MULTILINE 필요)
    blocks = re.split(r"^-+\s*$", text, flags=re.MULTILINE)
    out: Dict[str, CameraIntrinsics] = {}
    for block in blocks:
        m = re.search(r"\[(\w+)_camera_intrinsics\]", block)
        if not m:
            continue
        name = m.group(1)
        w = int(re.search(r"image_width:\s*(\d+)", block).group(1))
        h = int(re.search(r"image_height:\s*(\d+)", block).group(1))
        # camera_matrix: 블록 다음 3 줄의 숫자를 한 스트링으로 합쳐 파싱
        km = re.search(r"camera_matrix:\s*\n([\s\S]+?)\ndistortion_model", block)
        K = np.fromstring(km.group(1).replace("\n", " "), sep=" ").reshape(3, 3)
        # distortion_model: "equidistant" / "plumb_bob" / "rational_polynomial" / "none" / 미지정.
        # 미지정 시 "equidistant" 가정 — 이전 동작 호환 (이 데이터셋 모두 fisheye).
        mm = re.search(r"distortion_model:\s*([A-Za-z_]+)", block)
        model = mm.group(1).strip().lower() if mm else "equidistant"
        if model not in SUPPORTED_DISTORTION_MODELS:
            raise ValueError(
                f"[calib] {name}: 미지원 distortion_model={model!r}.  "
                f"지원 모델: {SUPPORTED_DISTORTION_MODELS}")
        dm = re.search(r"distortion_coefficients:\s*([-\deE.\s]+)", block)
        D_str = dm.group(1).strip() if dm else ""
        D = (np.fromstring(D_str, sep=" ").astype(np.float64)
             if D_str else np.zeros(0, dtype=np.float64))
        # 모델별 D 길이 sanity check — 너무 짧으면 0 패딩, 길면 경고.
        expected_D_len_map = {
            "equidistant": 4,
            "plumb_bob": 5,
            "rational_polynomial": 8,
        }
        # 핀홀류는 D 길이 무관 — 어차피 사용 안 함.
        expected_D_len = (0 if model in PINHOLE_MODEL_ALIASES
                          else expected_D_len_map[model])
        if expected_D_len > 0 and len(D) < expected_D_len:
            # ROS 류 calib 가 trailing zero 생략하는 경우 처리 (plumb_bob 의
            # k3=0 을 빼고 4개만 쓰는 등).  0 패딩으로 보강.
            print(f"[calib] {name}: model={model} 인데 D 길이 {len(D)} "
                  f"(기대 {expected_D_len}). 0 패딩.")
            D = np.concatenate([D, np.zeros(expected_D_len - len(D))])
        elif expected_D_len > 0 and len(D) > expected_D_len:
            print(f"[calib] {name}: model={model} 인데 D 길이 {len(D)} "
                  f"(기대 {expected_D_len}). 앞 {expected_D_len}개만 사용.")
            D = D[:expected_D_len]
        # raw_width/height 는 원본 BMP 사이즈, width/height 는 출력 사이즈.
        # downscale 미적용 (=1) 이면 둘이 같음.  apply_downscale() 호출 시 변경됨.
        out[name] = CameraIntrinsics(
            name=name, width=w, height=h, K=K, D=D,
            distortion_model=model,
            raw_width=w, raw_height=h, downscale=1,
        )
    return out


def _parse_sensor_tf(path: Path):
    """sensor_tf.txt → (T_base_cam 딕셔너리, T_base_lidar 행렬).

    파일 형식:
        [base_to_front_optical]
        r00 r01 r02 tx
        r10 r11 r12 ty
        r20 r21 r22 tz
        0   0   0   1
        -------------------------------------------
        [base_to_left_optical]
        ...
    """
    text = path.read_text()
    blocks = re.split(r"^-+\s*$", text, flags=re.MULTILINE)
    mats: Dict[str, np.ndarray] = {}
    for block in blocks:
        m = re.search(r"\[base_to_(\w+)\]", block)
        if not m:
            continue
        key = m.group(1)            # 'front_optical' | 'left_optical' | 'right_optical' | 'lidar_3d'
        # 숫자 행만 추출 (헤더 제외)
        rows = [
            list(map(float, ln.split()))
            for ln in block.splitlines()
            if ln.strip() and not ln.strip().startswith("[")
        ]
        mats[key] = np.asarray(rows, dtype=np.float64).reshape(4, 4)
    T_base_cam = {
        "front": mats["front_optical"],
        "left":  mats["left_optical"],
        "right": mats["right_optical"],
    }
    return T_base_cam, mats["lidar_3d"]


def load_calibration(root: Path | str,
                     downscale: Optional[Dict[str, int]] = None) -> Calibration:
    """calibration/ 하위 두 파일을 읽어 Calibration 객체 반환.

    downscale: {cam_name → int factor}.  주어진 카메라만 출력 해상도를 1/s 배
        축소.  4032×3036 같은 거대 카메라용 (RTX 3060 6GB OOM 방지).
        예: {"left": 4, "right": 4} → left/right 만 1008×759 로 출력, front
        는 그대로.  s=1 인 카메라는 변경 없음.
    """
    root = Path(root)
    intr = _parse_intrinsics(root / "calibration" / "v4l2_intrinsic.txt")
    if downscale:
        for name, s in downscale.items():
            if name not in intr:
                continue
            if s < 1:
                raise ValueError(f"downscale factor must be ≥ 1, got {s} for {name}")
            cam = intr[name]
            # 출력 사이즈만 갱신. K/D 는 그대로, build_undistort_maps 가
            # new_K 계산 단계에서 1/s 스케일 적용한다.
            cam.downscale = int(s)
            cam.width = cam.raw_width // int(s)
            cam.height = cam.raw_height // int(s)
    T_base_cam, T_base_lidar = _parse_sensor_tf(root / "calibration" / "sensor_tf.txt")
    return Calibration(intrinsics=intr, T_base_cam=T_base_cam, T_base_lidar=T_base_lidar)


# =============================================================================
# 2. 키프레임 (KeyFrame) + 파싱 유틸
# =============================================================================
@dataclass
class KeyFrame:
    """한 키프레임 = LiDAR 스캔 + 3 카메라 이미지 + pose + 메타데이터."""
    idx: int                                                  # 폴더 번호
    ts_ns: int                                                # 타임스탬프 (nanosec)
    T_world_base: np.ndarray                                  # [4, 4]
    points_base: np.ndarray                                   # [N, 3] base_link 기준
    intensities: np.ndarray                                   # [N]    LiDAR reflectance
    images: Dict[str, np.ndarray] = field(default_factory=dict)  # {'front': BGR uint8, ...}
    flags: Dict[str, int] = field(default_factory=dict)          # {'outdoor', 'elevator', 'type'}
    descriptor: Optional[np.ndarray] = None                      # NetVLAD 류 place descriptor

    def points_in(self, frame: str, calib: Calibration) -> np.ndarray:
        """LiDAR 포인트를 요청한 좌표계로 변환.

        frame = 'base'  : 원본 그대로 (PCD 가 base 기준으로 저장됨)
        frame = 'world' : T_world_base 적용
        """
        if frame == "base":
            return self.points_base
        if frame == "world":
            # 동차좌표로 올린 뒤 4×4 행렬 × Nx4 → Nx3
            return (self.T_world_base @ _h(self.points_base).T).T[:, :3]
        raise ValueError(f"unknown frame: {frame}")


# -----------------------------------------------------------------------------
# 저수준 파싱 helpers
# -----------------------------------------------------------------------------
def _h(p: np.ndarray) -> np.ndarray:
    """[N, 3] → [N, 4] 동차좌표 (w=1 을 붙임).  변환 행렬 곱셈을 위한 준비."""
    return np.hstack([p, np.ones((p.shape[0], 1), dtype=p.dtype)])


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """쿼터니언 (x, y, z, w) → 3×3 회전 행렬.

    keyframe_pose.txt 는 (qx, qy, qz, qw) 순서 (ROS 규약) 로 저장.
    주의: chapter2.py 의 _quat_to_rot 은 (w, x, y, z) 순서 — 모델 파라미터 쪽 규약과 다름.
    여기선 ROS 관용을 따라 (x, y, z, w) 로 받는다.

    수식 (Hamilton, 정규화 후):
        s = 2 / (x²+y²+z²+w²)
        R[0,0] = 1 - s(y² + z²),   R[0,1] = s(xy - wz),   R[0,2] = s(xz + wy)
        R[1,0] = s(xy + wz),       R[1,1] = 1 - s(x²+z²), R[1,2] = s(yz - wx)
        R[2,0] = s(xz - wy),       R[2,1] = s(yz + wx),   R[2,2] = 1 - s(x²+y²)
    """
    x, y, z, w = qx, qy, qz, qw
    n = x * x + y * y + z * z + w * w         # |q|²
    s = 0.0 if n == 0.0 else 2.0 / n
    xx, yy, zz = s * x * x, s * y * y, s * z * z
    xy, xz, yz = s * x * y, s * x * z, s * y * z
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ], dtype=np.float64)


def _parse_pose_txt(path: Path) -> tuple[int, np.ndarray]:
    """keyframe_pose.txt 한 줄 파싱.

    형식:  "<id> <frame> <ts_ns> tx ty tz qx qy qz qw"
    반환: (ts_ns, T_world_base [4, 4])
    """
    toks = path.read_text().split()
    ts = int(toks[2])
    tx, ty, tz, qx, qy, qz, qw = map(float, toks[3:10])
    T = np.eye(4)
    T[:3, :3] = _quat_to_R(qx, qy, qz, qw)
    T[:3,  3] = (tx, ty, tz)
    return ts, T


def _parse_flag(path: Path) -> int:
    """outdoor.txt / elevator.txt / keyframe_type.txt 류.
    형식: "<id> <value>".  value 가 숫자면 int, 1글자 문자면 ord 로 변환해 반환.
    """
    tok = path.read_text().split()[1]
    try:
        return int(tok)
    except ValueError:
        return ord(tok) if len(tok) == 1 else -1


def load_pcd_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """PCD v0.7 binary (fields = x y z intensity, f32) 읽기.

    PCD 포맷 요약:
      * 헤더는 ASCII (첫 11 줄 정도).  "DATA binary" 라인 이후가 바이너리 바디.
      * 여기선 intensity 까지 4 float = 16 byte per point.
      * NaN/Inf 포인트는 거름 (LiDAR 드라이버가 종종 생성).
    """
    with open(path, "rb") as f:
        # 헤더를 라인 단위로 읽다가 "DATA" 라인을 만나면 종료
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            if line.startswith(b"DATA"):
                break
        header = header_bytes.decode("ascii", errors="ignore")
        n = int(re.search(r"POINTS\s+(\d+)", header).group(1))
        data_kind = re.search(r"DATA\s+(\w+)", header).group(1)
        if data_kind != "binary":
            raise NotImplementedError(f"PCD DATA '{data_kind}' 미지원")
        # 바디 끝까지 16 바이트 × n 포인트
        raw = f.read(n * 16)

    # 메모리 뷰 → [N, 4] float32.  [x, y, z, intensity] 순.
    buf = np.frombuffer(raw, dtype=np.float32).reshape(n, 4)
    points = buf[:, :3].astype(np.float32)
    inten = buf[:, 3].astype(np.float32)

    # NaN/Inf 는 유효 마스크로 걸러낸다 — 이후 투영/변환에서 터지지 않도록.
    valid = np.isfinite(points).all(axis=1) & np.isfinite(inten)
    return points[valid], inten[valid]


def load_keyframe(root: Path | str, idx: int) -> Optional[KeyFrame]:
    """단일 키프레임 디렉터리를 읽어 KeyFrame 객체로 반환.  없으면 None."""
    d = Path(root) / str(idx)
    if not d.is_dir():
        return None
    pcd = d / "point_cloud.pcd"
    pose = d / "keyframe_pose.txt"
    # LiDAR + pose 는 필수, 나머지(이미지·flag·descriptor)는 옵셔널.
    if not (pcd.exists() and pose.exists()):
        return None

    ts_ns, T_wb = _parse_pose_txt(pose)
    pts, inten = load_pcd_binary(pcd)

    # 3 카메라 — 있는 것만 읽어들인다.  BMP 는 BGR uint8.
    images: Dict[str, np.ndarray] = {}
    for cam in CAM_NAMES:
        p = d / f"{cam}_color.bmp"
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                images[cam] = img

    # 플래그 3 종 — 파일 없으면 -1 (알 수 없음).
    flags = {
        "outdoor":  _parse_flag(d / "outdoor.txt")       if (d / "outdoor.txt").exists()       else -1,
        "elevator": _parse_flag(d / "elevator.txt")      if (d / "elevator.txt").exists()      else -1,
        "type":     _parse_flag(d / "keyframe_type.txt") if (d / "keyframe_type.txt").exists() else -1,
    }

    # NetVLAD descriptor — 첫 토큰이 id 인 경우가 있어 float 파싱 실패는 무시.
    desc = None
    dp = d / "descriptor.txt"
    if dp.exists():
        toks = dp.read_text().split()
        vals = []
        for t in toks:
            try:
                vals.append(float(t))
            except ValueError:
                pass
        desc = np.asarray(vals, dtype=np.float32) if vals else None

    return KeyFrame(
        idx=idx, ts_ns=ts_ns, T_world_base=T_wb,
        points_base=pts, intensities=inten,
        images=images, flags=flags, descriptor=desc,
    )


class BaseDataDataset:
    """폴더 루트를 순회하며 KeyFrame 객체를 이터레이트.

    __init__ 에서 calibration + undistort map 을 준비해두므로 생성자 비용이
    조금 있다 (수 ms).  이후 순회는 lazy — 한 번에 한 키프레임씩 읽는다.

    require_image=True 면 이미지가 아예 없는 kf (예: 0번) 를 자동 스킵.
    """

    def __init__(self, root: Path | str, require_image: bool = False,
                 cam_downscale: Optional[Dict[str, int]] = None):
        self.root = Path(root)
        # cam_downscale: 큰 해상도 카메라 (예: left/right 4032×3036) 의 출력
        # 사이즈를 1/s 로 축소.  remap 단계에서 직접 작은 이미지를 생성하므로
        # full-res 중간 버퍼 메모리를 쓰지 않는다.
        self.calib = load_calibration(self.root, downscale=cam_downscale)
        self.calib.build_undistort_maps()            # 언디스토션 테이블 미리 준비
        # 숫자 폴더만 인덱스로.  calibration/ 등은 자연스럽게 배제.
        self.indices: list[int] = sorted(
            int(p.name) for p in self.root.iterdir()
            if p.is_dir() and p.name.isdigit()
        )
        self.require_image = require_image

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[KeyFrame]:
        """idx 순서대로 KeyFrame 을 yield.  이미지 필수 플래그가 걸려 있으면
        이미지 없는 kf 는 스킵 (lazy)."""
        for i in self.indices:
            kf = load_keyframe(self.root, i)
            if kf is None:
                continue
            if self.require_image and not kf.images:
                continue
            yield kf

    def __getitem__(self, i: int) -> KeyFrame:
        """i 번째 유효 인덱스 (리스트 자체 i) 접근.  순차/랜덤 모두 가능."""
        kf = load_keyframe(self.root, self.indices[i])
        if kf is None:
            raise IndexError(i)
        return kf


# =============================================================================
# 3. chapter2.py 어댑터 — LiDAR → 이미지 투영 / 학습 샘플 빌드 / 씨앗 컬러링
# =============================================================================
def project_lidar_to_camera(
    points_base: np.ndarray,
    cam_name: str,
    calib: Calibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """base 프레임 LiDAR 포인트를 **언디스토션된 카메라 이미지 평면** 으로 투영.

    파이프라인:
      1) p_cam = T_cam_base · p_base      (카메라 좌표계로 회전·이동)
      2) 앞쪽(z > 0) 필터                (카메라 뒤쪽 점 제거)
      3) 핀홀 투영 u = fx·x/z + cx, v = fy·y/z + cy  (new_K 사용)
      4) 이미지 범위 내 필터

    Returns:
        uv    [M, 2]  픽셀 좌표 (float32).  M ≤ N.
        depth [M]     카메라 z (meters, > 0).
        mask  [N]     원본 포인트 중 이미지 안에 들어간 인덱스 bool.
                      color 샘플링 등에서 인덱스 대응용으로 사용.
    """
    cam = calib.intrinsics[cam_name]
    assert cam.new_K is not None, "calib.build_undistort_maps() 먼저 호출 필요"

    # 1) 카메라 좌표계로 변환: p_cam = inv(T_base_cam) · p_base
    T_base_cam = calib.T_base_cam[cam_name]
    T_cam_base = np.linalg.inv(T_base_cam)
    p_cam = (T_cam_base @ _h(points_base).T).T[:, :3]           # [N, 3]

    # 2) z > 0 : 카메라 앞쪽 점만.  1e-3 가드로 0 나눗셈 방지.
    z = p_cam[:, 2]
    in_front = z > 1e-3

    # 3) 핀홀 투영 (new_K = undistorted intrinsic)
    x = p_cam[in_front, 0] / z[in_front]
    y = p_cam[in_front, 1] / z[in_front]
    fx, fy = cam.new_K[0, 0], cam.new_K[1, 1]
    cx, cy = cam.new_K[0, 2], cam.new_K[1, 2]
    u = fx * x + cx
    v = fy * y + cy

    # 4) 이미지 안쪽 범위 필터
    in_img = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)

    # 원본 N 크기 mask 복원:  in_front 가 True 인 위치의 서브셋 중 in_img 만 True.
    mask = np.zeros(points_base.shape[0], dtype=bool)
    mask[np.where(in_front)[0][in_img]] = True

    uv = np.stack([u[in_img], v[in_img]], axis=1).astype(np.float32)
    depth = z[in_front][in_img].astype(np.float32)
    return uv, depth, mask


def build_chapter2_inputs(
    kf: KeyFrame,
    cam_name: str,
    calib: Calibration,
) -> dict:
    """한 (키프레임, 카메라) 쌍을 chapter2.LidarVisualGS 학습 입력으로 변환.

    반환 dict 의 6 개 키는 LidarVisualGS.train_step() 시그니처와 이름까지 맞춰져 있음.

    Keys:
        lidar_points_world [N, 3]    float32   모델 초기화용 (world).
        lidar_colors       [N, 3]    float32   (0~1). 투영 안 된 점은 회색 0.5.
        gt_image           [H, W, 3] float32   (0~1). undistorted + saturation boost.
        gt_depth           [H, W]    float32   (0 = invalid, meters).  LiDAR 투영 sparse.
        K                  [3, 3]    float32   undistorted new_K (핀홀).
        viewmat            [4, 4]    float32   T_cam_from_world.
    """
    if cam_name not in kf.images:
        raise ValueError(f"keyframe {kf.idx} 에 {cam_name} 이미지 없음")
    cam = calib.intrinsics[cam_name]

    # --- 1) 이미지 준비 (동일한 pipeline 으로 seed 컬러링과 정합) ---
    gt_image = _undistort_rgb(kf.images[cam_name], cam)

    # --- 2) LiDAR 투영 → uv, depth, mask ---
    pts_base = kf.points_in("base", calib)
    uv, depth, mask = project_lidar_to_camera(pts_base, cam_name, calib)

    # --- 3) Sparse depth map (H×W) 구성 ---
    # 같은 픽셀에 여러 점이 투영되면 더 가까운 쪽(= z 작음) 을 유지.
    # 방법: depth 를 내림차순(-depth) 으로 정렬 후 같은 픽셀을 덮어쓰면
    # 마지막에 남는 값이 가장 작은 depth = 더 가까운 점.
    H, W = cam.height, cam.width
    gt_depth = np.zeros((H, W), dtype=np.float32)
    ui = np.clip(uv[:, 0].astype(np.int32), 0, W - 1)
    vi = np.clip(uv[:, 1].astype(np.int32), 0, H - 1)
    order = np.argsort(-depth)             # 내림차순 인덱스
    gt_depth[vi[order], ui[order]] = depth[order]

    # --- 4) Seed 컬러: 투영된 점은 픽셀 색, 아니면 회색 0.5 ---
    N = pts_base.shape[0]
    colors = np.full((N, 3), 0.5, dtype=np.float32)
    proj_idx = np.where(mask)[0]
    colors[proj_idx] = gt_image[vi, ui]

    # --- 5) 위치는 world 좌표로 모델에 전달 ---
    pts_world = kf.points_in("world", calib).astype(np.float32)

    # --- 6) viewmat = T_cam_from_world = inv(T_world_base · T_base_cam)
    T_world_cam = kf.T_world_base @ calib.T_base_cam[cam_name]
    viewmat = np.linalg.inv(T_world_cam).astype(np.float32)

    return {
        "lidar_points_world": pts_world,
        "lidar_colors":       colors,
        "gt_image":           gt_image,
        "gt_depth":           gt_depth,
        "K":                  cam.new_K.astype(np.float32),
        "viewmat":            viewmat,
        # cam_index: CAM_NAMES 인덱스 (front=0, left=1, right=2). extrinsic refine 용.
        "cam_index":          CAM_NAMES.index(cam_name),
        # kf_index: 폴더명 기반 정수 id (0..680).  per-keyframe pose refine 용.
        "kf_index":           kf.idx,
    }


def colorize_keyframe(
    kf: KeyFrame,
    calib: Calibration,
    drop_unseen: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """kf 의 base 점들을 3-카메라 이미지에 투영해 world 좌표 + 컬러를 만든다.

    3DGS 초기 가우시안 seed 구성의 핵심 루틴.

    동작:
      1) 각 카메라의 언디스토션된 이미지를 준비
      2) 각 카메라에 대해 점을 투영, 보이는 점만 해당 픽셀 RGB 를 누적
      3) 여러 카메라에 동시에 찍힌 점은 **평균색**
      4) 어떤 카메라에도 안 찍힌 점은 회색 (0.5) — 또는 drop_unseen 이면 제외

    drop_unseen 의 의미:
      * True (기본) : 보이지 않는 점은 seed 에서 제외.  회색 유령 가우시안을
        방지해 학습 초반 loss 가 엉뚱한 방향으로 가지 않도록 한다.  대신 LiDAR 는
        보지만 카메라는 못 본 점 (천장/바닥/뒤쪽) 이 모두 빠진다.
      * False       : 옛 동작.  회색 seed 도 허용 (디버깅 / 특정 케이스용).
    """
    pts_base = kf.points_in("base", calib)
    N = pts_base.shape[0]
    # 각 점의 누적 RGB 와 몇 번 찍혔는지 카운트.
    color_acc = np.zeros((N, 3), dtype=np.float32)
    hit = np.zeros(N, dtype=np.int32)

    for cam_name in CAM_NAMES:
        if cam_name not in kf.images:
            continue
        cam = calib.intrinsics[cam_name]
        # 한 이미지에 대해 동일한 언디스토션·채도 부스트 거친 RGB 를 샘플 소스로 사용.
        img_rgb = _undistort_rgb(kf.images[cam_name], cam)

        uv, _, mask = project_lidar_to_camera(pts_base, cam_name, calib)
        ui = np.clip(uv[:, 0].astype(np.int32), 0, cam.width - 1)
        vi = np.clip(uv[:, 1].astype(np.int32), 0, cam.height - 1)
        # 보이는 점의 원본 인덱스로 RGB 누적
        idx = np.where(mask)[0]
        color_acc[idx] += img_rgb[vi, ui]
        hit[idx] += 1

    # 평균색 (누적 / 카운트).  미투영 점은 회색 0.5 로 남음.
    seen = hit > 0
    colors = np.full((N, 3), 0.5, dtype=np.float32)
    colors[seen] = color_acc[seen] / hit[seen, None]

    # world 좌표로 승격
    pts_world = (kf.T_world_base @ _h(pts_base).T).T[:, :3].astype(np.float32)

    if drop_unseen:
        return pts_world[seen], colors[seen]
    return pts_world, colors


# =============================================================================
# 4. 스모크 테스트 — 패키지가 import 없이 단독 실행 가능한지 확인
# =============================================================================
if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("base_data")

    # 이미지 없는 kf(예: 0번) 는 건너뛰어서 첫 유효 kf 를 집는다.
    ds = BaseDataDataset(root, require_image=True)
    print(f"[ok] {len(ds)} keyframes under {root}")
    kf = next(iter(ds))
    print(f"  kf#{kf.idx}  ts={kf.ts_ns}  pts={kf.points_base.shape}  "
          f"imgs={list(kf.images.keys())}  flags={kf.flags}")

    # 첫 유효 kf 를 front 카메라 기준 학습 샘플로 만들어 형상 검증.
    inp = build_chapter2_inputs(kf, "front", ds.calib)
    for k, v in inp.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:22s} shape={v.shape} dtype={v.dtype} "
                  f"min={v.min():.3f} max={v.max():.3f}")
    # sparse depth 의 유효 픽셀 밀도 — 높을수록 LiDAR·카메라 overlap 이 좋다는 뜻.
    valid = (inp["gt_depth"] > 0).sum()
    print(f"  projected pixels = {valid}/{inp['gt_depth'].size}  "
          f"({100*valid/inp['gt_depth'].size:.2f}%)")
