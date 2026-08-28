#!/usr/bin/env python3
"""
외형검사 노드 (CNN 기반)

동작 흐름:
  1. /worktable_top_rgb, /worktable_side_rgb 두 토픽을 항상 구독하며 "최신 프레임"만 계속 갱신해둔다.
  2. 로봇팔이 셀을 들어올린 신호와 함께 'inspect_cell' 서비스를 호출하면,
  3. 그 시점의 top/side 최신 프레임 두 장을 각각 CNN에 넣어 추론하고,
  4. 두 결과를 종합해서 정상/비정상을 response로 돌려준다.

서비스 타입: std_srvs/Trigger (커스텀 .srv 없이 바로 쓸 수 있음)
  request  : (없음)
  response : bool success   -> True = 정상, False = 비정상
             string message -> 상세 정보(신뢰도, 판정 근거)
"""

import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.models import resnet34
from PIL import Image as PILImage
import numpy as np


# ============================================================
# ⚠️ 학습 때(cnn_experiment_pipeline) 최종 채택한 설정과 반드시 동일하게 맞출 것
# ============================================================
CNN_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(CNN_DIR / "cell_dataset" / "cell_classifier_final.pt")
DEVICE_NAME = os.environ.get("CELL_INSPECTION_DEVICE", "cpu")

# ★ 실험 로그 확인 결과: 1_Baseline(ReLU,3x3)이 100% 로 최고 성능이었고,# ★ 실험 로그 확인 결과: 1_Baseline(ReLU,3x3)이 100% 로 최고 성능이었고,

#   실험 5(최종 학습)도 이 설정(activation=relu, kernel_size=3)으로 진행되어
#   cell_classifier_final.pt 가 이 구조로 저장되어 있음. (증강/ELU/5x5는 오히려 성능 저하)
ACTIVATION = "relu"
KERNEL_SIZE = 3
NUM_CLASSES = 2

# 프로젝트/실험 코드 사이에서 토픽 이름이 섞여 있다.
# - 현재 Isaac Sim worktable camera: /worktable_top_rgb, /worktable_side_rgb
# - 이 파일의 원래 주석: /top_rgb, /side_rgb
# - 이전 상수: /rgb_top, /rgb_side
# - 현재 m0609 camera USD 예시: /rgb
# 따라서 후보를 모두 구독하고, 실제 들어오는 최신 프레임을 사용한다.
TOP_TOPIC_CANDIDATES = ("/worktable_top_rgb", "/rgb_top", "/top_rgb", "/rgb")
SIDE_TOPIC_CANDIDATES = ("/worktable_side_rgb", "/rgb_side", "/side_rgb", "/rgb")
SERVICE_NAME = "inspect_cell"

# ★ 학습 로그 실제 출력과 정확히 일치함 (cell 15 출력: 클래스 매핑: {'abnormal': 0, 'normal': 1})
CLASS_TO_IDX = {"abnormal": 0, "normal": 1}
ABNORMAL_IDX = CLASS_TO_IDX["abnormal"]
NORMAL_IDX = CLASS_TO_IDX["normal"]
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}

# 이미지가 너무 오래된(로봇팔이 아직 안 올라온 시점) 것이면 판정에 안 쓰기 위한 최대 허용 시간
MAX_IMAGE_AGE_SEC = 1.0
TOP_NORMAL_CONFIDENCE_THRESHOLD = 0.50
SIDE_NORMAL_CONFIDENCE_THRESHOLD = 0.027
DEBUG_SAVE_DIR = os.environ.get(
    "CELL_INSPECTION_DEBUG_DIR",
    str(CNN_DIR / "inspection_debug"),
)

NORMALIZE_MEAN = (0.447, 0.440, 0.407)
NORMALIZE_STD = (0.260, 0.257, 0.271)


def build_model(activation: str, kernel_size: int, num_classes: int):
    model = resnet34(pretrained=False)  # 가중치는 어차피 저장된 state_dict로 덮어씀

    if kernel_size == 5:
        pad = 2
        model.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(pad, pad), bias=False)
        model.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=(pad, pad), bias=False)
        model.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=(pad, pad), bias=False)

    if activation == "elu":
        act_fn = nn.ELU()
        for layer_group in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in layer_group:
                block.relu = act_fn

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class CellInspectionNode(Node):
    def __init__(self):
        super().__init__('cell_inspection_node')

        # ── 모델 로드 (노드 시작 시 한 번만) ──
        requested_device = DEVICE_NAME.lower()
        if requested_device == "cuda" and not torch.cuda.is_available():
            self.get_logger().warn(
                "CELL_INSPECTION_DEVICE=cuda 요청됨, 하지만 CUDA 사용 불가 → CPU로 실행"
            )
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.get_logger().info(f"모델 로딩 중... (device={self.device})")

        self.model = build_model(ACTIVATION, KERNEL_SIZE, NUM_CLASSES)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            v2.Resize((96, 96)),
            v2.ToTensor(),
            v2.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ])
        os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

        self.get_logger().info("모델 로딩 완료")

        # ── top/side 이미지를 각각 계속 구독하며 최신 프레임만 보관 ──
        self.latest_top_image = None
        self.latest_top_time = 0.0
        self.latest_top_topic = ""
        self.latest_side_image = None
        self.latest_side_time = 0.0
        self.latest_side_topic = ""
        image_qos_profiles = (
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            ),
        )

        self.top_subs = [
            self.create_subscription(
                Image,
                topic,
                lambda msg, topic=topic, qos_name=qos_name: self.top_image_callback(
                    msg, f"{topic}/{qos_name}"
                ),
                qos,
            )
            for topic in TOP_TOPIC_CANDIDATES
            for qos_name, qos in (
                ("reliable", image_qos_profiles[0]),
                ("best_effort", image_qos_profiles[1]),
            )
        ]
        self.side_subs = [
            self.create_subscription(
                Image,
                topic,
                lambda msg, topic=topic, qos_name=qos_name: self.side_image_callback(
                    msg, f"{topic}/{qos_name}"
                ),
                qos,
            )
            for topic in SIDE_TOPIC_CANDIDATES
            for qos_name, qos in (
                ("reliable", image_qos_profiles[0]),
                ("best_effort", image_qos_profiles[1]),
            )
        ]

        # ── 서비스 서버 ──
        self.srv = self.create_service(
            Trigger, SERVICE_NAME, self.inspect_callback
        )

        self.get_logger().info(
            f"[준비 완료] '{SERVICE_NAME}' 서비스 대기 중... "
            f"(top_candidates={TOP_TOPIC_CANDIDATES}, "
            f"side_candidates={SIDE_TOPIC_CANDIDATES}, "
            "qos=reliable+best_effort)"
        )

    # ────────────────────────────────────────────────
    # 토픽 콜백 — 최신 프레임만 계속 갱신 (판정은 여기서 안 함)
    # ────────────────────────────────────────────────
    def top_image_callback(self, msg: Image, topic: str):
        self.latest_top_image = msg
        self.latest_top_time = time.time()
        self.latest_top_topic = topic

    def side_image_callback(self, msg: Image, topic: str):
        self.latest_side_image = msg
        self.latest_side_time = time.time()
        self.latest_side_topic = topic

    # ────────────────────────────────────────────────
    # 추론 함수 — ROS Image 메시지 하나를 받아 (정상확률, 비정상확률) 반환
    # ────────────────────────────────────────────────
    def ros_image_to_rgb_array(self, ros_image: Image):
        """cv_bridge 없이 sensor_msgs/Image를 RGB uint8 numpy array로 바꾼다.

        Isaac Sim 내장 Python에는 cv_bridge가 없는 경우가 많다. 여기서는 CNN이
        필요한 rgb8 배열만 만들면 되므로 ROS Image의 data/encoding/step을 직접
        해석한다. /rgb_top, /rgb_side가 rgb8이면 그대로 쓰고, bgr/rgba 계열이면
        채널만 맞춘다.
        """
        height = int(ros_image.height)
        width = int(ros_image.width)
        encoding = str(ros_image.encoding).lower()
        if height <= 0 or width <= 0:
            raise ValueError(f"invalid image size: {width}x{height}")

        if encoding in ("rgb8", "bgr8"):
            channels = 3
        elif encoding in ("rgba8", "bgra8"):
            channels = 4
        elif encoding in ("mono8", "8uc1"):
            channels = 1
        else:
            raise ValueError(f"unsupported image encoding: {ros_image.encoding}")

        row_bytes = int(ros_image.step)
        expected_min_step = width * channels
        if row_bytes < expected_min_step:
            raise ValueError(
                f"invalid image step: step={row_bytes}, expected>={expected_min_step}"
            )

        raw = np.frombuffer(ros_image.data, dtype=np.uint8)
        expected_size = row_bytes * height
        if raw.size < expected_size:
            raise ValueError(
                f"image data too small: bytes={raw.size}, expected={expected_size}"
            )
        rows = raw[:expected_size].reshape(height, row_bytes)
        pixels = rows[:, :expected_min_step].reshape(height, width, channels)

        if encoding == "rgb8":
            return pixels.copy()
        if encoding == "bgr8":
            return pixels[:, :, ::-1].copy()
        if encoding == "rgba8":
            return pixels[:, :, :3].copy()
        if encoding == "bgra8":
            return pixels[:, :, [2, 1, 0]].copy()
        return np.repeat(pixels, 3, axis=2).copy()

    def run_inference(self, ros_image: Image, view_name: str):
        rgb_image = self.ros_image_to_rgb_array(ros_image)
        pil_image = PILImage.fromarray(rgb_image)

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)[0]  # [class0 확률, class1 확률]
        except RuntimeError as exc:
            if self.device.type != "cuda" or "CUDA" not in str(exc):
                raise
            self.get_logger().warn(
                f"CUDA 추론 실패({exc}) → 모델을 CPU로 옮겨 재시도합니다"
            )
            self.device = torch.device("cpu")
            self.model.to(self.device)
            input_tensor = input_tensor.to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)[0]

        probs_cpu = probs.detach().cpu()
        abnormal_prob = float(probs_cpu[ABNORMAL_IDX])
        normal_prob = float(probs_cpu[NORMAL_IDX])
        pred_idx = int(torch.argmax(probs_cpu).item())
        pred_label = IDX_TO_CLASS.get(pred_idx, f"class_{pred_idx}")
        self.get_logger().info(
            f"[CNN RAW {view_name}] "
            f"probs[class0_abnormal,class1_normal]="
            f"[{abnormal_prob:.4f}, {normal_prob:.4f}], "
            f"pred={pred_label}, mapping={CLASS_TO_IDX}"
        )
        return {
            "abnormal_prob": abnormal_prob,
            "normal_prob": normal_prob,
            "pred_label": pred_label,
            "rgb_image": rgb_image,
        }

    # ────────────────────────────────────────────────
    # 서비스 콜백 — 여기가 실제 트리거 되는 지점
    # ────────────────────────────────────────────────
    def inspect_callback(self, request, response):
        now = time.time()

        # ── 두 이미지가 다 준비됐는지, 너무 오래된 건 아닌지 확인 ──
        if self.latest_top_image is None or self.latest_side_image is None:
            # 현재 통합 씬에는 wrist camera /rgb 하나만 있는 구성이 흔하다. 후보
            # 중 한쪽만 들어왔으면 같은 프레임을 양쪽 검사 입력으로 써서 공정을
            # 멈추지 않는다. 두 대 카메라가 실제로 연결되면 각각의 최신 프레임이
            # 자동으로 사용된다.
            if self.latest_top_image is not None:
                self.latest_side_image = self.latest_top_image
                self.latest_side_time = self.latest_top_time
                self.latest_side_topic = self.latest_top_topic + " (fallback)"
            elif self.latest_side_image is not None:
                self.latest_top_image = self.latest_side_image
                self.latest_top_time = self.latest_side_time
                self.latest_top_topic = self.latest_side_topic + " (fallback)"
            else:
                response.success = False
                response.message = (
                    "이미지가 아직 수신되지 않았습니다 "
                    f"(top candidates={TOP_TOPIC_CANDIDATES}, "
                    f"side candidates={SIDE_TOPIC_CANDIDATES})"
                )
                self.get_logger().warn(response.message)
                return response

        top_age = now - self.latest_top_time
        side_age = now - self.latest_side_time
        if top_age > MAX_IMAGE_AGE_SEC or side_age > MAX_IMAGE_AGE_SEC:
            response.success = False
            response.message = (
                f"이미지가 너무 오래됨 (top_age={top_age:.2f}s, side_age={side_age:.2f}s) "
                f"- 로봇팔이 카메라 앞에 도달했는지 확인 필요"
            )
            self.get_logger().warn(response.message)
            return response

        # ── 두 뷰 각각 추론 ──
        top_result = self.run_inference(self.latest_top_image, "top")
        side_result = self.run_inference(self.latest_side_image, "side")
        top_normal_prob = top_result["normal_prob"]
        side_normal_prob = side_result["normal_prob"]
        top_abnormal_prob = top_result["abnormal_prob"]
        side_abnormal_prob = side_result["abnormal_prob"]

        # ── 종합 판정: 둘 중 하나라도 정상 confidence가 부족하면 비정상 ──
        top_is_normal = top_normal_prob >= TOP_NORMAL_CONFIDENCE_THRESHOLD
        side_is_normal = side_normal_prob >= SIDE_NORMAL_CONFIDENCE_THRESHOLD
        final_is_normal = top_is_normal and side_is_normal  # 둘 다 정상이어야 최종 정상

        label = "normal" if final_is_normal else "abnormal"
        stamp_ms = int(now * 1000)
        try:
            top_path = os.path.join(
                DEBUG_SAVE_DIR,
                f"{stamp_ms}_{label}_top_n{top_normal_prob:.4f}_a{top_abnormal_prob:.4f}.png",
            )
            side_path = os.path.join(
                DEBUG_SAVE_DIR,
                f"{stamp_ms}_{label}_side_n{side_normal_prob:.4f}_a{side_abnormal_prob:.4f}.png",
            )
            PILImage.fromarray(top_result["rgb_image"]).save(top_path)
            PILImage.fromarray(side_result["rgb_image"]).save(side_path)
        except Exception as exc:
            self.get_logger().warn(f"검사 디버그 이미지 저장 실패: {exc}")

        response.success = final_is_normal
        response.message = (
            f"판정={'정상' if final_is_normal else '비정상'} | "
            f"top_raw=[abnormal={top_abnormal_prob:.3f}, normal={top_normal_prob:.3f}], "
            f"side_raw=[abnormal={side_abnormal_prob:.3f}, normal={side_normal_prob:.3f}] | "
            f"top_pred={top_result['pred_label']} | "
            f"side_pred={side_result['pred_label']} | "
            f"top_normal_prob={top_normal_prob:.3f} | "
            f"side_normal_prob={side_normal_prob:.3f} | "
            f"top_normal_threshold={TOP_NORMAL_CONFIDENCE_THRESHOLD:.3f} | "
            f"side_normal_threshold={SIDE_NORMAL_CONFIDENCE_THRESHOLD:.3f} | "
            f"debug_dir={DEBUG_SAVE_DIR} | "
            f"top_topic={self.latest_top_topic} | "
            f"side_topic={self.latest_side_topic}"
        )

        self.get_logger().info(f"[검사 결과] {response.message}")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CellInspectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
