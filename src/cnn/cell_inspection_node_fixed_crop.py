#!/usr/bin/env python3
"""
외형검사 노드 (CNN 기반)

동작 흐름:
  1. /top_rgb, /side_rgb 두 토픽을 항상 구독하며 "최신 프레임"만 계속 갱신해둔다.
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
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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

TOP_TOPIC_CANDIDATES = ("/worktable_top_rgb", "/rgb_top", "/top_rgb", "/rgb")
SIDE_TOPIC_CANDIDATES = ("/worktable_side_rgb", "/rgb_side", "/side_rgb", "/rgb")
SERVICE_NAME = "inspect_cell"

# ★ 학습 로그 실제 출력과 정확히 일치함 (cell 15 출력: 클래스 매핑: {'abnormal': 0, 'normal': 1})
CLASS_TO_IDX = {"abnormal": 0, "normal": 1}
NORMAL_IDX = CLASS_TO_IDX["normal"]

# 이미지가 너무 오래된(로봇팔이 아직 안 올라온 시점) 것이면 판정에 안 쓰기 위한 최대 허용 시간
MAX_IMAGE_AGE_SEC = 1.0
TOP_NORMAL_CONFIDENCE_THRESHOLD = 0.942
SIDE_NORMAL_CONFIDENCE_THRESHOLD = 0.50

NORMALIZE_MEAN = (0.447, 0.440, 0.407)
NORMALIZE_STD = (0.260, 0.257, 0.271)

# ============================================================
# 고정 Crop 영역 (원본 이미지 640 x 640 기준)
# PIL crop 형식: (left, top, right, bottom)
# ============================================================
TOP_CROP = (138, 263, 263, 388)       # top_view
SIDE_CROP = (243, 462, 368, 587)      # side_view


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

        self.bridge = CvBridge()

        # ── 모델 로드 (노드 시작 시 한 번만) ──
        requested_device = DEVICE_NAME.lower()
        if requested_device == "cuda" and not torch.cuda.is_available():
            self.get_logger().warn(
                "CELL_INSPECTION_DEVICE=cuda 요청됨, 하지만 CUDA 사용 불가 -> CPU로 실행"
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

        self.get_logger().info("모델 로딩 완료")

        # ── top/side 이미지를 각각 계속 구독하며 최신 프레임만 보관 ──
        self.latest_top_image = None
        self.latest_top_time = 0.0
        self.latest_top_topic = ""
        self.latest_side_image = None
        self.latest_side_time = 0.0
        self.latest_side_topic = ""

        self.top_subs = [
            self.create_subscription(
                Image,
                topic,
                lambda msg, topic=topic: self.top_image_callback(msg, topic),
                10,
            )
            for topic in TOP_TOPIC_CANDIDATES
        ]
        self.side_subs = [
            self.create_subscription(
                Image,
                topic,
                lambda msg, topic=topic: self.side_image_callback(msg, topic),
                10,
            )
            for topic in SIDE_TOPIC_CANDIDATES
        ]

        # ── 서비스 서버 ──
        self.srv = self.create_service(
            Trigger, SERVICE_NAME, self.inspect_callback
        )

        self.get_logger().info(
            f"[준비 완료] '{SERVICE_NAME}' 서비스 대기 중... "
            f"(top_candidates={TOP_TOPIC_CANDIDATES}, side_candidates={SIDE_TOPIC_CANDIDATES})"
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
    # 수정됨
    # ────────────────────────────────────────────────
    def run_inference(self, ros_image: Image, view: str):
        cv_image = self.bridge.imgmsg_to_cv2(
            ros_image, desired_encoding='rgb8'
        )
        pil_image = PILImage.fromarray(cv_image)

        # 원본 카메라 영상은 640 x 640 기준.
        # 학습 때와 동일하게 view별 고정 Crop을 먼저 적용한다.
        if pil_image.size != (640, 640):
            self.get_logger().warn(
                f"{view} 이미지 크기가 {pil_image.size} 입니다. "
                "Crop 좌표는 640x640 기준입니다."
            )

        if view == "top":
            crop_box = TOP_CROP
        elif view == "side":
            crop_box = SIDE_CROP
        else:
            raise ValueError(f"알 수 없는 view: {view}")

        cropped_image = pil_image.crop(crop_box)

        # 고정 Crop(125x125) -> 96x96 -> Normalize
        input_tensor = self.transform(cropped_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]  # [class0 확률, class1 확률]

        normal_prob = float(probs[NORMAL_IDX])
        return normal_prob

    # ────────────────────────────────────────────────
    # 서비스 콜백 — 여기가 실제 트리거 되는 지점
    # ────────────────────────────────────────────────
    def inspect_callback(self, request, response):
        now = time.time()

        # ── 두 이미지가 다 준비됐는지, 너무 오래된 건 아닌지 확인 ──
        if self.latest_top_image is None or self.latest_side_image is None:
            response.success = False
            response.message = "이미지가 아직 수신되지 않았습니다 (top 또는 side)"
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
        top_normal_prob = self.run_inference(self.latest_top_image, view="top")
        side_normal_prob = self.run_inference(self.latest_side_image, view="side")

        # ── 종합 판정: 둘 중 하나라도 비정상 쪽으로 기울면 비정상 (보수적 판정) ──
        top_is_normal = top_normal_prob >= TOP_NORMAL_CONFIDENCE_THRESHOLD
        side_is_normal = side_normal_prob >= SIDE_NORMAL_CONFIDENCE_THRESHOLD
        final_is_normal = top_is_normal and side_is_normal  # 둘 다 정상이어야 최종 정상

        response.success = final_is_normal
        response.message = (
            f"판정={'정상' if final_is_normal else '비정상'} | "
            f"top_normal_prob={top_normal_prob:.3f} | "
            f"side_normal_prob={side_normal_prob:.3f} | "
            f"top_normal_threshold={TOP_NORMAL_CONFIDENCE_THRESHOLD:.3f} | "
            f"side_normal_threshold={SIDE_NORMAL_CONFIDENCE_THRESHOLD:.3f} | "
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
