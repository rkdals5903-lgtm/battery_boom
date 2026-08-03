#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import numpy as np


class ColorDetector(Node):
    def __init__(self):
        super().__init__('color_detector')

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            10
        )

        self.pub = self.create_publisher(Int32, '/color_id', 10)

        self.MIN_PIXELS = 500  # 노이즈 방지용 최소 픽셀 수

        self.get_logger().info('Color Detector 노드 시작 — /rgb 구독 중...')

    def image_callback(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        b, g, r = cv2.split(cv_image)
        b = b.astype(int)
        g = g.astype(int)
        r = r.astype(int)

        blue_mask = (b > g) & (b > r)
        green_mask = (g > b) & (g > r)

        blue_count = np.count_nonzero(blue_mask)
        green_count = np.count_nonzero(green_mask)

        if blue_count > green_count and blue_count > self.MIN_PIXELS:
            color_id = 1
        elif green_count > blue_count and green_count > self.MIN_PIXELS:
            color_id = 2
        else:
            self.get_logger().info(
                f'색상 발행 불가 (blue_px={blue_count}, green_px={green_count})'
            )
            return

        msg_out = Int32()
        msg_out.data = color_id
        self.pub.publish(msg_out)

        color_name = '파랑' if color_id == 1 else '초록'
        self.get_logger().info(
            f'감지: {color_name} (color_id={color_id}, '
            f'blue_px={blue_count}, green_px={green_count})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()