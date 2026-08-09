#!/usr/bin/env python3
"""Test server for grip_cell_v3.py's battery inspection Trigger service."""

import argparse

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


SERVICE_NAME = "/battery_inspection_result"


def parse_bool(value):
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "good", "ok"):
        return True
    if normalized in ("false", "0", "no", "bad", "ng"):
        return False
    raise argparse.ArgumentTypeError("use true or false")


class BatteryInspectionTestServer(Node):
    def __init__(self, result):
        super().__init__("battery_inspection_test_server")
        self.result = bool(result)
        self.service = self.create_service(Trigger, SERVICE_NAME, self.handle_request)
        self.get_logger().info(
            f"ready: service={SERVICE_NAME}, configured result={self.result}"
        )

    def handle_request(self, request, response):
        del request
        response.success = self.result
        response.message = "good cell" if self.result else "defective cell"
        self.get_logger().info(
            f"request received -> success={response.success}, message={response.message!r}"
        )
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        type=parse_bool,
        default=True,
        help="Trigger response: true sends the cell to new_case; false rejects it",
    )
    args = parser.parse_args()

    rclpy.init(args=[])
    node = BatteryInspectionTestServer(args.result)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
