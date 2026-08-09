#!/usr/bin/env python3
"""Mock ROS 2 Trigger server for the hijack-robot cleared handshake."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Trigger


SERVICE_NAME = "/hijack_robot_cleared"


class MockHijackRobotClearedServer(Node):
    def __init__(self):
        super().__init__("mock_hijack_robot_cleared_server")
        self.request_count = 0
        self.service = self.create_service(
            Trigger,
            SERVICE_NAME,
            self.handle_cleared_request,
        )
        self.get_logger().info(
            f"mock hijack robot ready: service={SERVICE_NAME}"
        )

    def handle_cleared_request(self, request, response):
        del request
        self.request_count += 1
        response.success = True
        response.message = "Hijack robot cleared the full new_case (mock)"
        self.get_logger().info(
            f"cleared request #{self.request_count} received -> success=True"
        )
        return response


def main():
    rclpy.init()
    node = MockHijackRobotClearedServer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
