#!/usr/bin/env python3
"""Mock ROS 2 Trigger server for the pallet robot start request."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Trigger


SERVICE_NAME = "/pallet_robot_start"


class MockPalletRobotStartServer(Node):
    def __init__(self):
        super().__init__("mock_pallet_robot_start_server")
        self.service = self.create_service(
            Trigger,
            SERVICE_NAME,
            self.handle_start_request,
        )
        self.request_count = 0
        self.get_logger().info(
            f"mock pallet robot ready: service={SERVICE_NAME}"
        )

    def handle_start_request(self, request, response):
        del request
        self.request_count += 1
        response.success = True
        response.message = "Pallet robot started successfully (mock)"
        self.get_logger().info(
            f"start request #{self.request_count} received -> success=True"
        )
        return response


def main():
    rclpy.init()
    node = MockPalletRobotStartServer()
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
