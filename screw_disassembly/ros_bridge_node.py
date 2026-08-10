# ros_bridge_node.py
import os
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

TRIGGER_FILE = "/tmp/screw_trigger.flag"

class ScrewBridgeServer(Node):
    def __init__(self):
        super().__init__('screw_bridge_server')
        self.srv = self.create_service(Trigger, 'start_screw_process', self.handle_request)
        self.get_logger().info("🤖 [ROS2 Bridge] 나사 분해 서비스 대기 중... (/start_screw_process)")

    def handle_request(self, request, response):
        self.get_logger().info("📥 트리거 수신! 시뮬레이터로 신호를 전달합니다.")

        # 시뮬레이터가 감지할 수 있도록 플래그 파일 생성
        with open(TRIGGER_FILE, "w") as f:
            f.write("trigger")

        response.success = True
        response.message = "Screw disassembly process triggered successfully!"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ScrewBridgeServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
