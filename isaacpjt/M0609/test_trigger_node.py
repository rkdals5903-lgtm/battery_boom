import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class ScrewProcessServer(Node):
    def __init__(self):
        super().__init__('screw_process_server')
        # 서비스 서버 생성 (토픽 이름: /start_screw_process)
        self.srv = self.create_service(Trigger, 'start_screw_process', self.handle_process_request)
        self.get_logger().info("🤖 나사 체결 공정 ROS2 서비스 서버 대기 중...")

    def handle_process_request(self, request, response):
        self.get_logger().info("📥 트리거 수신! 나사 체결 공정을 1회 시작합니다.")
        
        # ── 여기에 우리가 만든 5번 공정(홈정렬 -> 4개 순회 -> 홈복귀)이 들어갑니다 ──
        # run_screw_process_cycle() 
        
        response.success = True
        response.message = "Screw process completed successfully and robot returned home."
        self.get_logger().info("📤 공정 완료 응답(Response) 전송 완료!")
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ScrewProcessServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()