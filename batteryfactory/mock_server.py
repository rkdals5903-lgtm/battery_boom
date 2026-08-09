import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class MockSuctionServer(Node):
    def __init__(self):
        super().__init__('mock_suction_server')
        self.srv = self.create_service(Trigger, '/suction_cover_opened', self.trigger_callback)
        self.get_logger().info('가짜 흡착팔 서버가 준비되었습니다! 아이작 심의 호출을 기다립니다...')

    def trigger_callback(self, request, response):
        self.get_logger().info('아이작 심으로부터 트리거 확인! 뚜껑이 열렸다고 응답(True)합니다.')
        response.success = True
        response.message = 'Cover opened successfully'
        # 1회 응답 후 서버 종료를 원하면 아래 타이머 주석을 푸세요
        # self.timer = self.create_timer(1.0, lambda: exit(0))
        return response

def main():
    rclpy.init()
    node = MockSuctionServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
