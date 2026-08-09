import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class MockSuctionCloseServer(Node):
    def __init__(self):
        super().__init__('mock_suction_close_server')
        self.srv = self.create_service(Trigger, '/suction_cover_close', self.trigger_callback)
        self.get_logger().info('📦 [종료 대기] 흡착팔 서버 준비 완료... 아이작 심의 "뚜껑 닫아" 신호를 기다립니다.')

    def trigger_callback(self, request, response):
        self.get_logger().info('🔔 아이작 심 공정 종료 트리거 수신! 뚜껑을 닫습니다.')
        response.success = True
        response.message = 'Cover closed successfully'
        return response

def main():
    rclpy.init()
    node = MockSuctionCloseServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
