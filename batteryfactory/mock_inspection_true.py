import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class MockInspectionTrue(Node):
    def __init__(self):
        super().__init__('mock_inspection_true')
        self.srv = self.create_service(Trigger, '/battery_inspection_result', self.trigger_callback)
        self.get_logger().info('🟢 [정상] 카메라 검사 서버 대기 중... (무조건 True 반환)')

    def trigger_callback(self, request, response):
        self.get_logger().info('📸 아이작 심 검사 요청 수신 -> 정상(True) 반환!')
        response.success = True
        response.message = 'Inspection Pass'
        return response

def main():
    rclpy.init()
    node = MockInspectionTrue()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
