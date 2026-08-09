import rclpy
from rclpy.node import Node
import numpy as np # 정규분포 랜덤값 생성용(np.random.normal)과 범위 제한(np.clip)에 씀.

# from your_package.srv import CheckVoltage  # 추후 이름 교체 TODO


class BatteryVoltageServer(Node):
    def __init__(self):
        super().__init__('battery_voltage_server')

        self.srv = self.create_service(
            # CheckVoltage, TODO
            'check_voltage',
            self.check_voltage_callback
        )

        # ── 전압 생성 파라미터 ──
        self.MEAN_VOLTAGE = 11.0       # 생성 분포의 중심값
        self.STD_DEV = 1.5             # 분산값 조절 — 클수록 값이 넓게 퍼짐
        self.MIN_VOLTAGE = 0.0
        self.MAX_VOLTAGE = 12.0

        self.get_logger().info(
            f'Battery Voltage 서비스 서버 시작 (평균={self.MEAN_VOLTAGE}V, '
            f'표준편차={self.STD_DEV}) — check_voltage 대기 중...'
        )

    def check_voltage_callback(self, request, response):
        voltage = np.random.normal(loc=self.MEAN_VOLTAGE, scale=self.STD_DEV)
        voltage = float(np.clip(voltage, self.MIN_VOLTAGE, self.MAX_VOLTAGE))

        response.voltage = voltage

        self.get_logger().info(f'전압 체크 트리거 수신 → 응답: {voltage:.2f}V')

        return response


def main(args=None):
    rclpy.init(args=args)
    node = BatteryVoltageServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()