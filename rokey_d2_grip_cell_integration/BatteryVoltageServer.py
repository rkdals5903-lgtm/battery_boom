#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main.py 통합 가능한 Battery Voltage ROS2 node.

기존 파일은 custom CheckVoltage.srv 타입 import가 TODO 상태라 create_service()
인자 자체가 완성되지 않았다. 통합본은 별도 custom srv 생성 없이 사용할 수 있게
std_srvs/Trigger로 /check_voltage를 제공한다.

- main.py 내부 GripCellNode는 같은 프로세스에서 ``sample_voltage()``를 직접 호출한다.
  (단일 spin loop 안에서 자기 프로세스의 ROS service를 동기 호출해 deadlock 나는 것을
   피하기 위함.)
- 외부 테스트 노드는 /check_voltage Trigger를 호출할 수 있고, 전압은
  response.message에 문자열로 들어간다.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class BatteryVoltageServer(Node):
    MEAN_VOLTAGE = 11.0
    STD_DEV = 1.5
    MIN_VOLTAGE = 0.0
    MAX_VOLTAGE = 12.0

    def __init__(self, service_name: str = "/check_voltage") -> None:
        super().__init__("battery_voltage_server")
        self._service = self.create_service(Trigger, service_name, self._handle_check)
        self.last_voltage = None
        self.get_logger().info(
            f"Battery Voltage node ready: {service_name}, "
            f"range=[{self.MIN_VOLTAGE:.1f}, {self.MAX_VOLTAGE:.1f}] V, "
            f"mean={self.MEAN_VOLTAGE:.1f} V"
        )

    @property
    def decision_voltage(self) -> float:
        """통합 GripCellNode가 정상/불량 경계로 사용하는 분포 중심값."""
        return float(self.MEAN_VOLTAGE)

    def sample_voltage(self) -> float:
        voltage = np.random.normal(loc=self.MEAN_VOLTAGE, scale=self.STD_DEV)
        voltage = float(np.clip(voltage, self.MIN_VOLTAGE, self.MAX_VOLTAGE))
        self.last_voltage = voltage
        self.get_logger().info(f"전압 생성: {voltage:.3f} V")
        return voltage

    def _handle_check(self, request, response) -> Trigger.Response:
        voltage = self.sample_voltage()
        response.success = True
        response.message = f"{voltage:.6f}"
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BatteryVoltageServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
