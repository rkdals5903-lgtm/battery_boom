# M0609 나사 분해 공정

ROS2 Trigger를 받아 M0609 로봇이 1~4번 나사를 순서대로 분해하는 Isaac Sim 공정입니다.
저장소를 clone/pull한 위치를 기준으로 상대경로를 사용합니다.

## 실행 환경

- Isaac Sim 5.1 계열
- ROS2 Humble (`rclpy`, `std_srvs`)
- NumPy, SciPy

## 실행

터미널 1:

```bash
source /opt/ros/humble/setup.bash
cd <repo>/screw_disassembly
python3 ros_bridge_node.py
```

터미널 2:

```bash
cd <repo>/screw_disassembly
<ISAAC_SIM_PATH>/python.sh run_screw_disassembly.py
```

Isaac Sim 로딩 후 Play를 누르고 터미널 3에서 트리거합니다.

```bash
source /opt/ros/humble/setup.bash
ros2 service call /start_screw_process std_srvs/srv/Trigger
```

## 주요 튜닝값

- 로봇 베이스: world `+Y 0.12 m` 이동 반영
- 전체 상공/작업 Z: `-1 mm`
- 1, 4번 상공/복귀 Z: 추가 `-5 mm`
- 1, 4번 작업 Z: 추가 `-3 mm`
- 3번: 중간 상공 경유, 중간/최종 제한시간 각 120스텝
- 상공 복귀: 허용오차 50 mm, 제한시간 120스텝
- 드라이버: `angle_increment=+0.3` 단방향 회전

## 주의

- 현재 `+0.3`을 풀림 방향으로 사용합니다. 반대로 회전하면 코드에서 `-0.3`으로 바꾸십시오.
- 3번은 제한시간 후 XY 정렬이 완전하지 않아도 작업 위치로 이동하므로 첫 실행 때 접근 궤적을 확인하십시오.
- IPC 파일은 `/tmp/screw_trigger.flag`입니다.
