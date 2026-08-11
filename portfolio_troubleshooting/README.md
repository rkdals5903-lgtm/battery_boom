# Portfolio Troubleshooting Capture Pack

이 폴더는 최종 구현 코드를 건드리지 않고 포트폴리오용 문제 재현 영상을 찍기 위한 별도 실행 묶음이다.

## 동결 기준

- 최종 실행 코드: `/home/rokey/rokey_d2_gamin_4/main.py`
- 촬영용 래퍼 위치: `portfolio_troubleshooting/runners/`
- PPT 자료: `battery_factory_troubleshooting.pptx`
- 시각자료 원본: `/home/rokey/Videos/Screencasts/*.webm`

## 실행 방식

각 스크립트는 Isaac Sim Python으로 기존 실험 파일 또는 최종 파일을 실행한다.

```bash
cd /home/rokey/rokey_d2_gamin_4
./portfolio_troubleshooting/runners/00_run_final_frozen_demo.sh
./portfolio_troubleshooting/runners/01_capture_conveyor_gate_final.sh
./portfolio_troubleshooting/runners/01_reproduce_screw_path_issue.sh
./portfolio_troubleshooting/runners/02_reproduce_initial_pose_issue.sh
./portfolio_troubleshooting/runners/03_reproduce_ros_service_chain.sh
./portfolio_troubleshooting/runners/04_reproduce_cover_open_integration.sh
./portfolio_troubleshooting/runners/05_reproduce_cell_sorting_issue.sh
```

ROS 2 서비스 체인을 같이 촬영할 때는 별도 터미널에서 필요한 `ros2 service call`을 실행한다.
이 래퍼들은 최종 코드를 수정하지 않는다.

`01_capture_conveyor_gate_final.sh`는 최종 코드 기준의 컨베이어 게이트 동작 촬영용이다.
Action Graph의 과거 실패 상태(KeyError, collider mismatch, Write Prim Attribute 타입 충돌)는
최종 USD를 되돌리지 않고는 그대로 재현하지 않는다.
