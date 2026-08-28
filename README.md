## battery_boom

Isaac Sim 기반의 배터리 팩 재제조·검사 공정 디지털 트윈 프로젝트입니다.
컨베이어로 투입된 배터리를 다중 Doosan M0609 로봇이 순차 처리하며, 흡착(VG10)·RG2·스크류드라이버 엔드이펙터로 배터리 이송, 나사 분해, 커버 분해, 셀 추출, CNN 외형 검사, 정상 셀 적재, 나사 재체결, 케이스 배출까지 하나의 시나리오로 통합합니다.

## 프로젝트 개요

`battery_boom`은 Isaac Sim world와 ROS 2 노드들을 하나의 프로세스로 묶어 배터리 공장 라인을 자동 재현하는 워크스페이스입니다.

주요 처리 흐름은 다음과 같습니다.

1. `main.py`가 SimulationApp을 띄우고 ROS 2 bridge·conveyor·surface gripper 확장을 활성화
2. factory USD와 5대의 M0609 로봇 USD를 `/World` 아래에 로드하고 World·RMPFlow·컨트롤러 초기화
3. VG10 로봇이 팔레트의 배터리를 컨베이어로 적재
4. 컨베이어 게이트가 배터리 bbox 겹침을 감지하면 벨트 정지 후 작업대 이송 트리거
5. 스크류 로봇이 나사 4개를 분해하고, 작업대 VG10이 커버를 흡착 분리해 바닥에 투하
6. RG2 로봇이 셀을 하나씩 추출하며 전압 샘플링과 CNN `inspect_cell` 서비스 호출
7. 정상 셀은 새 케이스 슬롯에 적재, 불량 셀은 폐기
8. 새 케이스 커버를 닫고 나사를 재체결한 뒤 완성 케이스를 컨베이어로 배출
9. 출고 로봇이 완성 케이스를 출고 팔레트로 이송하고 다음 배터리를 대기

## 주요 기능

* Isaac Sim factory USD 로드 및 5대 M0609 로봇 배치 (RG2/카메라, VG10 작업대, 스크류, VG10 팔레트, VG10 출고)
* 컨베이어 벨트 ActionGraph 제어와 bbox 게이트 기반 공정 전환
* `std_srvs/Trigger` 서비스 체인으로 로봇 작업 순서 직렬화
* 나사 분해/체결 상태머신 (HOME_ALIGN → MOVE_WAYPOINT → APPROACH → STABILIZE → SCREW → RETRACT → RETURN_HOME)
* 조립 조인트(casecover_to_casebase 등) 활성/비활성으로 커버·셀 분리 시뮬레이션
* ResNet34 기반 셀 외형 CNN 검사와 전압 검사 결과를 적재/폐기 판정에 반영
* `/rosout` 로그를 구독해 공정 단계와 검사 결과를 시각화하는 웹 모니터링 UI
* 반복 Play/Stop·재실행 시 동일한 초기 상태 재현

## 설치 방법

저장소를 clone합니다.

```bash
git clone https://github.com/rkdals5903-lgtm/battery_boom.git
cd battery_boom
```

ROS 2 환경을 적용합니다.

```bash
source /opt/ros/humble/setup.bash
```

Python 의존성을 설치합니다.

```bash
pip install numpy scipy pyyaml pillow torch torchvision opencv-python
```

`isaacsim`, `omni`, `pxr`, `rclpy`, `sensor_msgs`, `std_srvs` 계열은 pip 대상이 아니라 Isaac Sim / ROS 2 Humble 환경에서 제공됩니다. UI 서버는 Python 표준 라이브러리와 ROS 2의 `rclpy`, `rcl_interfaces`만 사용합니다.

Isaac Sim 실행 경로를 환경 변수로 지정합니다.

```bash
export ISAAC_SIM_PYTHON=/path/to/isaacsim/python.sh
```

## 실행 방법

세 개의 터미널에서 각각 실행합니다.

CNN 외형검사 노드:

```bash
./scripts/run_cnn_inspection.sh
```

Isaac Sim 메인 시뮬레이션:

```bash
./scripts/run_main_isaac.sh
```

공정 모니터링 UI (기본 포트 8107):

```bash
./src/ui/run_ui.sh 8107
```

개별 실행도 가능합니다.

```bash
source /opt/ros/humble/setup.bash
python3 src/cnn/cell_inspection_node.py
$ISAAC_SIM_PYTHON src/mainfile/main.py
```

브라우저에서 UI에 접속합니다.

```text
http://127.0.0.1:8107
```

## 기본 실행 흐름

1. `cell_inspection_node` 실행 (모델 로드 후 `/inspect_cell` 서비스 대기)
2. `main.py` 실행 → Isaac Sim world와 모든 controller 노드가 한 프로세스에서 spin
3. UI 서버 실행 → `/rosout` 구독 시작, 브라우저에서 상태 확인
4. `VG10PalletNode`가 배터리를 컨베이어로 적재
5. 컨베이어 게이트 감지 → 벨트 정지 → `/vg10_worktable/run_pick_place` 호출
6. `/start_screw_process` → 나사 분해 → `/start_battery_cover_drop` → 커버 분리·투하
7. `/start_grip_cell_process` → 셀별 `/check_voltage` + `/inspect_cell` → 정상 적재 / 불량 폐기
8. `/suction_cover_close` → 새 케이스 커버 닫기
9. `/start_screw_tightening` → 나사 재체결 → `/start_case_outfeed` → 완성 케이스 컨베이어 이송
10. `/vg10_outfeed/run_belt_to_pallet` → 출고 팔레트 이송, 다음 배터리 대기

## 패키지 구조

```text
battery_boom/
├── README.md
├── scripts/
│   ├── run_cnn_inspection.sh
│   └── run_main_isaac.sh
└── src/
    ├── basic/
    ├── cnn/
    │   ├── cell_inspection_node.py
    │   ├── cell_inspection_node_fixed_crop.py
    │   ├── cell_inspection_node_rqt_visualization.py
    │   └── cell_dataset/
    │       └── cell_classifier_final.pt
    ├── ui/
    │   ├── index.html
    │   ├── main.js
    │   ├── style.css
    │   ├── server.py
    │   └── run_ui.sh
    └── mainfile/
        ├── main.py
        ├── test_vg10_hyunwoo2_fixed.py
        ├── AI_COLLABORATION.md
        ├── controller/
        │   ├── m0609_rmpflow_controller.py
        │   ├── pick_place_controller.py
        │   ├── vg10_suction_pick_place_controller.py
        │   ├── screw_control.py
        │   ├── vg10_pallet_node.py
        │   ├── vg10_worktable_node.py
        │   ├── vg10_outfeed_node.py
        │   ├── screw_disassembly_node.py
        │   ├── screw_tightening_node.py
        │   ├── battery_cover_drop_node.py
        │   ├── suction_cover_close_node.py
        │   ├── grip_cell_node.py
        │   ├── battery_voltage_server.py
        │   └── case_outfeed_node.py
        ├── docs/
        ├── rmpflow/
        │   ├── m0609_description.yaml
        │   └── m0609_rmpflow_common.yaml
        ├── urdf/
        │   └── m0609_isaac_sim.urdf
        └── usd/
            ├── factory/
            ├── m0609/
            ├── M0609_raw/
            ├── doosan-robot2/
            └── onrobot_rg2/
```

## 노드 설명

### `main.py` (오케스트레이터)

`BatteryFactoryTask` 하나로 USD 로드·Prim 탐색·Physics 설정·Scene 등록을 담당하고, controller 노드들을 같은 프로세스에서 생성해 단일 spin loop로 돌립니다.

주요 역할:

* SimulationApp 생성 및 `isaacsim.ros2.bridge`, `isaacsim.robot.surface_gripper`, `isaacsim.asset.gen.conveyor`, `omni.replicator.core`, `omni.physx.graph` 확장 활성화
* factory / 로봇 / 작업대 USD를 `/World` 아래에 참조로 로드
* World, RMPFlow, 각 로봇 컨트롤러 초기화
* 컨베이어 벨트 속도 제어와 bbox 게이트 감지
* 서비스 체인의 시작점 트리거 및 노드 간 콜백 연결

### `cell_inspection_node`

`/worktable_top_rgb`, `/worktable_side_rgb` 카메라 토픽을 상시 구독해 최신 프레임을 유지하다가, `inspect_cell` 서비스 호출 시점의 top/side 프레임을 CNN에 넣어 정상/비정상을 판정합니다.

주요 역할:

* ResNet34(activation=relu, kernel_size=3, 2-class) 모델을 시작 시 1회 로드
* 입력 이미지 96×96 resize + 정규화
* top/side 각각 추론 후 신뢰도 임계값으로 종합 판정
* `std_srvs/Trigger` 응답으로 결과 반환 (`success=True` → 정상)
* 디버그 이미지 저장 지원

### `battery_voltage_server`

`/check_voltage` 서비스로 셀 전압을 샘플링하고 임계 전압과 비교합니다. `main.py` 내부에서는 deadlock을 피하기 위해 `sample_voltage()`를 직접 호출할 수도 있습니다.

### `vg10_pallet_node` (`VG10PalletNode`)

`/vg10_pallet/run_pallet_to_conveyor` 서비스로 팔레트의 배터리를 컨베이어로 적재합니다. 검증된 `SuctionStatePickPlaceController` 상태머신을 재사용하며, 서비스 호출 한 번당 배터리 한 개만 옮깁니다.

### `vg10_worktable_node` (`VG10WorktableNode`)

`/vg10_worktable/run_pick_place` 서비스로 컨베이어 → 작업대 Pick & Place를 수행합니다. 서비스가 호출되면 완료(또는 world 정지)까지 컨트롤러 `forward()`와 `world.step()`을 내부에서 반복한 뒤 응답하고, 이어서 `/start_screw_process`를 호출합니다.

### `screw_disassembly_node` (`ScrewDisassemblyNode`)

`/start_screw_process` 서비스로 작업대 배터리의 나사 4개를 분해합니다. HOME_ALIGN → MOVE_WAYPOINT → APPROACH → STABILIZE → SCREW → RETRACT → RETURN_HOME 상태머신으로 동작하고, 완료 후 `/start_battery_cover_drop`을 호출합니다.

### `battery_cover_drop_node` (`BatteryCoverDropNode`)

`/start_battery_cover_drop` 서비스로 작업대 VG10이 `casecover`를 흡착합니다. 흡착 시작 순간 `casecover_to_casebase` 조인트를 비활성화해 커버(+매달린 나사)만 분리해 들어올린 뒤 공장 바닥에 투하합니다.

### `grip_cell_node` (`GripCellNode`)

`/start_grip_cell_process` 서비스로 RG2 로봇이 셀을 하나씩 추출합니다. `main.py`가 만든 `world`와 RG2 `robot`을 주입받아 새로 생성하지 않습니다.

주요 역할:

* 셀 추출 후 `/check_voltage`와 `/inspect_cell` 서비스 호출
* 전압·CNN 결과를 종합해 `[INSPECTION FINAL]` 판정
* 정상 셀은 새 케이스 슬롯에 적재, 불량 셀은 폐기
* 4개 셀 완료 후 `/suction_cover_close` 호출

### `suction_cover_close_node` (`SuctionCoverCloseNode`)

`/suction_cover_close` 서비스로 새 케이스의 커버를 닫고, 완료 후 `/start_screw_tightening`을 호출합니다.

### `screw_tightening_node` (`ScrewTighteningNode`)

`/start_screw_tightening` 서비스로 새 케이스의 나사 4개를 체결합니다. 나사 분해와 동일한 상태머신을 쓰고 회전 방향만 반대이며, 같은 스크류 로봇·공구를 재사용합니다. 완료 후 `/start_case_outfeed`를 호출합니다.

### `case_outfeed_node` (`CaseOutfeedNode`)

`/start_case_outfeed` 서비스로 완성 케이스를 흡착 로봇으로 컨베이어까지 옮깁니다. 조립이 끝나는 순간 진짜 조립체를 비활성화하고 `good_battery.usd` 단일 dynamic 프록시로 스왑합니다.

### `vg10_outfeed_node` (`VG10OutfeedNode`)

`/vg10_outfeed/run_belt_to_pallet` 서비스로 컨베이어 마지막 구간 → 출고 팔레트 이송을 수행합니다. `VG10PalletNode`의 상태머신을 방향만 반대로 재사용합니다. (일부 좌표는 TODO 상태)

### UI 서버 (`src/ui/server.py`)

`/rosout`의 `rcl_interfaces/msg/Log`를 구독해 로그 패턴에서 공정 단계·전압·검사 결과를 파싱하고, `GET /api/status`로 브라우저에 최신 상태를 제공합니다. 정적 UI와 공정 애니메이션은 `index.html`, `main.js`, `style.css`에서 렌더링합니다. UI는 상태 확인 전용이며 로봇/시뮬레이션에 제어 명령을 보내지 않습니다.

## 인터페이스

### Service

모든 공정 서비스는 커스텀 `.srv` 없이 `std_srvs/Trigger`를 사용합니다.

| 서비스 | 제공 노드 | 역할 |
| --- | --- | --- |
| `/inspect_cell` | `cell_inspection_node` | CNN 셀 외형 검사 (`success=True` → 정상) |
| `/check_voltage` | `battery_voltage_server` | 셀 전압 샘플링 및 임계값 비교 |
| `/vg10_pallet/run_pallet_to_conveyor` | `vg10_pallet_node` | 팔레트 → 컨베이어 적재 |
| `/vg10_worktable/run_pick_place` | `vg10_worktable_node` | 컨베이어 → 작업대 이송 |
| `/start_screw_process` | `screw_disassembly_node` | 나사 분해 |
| `/start_battery_cover_drop` | `battery_cover_drop_node` | 커버 흡착 분리·투하 |
| `/start_grip_cell_process` | `grip_cell_node` | 셀 추출·전압·CNN·분류 |
| `/suction_cover_close` | `suction_cover_close_node` | 새 케이스 커버 닫기 |
| `/start_screw_tightening` | `screw_tightening_node` | 새 케이스 나사 체결 |
| `/start_case_outfeed` | `case_outfeed_node` | 완성 케이스 → 컨베이어 |
| `/vg10_outfeed/run_belt_to_pallet` | `vg10_outfeed_node` | 컨베이어 → 출고 팔레트 |

### Topic

| 토픽 | 타입 | 방향 | 용도 |
| --- | --- | --- | --- |
| `/worktable_top_rgb` | `sensor_msgs/Image` | 구독 | 셀 상단 카메라 (CNN 입력) |
| `/worktable_side_rgb` | `sensor_msgs/Image` | 구독 | 셀 측면 카메라 (CNN 입력) |
| `/rosout` | `rcl_interfaces/msg/Log` | 구독 | UI 서버 공정 상태 파싱 |

### HTTP (UI)

| 엔드포인트 | 용도 |
| --- | --- |
| `GET /` | 정적 UI (`index.html`) |
| `GET /api/status` | 최신 공정 단계·전압·검사 결과 JSON |

## 개발 환경

권장 환경:

* Ubuntu 22.04 LTS
* ROS 2 Humble
* Isaac Sim 5.x 계열
* Python 3.10 (ROS 2) / Isaac Sim 번들 Python 3.11
* NVIDIA GPU 권장

필요 Python 패키지:

* `numpy`
* `scipy`
* `pyyaml`
* `pillow`
* `torch`
* `torchvision`
* `opencv-python`

ROS 2 패키지는 표준 패키지만 사용합니다.

* `rclpy`
* `std_srvs`
* `std_msgs`
* `sensor_msgs`
* `geometry_msgs`
* `rcl_interfaces` (UI)

## 환경 변수 설정

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `ISAAC_SIM_PYTHON` | `/home/rokey/.../python.sh` | Isaac Sim `python.sh` 경로 |
| `CELL_INSPECTION_DEVICE` | `cpu` | CNN 추론 디바이스 (`cpu` / `cuda`) |
| `CELL_INSPECTION_DEBUG_DIR` | `src/cnn/inspection_debug` | 검사 디버그 이미지 저장 경로 |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` | ROS 2 미들웨어 구현 |
| `ROS_DOMAIN_ID` | (미설정) | 시뮬레이션과 UI가 같은 값을 써야 함 |
| `ROS_LOCALHOST_ONLY` | (미설정) | 멀티 호스트 사용 시 `0` |
| `BATTERY_PROJECT_LOG` | (미설정) | UI가 추가로 읽을 로컬 로그 파일 경로 |

## 시뮬레이션 에셋

`main.py`가 로드하는 주요 경로 (`PROJECT_DIR = src/mainfile`):

| 상수 | 경로 |
| --- | --- |
| `FACTORY_USD_PATH` | `usd/factory/factory_clean_2.usd` |
| `M0609_RG2_USD_PATH` | `usd/m0609/m0609_camera_cube.usd` |
| `M0609_VG10_USD_PATH` | `usd/m0609/m0609_vg10_cube.usd` |
| `M0609_SCREW_USD_PATH` | `usd/m0609/m0609_screw_cube.usd` |
| `WORK_TABLE_USD_PATH` | `usd/factory/Collected_new_work_table/work_table.usd` |
| `GOOD_BATTERY_PROXY_USD_PATH` | `usd/factory/good_battery.usd` |
| `M0609_URDF_PATH` | `urdf/m0609_isaac_sim.urdf` |
| `M0609_DESCRIPTION_PATH` | `rmpflow/m0609_description.yaml` |
| `M0609_RMPFLOW_CONFIG_PATH` | `rmpflow/m0609_rmpflow_common.yaml` |

`FACTORY_ROOT_PRIM_PATH`는 `/World`이며, 컨베이어 정지 시간은 `CONVEYOR_STOP_DURATION_S`(기본 16초), 벨트 속도는 `CONVEYOR_RUN_VELOCITY`(기본 1.0)로 조정합니다.

## Dry-run 및 반복 실행

* CNN 노드를 먼저 띄워 `/inspect_cell`이 준비된 상태에서 `main.py`를 실행합니다.
* UI는 상태 확인 전용이므로, 시뮬레이션 없이 UI만 띄워 `/api/status` 연결과 로그 파싱을 먼저 확인할 수 있습니다.
* Isaac Sim에서 Play/Stop을 반복해도 동일한 초기 상태에서 시나리오가 재현되도록 설계되어 있습니다.

## 디버깅

* `cell_inspection_node`는 `CELL_INSPECTION_DEBUG_DIR`에 판정에 사용한 top/side 프레임을 저장합니다.
* UI 서버가 파싱하는 로그 패턴:
  * `[VOLTAGE] cell_N: X V / threshold=Y V -> TRUE|FALSE`
  * `[INSPECTION FINAL] cell_N: voltage_ok=..., cnn_ok=... -> TRUE|FALSE`
  * `[NEW CASE SLOT VERIFY] accepted_slot=N`
* 공정이 넘어가지 않으면 해당 단계 서비스(`/start_*`)가 응답을 반환했는지, 다음 노드가 클라이언트로 대기 중인지 확인합니다.

## 주의 사항

* `scripts/run_main_isaac.sh`의 실행 경로가 이전 구조 기준(`src/rokey_d2_gamin_4/main.py`)으로 남아 있을 수 있습니다. 현재 메인 파일은 `src/mainfile/main.py`이므로 스크립트의 경로를 맞춰야 합니다.
* Isaac Sim 번들 Python은 3.11, 시스템 ROS 2 Humble `rclpy`는 3.10용입니다. `main.py`는 시작 시 `sys.path`에서 Python 3.10 전용 ROS 경로를 제거하고 번들 `rclpy`를 우선하도록 처리합니다. 셸에서 `/opt/ros/humble/setup.bash`를 source한 뒤 Isaac Sim을 띄우는 순서를 지킵니다.
* 시뮬레이션과 UI가 같은 ROS 2 그래프를 쓰도록 `ROS_DOMAIN_ID` / `ROS_LOCALHOST_ONLY`를 맞춥니다.
* `vg10_outfeed_node`는 좌표·이름 규칙 일부가 TODO 상태입니다. 실제 출고 동작 전에 좌표를 채워야 합니다.
* 실제 로봇으로 옮길 경우 로봇/작업대 좌표 보정, z축 pen-up/pen-down 높이, 속도·가속도 제한, emergency stop 동작, 작업 영역 범위를 먼저 확인합니다.
* `src/cnn/cell_dataset/*.pt` 모델 파일과 `usd/` 텍스처가 커서 저장소 용량이 큽니다. 필요 시 Git LFS로 옮기는 것을 권장합니다.
