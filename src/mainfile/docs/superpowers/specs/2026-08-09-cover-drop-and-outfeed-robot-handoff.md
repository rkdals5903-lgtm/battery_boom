# 인계 문서: 뚜껑 분리(casecover) + RG2 셀 분류 통합, 5번째 로봇(출고) 스캐폴딩

작성일: 2026-08-09 (RG2 셀 분류 섹션은 같은 날 대화 후반에 추가)
작성자: Claude(이 대화 세션)
대상 독자: Isaac Sim + 완성된 배터리 모델이 있는 다른 컴퓨터에서 이어서 작업할 사람

**갱신 이력**: 처음에는 "그립퍼가 하는 일" 범위를 뚜껑 분리(VG10)로만 좁혀서
작업했는데, 사용자가 "RG2 그립퍼 동작이 빠졌다"고 지적해서 RG2 셀 검사/분류
(batteryfactory/grip_cell_v4.py, 파이프라인 7/9/10단계)를 추가로 포팅했다.
아래 §4가 그 내용이다.

이 문서는 이번 세션에서 무엇을, 왜 그렇게 결정했는지, 무엇이 검증되지 않았는지를
정리한다. 이 세션 환경에는 `pxr`/Isaac Sim이 없어서(터미널에서 `import pxr` 실패
확인함) **아무 코드도 실행/테스트하지 못했다** — 전부 정적 분석 + 기존 코드 패턴을
근거로 작성했다. 실제 동작 확인은 전부 다른 컴퓨터 몫이다.

## 사용자가 준 요청 원문 (요약)

1. `batteryfactory/`(원본 그립퍼 작업 소스)를 보고, main.py에 합칠 수 있게 컨트롤러만
   뽑아서 넣기. 뚜껑 버리는 코드는 이미 원본에 있는데 main.py에는 없음.
   - 없는 내용이 있으면 만들기 전에 물어보기
   - main.py와 원본이 다르면 원본 우선
   - 초기 자세가 다르면 main.py 초기 자세로 있다가 동작 시 원본 자세로 이동
2. 현재 등록된 로봇 4대(RG2/VG10 작업대/VG10 팔레트/스크류)에 +1 — 컨베이어 벨트
   마지막 부분에서 벨트 -> 팔레트로 옮기는 로봇 추가. 컨트롤러는 1번(VG10PalletNode,
   팔레트->컨베이어) 참고.
   - 좌표는 나중에 입력 가능하도록 파일만 만들고 placeholder만 남기기
   - 서비스 통신은 main.py 기준
   - 컨트롤러를 그대로 쓸 수 있으면 새 컨트롤러 만들지 말고 node만 만들기
3. 끝나면 사고 과정/수정 내용을 memory 같은 곳에 정리해서 위치 알려주기(다른 컴퓨터로
   전달해야 하므로) — 이 문서가 그 결과물이다.

## 대화 중 사용자가 준 추가 답변 (중요)

- "뚜껑 버리는 로직은 다른 컴퓨터에 있어서 그 부분은 공란이나 대충 채워서 넘어가라는
  뜻이었어" — 즉 batteryfactory 폴더의 코드는 **완성본이 아니다.** 진짜 완성된
  버전은 이 세션이 접근할 수 없는 다른 컴퓨터에 있다.
- 기준 소스는 `battery_open_sasumi_assembly_safe.py`로 확정(사용자 선택).
- 배터리 모델은 `batteryfactory/new_file_ready/*.usd`(billow_battery_1~4,
  boom_battery_1~4)로 바꿀 예정이지만 "전처리가 다 안 된 것"이라 애매하고, 다 처리된
  버전은 다른 컴퓨터에 있음.
- USD 참조 교체(팩토리 씬 자체 수정)는 이번에 코드로 처리하지 않기로 함(컨트롤러
  코드만 준비, 실제 배터리 모델 교체는 Isaac Sim에서 사용자가 직접).

**결론: 이번 작업은 "배터리 모델이 교체된 뒤에 자동으로 켜지는" 형태로 만들었다.**
지금 당장 실행해도 배터리 모델이 안 바뀌어 있으면 기존 동작(배터리 전체 폐기)이
그대로 유지되고, 아무것도 깨지지 않는다.

## 조사 과정에서 확인한 사실

- `batteryfactory/`에는 같은 기능의 파일이 여러 버전으로 존재한다
  (`battery_open_sasumi.py` ~ `_v5.py`, `_assembly_safe.py`; `grip_cell.py` ~
  `_v7.py`, `_final_찐찐찐.py` 등). `diff`로 확인한 결과
  `battery_open_sasumi.py` → v2 → v3 → v4 → v5 → `assembly_safe`는 물리 안정성이
  점진적으로 개선되는 선형 진화였고, `assembly_safe`가 전용 배터리 USD
  (`small_cell_battery_staged_meters_assembly_safe.usd`)까지 새로 참조하는 가장
  진화된 버전이었다. `grip_cell` 계열은 반대로 런타임 소스 문자열 치환 방식이라
  버전 관계가 훨씬 복잡했다 — 이번 작업 범위(뚜껑 분리)에는 포함하지 않았다.
- 이 프로젝트는 git 저장소(`rokey_d2/`)이고 `batteryfactory/`는 아직 `git add`되지
  않은 untracked 폴더다(`git status` 확인). 커밋 히스토리에 "뚜껑 여는 소스 추가
  초안"(92d9788), "뚜껑 노드 수정본"(3fb91af, 현재 HEAD)이 있다 — 이번 세션 이전에
  이미 `controller/battery_cover_drop_node.py`에 한 차례 시도가 있었다.
- main.py가 실제로 로드하는 `usd/factory/factory_clean_2.usd`를 PowerShell
  `Select-String`으로 정적 검색한 결과 `good_battery` 문자열이 **아예 없다**
  (배터리는 컨베이어가 런타임에 스폰하는 것으로 추정). `casecover`/`nasa_`는 각각
  1회씩만 나왔다 — 이걸로는 실제 배터리 모델 구조를 확정할 수 없었다(binary USD
  crate라 `strings`/`usdcat`/pxr 전부 이 셸에서 못 씀).
- `batteryfactory/new_file_ready/conversion_report.json`을 읽어서
  `/SmallCellBattery/casecover`, `/casebase`, `/nasa_1~4`, `/cell_1~4` 구조를
  확인했다 — `battery_open_sasumi_assembly_safe.py`가 기대하는 prim 이름과 정확히
  일치한다(defaultPrim=SmallCellBattery 기준 상대 경로).

## 이번에 한 일 (파일별)

### 1) `main.py`

- **뚜껑 분리 관련 상수** (`BATTERY_COVER_SUCTION_PENETRATION_M`,
  `BATTERY_COVER_SOFT_LANDING_TRIGGER_M`, `BATTERY_COVER_SOFT_LANDING_CLEARANCE_M`):
  `battery_open_sasumi_assembly_safe.py`에서 그대로 가져온 값. 좌표 자체(바닥 Z 등)는
  다른 씬 기준이라 베끼지 않고 main.py 기존 값(`BATTERY_DISCARD_POSITION`)을 그대로
  썼다.
- **`BatteryFactoryTask`에 메서드 추가**(기존 `get_battery_screw_prim_paths` 바로
  뒤): `get_battery_casecover_prim_path`, `get_battery_casecover_joint_path`,
  `get_battery_nasa_prim_paths`, `get_battery_casebase_anchor_prim_paths`,
  `has_battery_cover_assembly`, `prepare_battery_cover_physics`,
  `get_battery_casecover_pick_position`, `release_battery_cover_joint`,
  `soft_land_battery_cover`, 그리고 `_last_placed_battery_path` 기준 0-인자 래퍼 3개
  (`get_last_placed_battery_casecover_position`,
  `release_last_placed_battery_cover_joint`,
  `soft_land_last_placed_battery_cover`).
  - **핵심 설계**: 전부 `has_battery_cover_assembly()`(= 배터리 하위에 `casecover`
    prim이 실제로 있는지)를 먼저 확인한다. 없으면 조용히 아무것도 안 하거나(no-op)
    기존 "배터리 전체 위치" 값을 그대로 반환한다. 그래서 아직 배터리 모델을 안
    바꾼 지금도 실행 자체는 그대로 되고, 모델을 바꾸는 순간 자동으로 새 동작이
    켜진다.
- **`_create_scene()`**: 배터리마다 `prepare_battery_cover_physics(battery_path)`
  호출 추가(casecover가 있을 때만 실제로 뭔가 함).
- **`BatteryCoverDropNode` 생성부**: `get_picking_position`을
  `get_last_placed_battery_casecover_position`으로 교체, `release_cover_joint`/
  `soft_land_cover` 콜백 추가.
- **5번째 로봇(VG10 출고) 스캐폴딩**: USD/prim 경로 상수, 물리 설정
  (`ROBOT_PHYSICS_CONFIGS`), `_load_usd`/`_discover_prims`/`_register_scene_objects`에
  다른 VG10들과 동일한 패턴으로 블록 추가, `main()`에 `initialize_robot`
  (최초 1회 + Play 재시작 시 재초기화), `VG10OutfeedNode` 생성, `rclpy.spin_once`/
  `reset_controller`/`destroy_node` 호출 추가.
  - 좌표 상수 5개가 전부 placeholder다(0,0,0 또는 빈 dict/list) —
    `M0609_VG10_OUTFEED_POSITION`, `M0609_VG10_OUTFEED_SURFACE_LOCAL_OFFSET`(잠정
    재사용값), `OUTFEED_SOURCE_PRIM_PATHS`, `OUTFEED_ORDER`,
    `OUTFEED_PALLET_DESTINATION_POSITION`. 전부 `main.py`의 "6-1c"/"6-3" 주석
    섹션에 모아뒀다.

### 2) `controller/battery_cover_drop_node.py`

- 생성자에 `release_cover_joint`, `soft_land_cover` 콜백 파라미터 추가(둘 다
  optional, None이면 기존과 동일하게 동작).
- `_handle_run()` 루프 안에서 `did_pick_succeed()`가 처음 True가 되는 프레임에
  **딱 한 번** `release_cover_joint()`를 호출하도록 추가(원본
  `release_cover_at_contact()`와 같은 타이밍 — 흡착 직후, 들어올리기 전).
- 성공 응답 직전에 `soft_land_cover()` 호출 추가(실패해도 응답 자체는 성공으로 둔다
  — 이미 물체는 떨어진 뒤라 소프트 랜딩 실패가 곧 폐기 실패는 아니라고 판단).
- 나머지 로직(상태머신, 서비스 이름, 흡착 실패 처리 등)은 그대로 뒀다.

### 3) `controller/vg10_outfeed_node.py` (신규 파일)

- `controller/vg10_pallet_node.py`(VG10PalletNode)를 거의 그대로 복제하고 이름만
  방향에 맞게 바꿨다. **컨트롤러 클래스(`SuctionStatePickPlaceController`)는 전혀
  건드리지 않고 그대로 재사용** — 사용자 지시("컨트롤러는 같은 것을 써서 바꾸지
  않아도 된다면 node만 만들기")를 그대로 따른 것.
- 서비스 이름: `/vg10_outfeed/run_belt_to_pallet` (기존 `/vg10_pallet/run_pallet_to_conveyor`
  네이밍 패턴 `<로봇>/run_<출발>_to_<도착>`을 그대로 따름).
- `source_paths`/`order`가 비어 있으면(지금 상태) 서비스를 호출해도 "옮길 대상이
  없습니다"로 안전하게 끝난다 — 아직 완성 케이스 prim 이름 규칙이 없어서 채우지
  못했다.

## 4) RG2 셀 검사/분류 (batteryfactory/grip_cell_v4.py 이식)

사용자가 지적한 뒤 추가로 진행한 부분. `batteryfactory/grip_cell_final_찐찐찐.py`("최종본")를
먼저 열어봤는데, 이건 단순한 상태머신이 아니라 **뚜껑 작업(구버전) + RG2 셀 이송 +
외부 카메라 검사 요청 + 전압 라우팅 + 팔레트 로봇 트리거**까지 하나로 합친 670줄짜리
절차형 스크립트였고, `ros2 service call`을 리눅스 bash subprocess로 직접 쏘는 방식
(`/opt/ros/humble/setup.bash` 등 Linux 전용 경로 하드코딩)을 썼다. 게다가 그 안에
embedded된 뚜껑 로직은 `assembly_safe`가 아니라 그 이전 구버전이었다. 그래서
`grip_cell_final_찐찐찐.py`를 그대로 옮기지 않고, 그 안에 base64+gzip으로 통째로
embedded돼 있던 `grip_cell_v4.py`(diff로 확인 — 디스크의 `grip_cell_v4.py`와
완전히 동일함)의 **실제 상태머신 로직**만 뽑아서 main.py 패턴으로 새로 작성했다.
확정 기준 버전: `grip_cell_v4.py`(로직) + `grip_cell_final_찐찐찐.py` 상단의 소수
튜닝 상수(둘 다 값이 사실상 동일 — 겹치는 부분은 문제 없음).

### 새 파일: `controller/rg2_cell_sort_node.py`

- `RG2CellSortNode`: 서비스 `/start_cell_sorting`. cell_1~4를 하나씩 집어
  검사 위치로 옮기고, `/battery_inspection_result`(std_srvs/Trigger) 응답에 따라
  new_case에 쌓거나 반려 위치에 떨어뜨린다. 4개 다 성공하면
  `/start_case_close`(아직 구현 안 된 다음 단계용 placeholder 서비스)를
  fire-and-forget으로 호출한다.
- `_LinkFollower`: 원본 `PhysicsLinkCellFollower`를 이식한 kinematic 캐리 헬퍼.
  **단순화**: 원본은 non-physical 시각 프록시(별도 소스 USD 필요)를 만들어 그걸
  대신 들고 다녔는데, main.py에는 그 프록시용 소스 USD가 없다. 대신 실제 cell
  rigid prim을 직접 kinematic으로 구동한다 — cell은
  `BatteryFactoryTask.prepare_battery_cover_physics()`에서 이미 kinematic으로
  잡혀 있으므로 추가 설정 없이 바로 가능했다.
- 그리퍼는 main.py에 이미 등록된 RG2 `ParallelGripper`(finger_joint/
  right_inner_knuckle_joint 2개만 구동)를 안 쓰고, 원본처럼 6개 mimic 관절을
  전부 직접 `ArticulationAction`으로 명령한다(`_command_gripper`) — 실제 접촉
  판정(`accept_contact`, contact_min_rad 등)이 필요한 작업이라 원본과 동일한
  방식을 유지했다.
- 이동은 `SuctionStatePickPlaceController`를 재사용하지 않고 새로 만들었다
  (`_move`, RMPFlowController 직접 사용) — RG2는 SurfaceGripper가 아니라
  ParallelGripper 계열이고 픽/검사/배치의 단계 구성 자체가 VG10 계열과 달라서
  기존 상태머신 클래스에 끼워 맞추기 애매했다. `screw_disassembly_node.py`가
  이미 "이 프로젝트 전용 절차형 로직은 Node 안에 직접 둔다"는 선례라 그 패턴을
  따랐다.
- **단순화(원본 대비)**: 원본은 이동 구간마다 orientation을 정교하게
  구분했다(위치만 추종/측정된 재파지 orientation 유지 등). 여기서는 셀 하나를
  처리하는 동안 고정 orientation(아래를 향하고 90도 yaw) 하나만 쓴다 — 좌표
  자체가 placeholder라 지금 정교하게 맞춰봐야 의미가 없어서 단순화했다. 실제
  좌표가 정해지면 필요 시 원본처럼 세분화해야 할 수 있다.
- 검사 결과 대기(`_request_inspection_result`): 원본은 bash subprocess로 CLI를
  불렀지만, 이 노드는 자체 rclpy 클라이언트로 `call_async()` 후
  `rclpy.spin_once(self, timeout_sec=0.0)` + `world.step()`을 번갈아 돌며
  기다린다. **이 nested spin_once 호출이 실제로 안전한지는 미검증이다** — 이
  콜백 자체가 main.py 메인 루프의 `rclpy.spin_once(rg2_cell_sort_node, ...)` 안에서
  실행되는 중이라, 그 안에서 같은 노드를 다시 스핀하는 형태다. 이론상
  `rclpy.spin_once()`가 매번 임시 executor를 새로 만들고 버리는 방식이라
  단일 스레드 안에서 중첩 호출해도 안전할 것으로 보이지만, Isaac Sim + rclpy
  조합에서 직접 확인된 적은 없다. 문제가 있으면 별도 스레드에서 도는 executor로
  바꾸거나, `MultiThreadedExecutor`를 쓰는 방향을 검토해야 한다.
- RG2 손가락 콜라이더(`add_rg2_fingertip_proxy_colliders`, main.py의
  `BatteryFactoryTask` 메서드): 원본과 동일한 로직이지만 대상 로봇 경로만
  `M0609_RG2_PRIM_PATH`로 바꿨다. 같은 `m0609_camera_cube.usd` 에셋을 참조하므로
  내부 구조(`Xform/m0609_camera/m0609/onrobot_rg2ft/...`)가 같을 것으로
  추정했지만 실제로 확인은 못 했다 — 실패해도 예외를 잡아 경고만 남기고 계속
  진행하도록 방어적으로 짰다.

### main.py 변경

- "6-4" 섹션에 RG2 그리퍼 관절 상수(하드웨어 고유값이라 그대로 이식, 실측 불필요)와
  `RG2_CELL_INSPECTION_POSITION`/`RG2_CELL_NEW_CASE_POSITION`/`RG2_CELL_REJECT_POSITION`
  (전부 `[0,0,0]` placeholder — 검사대/new_case 오브젝트 자체가 아직 씬에 없다) 추가.
- `BatteryFactoryTask`에 `get_last_placed_battery_path`, `get_battery_cell_prim_path`,
  `get_battery_cell_joint_path`, `has_cell_sorting_ready`, `release_battery_cell_joint`,
  `set_battery_cell_carry_collision_filter`, `get_battery_cell_pick_position`,
  `add_rg2_fingertip_proxy_colliders` 추가. `release_battery_cover_joint`가 쓰던
  세션 레이어 조인트 비활성화 로직은 `_deactivate_joint()`로 뽑아서 셀 조인트
  해제와 공유하도록 리팩터링했다.
- `battery_cover_drop_node.py`가 폐기(뚜껑 분리)에 성공하면 `/start_cell_sorting`을
  트리거하도록 `_trigger_cell_sorting()` 추가 — **중요**:
  `task.clear_last_placed_battery()`를 부르기 **전에** 트리거해야 한다(순서를
  반대로 하면 RG2CellSortNode가 배터리 경로를 조회할 시점엔 이미 비어 있어서
  실패한다) — 코드에도 이 순서를 명시하는 주석을 남겼다.
- `main()`에 `RG2CellSortNode` 생성, `robot`(RG2 SingleManipulator, 기존에
  등록만 되고 실제로 쓰이지 않던 것) 전달, spin_once/reset_controller/destroy_node
  호출 추가.

## 검증 여부 정리 (중요 — 다른 컴퓨터에서 반드시 확인)

| 항목 | 상태 |
|---|---|
| Python 문법(`py_compile`) | ✅ 확인함(이 세션에서) |
| Isaac Sim에서 실제 실행 | ❌ 이 세션은 Isaac Sim 자체가 없어서 전혀 실행 못 함 |
| `casecover`/`casebase`/`nasa_N`/`AssemblyJoints/casecover_to_casebase` 경로 규칙이 실제 교체된 배터리와 일치하는지 | ❌ 미검증. `new_file_ready/conversion_report.json` 기준 추정치. 실제 배터리를 `factory_clean_2.usd`(또는 그 후속)에 넣었을 때 `{battery_path}/casecover`처럼 단순 자식으로 붙는지, 아니면 `good_battery/tn__Part19_g6`처럼 한 단계 더 중첩되는지 확인 필요 |
| `soft_land_battery_cover()`의 `SingleRigidPrim(prim_path=...)`를 `scene.add()` 없이 임시로 만들어 `set_world_pose`/`set_linear_velocity` 호출하는 게 실제로 동작하는지 | ❌ 미검증. 이 프로젝트의 다른 모든 `SingleRigidPrim` 사용처는 `scene.add()`로 등록한 뒤 쓴다 — 여기서는 배터리마다 매번 새로 만들어야 해서 그렇게 안 했다. 물리 뷰 바인딩이 안 되면 예외가 나거나 조용히 무시될 수 있음 |
| `BATTERY_COVER_SOFT_LANDING_*` 상수, `max_steps=240` | ❌ 미검증. 원본은 physics_dt=1/120 기준 3초였는데, main.py의 실제 physics_dt는 확인 안 됨(World 기본값 사용 추정) — 다시 튜닝 필요할 수 있음 |
| VG10 출고 로봇 좌표 5종 전부 | ❌ placeholder(0,0,0 / 빈 값). 반드시 실측 필요 |
| `get_battery_casecover_pick_position`의 yaw 처리 | ⚠️ 배터리 base 전체의 bbox로 yaw를 계산해서 casecover에도 그대로 적용(casecover가 배터리와 같은 방향으로 놓인다는 가정) — 근사치. 정밀하게 하려면 casecover 자체 bbox로 yaw를 다시 계산하는 게 나을 수 있음 |
| RG2 검사대/new_case/반려 위치 3종 | ❌ placeholder(0,0,0). 씬에 검사대·new_case 오브젝트 자체가 없음 — 오브젝트 배치부터 필요 |
| `add_rg2_fingertip_proxy_colliders`의 `gripper_root` 경로(`.../onrobot_rg2ft`) | ❌ 미검증. main.py의 RG2도 같은 `m0609_camera_cube.usd`를 참조하니 구조가 같을 것으로 추정했을 뿐, 실제 Stage에서 확인 못 함 |
| `RG2CellSortNode._request_inspection_result()`의 `rclpy.spin_once(self, ...)` 중첩 호출 | ❌ 미검증. 콜백 안에서 같은 노드를 다시 스핀하는 형태라 이론상은 안전해야 하지만 실제 rclpy+Isaac Sim 조합에서 확인 안 됨. 응답을 못 받고 멈추면 여기부터 의심 |
| `_LinkFollower`가 실제 cell rigid prim을 kinematic으로 직접 구동하는 방식(원본의 non-physical 프록시 생략) | ❌ 미검증. 물리적으로 이상하게 보이거나(다른 셀과 겹침 등) 충돌 필터가 예상과 다르게 동작할 수 있음 |
| RG2 6관절 직접 구동이 main.py의 기존 `ParallelGripper`(2관절만 구동) 설정과 충돌하지 않는지 | ❌ 미검증. 서로 다른 시점에 순차 호출되므로 이론상 문제 없어야 하지만 실제 확인 필요 |

## 다른 컴퓨터에서 할 일 (우선순위 순)

1. 완성된 배터리 모델(casecover/casebase/nasa_1~4 구조)을 `factory_clean_2.usd`
   (또는 그 후속 씬)의 배터리 참조로 실제로 연결한다. 연결 후
   `has_battery_cover_assembly()`가 True를 반환하는지, 그리고
   `get_battery_casecover_prim_path()`가 가리키는 경로가 실제 존재하는지부터
   확인한다 — 여기서 어긋나면 위 표의 "경로 규칙" 항목을 고쳐야 한다.
2. `/start_battery_cover_drop` 서비스를 한 번 호출해서 casecover만 분리돼 들어
   올려지는지, 바닥에 소프트 랜딩하는지 직접 눈으로 확인한다. 안 되면 위 표의
   "SingleRigidPrim" 항목부터 의심한다.
3. `M0609_VG10_OUTFEED_*`, `OUTFEED_*` 상수를 Isaac Sim에서 실측해 채운다.
   `OUTFEED_SOURCE_PRIM_PATHS`/`OUTFEED_ORDER`는 완성 케이스가 실제로 어떤
   이름으로 컨베이어 위에 나타나는지가 정해져야 채울 수 있다(아직 그 파이프라인
   자체가 미구현).
4. `/vg10_outfeed/run_belt_to_pallet` 서비스를 누가/언제 호출할지(트리거 방식)
   결정한다 — 지금은 아무도 자동으로 호출하지 않는다.
5. RG2 검사대/new_case 오브젝트를 씬에 배치하고 `RG2_CELL_INSPECTION_POSITION`/
   `RG2_CELL_NEW_CASE_POSITION`/`RG2_CELL_REJECT_POSITION`을 실측해 채운다.
6. `/start_cell_sorting`을 한 번 호출해서 셀 픽업/검사 위치 이동/재파지까지
   되는지 확인한다. `/battery_inspection_result` 서비스(외부 컴퓨터 YOLO/전압
   연동)가 아직 없으면 `_request_inspection_result()`가 "서비스 없음 -> false로
   대체" 경로를 타므로, 이 상태에서는 모든 셀이 반려 처리된다 — 정상이다.
7. RG2 nested `rclpy.spin_once` 방식이 실제로 응답을 받아오는지 확인한다(위
   검증 표 참고). 문제가 있으면 이 부분부터 다시 설계해야 할 수 있다.
