# gamin-2 Grip Cell 통합 패치

## 배치 위치

아래 파일을 `rokey_d2`에 덮어쓰거나 추가한다.

```text
rokey_d2/
├─ main.py                         # 이 패치의 main.py
├─ BatteryVoltageServer.py         # 수정된 voltage node
└─ controller/
   ├─ grip_cell_node.py            # 신규
   └─ battery_cover_open_node.py   # 신규
```

기존 `controller/battery_cover_drop_node.py`는 남아 있어도 되지만 새 main.py에서는 import하지 않는다.
`grip_cell_final.py`, `battery_open_sasumi.py`, `mock_server.py`,
`mock_inspection_true.py`, `mock_inspection_false.py`는 새 통합 경로에서 사용하지 않는다.

## 공정 연결

```text
VG10 worktable 완료
 -> /start_screw_process
 -> ScrewDisassemblyNode
 -> /start_battery_cover_drop        (기존 서비스 이름 유지)
 -> BatteryCoverOpenNode
 -> cover open 완료 callback
 -> GripCellNode.request_start()
 -> cell_1~4 추출 / voltage 판정
 -> 정상: new_case 다음 빈 slot
 -> 불량: reject
 -> 정상 4개 누적
 -> /suction_cover_close Trigger
```

`/start_battery_cover_drop`이라는 서비스 이름은 기존 ScrewDisassemblyNode와의 호환을 위해
그대로 유지했지만 실제 기능은 이제 “배터리 전체 폐기”가 아니라 “casecover 분리”다.

## Voltage 판정

BatteryVoltageServer의 생성 분포 중심값은 `MEAN_VOLTAGE = 11.0 V`다.
사용자가 말한 "중간값"은 이 분포 중심값을 기준으로 연결했다.

```text
threshold = MEAN_VOLTAGE = 11.0 V
voltage < 11.0 V  -> False / reject
voltage >= 11.0 V -> True / new_case
```

같은 프로세스의 GripCellNode는 `voltage_node.sample_voltage()`를 직접 호출한다.
이는 main.py가 `spin_once()` 기반 단일 loop라서, GripCellNode callback 안에서 같은 프로세스의
voltage ROS service 응답을 동기 대기할 때 생길 수 있는 deadlock을 피하기 위한 것이다.
외부 테스트용 `/check_voltage` Trigger service도 유지한다.

## 2개 카운터

- `cell_count`: 현재 처리할 source cell 번호. 1 -> 5로 증가한다.
- `stack_count`: new_case의 다음 정상-cell slot 번호. 정상 판정 때만 증가한다.

예를 들어 cell_2가 불량이면 cell_count는 3으로 가지만 stack_count는 그대로라서
cell_3 정상품이 비어 있는 다음 slot을 채운다.

## new_case가 4개 미만인 경우

요청대로 아직 실제 동작을 넣지 않았다. `grip_cell_node.py` 끝부분에 TODO 주석만 있다.
향후 구현 대상은 다음 세 단계다.

1. 기존 source casebase 제거
2. 새 배터리 공급 요청
3. 다음 배터리 cover-open 뒤 기존 stack_count부터 계속 적재

이 분기에서는 `/suction_cover_close`를 보내지 않는다.

## 반드시 확인할 Stage Prim

셀 공정 시작 시 이름 기반으로 아래 Prim/Joint를 재귀 검색한다.

```text
<last_placed_battery>/.../casebase
<last_placed_battery>/.../cell_1 ~ cell_4
<last_placed_battery>/.../cell_1_to_casebase ~ cell_4_to_casebase
/World/new_case/.../casebase
```

cover는 `casecover`, `casecover_to_casebase`, 나사는 `nasa_1~4` 이름을 우선 검색한다.
따라서 Reference/Payload 때문에 한 단계 중첩되어도 direct path 하드코딩보다 안전하다.

## 주의

이 환경에서는 Isaac Sim 5.1 자체를 실행할 수 없으므로 실제 physics/RMPFlow runtime 검증은 하지 못했다.
Python syntax 검사와 통합 의존성 검사는 수행했다. 최초 실행 시에는 cover-open만 먼저 확인하고,
그 다음 `/start_grip_cell_process`를 수동 호출해 RG2 cell_1 접근 좌표부터 검증하는 것을 권장한다.
