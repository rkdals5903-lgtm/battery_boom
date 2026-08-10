# VG10 데모 전용 실행 — RG2 자동 실행 비활성화

## 배경

시연에서는 VG10(컨베이어 → 작업대) pick&place 하나만 정상 동작하면 된다.
컨베이어 벨트 정지 감지/자동 트리거는 이번 스코프에서 제외한다 — 시연 때
터미널에서 `ros2 service call /vg10_worktable/run_pick_place std_srvs/srv/Trigger`로
수동 호출한다.

## 문제

현재 `main.py`는 시뮬레이션이 Play되는 즉시 RG2 `PickPlaceController`가
자동으로 pick&place를 실행하고, 완료되면 `my_world.pause()`로 시뮬레이션
전체를 정지시킨다. 이 정지가 VG10 서비스 호출 도중(블로킹 루프로
`world.step()`을 반복하는 중)에 겹치면 `VG10WorktableNode._handle_run()`의
`while self._world.is_playing() and ...` 조건이 깨져 VG10 동작이 중단된다.

## 변경 사항

`main.py`의 `create_controllers()`에서 RG2 `PickPlaceController` 생성 블록을
주석 처리하고 `"rg2_pick_place": None`으로 둔다.

- `update_process()`와 `reset_controllers()`는 이미 `if controller is not None`
  가드가 있으므로 이 변경만으로 RG2는 Play를 눌러도 움직이지 않는다.
- RG2가 움직이지 않으므로 `is_done()`도 True가 되지 않아 `my_world.pause()`가
  호출되지 않는다 → VG10 서비스 호출과 겹쳐서 멈추는 위험 제거.
- 컨베이어 벨트 관련 코드/USD는 건드리지 않는다.
- `VG10WorktableNode` / `SuctionStatePickPlaceController`는 기존 구현 그대로
  사용한다(변경 없음) — 이미 서비스 트리거를 받으면 pick&place를 수행한다.

## 범위 밖

- 컨베이어 벨트 정지 감지(watcher) 노드 신설 — 다음 스코프.
- VG10(팔레트 → 컨베이어) 활성화 — 좌표 TODO 미해결로 계속 보류.
- RG2 코드 삭제 — 나중에 다시 쓸 수 있으므로 주석 처리만 하고 남겨 둔다.

## 테스트 계획

- Isaac Sim에서 Play 시 RG2가 움직이지 않는지 육안 확인.
- `ros2 service call /vg10_worktable/run_pick_place std_srvs/srv/Trigger` 호출 시
  VG10이 배터리 하나를 pick&place까지 완료하는지 확인.
- 코드 리뷰로 `update_process()`/`reset_controllers()`가 `None` 가드로
  RG2 관련 로직을 건너뛰는지 확인(런타임 검증은 Isaac Sim 환경에서만 가능).
