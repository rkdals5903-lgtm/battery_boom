# Battery Pack Story UI Motion V19

## 수정 사항

- 셀 최종 검사 결과 표시 시간을 **2초 → 5초**로 변경
- 양품/불량 셀 이동 애니메이션도 결과 화면 5초 유지 후 시작
- 케이스 전체 폐기 조건을 강화:
  - 현재 배터리의 `cell_1 ~ cell_4`가 모두 `INSPECTION FINAL`까지 끝나야만 `discard_motion` 허용
  - 뚜껑/커버를 여는 단계에서 들어오는 `배터리 폐기...` 로그는 케이스 전체 폐기 모션으로 연결하지 않음
- 따라서 뚜껑 오픈 중에는 배터리 본체가 작업대에 그대로 유지됨
- 4개 셀 검사 완료 후에만 기존 케이스 전체 폐기 애니메이션 실행

## 실행

```bash
cd /mnt/data
unzip -o battery_pack_story_ui_motion_v19.zip
cd battery_pack_story_ui_motion_v19
chmod +x run_ui.sh
./run_ui.sh 8107
```

브라우저:

```text
http://127.0.0.1:8107
```
