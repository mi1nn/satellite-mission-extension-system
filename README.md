## 26. `feature/add-more`

### 이슈와 수정

기존 Astrobee는 위성 관찰 카메라로만 움직여 MEP 포획·도킹 제어에 힘을 가하지 않았다. 이 브랜치의 두 커밋(`f6f74e5`, `af0af50`)은 `astrobee_assist.py`와 `vision_capture_demo.py`에 선택적 도킹 감쇠 보조를 추가했다. MEP 포획 전에는 probe 옆 대기 지점으로 이동하고, 포획 후 접근·파지하며, 정렬 단계에는 probe 뿌리 부근의 속도 편차를 감쇠하고 도킹축 횡방향 오차를 보정하는 힘을 MEP에 적용한다. 삽입 단계에 들어가기 전에 해제한다. `project/config/vision_capture.yaml`의 `astrobee.assist.enabled`는 기본 `false`이므로 명시적으로 켜야 한다.

Astrobee의 파지는 물리 조인트가 아니다. 비충돌 운동학적 Astrobee와 MEP에 가하는 외력으로 근사한다. 기본 추력 상한 `5 N`은 실제 Astrobee의 추력 모델이 아니라 3 t MEP을 위한 가상의 확대 모델이다. 이 브랜치에 위성 포인트 클라우드나 도킹 통로 안전 게이트는 없다.

### 실행 명령

저장소 루트에서 Isaac Sim 5.x 및 `project` 패키지를 준비한 뒤, 보조 기능을 켜고 임무를 실행한다. GUI 실행이므로 종료 시 창을 닫도록 `--exit_when_done`을 지정했다. ROS 2 브리지를 쓰지 않는 단독 실행은 `--no_ros`로 분리한다.

```bash
~/isaac-sim/python.sh -m pip install --editable project
~/isaac-sim/python.sh project/scripts/vision_capture.py --no_ros --exit_when_done --tag astrobee_assist --set astrobee.assist.enabled=true
```

보조 기능을 끄는 비교 실행은 위 명령의 마지막 옵션을 `--set astrobee.assist.enabled=false`로 바꾼다. 해당 스위치는 카메라 자체를 제거하는 `--no_astrobee`와 다르다.

### 결과

구현상 보조 상태는 대기·접근·파지·감쇠·해제 순으로 전이하며, 감쇠 힘은 `vision_capture_demo.py`에서 MEP에 적용된다. 실행 후 `project/logs/vision_capture/astrobee_assist_result.json`의 `metrics.astrobee_assist`에 파지 여부, 감쇠 시간, 임펄스와 최대 힘 등이 기록되도록 연결돼 있다. `project/tests/test_astrobee_assist.py`와 `project/tests/test_astrobee_arm.py`는 제어 및 팔 통합 경로를 다루지만, 이 README 작업에서는 Isaac Sim 임무나 테스트를 실행하지 않았으므로 성공률·감쇠 성능은 주장하지 않는다.
