## 24. `feature/reference-adopt`

### 이슈와 수정

기존 도킹 제어는 위치(XY)와 자세를 순차 정렬한 뒤 축 방향으로 접근합니다. 움직이는 위성의 도킹 포트를 추종하면서 두 오차를 함께 보정하기 위해 `project/srb/tasks/manipulation/debris_capture/coupled_dock.py`에 예측·결합 제어를 추가했습니다. `project/config/vision_capture.yaml`의 `docking_control.mode` 기본값은 기존 `legacy`로 유지하며, `coupled_predictive`는 명시적으로 선택해야 합니다. 위성 속도로 0.3초 뒤의 목표 포즈를 예측하고 위치·자세를 동시 보정하며, 접근 게이트·비상 정지·rollback hysteresis를 적용합니다. 결과 JSON의 `metrics.docking_control`과 도킹 CSV로 두 모드를 비교할 수 있습니다. 보정 속도 상한은 0.015 → 0.03 m/s로 조정했지만, 새 값은 Isaac Sim에서 재검증되지 않았습니다.

### 실행 명령

저장소 루트에서 실행합니다. 아래 시뮬레이션 명령에는 Isaac Sim 설치(`~/isaac-sim/python.sh`)가 필요합니다. 비교 명령은 **두 실행의 결과 JSON이 생성된 뒤**에만 사용합니다.

```bash
# 오프라인 제어 테스트: project 환경에 numpy, pytest 등 의존성이 설치되어 있어야 합니다.
cd project
python3 -m pytest tests/test_coupled_dock.py -q
cd ..

# 동일한 dynamic 시나리오를 모드별로 실행 (Isaac Sim 필요)
~/isaac-sim/python.sh project/scripts/vision_capture.py --scenario dynamic --headless --dock_only --tag cd_legacy
~/isaac-sim/python.sh project/scripts/vision_capture.py --scenario dynamic --headless --dock_only --tag cd_coupled \
  --set docking_control.mode=coupled_predictive
python3 project/scripts/compare_docking_control.py \
  project/logs/vision_capture/cd_legacy_result.json project/logs/vision_capture/cd_coupled_result.json
```

### 결과

`docs/prompt/coupled_predictive_docking.md`에 기록된 과거 검증에서 오프라인 `test_coupled_dock.py` 25개는 통과했습니다. 같은 기록의 2026-09-23 Isaac Sim 스모크에서는 두 실행 모두 수동 중단했습니다. `legacy`는 접근 후 rollback되어 정체했고, `coupled_predictive`는 정렬 중 lateral 오차가 423 → 394 mm로 줄었으나 도킹 성공 여부를 확인하기 전에 종료했습니다. 이는 **0.015 m/s 설정의 부분 스모크 관찰**이지 새 0.03 m/s 설정의 성공 판정이 아닙니다. 오프라인 toy 모델에서는 상한 변경으로 정렬 확인까지 41 → 27초였지만 실제 팔·물리·전체 임무의 성능 증거로 볼 수 없습니다. 이번 README 작성 과정에서 Isaac Sim 실행 또는 테스트를 새로 수행하지 않았습니다. 실제 도킹 완료, 성공률, `full_6dof` 경로 및 새 게인은 여전히 Isaac Sim 검증이 필요합니다.
