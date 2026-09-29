## 27. `feature/pointcloud`

### 이슈와 수정

Astrobee RGB-D 영상의 위성 좌표계 3D 보셀 맵, 노즐 통로 판정, `CAMERA 4 · ASTROBEE MAP`, PLY 내려받기는 앞선 `feature/3d-map`의 `0fac20a`에서 구현되어 이 브랜치로 병합된 기능이다. `feature/pointcloud` 고유 커밋 `0e11875`는 이를 새로 만든 것이 아니라 도킹 불가 사유의 전달·표시를 보강했다. 통로의 장애물뿐 아니라 관측 부족으로 `DOCKING_UNAVAILABLE`에 도달해도 `backend/app.py`의 상태 정규화가 `failure`를 보존하고 `frontend/js/live.js`가 해당 사유를 임무 상태 패널에 표시하도록 바꿨다. `checks/map_check.py`에도 두 사유의 회귀 확인을 추가했다. `astrobee.py`와 `astrobee_map.py`의 판정 설명도 관측 횟수 조건과 삽입 전 게이트에 맞춰 수정했다.

최신 브랜치 tip `6be5245`는 main의 coupled-predictive 도킹 제어 변경을 병합했다. 이 도킹 제어는 포인트 클라우드 센서 구현과 별개의 상속된 변경이다.

### 실행 명령

맵 패널과 사유 표시의 독립 점검은 저장소 루트에서 대시보드 의존성을 설치한 환경으로 실행한다. `map_check.py`는 ROS 2 없이 FastAPI TestClient와 스텁 맵을 검사하며, Playwright/Chromium이 설치된 경우 브라우저 패널도 검사한다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r mep_dashboard/requirements.txt
cd mep_dashboard
../.venv/bin/python checks/map_check.py
```

실제 장애물 경로를 확인하려면 저장소 루트에서 Isaac Sim 및 `project` 패키지를 준비한 뒤 다음을 실행한다. `--dock_only`는 MEP가 초기부터 노즐을 가릴 수 있어 사용하지 않는다. GUI를 계속 띄우지 않도록 `--exit_when_done`을 지정한다.

```bash
~/isaac-sim/python.sh -m pip install --editable project
~/isaac-sim/python.sh project/scripts/vision_capture.py --no_moving_dock --nozzle_obstruction --exit_when_done --tag blocked
```

### 결과

구현상 도킹 통로가 장애물로 확정되면 임무는 `DOCKING_UNAVAILABLE`에서 멈추고, 관측이 부족한 경우도 삽입 전 게이트에서 도킹 불가가 된다. 두 경우의 `failure` 문자열을 상태 정규화·웹 임무 패널에 전달하는 것이 이 브랜치 고유 변경이다. 기본 설정의 Astrobee 맵은 활성화되어 있고 실행이 끝나면 `project/logs/vision_capture/<tag>_astrobee_map.ply` 저장 경로가 결과 JSON의 `metrics.astrobee_map.ply`에 반영될 수 있다. 이 README 작업에서 대시보드 점검이나 Isaac Sim 실행은 하지 않았으므로 실측 판정·테스트 통과를 주장하지 않는다.
