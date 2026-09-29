## 25. `feature/3d-map`

### 이슈와 수정

기존 Astrobee 카메라 영상만으로는 위성 형상과 도킹 노즐 통로의 이물질 여부를 3D로 확인할 수 없었습니다. `project/srb/tasks/manipulation/debris_capture/astrobee_map.py`가 카메라 depth 표본을 위성 좌표계의 voxel 맵에 축적·갱신하고, 확인된 점이 위성/MEP 모델로 설명되지 않으며 노즐의 keep-out 구역에 있으면 방해물로 판정합니다. 통로를 충분히 관찰하지 않은 상태도 안전한 통로로 간주하지 않습니다. 설정(`project/config/vision_capture.yaml`의 `astrobee.map`)에 따라 미확인/차단 시 `DOCKING_UNAVAILABLE`로 임무를 중단합니다. ROS 2 `/astrobee/map/points`와 `/astrobee/map/clearance`를 통해 대시보드 CAMERA 4에 맵·판정을 표시하고, 브리지는 실행별 `.ply`를 영상과 같은 저장 위치에 보존합니다. 검증 화면의 `/api/validation/runs/{session_id}/pointcloud`에서도 해당 파일을 제공합니다. 카메라 포즈는 시뮬레이터에서 얻으며 실제 SLAM/자기 위치 추정은 구현하지 않습니다.

### 실행 명령

저장소 루트 기준입니다. 오프라인 검증에는 Python의 numpy·pytest, 웹 체크에는 대시보드 의존성이 필요합니다. 맵 시뮬레이션은 Isaac Sim 및 관련 자산이 설치된 환경에서 실행합니다. `--nozzle_obstruction`은 시각적 테스트 이물질을 추가합니다(물리 collider는 없음).

```bash
# Isaac Sim 없이 수학/경로 검사
python3 -m venv .venv
.venv/bin/python -m pip install numpy pytest
cd project
../.venv/bin/python -m pytest tests/test_astrobee_map.py tests/test_astrobee_observer.py -q
cd ..

# ROS 2 없이 FastAPI·WebSocket·포인트클라우드 API 검사;
# Playwright/Chromium이 있으면 CAMERA 4 브라우저 검사도 수행
cd mep_dashboard
../.venv/bin/python -m pip install -r requirements.txt
../.venv/bin/python checks/map_check.py
cd ..

# 실물 시뮬레이터 검증: 노즐 방해물 시나리오 (dock_only 사용 금지)
~/isaac-sim/python.sh project/scripts/vision_capture.py --no_moving_dock --nozzle_obstruction --tag blocked --exit_when_done
```

### 결과

코드에는 depth 역투영, 위성 프레임 voxel 맵, 통로 판정, ROS 2 발행, `.ply` 저장·전송 및 CAMERA 4 렌더링이 연결되어 있습니다. `project/tests/test_astrobee_map.py`와 `test_astrobee_observer.py`는 Isaac Sim 없이 계산을 검사하고 `mep_dashboard/checks/map_check.py`는 ROS 2 없이 서버 경로를 검사합니다. 브라우저 부분은 Playwright와 Chromium이 없으면 건너뛰므로 서버 체크 통과만으로 UI 렌더링을 검증했다고 볼 수 없습니다. 이번 README 작성 과정에서 해당 명령 또는 Isaac Sim 실제 장면을 실행하지 않았으며 통과 건수·실제 장애물 검출/도킹 중단 결과는 주장하지 않습니다. 실제 depth 영상 품질, 위성/MEP 모델 제외 및 장애물 시나리오에서의 end-to-end 판정은 Isaac Sim에서 따로 확인해야 합니다.
