# MEP Mission Extension Dashboard

Automated Satellite Mission Extension System 의 웹 대시보드.
ROS 2 실시간 텔레메트리와 Firestore 실행 이력을 한 화면에서 보여주고,
시뮬레이터에 임무 제어 명령을 보냅니다.

HTML / CSS / Vanilla JavaScript / Chart.js / FastAPI. 빌드 도구·번들러 없음.
저장소 전체 설치와 실행 순서는 상위 [README.md](../README.md) 를 참고하세요.

## 실행

`rclpy` 는 ROS 2 설치본을 쓰므로 venv 를 `--system-site-packages` 로 만들어야 합니다.

```bash
source /opt/ros/jazzy/setup.bash          # ROS_DOMAIN_ID 를 시뮬레이터 PC 와 맞출 것
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

브라우저에서 **http://127.0.0.1:8000**. 종료는 `Ctrl+C`.
8000번이 사용 중이면 `--port 8001` 로 바꿉니다.

- `rclpy` 가 없어도 서버는 뜹니다. LIVE 탭만 OFFLINE 이 되고 VALIDATION 탭은 정상 동작합니다.
- Firebase 자격 증명은 저장소 루트의 `serviceAccount.json` 을 자동으로 찾습니다.
  다른 경로를 쓰려면 `FIREBASE_CREDENTIALS` 환경 변수로 지정합니다.
- Chart.js 4.4.8 은 `frontend/js/vendor/chart.umd.js` 에 포함돼 있어 외부 인터넷이 필요 없습니다.

## 화면

상단 탭으로 전환합니다 (방향키 / Home / End 지원).

### LIVE MISSION — ROS 2 실시간

- Isaac Sim 뷰포트와 카메라 3면: MEP 포착(cam_wrist), 위성 도킹(cam_probe), Astrobee RGB 관찰
- CAMERA 4 · ASTROBEE MAP: 위성 좌표계 포인트 맵, 노즐 금지 구역 윤곽, 빨간 장애물 점과 `NOT OBSERVED` / `DOCKING AVAILABLE` / `DOCKING UNAVAILABLE` 판정. 맵이 없으면 WAITING
- 임무 상태와 7단계 진행률(7단계 궤도 이송은 미구현), 총 속도 / 각속도 / 잔여 거리. `DOCKING_UNAVAILABLE`일 때 실제 중단 사유를 표시
- 위치 오차 시계열 차트와 **PLAY / PAUSE / STOP** → `/mrv/cmd/start`, `/cmd/pause`, `/cmd/abort` 발행

### TECHNOLOGY VALIDATION — Firestore 기록

- KPI 5개: 전체 성공률, 총 실행 횟수, 평균 임무 시간, MEP 포착 반복 정밀도, 위성 도킹 정밀도
- 실행 이력 표 → 행을 고르면 상세가 갱신됩니다
- 단계별 텔레메트리 그래프. 그래프를 드래그하거나 슬라이더로 구간을 고르면
  구간 통계와 **EXPORT JSON** 이 그 구간만 다룹니다
- MISSION VIDEO: 고른 구간의 임무 영상으로 이동합니다 (아래 참고)
- 실행 종료 시 저장된 Astrobee 포인트 맵이 있으면 RUN INFORMATION에 `.ply` 다운로드 링크가 나타납니다. 서버는 파일이 없으면 404를 반환합니다

## 데이터 경로

```text
GPU PC: Isaac Sim (vision_capture.py)
   │  /mrv/** (sensor_msgs/Image, PoseStamped, String ...)
   │  /astrobee/camera/image_raw + /astrobee/map/{clearance,points}
   │  ROS 2 DDS · 같은 ROS_DOMAIN_ID
   ├─► backend/app.py  RosLiveBridge ──► /api/live/*.jpg, /ws/live (JSON + 맵 바이너리) ──► LIVE 탭
   └─► firebase_bridge.py ──► Firestore + 세션 .ply ──► /api/validation/* ──► VALIDATION 탭
```

### API

| 엔드포인트 | 내용 |
|---|---|
| `GET /` , `/static/*` | 화면과 정적 파일 |
| `GET /health` | 서버·ROS 브리지 상태 |
| `GET /api/live/{viewport,camera1,camera2,camera3}.jpg` | 최신 프레임. 없으면 `204` → `STREAM OFFLINE` |
| `WS /ws/live` | 실시간 상태·텔레메트리 |
| `POST /api/live/command/{start,pause,resume,abort}` | 시뮬레이터 제어 |
| `GET /api/validation/runs` | 실행 목록 |
| `GET /api/validation/runs/{id}/telemetry` | 실행 1건의 시계열 |
| `GET /api/validation/runs/{id}/video-metadata` , `/video` | 임무 영상 |
| `GET /api/validation/runs/{id}/pointcloud` | 종료 실행의 Astrobee 맵 `.ply`; 없으면 404 |

### 읽기 캐시

Firestore 는 문서 단위로 과금되고, 실행 1건의 텔레메트리가 800~2,000 문서입니다.
캐시가 없으면 실행을 몇십 번 클릭하는 것만으로 무료 한도(하루 읽기 50,000건)가 소진됩니다.

- 끝난 실행은 `.cache/validation/` 에 저장하고 이후 Firestore 를 읽지 않습니다.
- 기록 중인 실행(`is_running: true`)은 캐시하지 않고 매번 새로 읽습니다.
- Firestore 에 접근할 수 없으면 캐시본을 내려주고 화면 상단에 `CACHED DATA` 를 표시합니다.
- 초기화: `rm -rf .cache/`
- 조정: `MEP_DASHBOARD_CACHE_DIR`, `MEP_DASHBOARD_RUNS_TTL_S` (기본 60초)

### 임무 영상

`project/logs/vision_capture/<session_id>.mp4` 와 사이드카 `<session_id>.video.json`.
사이드카의 `t0_sim_s` 가 영상 시간 0에 해당하는 sim time 이라, 텔레메트리 `sim_time` 과 맞물립니다.
경로는 `MRV_RUN_VIDEO_DIR` 로 바꿀 수 있습니다.

**GUI 로 실행한 시뮬레이션에서만 생성됩니다.** headless 에서는 뷰포트 캡처가 동작하지 않습니다.
영상이 없는 실행은 `VIDEO NOT AVAILABLE` 로 표시됩니다.

## 파일 구조

```text
mep_dashboard/
├── backend/
│   └── app.py              # Firestore API + 읽기 캐시 + ROS 2 라이브 + 영상 서빙
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── app.js          # 공통 차트 설정과 탭 전환
│       ├── live.js         # LIVE 탭 (WebSocket 텔레메트리 + 맵 전달)
│       ├── camera3.js      # Astrobee 카메라 폴링
│       ├── map3d.js        # CAMERA 4 포인트 맵·도킹 통로 렌더링 (로컬 three.js)
│       ├── firebase.js     # 서버 Firestore API 브라우저 클라이언트
│       ├── validation.js   # VALIDATION 탭 (KPI·이력·그래프·영상·.ply 링크)
│       ├── ui.js
│       └── vendor/         # 로컬 Chart.js, three.js
├── checks/
│   ├── cache_check.py      # 캐시 동작 (Firestore 스텁, 네트워크 불필요)
│   ├── map_check.py        # 맵 WebSocket·.ply·브라우저 스모크
│   └── browser_check.py    # 브라우저 스모크 (playwright)
├── requirements.txt
└── README.md
```

`.venv/`, `.cache/` 는 로컬 전용이며 버전 관리에서 제외됩니다.

## 검증

```bash
.venv/bin/python checks/cache_check.py     # 네트워크 없이 실행 가능
```

`.venv/bin/python checks/map_check.py`는 ROS 2 없이 맵 WebSocket·`.ply`를 확인하고, Playwright Chromium이 설치돼 있으면 CAMERA 4 화면도 확인합니다.

캐시 계층을 Firestore 스텁으로 검증합니다: 재조회 시 읽기 0건, 기록 중인 실행 캐시 제외,
`is_running` 필드가 없는 과거 실행 캐시, 할당량 초과 시 캐시 폴백, 캐시가 없을 때 503 유지.

```bash
.venv/bin/python -m pip install playwright==1.63.0
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" .venv/bin/python -m playwright install chromium
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" .venv/bin/python checks/browser_check.py
```

브라우저 스모크는 8000번 포트에서 서버가 실행 중이어야 합니다.
