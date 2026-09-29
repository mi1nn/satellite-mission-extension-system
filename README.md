## 20. `feature/web-v2`

### 이슈와 수정

- 기존 `feature/web`의 대시보드와 `feature/integration`의 시뮬레이션 파이프라인을 이어받았다. 이 브랜치의 변경(`9cbe505`, `a42e396`)은 별도로 ROS 2→Firestore 기록과 웹 화면을 연결하고, 대시보드 디자인·검증 화면·구간 영상 재생을 개편한 것이다.
- `project/scripts/firebase_bridge.py`가 ROS 2 상태·pose·카메라 메시지를 세션 요약과 시계열로 기록한다. `mep_dashboard/backend/app.py`는 Firestore 실행 목록/telemetry, `/ws/live` ROS 2 실시간 상태, 카메라 JPEG 및 실행별 MP4를 제공한다. 검증 화면은 실행 구간과 영상 시각을 맞춰 재생한다. 실행 영상은 `<session_id>.mp4`와 `<session_id>.video.json`이 있을 때만 표시된다.
- 이 브랜치의 `backend/app.py`는 실행 영상 기본 위치가 `/home/rokey/space_robotics_bench/project/logs/vision_capture`이므로 다른 설치 경로에서는 `MRV_RUN_VIDEO_DIR`을 지정해야 한다. `mep_dashboard/README.md`의 mock 전용 설명은 현재 웹 구현과 맞지 않는다.

### 실행 명령

저장소 루트에서 대시보드 실행(로컬 Python에 FastAPI·Uvicorn·Pillow·NumPy 설치 필요):

```bash
cd mep_dashboard
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
MRV_RUN_VIDEO_DIR="$(pwd)/../project/logs/vision_capture" .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

`http://127.0.0.1:8000` 및 `/health` 확인. ROS 2 라이브 구독에는 실행 전에 ROS 2 환경을 source하고 시뮬레이터와 `ROS_DOMAIN_ID`를 맞춘다. Firestore 실데이터에는 `FIREBASE_CREDENTIALS` 또는 `GOOGLE_APPLICATION_CREDENTIALS`(또는 emulator)가 필요하다. 기록 브리지는 별도 ROS 2 환경의 저장소 루트에서 `python3 project/scripts/firebase_bridge.py --dry_run`으로 쓰기 예정 메시지를 확인할 수 있다(실제 ROS 토픽이 있어야 출력됨).

### 결과

- 코드에서 확인: 실데이터 API·라이브 WebSocket·실행별 비디오 메타데이터/파일 경로와 구간 seek가 구현되어 있다. `stage_for_state`는 1~6단계만 실제 상태에 매핑하며 7단계 orbit transfer는 미구현이다.
- 미검증: 이 문서 작업에서 Isaac Sim, ROS 2, Firestore 연동이나 브라우저 재생을 실행하지 않았다. `checks/browser_check.py`는 실제 Firestore 실행 기록과 Chromium/Playwright가 있어야 의미 있는 검증이 된다.
