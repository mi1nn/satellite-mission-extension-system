## 22. `feature/web-integration-v2`

### 이슈와 수정

- `feature/web-v2`의 대시보드·영상/Firestore/ROS 2 연동은 병합으로 상속했다. 이 브랜치의 별도 수정(`8ba1086`, `8c98942`)은 세션 종료 KPI 판정과 검증 데이터 재조회 비용에 집중한다.
- `project/scripts/firebase_bridge.py`는 그리퍼가 해제되어 현재 `captured`가 false여도 과거 포착 성공을 유지하고, 짧게 지나간 `DOCKED` 상태를 놓친 경우 이후 도킹 완료 상태로 성공을 판정한다. 이는 Firestore `capture_success`/`docking_success`/`mission_success`가 정상 도킹 임무에서 잘못 false가 되던 경우를 보정한다.
- `mep_dashboard/backend/app.py`는 실행 목록을 기본 60초 동안 파일 캐시하고, 완료된 실행의 telemetry를 재사용한다. 기록 중인 실행은 재조회하며, Firestore 실패 시 기존 캐시가 있으면 `cache: stale`로 반환한다. 캐시가 없으면 503이다. 프런트엔드는 stale 표시를 한다. 캐시 위치는 기본 `mep_dashboard/.cache/validation`, `MEP_DASHBOARD_CACHE_DIR`로 변경 가능하다.
- 7단계 orbit transfer는 이 브랜치에서도 구현되지 않았다. 도킹 완료 단계 6의 표시와 Firestore 성공 KPI를 7단계 실행으로 혼동하지 않는다.

### 실행 명령

저장소 루트에서 대시보드 실행(ROS 2 실시간 데이터는 동일한 `ROS_DOMAIN_ID`와 ROS 2 환경이 필요):

```bash
cd mep_dashboard
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

`http://127.0.0.1:8000/health` 확인. Firestore는 `FIREBASE_CREDENTIALS`/`GOOGLE_APPLICATION_CREDENTIALS` 또는 emulator가 필요하다. 이 브랜치에서는 저장소 루트의 `serviceAccount.json`이 존재할 때 대시보드가 기본 인증 파일로 사용하지만, 비밀키 파일은 저장소에 포함되지 않는다. 영상 기본 위치는 이 브랜치에서 저장소의 `project/logs/vision_capture`로 변경되며 필요하면 `MRV_RUN_VIDEO_DIR`로 지정한다.

Firestore 없이 캐시 분기만 확인하는 기존 스텁 점검은 위 의존성 설치 후 `mep_dashboard`에서 실행한다:

```bash
.venv/bin/python checks/cache_check.py
```

### 결과

- 코드에서 확인: `checks/cache_check.py`는 완료 실행의 반복 조회 0회, 기록 중 실행 재조회, quota 오류 시 stale 응답, 캐시가 없을 때 503을 검사하도록 작성되어 있다. 검사 파일의 존재는 이번 작업에서 실행 성공을 뜻하지 않는다.
- 미검증: 이 문서 작업에서 캐시 점검 스크립트, 실제 Firestore/ROS 2, 브라우저 및 Isaac Sim 임무를 실행하지 않았다. 활성 임무의 최종 KPI와 동영상 동기화도 실환경 검증이 필요하다.
