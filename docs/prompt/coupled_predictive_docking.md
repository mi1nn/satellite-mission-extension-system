# Predictive Coupled Docking Control: 변경 근거와 비교 분석

- 브랜치: `feature/reference-adopt`
- 작성일: 2026-09-23
- 대상 코드: `project/srb/tasks/manipulation/debris_capture/`
- 관련 설정: `project/config/vision_capture.yaml`의 `docking_control:`, `docking_alignment:` 섹션
- 상태: 오프라인 검증 완료. Isaac Sim은 스모크 테스트 일부만 진행(6절 참고).

---

## 0. 요약

| 항목 | 이전 (`legacy`) | 변경 (`coupled_predictive`) |
|---|---|---|
| 목표 pose | 현재 docking port | Client 운동으로 예측한 `t+T` 시점의 docking port |
| 정렬 순서 | XY → Orientation 순차 | Position + Attitude 동시 |
| 자세 오차 | 축 기울기 + roll 각을 따로 판정 | SO(3) 회전 벡터 `Log(R_d R_c^T)` |
| 제어 법칙 | reference를 일정 속도로 목표 쪽 이동 | feed-forward + P(명령 pose) + D(측정 상대속도) |
| Client 각속도 | 사용 안 함 | 목표 예측과 feed-forward에 사용 |
| 접근 속도 | 코리도 밖이면 0 (hard gate) | 정렬 정도에 비례해 연속 감속 (soft gate) + 비상 정지 |
| `ALIGNMENT_CHECK` | 타이머 + 12 s 정지 창 | 네 조건이 `stable_duration_sec` 동안 연속 성립 |
| 복귀 조건 | 즉시 복귀 | 진입보다 넓은 기준 + 유지 시간 (hysteresis) |
| 상대속도 측정 | MEP PhysX 속도 | tip pose 차분 |

**기본값은 `legacy`입니다.** 새 방식은 설정으로 켭니다.

```bash
--set docking_control.mode=coupled_predictive
```

기존 검증 경로를 깨지 않기 위해서입니다. 과거에 공유 도킹 제어를 직접 수정했다가 `full_6dof` 파이프라인이 깨진 이력이 있습니다.

---

## 1. 배경: 이전 코드의 문제

### 1.1 현재 위치를 뒤늦게 따라가는 구조
`vision_capture_demo.py`의 `track_probe`(1984행)는 아래 두 가지만 합니다.

```python
pos = prev.pos + self.sat_velocity_at(self.dock_world().pos) * self.dt   # 선속도 feed-forward
rot = interp(prev.rot → goal.rot, align_speed_deg_s)                     # 현재 자세로 slerp
```

- 목표는 **현재** docking port입니다.
- Client의 **각속도는 reference 회전에 반영되지 않습니다**. Client가 회전하면 자세 목표가 계속 앞서 가고, reference는 `align_speed_deg_s` 속도로 뒤따라갑니다.
- 그 결과 텀블링 client에서는 자세 추종 지연이 구조적으로 생깁니다.

### 1.2 순차 정렬
`step_xy_align`(2108행)에서 lateral을 맞추고, `step_orientation_align`(2119행)에서 자세를 맞춥니다.

- 실제 reference는 두 상태 모두 같은 `align_track`이라서 위치와 자세를 함께 움직입니다. **상태 전이 조건만 순차**입니다.
- 긴 probe에서는 자세 오차가 tip의 위치 오차로 증폭됩니다. 그래서 XY를 먼저 맞춘 뒤 자세를 돌리면 XY가 다시 틀어지고, 상태가 왕복할 수 있습니다.

### 1.3 Binary hard gate
`step_z_approach`(2173행)는 `approach_still_aligned`(`probe_dock.py` 452행)가 거짓이면 즉시 정렬 상태로 돌아갑니다.

- 접근 속도는 코리도 안에서는 전부, 밖에서는 0입니다.
- 코리도 경계 근처의 흔들림이 곧바로 상태 전환으로 이어집니다.

### 1.4 타이머 기반 정렬 확인
`step_alignment_check`(2130행)의 조건은 세 가지입니다.

- `alignment_ok`
- `settled`: `settle_window_s` 동안 tip 변위가 `settle_window_m` 이하
- `align_hold_s` 타이머

"정지 상태"를 **변위 창**으로 판정하기 때문에, 정렬이 끝나도 창 길이만큼 기다려야 합니다. 상대속도를 직접 보는 조건은 없습니다.

### 1.5 상대속도 측정원
`dock_metrics`(1913행)의 `rel_speed`는 MEP의 PhysX 속도로 계산합니다.

- `moving_dock.py`의 `VelocityEstimator` 주석에 따르면, kinematic으로 움직이는 MRV base에 끌려가는 MEP는 PhysX 속도에 base 이동이 빠질 수 있습니다.
- 이 값에 D 게인을 걸면 client 속도만큼 편향된 명령이 나옵니다.

---

## 2. 레퍼런스에서 가져온 것과 가져오지 않은 것

### 2.1 Zhou, Liu, Cai, "Motion-planning and pose-tracking based rendezvous and docking with a tumbling target"
**가져온 핵심 아이디어**
- Chaser는 target의 **현재** pose가 아니라, target 운동으로부터 **계획한 원하는 docking pose**를 추종합니다.
- Motion planning(원하는 pose 생성)과 pose tracking(추종)을 분리합니다.

**이 코드에서의 구현**
- 원하는 pose 생성: `predict_body_frame` + `desired_tip`
- 추종: `CoupledPoseTracker`
- 두 부분은 인터페이스로 분리되어 있습니다(`coupled_dock.py`의 `TRACKERS` 사전).

**가져오지 않은 것**
- 논문의 전체 궤적 최적화와 계획 기법.
- 이번 단계는 등속 twist 가정의 짧은 horizon 예측(`T = 0.3 s`)입니다.

### 2.2 Ye, Lu, Mu, "Compound control for autonomous docking to a three-axis tumbling target"
**가져온 핵심 아이디어**
- Position과 Attitude를 **target 기준 상대 좌표에서 동시에** 추종합니다.
- 목표 상태는 상대 위치 오차, 상대 자세 오차, 상대 선속도, 상대 각속도가 모두 0으로 가는 것입니다.

**이 코드에서의 구현**
- 추종기는 `e_p`, `e_R`, `v_rel`, `ω_rel`을 한 스텝에서 함께 사용합니다.
- 정렬 판정도 네 값을 동시에 봅니다(`AlignmentGate.entry`).

**가져오지 않은 것**
- Compound Sliding Mode Control과 Nonsingular Terminal SMC.
- 이유 1: 요청 범위가 "단순하고 검증 가능한 PD부터"입니다.
- 이유 2: 현재 플랜트는 논문처럼 추력기로 직접 힘을 주는 강체가 아닙니다. 위치 제어되는 관절 드라이브와 3 t 페이로드의 저감쇠 모드(주기 약 20 s)를 거칩니다. SMC의 스위칭 항이 이 모드를 가진시킬 가능성이 높습니다.
- 나중에 SMC나 MPC는 `TRACKERS`에 같은 `reset`/`step` 인터페이스로 등록하면 FSM 수정 없이 비교할 수 있습니다.

> 논문 세부 수식은 이번 구현에서 재현하지 않았습니다. 위 두 아이디어(원하는 pose 계획, 위치·자세 결합 추종)만 적용했습니다.

---

## 3. 변경 항목별 근거

### 3.1 미래 docking pose 예측
구현: `coupled_dock.py` `predict_body_frame`(212행)

```
R_dock(t+T) = Exp([ω] T) · R_dock(t)
p_dock(t+T) = c + v_c T + Exp([ω] T) · (p_dock(t) − c)        c = client CoM
```

**왜 이렇게 했나**
- 요청 수식 `p + v·T`는 port가 접선 방향으로 직선 운동한다고 가정합니다. 텀블링 client의 port는 CoM을 중심으로 원을 그리므로, 직선 외삽은 궤적 바깥으로 벗어납니다. 강체 운동으로 예측하면 ω = 0일 때는 `p + v·T`와 같고, ω ≠ 0일 때도 정확합니다.
- 각속도는 PhysX가 world frame 벡터로 주므로 `Exp`를 **왼쪽에** 곱합니다. 요청의 `R · Exp(ωT)`는 body frame ω일 때의 형태입니다.
- 검증: `test_prediction_of_a_tumbling_port_follows_the_circle_about_the_com`

### 3.2 자세 오차: 회전 벡터
구현: `pose_error`(248행), `so3_log`(184행)

- `e_R = Log(R_d · R_c^T)`는 최소 회전 벡터입니다. 크기가 곧 오차 각도이고, 방향이 회전축입니다.
- Euler 뺄셈은 ±180° 경계에서 불연속이고 짐벌락이 있습니다.
- 검증: `test_orientation_error_is_a_rotation_vector_not_an_euler_difference`. 170°에서 −170°로 가는 오차를 Euler 뺄셈은 −340°로 계산하지만, 회전 벡터는 +20°입니다.
- Euler 값(`rpy_errors_deg`)은 기존처럼 로그 표시용으로만 남습니다.

### 3.3 Coupled PD 추종기
구현: `CoupledPoseTracker.step`(427행)

```
D     = 예측 dock 축 위의 원하는 tip pose (t+T)
C     = 명령 tip pose(IK reference)를 자기 twist로 t+T까지 전진시킨 pose
e_p   = D.pos − C.pos                  e_R = Log(D.rot · C.rot^T)
v_cmd = v_D + clip(Kp_p · e_p) − Kd_p · (v_tip − v_D)
ω_cmd = ω_c + clip(Kp_R · e_R) − Kd_R · (ω_tip − ω_c)
```

**각 항의 근거**

| 항 | 역할 | 근거 |
|---|---|---|
| `v_D`, `ω_c` | feed-forward | 움직이는 target을 추종할 때 정상 상태 지연을 없앱니다. 기존 코드에는 선속도 항만 있었습니다. |
| P | 위치·자세 오차 보정 | 두 오차를 같은 스텝에서 동시에 보정합니다. |
| D (상대속도) | 감쇠 | world 속도가 아니라 상대속도를 0으로 만듭니다. 요청 8절 요구사항입니다. |
| `clip` | 보정 속도 상한 | 기존 `align_speed_mps`와 같은 목적입니다. 빠른 보정은 20 s 모드를 가진시킵니다. |
| 가속도 제한 | 명령 속도 변화율 제한 | 기존 `accel_mps2` 주석 참고: 속도를 계단처럼 바꾸면 페이로드가 옆으로 20 mm 튑니다. |
| anti-windup | 명령 pose의 최대 선행량 제한 | 팔이 따라오지 못할 때 reference가 발산하는 것을 막습니다. |

**요청과 다른 점 1: P 항의 "현재 pose"는 명령 tip pose입니다.**

1-D toy 모델로 확인했습니다. tip은 reference를 2차계로 따라가고(ω_n = 2π/20, ζ = 0.02), 관절 드라이브는 속도 feed-forward를 받습니다. 결과(120~150 s 시뮬레이션의 마지막 10 s 최대 오차):

| P 항 입력 | Kp [1/s] | Kd | 결과 |
|---|---|---|---|
| 측정 tip | 0.05 ~ 0.3 | 0 ~ 1.0 | 11 ~ 93 mm limit-cycle, 16개 조합 모두 수렴하지 않음 |
| 측정 50 % + 명령 50 % 혼합 | 0.2 | 0.3 | 780 mm, 발산 |
| **명령 tip** | **0.2** | **0.6** | **0.0 mm, 수렴 (overshoot 12 mm)** |

(목표 0.1 m 계단 입력, 보정 속도 상한 15 mm/s, 가속도 제한 0.01 m/s²)

- 이유: reference가 적분기처럼 동작하기 때문에, 측정 tip을 P 항에 넣으면 적분기와 저감쇠 2차 모드가 직렬로 연결됩니다. 이 루프는 위상 여유가 부족합니다.
- 측정 tip은 D 항(상대속도)과 모든 게이트 판정에 씁니다. 실제로 도킹되는 것은 측정 tip이기 때문입니다.
- `tip_feedback_weight`(기본 0)로 측정 tip을 섞어 실험할 수 있습니다.
- 검증: `test_measured_tip_feedback_alone_limit_cycles_on_the_payload_mode`

**요청과 다른 점 2: 미래 목표와 비교하는 대상도 미래 tip입니다.**

- "미래 port − 현재 tip"으로 비교하면, 정상 상태에서 tip이 port보다 `v·T`만큼 앞서 있는 편향이 생깁니다. 텀블링 client에서는 이 편향이 lateral 오차가 됩니다.
- 양쪽 모두 `t+T`로 옮기면 `e = e_now − v_rel · T`가 됩니다. 편향 대신 약한 D 효과가 남습니다.

**예측 효과(toy 모델, 3D, 마지막 10 s 최대 위치 오차)**

| 시나리오 | T = 0 | T = 0.3 s | T = 1.0 s |
|---|---|---|---|
| 정지 client | 0.27 mm | 0.34 mm | 0.47 mm |
| 20 mm/s 표류 + 0.01 rad/s 텀블링 | 1.23 mm | 0.87 mm | 1.30 mm |
| 0.02 rad/s 텀블링 | 6.15 mm | 4.35 mm | 3.06 mm |

- 텀블링이 빠를수록 예측의 이득이 커집니다.
- 정지 client에서는 예측이 잡음만 약간 늘립니다(0.1 mm 수준).
- 검증: `test_prediction_horizon_reduces_the_tracking_error_of_a_tumbling_client`

### 3.4 Soft gate와 비상 정지
구현: `alignment_scale`(280행), `emergency_stop`(286행), `step_coupled_approach`(2433행)

```
scale = ramp(lateral; 10 mm → 80 mm) × ramp(orientation; 1° → 7°)     (smoothstep)
speed = approach_speed(remaining) × scale
```

**왜 이렇게 했나**
- 이진 게이트는 속도를 0과 최대값 사이에서 전환합니다. 기존 코드 주석대로 속도 계단은 페이로드 모드를 가진시킵니다.
- smoothstep은 C1 연속이라 속도 명령이 튀지 않습니다.
- 두 오차를 곱하므로, 한쪽만 나빠도 감속합니다. 결합 추종 개념과 일치합니다.
- 비상 정지는 hard gate를 없애지 않고 안전 조건으로 남긴 것입니다(요청 5절).

**비상 정지 조건**

| 조건 | 기준 |
|---|---|
| lateral | > 0.15 m |
| 자세 | > 12° |
| 노즐 벽 간격 | `min_wall_clearance_m` 미만 |
| 노즐 안 상대속도 | > 0.1 m/s |

- 상대속도 조건은 **노즐 안에서만** 봅니다. 먼 거리에서는 접근 속도 자체가 `approach_speed_mps` 0.15 m/s라서, 이 조건이 항상 발동하기 때문입니다.
- 검증: `test_soft_gate_is_continuous_and_monotonic`, `test_emergency_stop_only_on_unsafe_states`

### 3.5 조건 + 유지 시간 기반 `ALIGNMENT_CHECK`
구현: `AlignmentGate.entry`(338행), `step_coupled_alignment_check`(2402행)

```
lateral < 0.05 m  AND  자세 < 4°  AND  |v_rel| < 0.02 m/s  AND  |ω_rel| < 1°/s
→ stable_duration_sec(0.5 s) 동안 연속 성립
```

**왜 이렇게 했나**
- 기존 `settle_window`는 "변위가 작다"로 정지를 판정합니다. 느린 진동에서도 창을 채울 때까지 기다려야 합니다.
- 상대속도를 직접 보면 정지 판정이 더 직접적입니다. 요청 6절입니다.
- 시간은 스텝 수가 아니라 sim time으로 셉니다. 한 스텝 만족으로는 전이하지 않습니다.
- MRV station 오차와 depth 보정 대기도 `blocked` 인자로 같은 유지 시간에 포함합니다. 외부 조건이 잠깐 깨지면 유지 시간도 처음부터 다시 셉니다.
- 검증: `test_alignment_needs_the_stable_duration_not_one_step`, `test_alignment_stable_interval_restarts_when_a_condition_breaks`

### 3.6 Rollback hysteresis
구현: `AlignmentGate.rollback`(349행), `validate_docking_control_cfg`(126행)

```
진입:  lateral < 0.05 m,  자세 < 4°
복귀:  lateral > 0.08 m  또는  자세 > 7°,  0.2 s 이상 지속
검사:  진입 < 복귀 ≤ 비상 정지  (위반 시 설정 로드 실패)
```

**왜 이렇게 했나**
- 진입과 복귀 기준이 같으면 경계 근처 잡음에 상태가 왕복합니다(요청 7절).
- 유지 시간 0.2 s는 한 스텝짜리 튐을 무시하기 위한 것입니다.
- 검증: `test_rollback_hysteresis_ignores_noise_between_the_thresholds`, `test_config_defaults_keep_legacy_and_validate_thresholds`

### 3.7 상대속도: pose 차분
구현: `TwistEstimator`(360행)

- 연속한 tip pose에서 `v = Δp/Δt`, `ω = Log(R_k R_{k-1}^T)/Δt`를 구하고 저역통과(τ = 0.1 s)합니다.
- 1.5절의 PhysX 속도 문제를 피합니다. pose는 이동하는 MRV에서도 정확합니다.
- Client 속도는 기존처럼 PhysX 값을 씁니다. Client는 실제 동역학 강체이기 때문입니다.
- 검증: `test_twist_estimator_from_poses`

---

## 4. FSM 변경

```
legacy (변경 없음)
PRE_DOCK_APPROACH → XY_ALIGN → ORIENTATION_ALIGN → ALIGNMENT_CHECK → Z_APPROACH → FINAL_INSERTION → DOCK_READY

coupled_predictive
PRE_DOCK_APPROACH → POSITION_ATTITUDE_ALIGN → ALIGNMENT_CHECK → Z_APPROACH → FINAL_INSERTION → DOCK_READY
                            ▲                        │               │
                            └──── rollback / 비상 정지 ┴───────────────┘
```

- 추가된 상태는 `POSITION_ATTITUDE_ALIGN` 하나입니다. 기존 FSM을 전면 삭제하지 말라는 요청에 따른 것입니다.
- 새 상태 이름은 대시보드 단계 매핑에도 추가했습니다(`mep_dashboard/backend/app.py`, `mep_dashboard/frontend/js/validation.js`).

**기존 함수에서 바뀐 곳(모두 `self.coupled`일 때만 동작)**
- `step_pre_dock`: 새 모드면 `POSITION_ATTITUDE_ALIGN`으로 전이하고, `settle_window` 대기를 하지 않습니다.
- `step_dock_ready`: 새 모드면 coupled 추종기를 사용합니다.
- `goto`: 상태 전환 횟수를 셉니다(읽기 전용).
- dispatch: `control_measure`와 `log_control_row`를 두 모드 모두에서 실행합니다(읽기 전용).

**바뀌지 않은 것**
- `track_probe`, `step_xy_align`, `step_orientation_align`, `step_alignment_check`, `step_z_approach`, `forced_alignment_check`.
- 운반 구간(`PRE_DOCK_APPROACH`)은 두 모드 모두 기존 `track_probe`를 씁니다.
- `DOCK_READY`의 도킹 조건(`probe_dock.dock_ready`)과 moving client 게이트.

---

## 5. 성능 비교 방법

두 모드 모두 같은 형식으로 기록합니다.

- **CSV** `<tag>_docking_control.csv`: 요청 11절 항목 전부와 명령 선속도·각속도.
- **결과 JSON** `metrics.docking_control`: 요청 10절 metric 전부.

```bash
~/isaac-sim/python.sh project/scripts/vision_capture.py --scenario dynamic --headless --dock_only --tag cd_legacy
~/isaac-sim/python.sh project/scripts/vision_capture.py --scenario dynamic --headless --dock_only --tag cd_coupled \
    --set docking_control.mode=coupled_predictive
python3 project/scripts/compare_docking_control.py \
    project/logs/vision_capture/cd_legacy_result.json project/logs/vision_capture/cd_coupled_result.json
```

| metric | JSON 키 |
|---|---|
| Docking success rate | `docking_success` |
| Docking time | `docking_time_s` (도킹 상태 진입부터) |
| 도킹 시점 tip 위치·자세 오차 | `probe_tip_position_error_at_dock_m`, `_orientation_error_at_dock_deg` |
| 도킹 시점 상대속도 | `relative_velocity_at_dock_mps` |
| 최대 위치·자세 오차 | `max_position_error_m`, `max_orientation_error_deg` (정렬·접근 상태만) |
| Rollback 횟수 | `rollback_count` |
| 비상 정지 횟수 | `emergency_stop_count` |
| 정렬↔접근 전환 횟수 | `alignment_state_switching_count` |
| 최대 접촉 속도 | `max_contact_velocity_mps` (노즐 안) |

---

## 6. 검증 상태

### 6.1 오프라인 테스트 (`python3 -m pytest`, 시뮬레이터 불필요)

| 항목 | 결과 |
|---|---|
| `tests/test_coupled_dock.py` (신규) | 25 / 25 통과 (9절 회귀 테스트 포함) |
| `test_probe_dock.py`, `test_moving_dock.py`, `test_astrobee_observer.py` (기존) | 전부 통과 |
| `test_vision_math.py` | 26 통과, 4 실패. 변경 전 코드에서도 같은 4개가 실패합니다(기본 설정 기대값이 오래됨, cv2 버전). |

`test_dynamic_intercept.py` 등 시뮬레이터를 쓰는 시나리오 테스트는 이번에 실행하지 않았습니다.

### 6.2 Isaac Sim 스모크 테스트 (2026-09-23)

**조건:** `--scenario dynamic --headless --dock_only`, moving client 기본값(client 20 mm/s, forced insertion, concurrent docking). 두 실행 모두 같은 초기 조건이고, SIGINT로 종료해 결과 JSON을 저장했습니다.

```bash
~/isaac-sim/python.sh project/scripts/vision_capture.py --scenario dynamic --headless --dock_only --tag cd_legacy
~/isaac-sim/python.sh project/scripts/vision_capture.py --scenario dynamic --headless --dock_only --tag cd_coupled \
    --set docking_control.mode=coupled_predictive
```

| 항목 | legacy (`cd_legacy`) | coupled_predictive (`cd_coupled`) |
|---|---|---|
| 종료 사유 | sim t≈112 s, 정체되어 수동 중단 | sim t≈90 s, 사용자 요청으로 중단 |
| 상태 전이 | `PRE_DOCK` → `ALIGNMENT_CHECK`(47.9 s) → `Z_APPROACH`(48.4 s) → `ALIGNMENT_CHECK`(64.5 s, rollback) | `PRE_DOCK` → `POSITION_ATTITUDE_ALIGN`(47.9 s) |
| 도킹 성공 | 실패 | 판단 불가(도킹 전 단계에서 중단) |
| 최대 lateral 오차 | 244 mm | 730 mm (정렬 시작 시점 오차) |
| 최대 자세 오차 | 1.9° | 1.1° |
| Rollback / 비상 정지 | 1 / 0 | 0 / 0 |
| 정렬↔접근 전환 | 2 | 0 |
| 최대 접촉 속도(노즐 안) | 139 mm/s | 0 (삽입 전) |

**legacy 관찰**
- 운반 후 바로 `Z_APPROACH`에 들어갔습니다. 삽입 중 lateral이 코리도(200 mm) 밖인 235 mm로 벌어져 rollback됐습니다.
- 그 뒤 `ALIGNMENT_CHECK`에서 sim 66~112 s 동안 lateral이 235~243 mm, 남은 거리가 약 0.09 m에서 변하지 않아 중단했습니다. forced insertion 코리도(200 mm) 밖이라 접근이 재개되지 않았습니다.
- 제 변경은 legacy 제어 경로에 읽기 전용 로깅만 추가했습니다. 다만 변경 전 코드로 같은 실행은 하지 않았으므로, 이 정체가 원래 동작이라는 것은 **확인되지 않았습니다**.

**coupled_predictive 관찰**
- 운반 구간은 legacy와 같게 동작했습니다(둘 다 47.9 s에 전이).
- 새 상태 전이, 로깅, 결과 JSON 저장까지 예외 없이 동작했습니다.
- 정렬 추이(`POSITION_ATTITUDE_ALIGN`):

| sim t | lateral | 축 방향 거리 | 자세 오차 | 상대속도 |
|---|---|---|---|---|
| 86.1 s | 423 mm | 1.222 m | 0.03° | 9.2 mm/s |
| 88.1 s | 407 mm | 1.221 m | 0.01° | 5.5 mm/s |
| 90.1 s | 394 mm | 1.220 m | 0.00° | 4.2 mm/s |

- 자세 오차는 거의 0까지 줄었고, lateral도 줄고 있지만 느립니다(약 7 mm/s). 보정 속도 상한 `max_correction_speed_mps`(15 mm/s)에 걸려 있는 것으로 보입니다.
- 이 속도라면 진입 기준(50 mm)까지 약 50 s가 더 필요합니다.

**참고: 첫 legacy 실행 중단 원인**
- 첫 실행은 carb mutex assertion(`unlock() called by non-owning thread`)으로 sim 시작 직후 중단됐습니다.
- 시뮬레이터를 쓰는 `test_dynamic_intercept.py`를 동시에 돌리고 있었고, 단독으로 다시 실행하니 재현되지 않았습니다.

### 6.3 미검증 항목

| 항목 | 상태 |
|---|---|
| coupled `ALIGNMENT_CHECK` 이후(soft gate 접근, 도킹 성공) | 미확인 |
| 자세 진입 기준 4°의 실제 통과 여부 | 자세는 통과 수준(0.0°), lateral 수렴 대기 |
| 변경 전 코드로 한 legacy 기준 실행 | 미실행 |
| 전체 파이프라인(`full_6dof`) coupled 모드 | 미실행 |


---

## 7. 한계와 남은 작업

1. **게인은 toy 모델 기준입니다.** 실제 팔의 IK, 관절 스텝 제한, 6자유도 결합에서 수렴하는지 Isaac 결과로 확인해야 합니다.
   - 2026-09-28: `max_correction_speed_mps`를 0.015 → **0.03**으로 변경했습니다. 먼 거리 정렬 속도는 약 `상한 / (1 + Kd)`라서, 0.015에서는 Isaac 실측 약 7 mm/s(toy 8 mm/s)였습니다.
   - toy 모델(좌우 420 mm 시작, client 20 mm/s 표류)에서 정렬 확인 통과 시간은 41 s → 27 s입니다. 50 mm 안에 들어간 뒤 다시 벌어지지 않았고, 마지막 10 s 잔여 오차(0.7 mm)도 같습니다.
   - 0.02는 표류 + 텀블링에서 79 mm까지 다시 벌어졌고(복귀 기준 80 mm), Kd 0.3은 0.03과 조합하면 86 mm로 복귀 기준을 넘어 제외했습니다.
   - Isaac에서 `rollback_count`가 생기면 0.025로 낮춥니다(`--set docking_control.max_correction_speed_mps=0.025`).
2. **자세 진입 기준 4°는 roll을 포함한 3축 전체 각도입니다.** legacy 설정은 기울기 12°, roll 5°로 느슨합니다. 팔이 4°까지 도달하지 못하면 `ALIGNMENT_CHECK`에서 시간 초과가 날 수 있습니다. 그 경우 `docking_alignment.orientation_threshold_deg`를 조정합니다.
3. **depth 보정 규칙은 legacy와 같습니다.** 축 위 15 mm, 1° 이내에서만 보정합니다. non-forced 모드에서는 보정 전에는 접근하지 않습니다.
4. **운반 구간은 예측을 쓰지 않습니다.** 4.8 m 자유 공간 이동이라 이번 범위에서 제외했습니다.
5. **다음 Phase 후보**: `TRACKERS`에 SMC 또는 MPC 추종기를 추가하고, 같은 metric으로 비교합니다.

---

## 8. 변경 파일

| 파일 | 내용 |
|---|---|
| `project/srb/tasks/manipulation/debris_capture/coupled_dock.py` | 신규. 예측, SO(3), 추종기, 게이트, metric, CSV 컬럼 |
| `project/srb/tasks/manipulation/debris_capture/vision_capture_demo.py` | 새 상태, coupled step 함수, 로깅, 결과 metric |
| `project/srb/tasks/manipulation/debris_capture/vision.py` | 설정 섹션 등록과 검증 |
| `project/srb/tasks/manipulation/debris_capture/probe_dock.py` | `PHASES`에 새 상태 추가 |
| `project/srb/tasks/manipulation/debris_capture/mrv_approach.py` | 플룸 스캔 캐시를 분사 면 단위로 변경(9절) |
| `project/config/vision_capture.yaml` | `docking_control:`, `docking_alignment:` (기본 `mode: legacy`) |
| `project/tests/test_coupled_dock.py` | 신규. 오프라인 테스트 25개 |
| `project/scripts/compare_docking_control.py` | 신규. 모드별 비교표 |
| `mep_dashboard/backend/app.py`, `mep_dashboard/frontend/js/validation.js` | 새 상태를 "도킹 준비" 단계에 매핑 |

---

## 9. 이슈: 도킹 후 MRV가 물러나지 않고 시뮬레이션이 멈춤 (2026-09-23)

### 증상
- 실행: `~/isaac-sim/python.sh project/scripts/vision_capture.py --set docking_control.mode=coupled_predictive` (GUI, 전체 파이프라인)
- coupled 모드로 도킹에 성공했습니다(`docking_success=True`, 도킹 시간 114 s, 도킹 시점 tip 오차 20.9 mm / 0.08°, 상대속도 2.3 mm/s).
- 하지만 `DOCKED → ROBOT_RELEASE`(sim t=188.2 s) 뒤로 MRV가 분리되지 않았고, Ctrl+C로 종료했습니다(`MOVE4~6: not reached`).

### 원인
1. **sim 시간이 188.2 s에서 멈췄습니다.** `full_6dof_moving.csv`(0.1 s 간격)가 `ROBOT_RELEASE` 첫 행에서 끝납니다. 그동안 `[MRV-VFX] exhaust face ...`가 6번 이상 찍혔습니다.
2. **플룸 배치 스캔이 매 스텝 반복됐습니다.** `ThrusterVfx.fire_anchors`는 선체 벽면 스캔(`_wall_anchors`)을 분사 방향을 소수 셋째 자리까지 반올림한 값으로 캐시했습니다. 방향이 조금만 바뀌어도 다시 스캔했고, 주축에서 벗어난 방향은 13×13 레이를 하나씩 쏘는 느린 경로를 탔습니다.
3. **방향이 계속 바뀐 이유:** `DOCK_READY` 중 coupled 추종기가 도킹점 쪽으로 계속 밀어 client를 가속시켰습니다(0.3 s 동안 19.46 → 19.68 mm/s). station keeping 중인 MRV가 따라 가속하면서(최대 0.88 mm/s², 방향 `[1, −0.028, 0]`) 플룸 기준값 1e-4 m/s²를 넘었습니다.

| sim t | 상태 | MRV 가속도 | client 속도 |
|---|---|---|---|
| ~187.8 s | DOCK_READY | 0 | 19.46 mm/s |
| 187.9 s | DOCK_READY | 0.32 mm/s² | 19.46 mm/s |
| 188.1 s | DOCK_READY | 0.88 mm/s² | 19.60 mm/s |
| 188.2 s | ROBOT_RELEASE | 0.77 mm/s² | 19.68 mm/s |

### 조치
| 조치 | 파일 | 적용 모드 |
|---|---|---|
| 스캔 캐시 키를 분사 면(주축 + 부호)으로 바꾸고, 스캔 방향을 그 주축에 맞춤. 노즐 위치는 원래 면에 의해서만 정해지므로 배치 결과는 같고, 면당 한 번만 빠른 경로로 스캔합니다. | `mrv_approach.py` `fire_anchors` | **모든 모드** (legacy 포함) |
| `DOCK_READY` 진입 시 축 방향 목표를 측정 tip 위치에 고정하고, 더 밀지 않음 | `vision_capture_demo.py` `step_dock_ready` | coupled_predictive |
| 노즐 안에서 축 방향 선행량을 5 cm가 아닌 1 cm로 제한(`docking_control.max_axial_lead_inside_m`) | `vision_capture_demo.py` `_advance_axial`, `coupled_dock.py`, `vision_capture.yaml` | coupled_predictive |

### 검증
- 회귀 테스트 `test_plume_scan_runs_once_per_face_not_per_direction`: 측정된 흔들림(최대 0.028 비축 성분)으로 50번 호출해도 스캔은 1번입니다. **수정 전 코드에서는 실패하고, 수정 후에는 통과합니다.**
- 오프라인 테스트 69개 통과(`test_coupled_dock.py`, `test_moving_dock.py`, `test_probe_dock.py`).
- **Isaac 재실행은 아직 하지 않았습니다.** 도킹 후 `ROBOT_RELEASE → MRV_SEPARATION` 전이와 legacy 모드 플룸 표시를 확인해야 합니다.
