# 06. 논문 레퍼런스와 이를 바탕으로 개선한 기능

기준: 현재 `feature/clean-files`의 legacy controller와 **미병합** `feature/reference-adopt` (`10a45ae`)의 `coupled_predictive` controller. `coupled_predictive`는 기본값이 아니다. Isaac Sim end-to-end 검증이 끝나기 전에는 legacy를 유지한다.

## 1. 문제와 수치

| 항목 | 값 | 상태 |
|---|---:|---|
| 페이로드 | 3,000 kg | 설계값 (`vision_capture.yaml:10-11`) |
| 지배 arm+payload 모드 | 약 20 s, 0.05 Hz, $\omega_n≈0.314$ rad/s | 관찰값 (`docking-control-references.md:16-27`) |
| 도킹 공차 | 축/반경 40/40 mm, 축각/roll 2°/4° | 설계값 (`yaml:380-386`) |
| 속도 0→0.15 m/s step | 1 s 안에 횡 ±20 mm | 실측 기록 |
| 0.25 m/s 이송 | pre-dock에서 횡 ±150 mm | 실측 기록 |
| 0.02 m/s 횡 보정 | ±12 mm; 기존 10 mm gate 초과 | 실측 기록 |
| active damping gain 1.0 | 0.1 s 안에 발산 | 실측 기록 |

현재 legacy는 진동을 상쇄하지 않고, 가속도 제한 0.01 m/s²·연속 감속·정렬 저속 0.015 m/s로 **자극을 회피**한다. 이동 Client 도킹 반경오차는 39.97–39.98 mm로 40 mm 결합 한계에 근접했다 ([01_business_requirements.md](01_business_requirements.md)).

## 2. 레퍼런스와 현재 반영

| 레퍼런스 | 가져온 개념 | 구현 | 수치 효과/제약 |
|---|---|---|---|
| Singer & Seering (1990), ZV/ZVD Input Shaping | 불연속 명령이 잔류 진동을 가진 | `probe_dock.py` ramp와 `accel_mps2` | 0→0.15 m/s ramp 15 s; ZV/ZVD 자체는 미구현 |
| Chen & Sun (2020), *Nonlinear Control of Underactuated Systems Subject to Both Actuated and Unactuated State Constraints* | 연속 결합·상태 제약 | legacy의 continuous deceleration; coupled의 soft gate | `decel_gain_hz=0.08`, z=0.30; 이진 gate의 진동 펌핑 회피 |
| Aghili (2009), heavy-payload impedance control | 관절 탄성과 중량 payload의 안정 감쇠 | `swing_damping` 설정은 남기되 0으로 비활성 | gain 1.0이 0.1 s에 불안정; 안정 게인 미도출 |
| Zhou, Liu, Cai, *Motion-planning and pose-tracking based rendezvous and docking with a tumbling target* | 미래 dock pose 계획과 추종 분리 | `coupled_dock.py:predict_body_frame`, `CoupledPoseTracker` | $T=0.3$ s의 constant-twist 예측; 전체 궤적 최적화는 미도입 |
| Ye, Lu, Mu, *Compound control for autonomous docking to a three-axis tumbling target* | 위치·자세·상대 선/각속도 동시 추종 | `CoupledPoseTracker`, `AlignmentGate` | SMC/terminal SMC는 20 s 저감쇠 모드 가진 위험 때문에 미도입 |
| Nenchev et al. (1999), Reaction Null-Space | 7-DoF redundancy | 미도입 | base는 world-fixed; 지배 문제는 base reaction이 아닌 payload swing |
| Singhose (2009), multi-mode shaper | 다중 모드 shaping | 미도입 | 관찰된 지배 모드가 약 20 s 단일 모드 |
| Zhang et al. (2022), macro/micro + backlash | backlash·lag 보상 | 미도입 | Implicit PD 모델에 backlash 없음 |

`Flatness-based Trajectory Planning`은 사전 궤적 산출을 전제하므로, 매 simulation step마다 표류 target pose를 재읽는 폐루프 도킹과 맞지 않아 채택하지 않았다.

## 3. legacy controller

| 메커니즘 | 설정 / 위치 | 의도 |
|---|---|---|
| 가속도 ramp | `accel_mps2=0.01` | 속도 계단을 피함 |
| 연속 감속 | `decel_gain_hz=0.08`, `z_decel_gain_hz=0.30`, creep 0.005 m/s | 목표 근처의 abrupt stop 제거 |
| 느린 정렬 | 0.015 m/s, 2°/s | 20 s 모드 재가진 회피 |
| corridor rollback | lateral >0.20 m 또는 axis >12° | 안전 실패 시 재정렬 |
| depth gate | 9 px patch, agreement ≤150 mm, 10 calibration samples | blind insertion 금지 |

**구성 불일치:** 설정 `settle_window_s=3.0 s`는 주석의 “20 s 모드의 반주기 이상”과 맞지 않는다. 3 s는 전환점에서 남은 진동을 통과시킬 수 있으므로, reference-adopt의 상대속도+유지시간 gate와 비교 검증해야 한다.

## 4. `coupled_predictive` 개선안

### 4.1 legacy 대비

| 항목 | legacy | coupled_predictive |
|---|---|---|
| 목표 | 현재 dock pose | $t+T$ 예측 dock pose |
| 정렬 | XY → orientation 순차 | 위치+자세 동시 |
| 자세 오차 | 축 기울기/roll 분리 | $e_R=\log(R_dR_c^T)$ 회전벡터 |
| 제어 | reference 이동 | feed-forward + P + D |
| approach gate | binary corridor | smooth soft gate + emergency stop |
| 정렬 확인 | timer/settle window | 4 조건이 0.5 s 연속 성립 |
| rollback | 즉시 | 진입보다 넓은 hysteresis + 0.2 s 유지 |
| tip 속도 | PhysX 값 의존 | 연속 pose 차분, low-pass $\tau=0.1$ s |

예측은 Client CoM $c$, linear velocity $v_c$, angular velocity $\omega$에서 계산한다.

$$R_d(t+T)=\exp([\omega]T)R_d(t)$$
$$p_d(t+T)=c+v_cT+\exp([\omega]T)(p_d(t)-c)$$

추종 명령은 다음 구조다.

$$v_{cmd}=v_D+\operatorname{clip}(K_pe_p)-K_d(v_{tip}-v_D)$$
$$\omega_{cmd}=\omega_c+\operatorname{clip}(K_Re_R)-K_{dR}(\omega_{tip}-\omega_c)$$

| parameter | 값 | 상태 |
|---|---:|---|
| prediction horizon | 0.3 s | 설정값 |
| $K_p$ position/attitude | 0.15 Hz / 0.15 Hz | 설정값 |
| $K_d$ position/attitude | 0.6 / 0.6 | 설정값 |
| 최대 보정 선속도/각속도 | 0.015 m/s / 2°/s | 설정값 |
| soft-gate ramp | lateral 10–80 mm, orientation 1–7° | 설정값 |
| emergency stop | lateral >150 mm, orientation >12°, nozzle clearance 부족, nozzle 내 상대속도 >0.1 m/s | 설정값 |
| alignment entry | lateral <50 mm, attitude <4°, \|vrel\|<0.02 m/s, \|ωrel\|<1°/s | 0.5 s 유지 |
| rollback | lateral >80 mm 또는 attitude >7° | 0.2 s 유지 |

근거는 `git show feature/reference-adopt:docs/prompt/coupled_predictive_docking.md` 및 해당 branch의 `coupled_dock.py`, `vision_capture.yaml`이다.

## 5. 검증 현황

| 구분 | 결과 | 해석 |
|---|---|---|
| offline `test_coupled_dock.py` | 21개 test function (문서상 pytest 25/25) | prediction, SO(3), gate, hysteresis, twist estimator 검증 |
| 1-D payload toy model | command-pose P, Kp=0.2/Kd=0.6: 0.0 mm 수렴, overshoot 12 mm | 실제 Isaac Sim 물리의 증거 아님 |
| 예측 toy model | 20 mm/s + 0.01 rad/s: T=0 1.23 mm → T=0.3 0.87 mm | 모델 기반 결과 |
| Isaac Sim coupled | 도킹 전 중단 | 도킹 성공·오차 개선 **미검증** |

다음 비교는 동일 scenario·seed·초기 pose에서 `legacy`와 `coupled_predictive`를 각각 여러 번 실행하고 성공률, axial/radial/angle/roll 오차, realign 수, 도킹 구간 sim time으로 평가해야 한다. 결과가 없으면 coupling 개선 효과를 주장하지 않는다.
