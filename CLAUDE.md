# GraspGrade: Claude Code 작업 안내

이 파일은 새 Claude Code 세션이 바로 이어서 작업할 수 있도록 쓴 안내다.
작업 전에 아래 문서를 이 순서로 읽는다.

1. [docs/PROGRESS.md](docs/PROGRESS.md): 어디까지 했고 다음에 무엇을 할지
2. [docs/MEASUREMENT_PLAN.md](docs/MEASUREMENT_PLAN.md): 무엇을 무엇과 비교하는지, 어느 데이터로. 다른 문서와 다르면 이 문서가 맞다
3. [docs/PROPOSAL.md](docs/PROPOSAL.md): 10/28 제안서 (용도, 범위, 부드러운 물체 확인, 비교 6)
4. [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md): 프로젝트 정의, 파이프라인, 10일 계획, 위험 (2026-10-21 원본)
5. [docs/DATA_NOTES.md](docs/DATA_NOTES.md): 데이터에서 확인한 사실
6. [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md): 환경, 버전 고정 이유, 함정

## 작업 규칙

- 답은 한국어로 짧고 냉정하게 쓴다. 추측하지 않고 물어본 것만 답한다.
- 리스크는 맨 끝에 "리스크:" 한 줄, "극복방안:" 한 줄로만 쓴다.
- 영어 산출물: 쉬운 영어. em-dash, 축약형, 세미콜론 금지.
- 결정 트리 금지. 통합 점수는 로지스틱 회귀.
- Meta API 호출은 크레딧($150)을 쓴다. 사용자 승인 없이 호출하지 않는다.
- `.env` 값과 HOT3D 서명 URL은 절대 출력하지 않는다.
- sudo는 사용자 비밀번호가 필요하다. apt 작업은 사용자에게 넘긴다.
- 실험을 하면 같은 턴에 `docs/experiments/`에 제출용 영어 기록(설정, 수치, 비용, 이미지, 결론)을 쓰고 README 표에 한 줄 추가한다. 이미지는 `docs/experiments/img/`.

## 위치

- 실제 작업 위치: WSL2 Ubuntu-24.04, `~/graspgrade` (사용자 `joonse`). venv는 `~/graspgrade/.venv`.
- Windows 폴더 `C:\Users\USER\Documents\works\hackathon_meta`는 편집용 사본이다.
  스크립트와 문서는 여기서 고치고 WSL로 복사한다. 데이터(`data/`, `out/`)는 WSL에만 있다.
- `.env`를 복사할 때는 `tr -d '\r' < .env > ~/graspgrade/.env && chmod 600 ~/graspgrade/.env`.

## Windows에서 WSL 명령 실행

Git Bash에서:

    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -e bash -c 'cd ~/graspgrade && .venv/bin/python scripts/...'

- `-e`를 빼면 `$VAR`가 미리 확장되어 빈 값이 된다.
- `MSYS_NO_PATHCONV=1`을 빼면 `/mnt/c/...` 경로가 Windows 경로로 바뀐다.
- 시스템 `python3`가 아니라 `.venv/bin/python`을 쓴다.

Windows 사본을 WSL로 동기화:

    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -e bash -c 'cp -r /mnt/c/Users/USER/Documents/works/hackathon_meta/{scripts,docs,splits,viewer,CLAUDE.md,README.md,requirements.txt} ~/graspgrade/ && mkdir -p ~/graspgrade/web && cp -r /mnt/c/Users/USER/Documents/works/hackathon_meta/web/tools ~/graspgrade/web/'

## 스크립트

| 파일 | 하는 일 |
| --- | --- |
| `scripts/setup_wsl.sh` | 환경 구축 (멱등) |
| `scripts/verify_env.py` | import 확인 + dex-retargeting 샘플 리타게팅 |
| `scripts/cut_episodes.py` | OpenGraph AV1 원본을 에피소드별 720p H.264로 자름 |
| `scripts/mediapipe_check.py` | 장갑 손 MediaPipe 검출률 측정 |
| `scripts/hot3d_download.py` | HOT3D 서명 URL로 다운로드 + sha1 검증 |
| `scripts/make_splits.py` | 튜닝 / held-out 분할 생성 (seed 20261026) |
| `scripts/meta_smoke.py` | Meta API 스모크 테스트 (Muse Spark 이미지 1장, SAM 3.1 5초 클립). `--step 3`이면 10fps 클립. 크레딧 사용 |
| `scripts/sam_prompt_compare.py` | 같은 클립에서 SAM 물체 프롬프트별 검출, id 안정성, 손과의 겹침 비교 (API 호출 없음) |
| `scripts/sam_fps_compare.py` | 같은 클립의 SAM 10fps 결과와 30fps 결과 비교, 그림 저장 (API 호출 없음) |
| `scripts/sam_episodes.py` | OpenGraph 에피소드 전체를 10fps로 SAM에 보냄 (`hand`, `small object`). `--send` 없으면 클립 생성과 비용 출력만 |
| `scripts/sam_fast.py` | SAM 출력 빠른 리더. 공식 파서와 같은 디코더를 numba로 컴파일 (마스크당 약 3ms, 공식은 400ms). `--check`로 공식 파서와 대조 |
| `scripts/contact_cut.py` | 숫자 1: SAM 마스크로 손별 접촉 구간 추정 (특징 + 로지스틱 회귀 + 히스테리시스), boundary F1 / contact IoU 평가. `rule`은 B 규칙(겹침 + 동반 이동)과 참고 행을 leave-one-out으로 평가 (`--episodes`, `--name`으로 일부 에피소드와 출력 이름 지정, 격자는 병렬로 한 번만 계산) |
| `scripts/sam_track_check.py` | 에피소드 전체 SAM 추적 점검 (검출률, id 수명, 손 좌우, 접촉 프레임에서 손 검출), 타임라인 그림 |
| `scripts/hot3d_clip_count.py` | 모캡 물체 이동으로 HOT3D 잡기 클립 수와 길이를 세고 숫자 2 SAM 비용 계산 (API 호출 없음) |
| `scripts/hot3d_calib_check.py` | HOT3D 보정 방향 확인: 물체 모델 투영 vs 2D 박스, MediaPipe 손 vs 2D 손 박스 (방향, 프레임 밀림) |
| `scripts/hot3d_hands.py` | UmeTrack 프로필 순운동학으로 HOT3D 손 랜드마크 21개(손가락 끝 포함) 계산, 손 박스/영상/물체 거리로 검증 |
| `scripts/hot3d_copyable.py` | copyable 라벨: UmeTrack 손 → dex-retargeting Allegro → 표면 부호거리, 손목 이동 격자. `run` / `report` / `figures`, `--source clips`면 hot3d_clips.py 클립에 라벨, `--method` (ratio, scale1, position) |
| `scripts/hot3d_clips.py` | 숫자 2 클립: 모캡 손가락 끝-물체 거리로 접촉 구간을 찾아 클립 생성 (전 프레임, 병렬). `--plot SEQ OBJECT` 그림 |
| `scripts/hot3d_local_checks.py` | 숫자 2 로컬 채점 3개 항목 (카메라 안정성, 리타게팅 복원성, 동작 완결성). `video` (시퀀스별 MediaPipe + 광학 흐름, 캐시) → `clips` |
| `scripts/hot3d_signal_check.py` | 튜닝 클립에서 항목별 AUC, 물체 기준선 AUC (사람 하나씩 빼기), 사람별 AUC |
| `scripts/hot3d_hand_shape.py` | 접촉 중 손 모양 특징 10개 (영상 MediaPipe, 모캡 상한) → 튜닝 AUC, 물체 기준선과 비교 |
| `scripts/hot3d_label_compare.py` | copyable 라벨 방식 비교 (ratio / scale1 / position): 충실도, 손 크기 상관, 비율, 캔 잡기 그림 |
| `scripts/hot3d_copyable_v2.py` | copyable 라벨 v2 (position + 링크 메쉬 충돌 + 접촉점 도달). `run tune` 또는 `run held_out`, `report [split]`, `figures`. 숫자 2의 정답 |
| `scripts/hot3d_v2_signal.py` | v2 라벨로 무료 항목, 물체 기준선, 손 모양(영상/모캡) AUC를 튜닝에서 계산, 깊이 2/5/10 mm |
| `scripts/hot3d_object_geometry.py` | A안 확인: 3D 모델 기하 속성으로 v2 라벨 AUC (사람 하나씩 빼기, 물체 하나씩 빼기), 물체별 그림 |
| `scripts/hot3d_object_items.py` | Muse 물체 항목 6개를 정답 대리값으로 등급화하고 AUC 점검 (사람/물체 하나씩 빼기) |
| `scripts/hot3d_object_sheet.py` | 물체별 접촉 장면 한 장씩 모은 사진 (손잡이 판단용) |
| `scripts/hot3d_v2_heldout_summary.py` | v2 라벨의 튜닝 vs held-out 비율 비교 (사람별, 물체별). 라벨만, 모델 평가 없음 |
| `scripts/hot3d_number2_inputs.py` | 숫자 2 입력 준비: `select` (최종 클립과 역할), `windows` (SAM 윈도우, 비용), `cut` (10fps mp4), `frames` (Muse 프레임) |
| `scripts/sam_hot3d.py` | 숫자 2 SAM 윈도우 전송. `--send` 없으면 비용만. 크레딧 사용 |
| `scripts/muse_hot3d.py` | 숫자 2 Muse 체크리스트/기준선 호출. `--send` 없으면 비용만. 크레딧 사용 |
| `scripts/number2_sam_checks.py` | SAM 결과로 가시성, 접촉 명확성 등급 |
| `scripts/number2_pilot.py` | P1 파일럿 점검 (EXP-23): SAM이 고른 물체 vs 데이터셋 2D 박스, Muse 물체 등급 vs 3D 모델 대리값, 실제 호출 단가 (API 호출 없음) |
| `scripts/hot3d_fisheye_check.py` | 어안 왜곡 보정 점검 (EXP-25): 원본 vs 보정 프레임에서 MediaPipe 손 검출, 2D 오차, 손 모양 오차를 모캡과 비교. `FISHEYE_FOCAL=1.0`은 보조 확인 |
| `scripts/number2_score.py` | 숫자 2 점수 학습과 비교. `--check` (튜닝만, 코드 점검), `--gate` (P7a 뒤 게이트, `--standin`은 대리값 점검), `--final` (held-out, 마지막에 한 번). 출력에 물체 하나씩 빼기, 비교 5, 비교 6 (물체 비율 추가), extra (물체 안 AUC, 실패 원인별 AUC) 포함 |
| `scripts/number2_depth.py` | 숫자 2 held-out을 깊이 2 mm / 10 mm 라벨로도 채점 (EXP-41). 5 mm로 배운 점수 그대로, 5 mm 행이 EXP-40과 같지 않으면 중단. API 호출 없음 |
| `scripts/number2_power.py` | 비교 5 (학습 가중치 vs 같은 가중치) 검정력: 튜닝 3명짜리 가짜 held-out 20개, 대리값 (EXP-27, API 호출 없음) |
| `scripts/benchmark_export.py` | 정답을 공개 벤치마크로 내보냄 (`benchmark/number1/ground_truth.json`, `benchmark/number2/clips.csv`, 그림). 채점은 `benchmark/score_number1.py`, `score_number2.py` (프로젝트 코드와 독립, EXP-28) |
| `scripts/tuning_muse_check.py` | 진짜 Muse 답으로 튜닝 확인 (EXP-29): A 보정 (에피소드 하나씩 빼기), 숫자 2 R1/R2/R3 (사람 하나씩 빼기), 처음 보는 물체 |
| `scripts/prompt_revision_check.py` | Muse 오류 분석과 프롬프트 1회 수정 비교 (EXP-30): 체크리스트 v1/v2, A v1/v2, 측정 계획 규칙으로 held-out 버전 선택. `muse_hot3d.py --task checklist_v2`, `muse_contact.py --a-version 2` |
| `scripts/number2_gate_noise.py` | 대리값 등급에 무작위 오류를 섞어 게이트 결과가 어떻게 떨어지는지 (EXP-26, API 호출 없음) |
| `scripts/muse_contact.py` | 숫자 1 Muse 비교군 A, C. `a`, `c` (`--send` 없으면 비용만, `--sample N`은 C 보정용 튜닝 경계), `score` (A 보정은 `--a-shift-episodes` 0,30,53, C 보정은 `--c-shift-episodes` 튜닝 50개에서 배워 적용, 짝지은 부트스트랩, EXP-36), `selftest` (가상 Muse로 상한, `--lag`는 보정 점검) |
| `scripts/contact_cut.py rule-apply` | 튜닝에서 고른 B를 다른 에피소드에 적용해 구간 내보내기 (C의 입력). `--train tune --apply held_out` |
| `scripts/demo_figures.py` | 데모 영상용 그림 4장(최종 결과 파일에서)과 장면 1, 2, 6 글자 카드 3장 (1920x1080): `out/demo/`, 사본 `docs/demo/`. API 호출 없음 |
| `scripts/viewer_data.py` | 뷰어 데이터 (최종 held-out 결과, EXP-37, EXP-40, 보류 목록 EXP-42): `viewer/data/*.json`, `viewer/media/*.mp4` (git 제외). 숫자 2 점수와 숫자 1 행이 최종 실행과 같은지 확인 |
| `scripts/refine_check.py` | API 없는 점검: 촉각 임계값 vs 라벨, 짧은 구간, 머리 IMU 흔들림, 10fps 비용 |
| `scripts/refine_plots.py` | refine_check 그림을 `docs/experiments/img/`에 저장 |
| `scripts/regression_check.py` | 회귀 점검: 보고한 수치의 무료 명령 12개를 돌려 `out/regression/baseline/`과 바이트 비교. `save` / `compare [--only ...]`. 코드 정리 전후에 쓴다. 숫자 1은 파일럿 3개 전용 파일(`rule_pilot_*`)과 빈 C 폴더로 돌아 50개 B 출력을 덮어쓰지 않는다. `score_tune50`은 진짜 튜닝 C 답으로 최종과 같은 보정 |
| `scripts/number1_final_report.py` | 숫자 1 held-out 최종 보고 (EXP-37): `score` 결과와 같은 행을 다시 만들어 일치 확인, 참고 행 2개, 손별, B 지연, 사후 비교 (C-B, A+보정 - C+보정), 짧은 에피소드 9개 왼손. API 호출 없음 |
| `scripts/check_paid_cache.py` | 가짜 클라이언트로 유료 캐시 점검 (실패/해석 불가 응답은 완료로 저장 안 됨, arm C 구간 변경 시 중단, `--final` 빈 등급 시 중단, 보정 에피소드에 held-out이 있으면 중단). API 호출 없음 |
| `web/tools/demo_video.py` | 데모 영상 3분을 docs/DEMO_VIDEO.md대로 합성 (뷰어 데이터와 영상, `out/demo/` 그림과 카드). WSL에서 실행, 결과 `out/demo/graspgrade_demo.mp4`. `--only 3a`는 샷 하나, `--voice-text`는 샷별 내레이션 문구만 씀. 샷별 WAV가 있으면 길이를 맞추고 소리를 섞음. `--lang ko`는 한국어판 (`voice_ko/`, Heami 음성 -Rate 3, 결과 `graspgrade_demo_ko.mp4`). API 호출 없음 |
| `web/tools/demo_voice_el.py` | 영어 내레이션을 ElevenLabs Eric 음성으로 (`out/demo/voice_el/`, 이미 있는 샷은 건너뜀). `--send` 없으면 크레딧만 출력. 무료 크레딧 사용, 사용자가 호출하자고 할 때만. 영상은 `demo_video.py --voice el` |
| `web/tools/demo_voice.ps1` | 샷별 내레이션을 Windows 내장 합성 음성으로 WAV로 만듦 (PowerShell, WSL `out/demo/voice/`) |
| `web/tools/build_en.py` | 영어 보고서 페이지 `en.html`을 `index.html`과 `tools/en/`에서 만듦 |
| `web/tools/build_logs.py` | 실험 기록 전체 페이지 `logs.html`을 `docs/experiments/`에서 만들고 그림을 `assets/logs/`로 복사 |
| `web/tools/label_media.py` | 부드러운 물체 라벨링 페이지용 물체별 사진과 짧은 클립 (`label-media/`) |
| `scripts/muse_soft.py` | 부드러운 물체 확인: `segments` (OpenGraph 장갑 접촉 구간과 물체), `frames` (구간당 3장), `send --set pilot/hot3d/opengraph` (`--send` 없으면 비용만, 라벨 시트 미완성이면 중단, 크레딧 사용), `score` (잡은 비율, 보류 비율, 합격) |
