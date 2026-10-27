# GraspGrade: Claude Code 작업 안내

이 파일은 새 Claude Code 세션이 바로 이어서 작업할 수 있도록 쓴 안내다.
작업 전에 아래 문서를 이 순서로 읽는다.

1. [docs/PROGRESS.md](docs/PROGRESS.md): 어디까지 했고 다음에 무엇을 할지
2. [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md): 프로젝트 정의, 파이프라인, 10일 계획, 위험 (2026-10-21 원본)
3. [docs/DATA_NOTES.md](docs/DATA_NOTES.md): 데이터에서 확인한 사실
4. [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md): 환경, 버전 고정 이유, 함정

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

    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -e bash -c 'cp -r /mnt/c/Users/USER/Documents/works/hackathon_meta/{scripts,docs,splits,CLAUDE.md,requirements.txt} ~/graspgrade/'

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
| `scripts/contact_cut.py` | 숫자 1: SAM 마스크로 손별 접촉 구간 추정 (특징 + 로지스틱 회귀 + 히스테리시스), boundary F1 / contact IoU 평가 |
| `scripts/sam_track_check.py` | 에피소드 전체 SAM 추적 점검 (검출률, id 수명, 손 좌우, 접촉 프레임에서 손 검출), 타임라인 그림 |
| `scripts/hot3d_clip_count.py` | 모캡 물체 이동으로 HOT3D 잡기 클립 수와 길이를 세고 숫자 2 SAM 비용 계산 (API 호출 없음) |
| `scripts/hot3d_calib_check.py` | HOT3D 보정 방향 확인: 물체 모델 투영 vs 2D 박스, MediaPipe 손 vs 2D 손 박스 (방향, 프레임 밀림) |
| `scripts/hot3d_hands.py` | UmeTrack 프로필 순운동학으로 HOT3D 손 랜드마크 21개(손가락 끝 포함) 계산, 손 박스/영상/물체 거리로 검증 |
| `scripts/refine_check.py` | API 없는 점검: 촉각 임계값 vs 라벨, 짧은 구간, 머리 IMU 흔들림, 10fps 비용 |
| `scripts/refine_plots.py` | refine_check 그림을 `docs/experiments/img/`에 저장 |
