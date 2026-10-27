# GraspGrade 진행 상황

마지막 갱신: 2026-10-27 (해커톤 Day 2). 마감 2026-11-04 정오 ET.

## 한 줄 상태

Meta API 호출 직전까지 끝났다. Meta API는 아직 한 번도 호출하지 않았다.
git 저장소는 `~/graspgrade`에 init만 했고 커밋은 0개다.

## 끝난 것

| 단계 | 결과 | 근거 |
| --- | --- | --- |
| WSL2 + 시스템 패키지 | Ubuntu-24.04, ffmpeg 6.1.1 (libdav1d 포함) | docs/ENVIRONMENT.md |
| venv + 패키지 | 버전 고정 설치 완료 | requirements.txt |
| 환경 검증 | 검출률 1.000, jitter 1.23 mm/frame, 리타게팅 오차 중앙값 17.0% | `scripts/verify_env.py` |
| 키 | `~/graspgrade/.env`에 MODEL_API_KEY, HUGGING_FACE_TOKEN (600, gitignore) | |
| OpenGraph 다운로드 | 72 에피소드, 138,629 프레임, 30fps, 5.9 GB | `data/ego-tactile` |
| OpenGraph 자르기 | 72개 전부 720p H.264, 프레임 수 일치, 시작 프레임 정렬 확인 | `data/episodes_720p`, `cut_report.json` |
| MediaPipe 검출률 | 8 에피소드 평균 82.3%, 배경에 따라 14%~100% | `out/mediapipe_check/summary.json` |
| HOT3D | 198 시퀀스 전부 ground_truth + hand_data, 물체 3D 모델, 영상은 P0001_10a27bf7 1개 | `data/hot3d` (7.3 GB) |
| 분할 고정 | OpenGraph 50/22, HOT3D 사람 6/3 | `splits/*.json` |

브리프 1-7 설치 확인 순서 기준: 1, 2, 5번 완료. 3, 4번(Meta API)은 대기. 6번(HOT3D 좌표 정렬)은 미완.

## 계획과 달랐던 사실 (자세한 내용은 DATA_NOTES.md)

1. OpenGraph 영상은 AV1이다. OpenCV는 못 읽는다. 항상 `data/episodes_720p`의 H.264를 쓴다.
2. OpenGraph 라벨은 촉각 신호에서 자동 생성됐다. 접촉 = `annotation.anchor != "transition"`.
3. ±5프레임 = ±0.167초 (30fps, 간격 정확히 1/30초).
4. HOT3D Aria에서 정답(포즈)이 있는 사람은 9명, 136 시퀀스다. 19명은 Quest 포함 숫자다. 검증 세트는 3명, 51 시퀀스.
5. HOT3D `preview_rgb.mp4`가 원본 해상도(1408x1408, 30fps, 똑바로 선 방향)다. VRS(329 GB)는 필요 없다.
6. 장갑 손 MediaPipe 미검출은 좌우 라벨 문제가 아니라 손을 놓친 것이다. 베이지 벽 앞 스프레이병 장면(데이터의 약 20%)이 가장 나쁘다.
7. HOT3D 다운로드 URL은 약 2026-11-10까지 유효하다.

## 분할

- OpenGraph held-out 22개: 7, 14, 18, 24, 27, 34, 36, 37, 39, 40, 41, 42, 43, 45, 48, 49, 52, 57, 58, 64, 66, 67
- HOT3D 튜닝: P0002, P0003, P0009, P0011, P0014, P0015 (85 시퀀스)
- HOT3D held-out: P0001, P0010, P0012 (51 시퀀스)
- 포즈 파일이 빈 62 시퀀스(P0004, P0005, P0008, P0016, P0020 등)는 제외.

held-out은 최종 측정 전까지 튜닝에 쓰지 않는다.

## 다음 할 일 (순서대로)

1. **Meta API 스모크 테스트 (사용자 승인 필요)**
   - Muse Spark: chat completions, `muse-spark-1.3`, 이미지 1장, `reasoning_effort="low"` (`"none"`은 400 에러).
   - SAM 3.1: Files API로 짧은 mp4 업로드(`purpose="user_data"`), responses API `sam-3.1`, 텍스트 프롬프트 `"hand"`, `stream=True`.
   - 클라이언트: `OpenAI(base_url="https://api.meta.ai/v1", api_key=MODEL_API_KEY)`.
   - 입력 후보: 튜닝 split의 `data/episodes_720p` 에피소드 일부를 몇 초로 자른 것.
   - 확인할 것: 실제 응답 형식(브리프 1-5의 `<Nf>` 박스/마스크 형식과 대조), 토큰/비용, 걸린 시간.
   - 결과를 바탕으로 에피소드 1개 전체 비용과 시간을 추정한다 (Day 1~2 목표).
2. Day 3~4: OpenGraph 접촉 자르기 (SAM 박스 겹침 → 접촉 신호 → Muse Spark 경계 확인), 보이는 접촉과 장갑 압력 사이 시간차 보정. 숫자 1.
3. Day 5 첫 작업: HOT3D 카메라 보정 방향 확인 (아래 미해결 1). 그다음 copyable 라벨 쏠림 확인 (브리프 8-1).
4. 남은 HOT3D 영상 다운로드: 정답 있는 136 시퀀스의 `preview_rgb.mp4`. `scripts/hot3d_download.py`에 preview 파일 그룹을 지정해서 받는다 (실행 전 JSON의 그룹 이름 확인).

## 미해결

1. HOT3D `camera_models.json`(FISHEYE624)이 똑바로 선 preview 기준인지, 90도 돌아간 Aria 원본 기준인지 모른다.
   확인 방법: `box2d_hands.csv`(스트림 214-1)의 정답 박스와 투영한 MANO 손을 겹쳐 본다.
2. HOT3D 툴킷 / projectaria_tools / MANO 모델 등록은 아직 안 했다.
3. 브리프 9장 할 일 중 Official Rules 확인, 제안서 프레이밍 반영, one-pager 이름 변경, 이인석 지원서, 한국 일정은 이 환경 작업과 무관해서 손대지 않았다.

## 디스크

WSL `~/graspgrade/data` 14 GB (ego-tactile 5.9G, hot3d 7.3G, episodes 511M). WSL 가상 디스크는 C: 드라이브에 있으므로 실제 한도는 C: 여유 공간이다 (2026-10-27 시작 시 170 GB).
`data/hot3d/aria/*/` 안의 원본 zip은 압축을 푼 뒤에도 남아 있다. 공간이 필요하면 지워도 된다.
