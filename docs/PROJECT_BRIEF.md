# GraspGrade 데스크톱 작업 환경 설정 + 프로젝트 히스토리

작성일: 2026-10-27. 대상 장비: 데스크톱 (RTX 3070, RAM 32GB). 사용자는 현재 한국에 있다.
이 파일은 Claude Code가 처음 읽을 문서다. 1부는 환경 설정, 2부는 프로젝트 히스토리와 결정 사항, 3부는 프로젝트 컨텍스트 전체이다.

---

# 1부. 데스크톱 환경 설정

## 1-0. 먼저 확인할 것
- 데스크톱 OS가 Windows인지 Linux인지 확인한다. Windows라면 모든 작업을 WSL2 Ubuntu 안에서 한다. HOT3D 툴킷과 dex-retargeting은 Linux와 macOS를 기준으로 만들어져 있다.
- 디스크 여유 공간은 100GB 이상 필요하다. 데이터, 에피소드 mp4, 클립, 캐시가 여기에 들어간다.
- GPU는 필수가 아니다. SAM 3.1과 Muse Spark는 Meta 서버에서 돌아간다. GPU는 API가 막히거나 크레딧이 모자라서 SAM을 로컬에서 돌려야 할 때를 대비한 것이다. 3070의 VRAM은 8GB라서 SAM 3 계열이 들어가는지는 아직 확인하지 않았다.

## 1-1. WSL2 (Windows인 경우만)
```powershell
# 관리자 PowerShell
wsl --install -d Ubuntu-24.04
```
- 재부팅한 뒤 Ubuntu 사용자를 만든다.
- 프로젝트는 반드시 WSL 파일시스템(`~/`) 안에 둔다. `/mnt/c/...` 아래에 두면 디스크 I/O가 매우 느려진다.
- GPU를 쓸 경우: NVIDIA 드라이버는 Windows 쪽에만 설치한다. WSL 안에 드라이버를 따로 설치하지 않는다. 설치 후 WSL에서 `nvidia-smi`가 동작하는지 확인한다.

## 1-2. 시스템 패키지
```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv git ffmpeg build-essential
```
- Python은 3.12를 쓴다. 3.11도 되지만 3.13은 안 된다.

## 1-3. 프로젝트와 가상환경
```bash
mkdir -p ~/graspgrade && cd ~/graspgrade
# 기존 프로토타입(touchgrain.zip)을 쓸지는 Official Rules의 사전 작성 코드 허용 여부에 따라 결정한다
# 허용되면: unzip touchgrain.zip
# 금지되면: 빈 repo에서 새로 시작한다
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "mediapipe==0.10.21" "dex_retargeting>=0.4,<0.5" "numpy==1.26.4" "pandas<3" "pyarrow<20" \
    opencv-python scikit-learn openai huggingface_hub
git clone --recurse-submodules https://github.com/dexsuite/dex-retargeting ext/dex-retargeting
```
기존 프로토타입에 `requirements.txt`가 있으면 위의 개별 설치 대신 `pip install -r requirements.txt`를 쓴다.

### 버전을 고정하는 이유 (직접 겪은 문제들)
- MediaPipe 최신 버전에서는 dex-retargeting 예제가 쓰는 옛 API(`mp.solutions`)가 빠졌다. 그래서 `mediapipe==0.10.21`로 고정한다.
- `mediapipe==0.10.21`은 numpy 2를 지원하지 않고, dex-retargeting 0.5.x는 numpy 2를 요구한다. 그래서 `dex_retargeting` 0.4.x와 `numpy==1.26.4`, `pandas<3`, `pyarrow<20`로 맞춘다.
- dex-retargeting은 torch가 없으면 import 단계에서 실패한다. CPU 버전으로 충분하다. 그냥 `pip install torch`를 하면 Linux에서는 수 GB짜리 CUDA 버전이 설치된다. 로컬 SAM이 필요해지면 그때 CUDA 버전으로 바꾼다.
- `--recurse-submodules`를 빼먹으면 로봇 URDF 폴더(Allegro 포함)가 비어 있게 된다.

## 1-4. 계정과 키
```bash
huggingface-cli login            # Hugging Face 토큰
echo "MODEL_API_KEY=..." > .env  # dev.meta.ai에서 발급받은 Meta Model API 키
export $(cat .env)
```
- `.gitignore`에 `.env`, `data/`, `out/`을 넣는다.
- HOT3D: projectaria.com/datasets/hot3D 에서 이메일로 등록하면 다운로드 URL이 담긴 JSON을 받는다. 이 JSON은 14일 동안 유효하다. 툴킷은 https://github.com/facebookresearch/hot3d 이고, 설치 방법은 이 repo의 README를 따른다. 손 포즈가 MANO 포맷이라 MANO 모델 등록(mano.is.tue.mpg.de)이 필요할 수 있다.

## 1-5. Meta API 연결
```python
from openai import OpenAI
import os
client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])

# Muse Spark: chat completions, 모델 muse-spark-1.3
# reasoning_effort="none"을 넣으면 400 에러가 난다. "low"를 쓴다.

# SAM 3.1: responses API, 모델 sam-3.1. 영상은 Files API로 먼저 올린다.
f = client.files.create(file=open("clip.mp4", "rb"), purpose="user_data")
stream = client.responses.create(
    model="sam-3.1",
    input=[{"role": "user", "content": [
        {"type": "input_text", "text": "hand"},
        {"type": "input_video", "file_id": f.id},
    ]}],
    stream=True,
)
```
- SAM 3.1은 텍스트 프롬프트만 받는다(점이나 박스는 안 된다). 입력은 MP4/MOV이고, 요청당 최대 15,000프레임, 프레임당 최대 16개 객체까지 처리한다.
- SAM 출력은 프레임마다 `<Nf>` 뒤에 `객체id, <|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|>, <|mask;...|>` 형식으로 나온다. 마스크는 base85로 인코딩되어 있다.
- 정확한 필드는 https://dev.meta.ai/docs/sam/segmenting 과 https://dev.meta.ai/docs/sam/reading-segmentation 에서 확인한다.
- `muse-spark-1.3-contributor` 모델은 더 싸지만 Meta가 프롬프트와 응답을 학습에 쓴다. CC-BY 공개 데이터에만 쓴다.

## 1-6. 데이터
```bash
# OpenGraph (6.3GB 전체)
huggingface-cli download OpenGraphLabs-Research/ego-tactile-manipulation \
    --repo-type dataset --local-dir data/ego-tactile
```
- LeRobot v3.0은 여러 에피소드를 mp4 하나에 이어 붙이고, 각 에피소드의 시작 시각을 메타데이터에 둔다. 에피소드별 mp4는 그 시작 시각으로 ffmpeg 절단해서 만든다. SAM 업로드 크기와 비용을 줄이기 위해 720p로 줄인다.
- 라벨 확인용 컬럼(이전 계획 기준, 실제 값은 다운로드 후 확인): `episode_index`, `annotation.anchor`, `annotation.hand`
- HOT3D는 Aria RGB 스트림만 쓴다(Quest 3 영상은 흑백). 어안 렌즈라 툴킷으로 왜곡을 보정해야 한다.

## 1-7. 설치 확인 순서
1. `python -c "import mediapipe, dex_retargeting, cv2, sklearn, openai; print('ok')"`
2. dex-retargeting 샘플 영상으로 리타게팅 확인:
   `ext/dex-retargeting/example/vector_retargeting/data/human_hand_video.mp4`
   이전 검증값: detection_rate 1.0, jitter 2.82mm, retarget_err 중앙값 15.7%
3. Muse Spark에 이미지 1장 호출해서 응답 확인
4. SAM 3.1에 짧은 mp4 1개 호출해서 실제 응답을 파서와 대조하고, 비용과 시간을 측정
5. OpenGraph 에피소드 1개로 MediaPipe 검출률 확인 (장갑 낀 손이라 낮을 수 있다)
6. HOT3D 시퀀스 1개를 받아서 MANO 손과 Aria RGB 좌표가 정렬되는지 확인

## 1-8. 주의사항
- MediaPipe는 셀카 기준으로 좌우를 붙인다. 그래서 거울 반전이 없는 1인칭 영상에서는 오른손이 "Left" 라벨로 나온다. 이전 샘플 테스트에서 반대로 설정했을 때 300프레임 중 0프레임이 검출됐다.
- dex-retargeting의 기본 설정은 `allegro_hand_right.yml`이다. 왼손을 따로 처리하지 않으면 데이터의 절반을 잃는다.
- repo를 public으로 바꾸기 전에 `.env`, `data/`, `out/`이 커밋되지 않았는지 확인한다.

---

# 2부. 프로젝트 히스토리 (어떻게 여기까지 왔나)

1. **원 아이디어 (이인석)**: 로봇 손끝 촉각 센서 데이터에 granularity를 조절할 수 있는 라벨링 툴. 준세는 원 아이디어에서 벗어나도 된다고 했다.
2. **1차 계획서 (touchgrain_hackathon_plan.md, 10/19)**: L1~L4 라벨, V0~V4 채점 방식 비교, HaWoR 손 포즈 추정, 숫자 3개. 폐기 이유: 정답 라벨이 사실상 HaWoR 품질을 측정했고, 10일 안에 하기에는 범위가 너무 컸으며(직렬로 합치면 11.5~13일), 숫자 3은 Do as I Do의 5%와 비교할 수 없는 지표였다.
3. **2차 계획서 (touchgrain_mvp_plan, 10/20)**: OpenGraph 하나로 축소. 폐기 이유: 숫자 2의 정답("접촉 경계가 ±5프레임 이내인가")이 우리 추정기가 맞았는지를 묻는 자기참조 구조였고, baseline의 질문과 정답이 맞지 않았다.
4. **현재 구조 (10/21)**: 정답을 두 개로 나눈다. OpenGraph 촉각으로 "언제", HOT3D 모캡으로 "로봇이 따라할 수 있는가"를 검증한다. HaWoR는 쓰지 않는다.
5. **포지셔닝 전환**: 우리 방법은 기존 필터를 대체하지 않고 보완한다. 기존 필터 위에 얹는 plug-in layer로 설명한다.
6. **점수 방식**: 로지스틱 회귀를 쓴다. 결정 트리는 쓰지 않는다. 통합 점수와 7개 등급을 둘 다 유지한다.
7. **뷰어**: 영상 위에 박스를 overlay하고, 접촉 신호 차트를 추가한다.
8. **이름 변경**: TouchGrain에서 GraspGrade로 바꿨다.
9. **지원서 제출 완료 (10/21)**: Indie Dev Track으로 냈다. 이인석은 별도로 제출해야 한다.
10. **작업 장비**: 데스크톱(3070, 32GB)을 메인으로 쓰고, 맥북에어는 보조로 쓴다.

---

# 3부. 프로젝트 컨텍스트

작성일: 2026-10-21. 이전 이름은 TouchGrain이다. 코드나 문서에 TouchGrain이 남아 있으면 GraspGrade로 바꾼다.

## 0. 작업 규칙

- 사용자: 임준세 (Joonse Lim). Duke Fuqua MBA 2027, CS 전공, 전 Shinhan Bank SWE.
- 팀: 준세 + 이인석 (fuqua.app 공동 제작자) 2인.
- 응답 스타일: 냉정하게, 추측 금지, 물어본 것만 답한다. 리스크는 맨 끝에 "리스크:" 한 줄과 "극복방안:" 한 줄로만 쓴다. 같은 내용을 두 번 쓰지 않는다.
- 영어 산출물: 쉬운 영어로 쓰고 em-dash, 축약형, 세미콜론을 쓰지 않는다.
- 사용자가 결정 트리는 원하지 않는다. 통합 점수는 로지스틱 회귀로 만든다.

## 1. 해커톤

- Meta Global AI Developer Hackathon, 2026-10-26 ~ 11-04. 최종 제출은 11/4 정오 ET.
- 트랙: Indie Dev Track (지원서 제출 완료).
- 선발자에게 Meta Model API 크레딧 $150가 지급된다.
- Official Rules는 10/26 전에 공개된다. 확인할 것: 사전 작성 코드 허용 여부, 팀 규모 제한. 사전 코드가 금지되면 기존 touchgrain.zip 프로토타입은 버리고 새 repo로 시작한다.
- 테마: Push the Frontier (robotics).

### 제출한 지원서 답변 (2026-10-21)

- LinkedIn: https://www.linkedin.com/in/joonse-lim
- GitHub: https://github.com/joonselim
- Twitter: 비움
- Location: Durham, NC
- Study/work: Duke University, Fuqua School of Business (MBA)
- Meta Model API 사용 경험: No / Very likely / Very favorable / Instagram ads로 알게 됨
- What to build:
  > GraspGrade finds the parts of hand videos that a robot hand can actually learn from. It cuts each video at the moment of touch and scores each clip, using SAM 3.1 and Muse Spark. We test both steps against real touch sensor and motion capture data.
- AI project:
  > I built fuqua.app with Inseok, my teammate for this hackathon. It is an app for Fuqua MBA students, and about 30% of the school uses it. Its course recommender uses a CART model trained on past course ratings and each student's concentration to rank classes for each student. Its grading tool uses CART on past grade distributions to estimate the grade curve of each class. I also built space.fuqua.app and ran a vibe coding workshop, so classmates with no coding background could build their own portfolio site with AI and publish it at their own fuqua.app address, like joonse.fuqua.app.
  - 주의: CART 모델의 입력 데이터("past course ratings and each student's concentration", "past grade distributions")는 Claude가 지어낸 문구이다. 실제와 다르면 질문을 받았을 때 바로 드러난다.

## 2. 한 줄 정의

1인칭 손 조작 영상을 손이 물체에 닿는 순간 기준으로 클립으로 자르고, 각 클립에 "로봇 손이 따라할 수 있는가" 점수(0~100)를 매긴다. 자르기는 촉각 장갑 데이터로, 점수는 모션캡처 데이터로 검증한다. 입력은 영상만 쓰고, 촉각과 모캡 데이터는 채점에만 쓴다.

포지셔닝: 우리 방법은 큰 랩들이 가진 기존 필터를 대체하지 않고 보완한다. 기존 필터 위에 센서로 검증된 접촉 타이밍과 설명 가능한 점수를 더하는 plug-in layer로 설명한다. 공개된 stand-in(Muse Spark 단독 또는 논문의 필터 규칙)을 써서 "기존 필터 단독 vs 기존 필터 + GraspGrade"로 비교한다. (제안서에는 아직 이 프레이밍이 반영되지 않았다.)

새 모델은 만들지 않는다. 우리가 기여하는 것은 세 가지이다.
1. 7개 체크와 학습된 가중치로 만든 점수
2. "로봇이 따라할 수 있는가"의 정답 정의 (손가락 끝과 물체의 기하 체크)
3. 센서 데이터로 이 방법이 AI 단독 판정보다 낫다는 것을 증명

## 3. 데이터셋

| 데이터 | 용도 | 비고 |
| --- | --- | --- |
| [OpenGraph Ego-Tactile](https://huggingface.co/datasets/OpenGraphLabs-Research/ego-tactile-manipulation) | 숫자 1 (접촉 시점) | 6.3 GB, 72 에피소드, 1.28시간, 1인칭 1080p, 양손 촉각 장갑(손당 80 택셀), LeRobot v3.0, CC-BY-4.0, 승인 불필요 |
| [HOT3D](https://projectaria.com/datasets/hot3D) | 숫자 2 (로봇 복제 가능성) | 833분 이상, 피험자 19명, 단단한 물체 33종, 모캡 3D 손/물체/카메라 포즈와 3D 모델. 촉각 데이터는 없다. 다운로드 링크는 14일 동안 유효하다 |
| [HOT3D-Clips](https://huggingface.co/datasets/bop-benchmark/hot3d) | HOT3D 대안 포맷 | webdataset 이미지 묶음이라 SAM 입력용 mp4를 다시 만들어야 한다 |

도구:
- HOT3D 툴킷: https://github.com/facebookresearch/hot3d
- dex-retargeting: https://github.com/dexsuite/dex-retargeting

쓰지 않는 데이터: EgoDex (NC-ND라서 결과를 공개할 수 없다), RoCo, FreeTacMan.

## 4. 파이프라인

### 처리 흐름
1. 영상 전체를 SAM 3.1에 넣어 프레임별 손/물체 박스와 마스크를 받는다. 깊이 정보 손실을 줄이기 위해 박스뿐 아니라 마스크도 저장한다.
2. 프레임마다 박스 겹침과 동반 이동을 계산해 0~1 접촉 신호를 만든다.
3. 각 접촉의 시작/끝 주변 약 7프레임만 Muse Spark에 보내 경계를 확정한다. 이유와 confidence도 저장한다.
4. ffmpeg로 앞뒤 여유를 두고 클립을 자른다.

### 7개 채점 항목
가시성, 카메라 안정성, 접촉 명확성, 동작 완결성(reach/release), 손가락 가림(Muse Spark), 물체 성질(Muse Spark), 리타게팅 복원성(MediaPipe + dex-retargeting, Allegro).

게이트: 장면 전환이 있거나 접촉이 없으면 0점.

### 점수
- 등급을 high=1, mid=0.5, low=0으로 바꿔 7개 숫자로 만든다.
- scikit-learn 로지스틱 회귀로 모캡 정답(copyable 여부)을 학습한다.
- 확률에 100을 곱해 0~100 점수로 쓴다. 가중치를 읽을 수 있다는 점이 장점이다.
- 통합 점수는 정렬과 숫자 2 검증에 쓰고, 7개 등급은 설명에 쓴다.
- 프리셋: 엄격 / 균형 / 물량.

## 5. 측정 (두 개의 정답)

| # | 질문 | 정답 | 비교 | 지표 |
| --- | --- | --- | --- | --- |
| 1 | 영상만으로 접촉 시작과 끝을 찾는가 | OpenGraph 촉각 장갑 | Muse Spark 단독 / SAM 3.1 규칙 / SAM 3.1 + Muse Spark 경계 확인 | boundary F1 (두 개의 허용 창), contact IoU |
| 2 | 우리 점수가 로봇이 따라할 수 있는 클립을 고르는가 | HOT3D 모캡 → dex-retargeting → Allegro 기하 체크 | 우리 점수 vs 체크리스트 없이 같은 질문을 받은 Muse Spark | AUC (부트스트랩 신뢰구간 포함), 상위 20% 양성 비율 vs base rate |

copyable 정의: 모캡 손을 Allegro로 매핑하고 Allegro 손목을 사람 손목 위치에 놓는다. 접촉 프레임에서 엄지와 다른 손가락 하나 이상이 물체 표면 가까이에 있고, 어느 손가락도 물체 안으로 깊이 들어가지 않으면 copyable로 본다. 거리 기준은 튜닝 split에서 정한 뒤 고정한다.

분할: OpenGraph는 에피소드 단위로 50개 튜닝, 22개 held-out으로 나눈다. HOT3D는 사람 단위로 나눈다. 분할은 Day 1에 고정한다.

## 6. 10일 계획

1. Day 1~2: OpenGraph와 HOT3D 일부를 다운로드하고, split을 고정하고, SAM 3.1과 Muse Spark를 연결한다. 에피소드 1개로 비용과 속도를 측정한다.
2. Day 3~4: OpenGraph로 접촉 자르기를 하고, "보이는 접촉"과 "장갑 압력" 사이의 시간차를 측정해 보정한다. 숫자 1.
3. Day 5~6: HOT3D 라벨링. 어안 영상을 왜곡 보정하고, 모캡 손을 dex-retargeting에 넣고, 손가락 끝과 물체 모델을 비교한다.
4. Day 7: HOT3D 클립에 7개 체크와 baseline을 돌리고 가중치를 학습한다. 숫자 2.
5. Day 8~9: 뷰어, 데모 영상, README.
6. Day 10: 최종 점검 후 정오 ET 전에 제출한다.

## 7. 기술 스택

- Python (venv). Meta API는 OpenAI Python 라이브러리에서 base URL만 Meta 주소로 바꿔 쓴다. SAM 3.1과 Muse Spark는 Meta 서버에서 돌아가므로 GPU가 필요 없다.
- 영상 처리: ffmpeg (자르기), OpenCV (프레임 읽기, 카메라 흔들림 측정)
- 손: MediaPipe (21개 관절, 로컬 CPU), dex-retargeting (Allegro)
- 데이터: huggingface-cli, pandas, scikit-learn
- 뷰어: GitHub Pages 정적 페이지 (HTML/JS 또는 React), 백엔드 없이 Python이 만든 JSON 2개를 읽는다.
- 노트북 사양: RAM 16GB, 디스크 50~100GB 여유.

### 뷰어 구성
- 왼쪽: 점수 순으로 정렬한 클립 목록
- 가운데: 비디오 플레이어 + canvas 박스 오버레이(접촉 시 색 변경) + 접촉 신호 라인 차트 + 우리 추정과 장갑 정답 타임라인
- 오른쪽: 7개 등급 카드와 Muse Spark 한 줄 이유
- 상단: 프리셋 3개

## 8. 알려진 기술 위험

1. HOT3D 정답 라벨이 한쪽으로 쏠릴 수 있다 (가장 큰 위험). Allegro 손이 사람 손보다 커서, 기준이 엄격하면 거의 전부 "불가", 느슨하면 거의 전부 "가능"이 될 수 있다. 손목 위치를 몇 cm 범위에서 최적화한 뒤 판정한다. 튜닝 세트에서 가능 비율이 20~80% 밖이면 정의를 다시 조정한다. Day 5에 가장 먼저 할 작업이다.
2. 숫자 2용 클립은 우리 파이프라인이 아니라 모캡 손과 물체 사이 거리로 자른다. 자르기 오류가 섞이지 않게 하기 위해서다.
3. HOT3D는 Aria RGB만 쓴다 (Quest 3 영상은 흑백). Aria는 어안이라 툴킷으로 왜곡 보정이 필요하다. 손 포즈는 MANO 포맷이다.
4. dex-retargeting 기본 설정은 Allegro 오른손이다. 왼손을 처리하지 않으면 데이터의 절반을 잃는다.
5. HOT3D는 단단한 물체뿐이고 실험실에서 고정 촬영했다. 그래서 "물체 성질"과 "카메라 안정성" 항목은 분산이 0일 수 있다. 그런 경우 회귀에서 빼고 데모에서 설명한다.
6. 피험자가 19명뿐이라 검증 세트는 4~5명이 된다. AUC에는 부트스트랩 신뢰구간을 함께 보고한다.
7. OpenGraph는 촉각 장갑을 낀 손이라 MediaPipe 검출률이 떨어질 수 있다.
8. 영상상 접촉 시점과 촉각 압력 임계 시점 사이에 체계적 지연이 있어 ±5프레임 허용 범위를 잠식할 수 있다.
9. LeRobot은 여러 에피소드를 하나의 mp4에 담는다. 메타데이터의 시작 시각을 기준으로 ffmpeg로 잘라야 한다.
10. 우리 문제 제기는 인터넷 영상이지만 검증은 실험실 데이터로 한다. 데모 끝에 직접 찍은 1인칭 영상 몇 개를 보여준다.

## 9. 할 일

- [ ] Official Rules 확인: 사전 코드 허용 여부, 팀 규모
- [ ] Meta Model API 키 발급 (dev.meta.ai), Hugging Face 토큰 발급
- [ ] HOT3D-Clips 클립 1개로 MediaPipe를 돌려 좌표 정렬이 가능한지 확인
- [ ] OpenGraph를 다운로드해 실제 fps, 라벨 생성 방식, 타임스탬프 간격을 확인하고 ±5프레임 기준을 초 단위로 다시 정의
- [ ] SAM 3.1 실제 응답을 기존 파서와 대조하고, 에피소드 1개의 비용과 시간 측정
- [ ] 제안서와 one-pager에 "기존 필터의 보완재" 프레이밍 반영
- [ ] one-pager 이름을 TouchGrain에서 GraspGrade로 변경
- [ ] 이인석 지원서 별도 제출 여부 확인
- [ ] 10/25~12/2 한국 체류 일정과 해커톤 기간이 겹친다. 행사가 온라인인지, 마감이 한국 시간으로 몇 시인지 확인

## 10. 산출물 링크

- 제안서 (Claude Doc): https://claude.ai/artifact/5ZA5Tc99Rjt5UbjhxguMzY
- 제안서 (HTML): https://claude.ai/artifact/DRwFXtPtJM6UjBGk4fLBN2
- One-pager (HTML, 이름 변경 전): https://claude.ai/artifact/4so1DrQxogYfb5wsvTmBMv
- GitHub Pages 배포 예정 경로: https://joonselim.github.io/graspgrade (repo `joonselim/joonselim.github.io`, 브랜치 `main`, 루트에 index.html만 있는 정적 사이트, 파일 위치 `graspgrade/index.html`)
