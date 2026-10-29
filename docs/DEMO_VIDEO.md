# 데모 영상 구성안

길이 목표 3:00. 공식 규칙의 길이 제한은 아직 확인하지 않았다 (사용자 확인 항목).
화면은 뷰어(`viewer/`) 녹화와 실험 그림 3장이 전부다. 새 계산이나 API 호출은 없다.
내레이션과 화면 글자는 영어(쉬운 영어, em-dash, 축약형, 세미콜론 없음). 내레이션은 454단어로, 분당 150단어로 읽으면 약 3분이다.

## 장면 순서

| # | 시간 | 장면 | 화면 |
| --- | --- | --- | --- |
| 1 | 0:00-0:15 | 문제 | HOT3D 클립 2개 전체 화면 (컵, 키보드) |
| 2 | 0:15-0:35 | GraspGrade가 하는 일 | 제목 카드, 입력과 출력 3줄 |
| 3 | 0:35-1:25 | 숫자 1: 접촉 시점 | 뷰어 Contact timing, 에피소드 36과 7, 그림 `demo_number1.png` |
| 4 | 1:25-2:25 | 숫자 2: 로봇 손 점수 | 뷰어 Clip scores, 클립 3개, 그림 `demo_number2_ours.png` |
| 5 | 2:25-2:48 | 안 되는 것 | 같은 물체 컵 클립 2개 나란히, 그림 `demo_number2_limits.png`, `demo_soft.png` |
| 6 | 2:48-3:00 | 공개하는 것, 마무리 | 끝 카드 |

## 보여줄 클립과 에피소드

모두 held-out이다. 숫자는 `viewer/data/*.json`과 최종 기록(EXP-37, EXP-40)에서 옮겼다.

### 숫자 1 (OpenGraph, 뷰어 Contact timing 탭)

| 에피소드 | 길이 | 장갑 접촉 수 (왼/오) | F1 ±5: A+shift / B / C+shift | 쓰는 이유 |
| --- | --- | --- | --- | --- |
| 36 | 68.8 s | 8 / 12 | 0.229 / 0.065 / 0.163 | 전체 순위(A+shift > C+shift > B)와 같다. one-pager 그림과 같은 에피소드 |
| 7 | 77.8 s | 4 / 34 | 0.040 / 0.197 / 0.250 | 짧은 접촉이 많으면 순위가 뒤집힌다. 정직한 반례 |

### 숫자 2 (HOT3D, 뷰어 Clip scores 탭)

| 클립 | 물체 | 점수 | 정답 | Muse Spark 단독 | 쓰는 이유 |
| --- | --- | --- | --- | --- | --- |
| `P0010_99385ad8_left_00073` | mug_white | 78.3 | copyable | 0.85 | 기본 예시. 등급 12개와 가중치를 보여 준다 |
| `P0012_d6272ce1_left_00290` | keyboard | 9.8 | 실패 (collision) | 0.85 | 우리 점수는 맞고 Muse 단독은 틀림 |
| `P0001_b2bcbe28_left_03432` | aria_small (안경) | 77.6 | copyable | 0.05 | 반대 방향으로 Muse 단독이 틀림 |
| `P0010_1573a837_right_02542` | mug_patterned | 74.1 | copyable | 0.90 | 장면 5. 아래 클립과 짝 |
| `P0001_b2bcbe28_left_01861` | mug_patterned | 74.6 | 실패 (grip) | 0.85 | 장면 5. 점수는 위 클립과 거의 같은데 정답이 다름 (물체 안에서는 우연 수준). 등급 12개 중 9개가 같다 |

장면 1과 5의 클립 4개는 접촉 구간만 자르면 3.0~4.3초라 장면을 못 채운다. `viewer_data.py`의 `DEMO_PAD`가 이 4개에만 앞뒤로 3초(90프레임)씩 더 붙여 8.7~10.3초로 만든다 (영상 처음이나 끝에 걸리면 있는 만큼만). 뷰어가 그리는 접촉 구간도 같이 밀린다. 나머지 클립과 에피소드 영상은 그대로다.

참고: 뷰어 상위 15개 중 11개가 copyable, 하위 15개 중 14개가 실패다. 이 30개는 극단값이라 영상에서는 이 비율 대신 전체 수치(상위 20% 중 59.4%, 기준 46.4%)를 쓴다.

### 그림과 카드 (영상용 1920x1080)

`scripts/demo_figures.py`가 최종 결과 파일(EXP-37, EXP-40, EXP-42)에서 그림 4장을 다시 그리고, 글자 카드 3장도 그린다. 원본은 WSL `out/demo/`, 사본은 `docs/demo/`.

- `docs/demo/demo_number1.png`: 숫자 1. 경계 F1 ±5와 IoU, A + shift(파랑)와 B(주황) 강조
- `docs/demo/demo_number2_ours.png`: 숫자 2 AUC. 우리 점수(파랑)와 Muse Spark 단독(주황) 강조
- `docs/demo/demo_number2_limits.png`: 같은 막대. 물체 이름(주황)과 처음 보는 물체(파랑) 강조
- `docs/demo/demo_soft.png`: 부드러운 물체 확인. 합격선 대비 51.7%(미달), 2.1%(충족)
- `docs/demo/demo_card_scene1.png`: 장면 1 화면 글자. 배경이 투명한 PNG이고, 아래쪽 반투명 검은 띠 위에 흰 글자. 영상 위에 겹친다
- `docs/demo/demo_card_scene2.png`: 장면 2 제목 카드. 입력 상자, 화살표, 출력 3줄, 아래에 "SAM 3.1 · Muse Spark"
- `docs/demo/demo_card_scene6.png`: 장면 6 끝 카드. 이름, 한 줄 소개 "Find the hand videos a robot can learn from.", Code / Viewer / Benchmark, 데이터 출처. 저장소 주소는 공개 범위가 정해지지 않아 넣지 않았다

## 장면별 화면과 내레이션

### 1. 문제 (0:00-0:15)

화면: `P0010_99385ad8_left_00073` (컵, 8.7초) 재생, 이어서 `P0012_d6272ce1_left_00290` (키보드, 10.0초). 소리 없음. 두 개로 18.7초라 15초 장면을 채우고 남는다.

화면 글자: **Not every hand video can teach a robot hand.** (`demo_card_scene1.png`를 영상 위에 겹침)

> Robot hands can learn to grasp by watching people. But much hand video cannot teach a robot hand. The fingers are hidden, or the grasp is one a robot hand cannot copy.

### 2. GraspGrade가 하는 일 (0:15-0:35)

화면: `demo_card_scene2.png`. 왼쪽 입력 "First-person hand video", 오른쪽 출력 3줄 "Clips cut at each touch", "A 0 to 100 score: can a robot hand copy it?", "A one-line reason". 아래에 "SAM 3.1 · Muse Spark".

> GraspGrade takes first-person hand video, cuts it into clips at the moment of touch, and gives each clip a score from 0 to 100 for how likely a robot hand can copy it. SAM 3.1 tracks the hands and objects. Muse Spark answers a short checklist about each clip. We check both steps against real sensor data.

### 3. 숫자 1: 접촉 시점 (0:35-1:25)

화면 순서:
1. `http://localhost:8765/#episode=36&hand=right`: 에피소드 36, 오른손 트랙 4줄(glove, A + shift, B, C + shift)만 보인다.
2. glove 구간 하나를 클릭해 영상이 그 접촉으로 넘어가는 것을 보여 준다.
3. 그림 `demo_number1.png` (5초).
4. `http://localhost:8765/#episode=7&hand=right`: 에피소드 7 오른손 (5초).

화면 글자: **Held-out: 22 episodes, 574 glove boundaries**

> First, when does a touch start and end? We check this against a touch-sensing glove. The top track is the glove. Below it are three methods: Muse Spark alone, our SAM rule, and the SAM rule checked by Muse Spark.
>
> Video shows a touch before the glove feels pressure, so we learn one timing shift, on tuning episodes only. On 22 held-out episodes, Muse Spark with the shift is best. Its boundary F1 within five frames is 0.195, against 0.090 for the SAM rule.
>
> All methods are low at five frames, because the glove measures pressure and video cannot see pressure. When touches are many and short, like in this episode, the SAM rule checked by Muse Spark does better.

### 4. 숫자 2: 로봇 손 점수 (1:25-2:25)

화면 순서:
1. `http://localhost:8765/#clip=P0010_99385ad8_left_00073` (컵, 78). 오른쪽에 등급 12개와 가중치가 보인다.
2. `http://localhost:8765/#clip=P0012_d6272ce1_left_00290` (키보드, 10). 편집에서 정답 "collision"과 "85% copyable" 칸을 확대.
3. `http://localhost:8765/#clip=P0001_b2bcbe28_left_03432` (안경, 78). 편집에서 "5% copyable" 칸을 확대.
4. 그림 `demo_number2_ours.png` (8초).

화면 글자: **Held-out: 1,426 clips from 3 people**

> Second, can a robot hand copy this grasp? The ground truth comes from motion capture in the HOT3D dataset from Meta. We place an Allegro robot hand on each human grasp and check if it reaches the same contact points without passing through the object.
>
> This mug clip scores 78, and the robot hand can copy it. The score is a logistic regression on twelve grades, so every weight is visible.
>
> This keyboard clip scores 10. The robot hand would pass through the keyboard. Asked alone, Muse Spark said yes, at 85 percent. This glasses clip scores 78 and is copyable. Muse Spark alone said 5 percent.
>
> On 1,426 held-out clips from three people, our score reaches an AUC of 0.624. Muse Spark alone, asked the same question, is at 0.501, which is chance. The gain comes from asking a checklist instead of one question.

### 5. 안 되는 것 (2:25-2:48)

화면 순서:
1. 그림 `demo_number2_limits.png` (6초).
2. `http://localhost:8765/#compare=P0010_1573a837_right_02542,P0001_b2bcbe28_left_01861`: 컵 클립 2개 나란히, 74점 copyable과 75점 grip 실패. 나란히 재생이라 길이는 긴 쪽인 10.3초다. 장면에 배정된 약 13초보다 짧으므로 편집에서 마지막 프레임을 멈춰 맞춘다 (결정 2026-10-29).
3. 그림 `demo_soft.png` (4초).

화면 글자: **Rules written before held-out scoring. Held-out scored once.**

> We also show what does not work. When an object was seen in training, its name alone does better, at 0.771. On new objects our score keeps 0.571. Inside one object it is at chance. These two mugs get almost the same score, but only one grasp is copyable. Our soft object filter also failed its pass rule. Every rule was written before we scored the held-out data, and we scored it once.

### 6. 공개하는 것, 마무리 (2:48-3:00)

화면: `demo_card_scene6.png`. "Code", "Viewer", "Benchmark"와 각 한 줄 설명, 아래 작은 글씨로 데이터 출처 "OpenGraph Ego-Tactile (CC-BY-4.0) · HOT3D (Meta)".

> The code, the viewer, and both ground truths are open, as a benchmark with its own scorers. GraspGrade finds the hand videos a robot can learn from.

## 녹화 준비

1. 뷰어 데이터와 영상은 WSL에만 있다 (`~/graspgrade/viewer/media/`). WSL에서 서버를 띄우고 Windows 브라우저로 `http://localhost:8765`를 연다. 같은 명령이 `.claude/launch.json`의 `viewer`에도 있다.

   ```bash
   MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -e bash -c 'cd ~/graspgrade/viewer && python3 -m http.server 8765'
   ```

2. 장면마다 위의 링크(`#clip=`, `#episode=…&hand=right`, `#compare=`)로 바로 연다. 해시 없이 열면 예전처럼 첫 탭이 나온다. `viewer_data.py`를 다시 돌린 뒤에는 캐시를 비우고 새로 고친다 (Ctrl+Shift+R). 안 그러면 예전 길이의 영상이 그대로 나온다.
3. 화면 1920x1080, 브라우저 확대 100%, 라이트 테마, 전체 화면(F11). 이 크기에서 뷰어 화면 6개(클립 3개, 비교 1개, 에피소드 2개)가 모두 세로 1080 안에 들어가는 것을 확인했다. 확대하면 오른쪽 등급 목록이 잘린다. 장면마다 따로 녹화해 편집에서 잇는다.
4. 내레이션은 장면별로 따로 녹음해 길이를 맞춘다. 자막은 내레이션 문구 그대로 쓴다.
5. 루트의 `demo.html`은 쓰지 않는다. 4번 장면이 "Illustration. The scoring model is being built"와 "seven checks"로 최종 결과와 맞지 않는다.

## 사용자 결정 필요

- 공식 규칙의 영상 길이 제한과 형식. 3분보다 짧으면 장면 5의 그림 `demo_soft.png`와 장면 3의 에피소드 7을 먼저 뺀다.
- HOT3D 영상을 공개 영상에 넣어도 되는지 (라이선스 미확인). 안 되면 장면 1, 4, 5의 HOT3D 화면을 그림과 트랙만으로 바꾼다.
- 내레이션 목소리 (직접 녹음 또는 음성 합성).
- 제안서 위험 10의 "직접 찍은 1인칭 영상"은 넣지 않았다. 넣으려면 SAM과 Muse 호출이 필요하고, 남은 크레딧은 $6.73이다.
