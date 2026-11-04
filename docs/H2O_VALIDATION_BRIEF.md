# H2O 외부 검증: 검증 세션 인계 메모

작성 2026-10-30, 실험 세션. 검증 세션이 이 파일만 읽고 바로 시작할 수 있게 썼다.
아직 아무것도 받지 않았고, 계산이나 API 호출도 하지 않았다.

> 상태 (2026-11-04): 중단. H2O는 ETH 서버가 다운로드를 막아 받지 못했다. 검증 세션은 OakInk2(`E:\oakink2`, CC BY-SA 4.0)로 바꿨지만 MANO 파일을 기다리는 단계에서 멈췄다. 측정 계획 절, 라벨, 유료 호출, EXP-43 기록 모두 없다. 아래 계획은 그대로 다시 쓸 수 있다.

## 1. 목적

숫자 2(로봇 손이 따라 할 수 있는 잡기 점수)가 HOT3D 밖에서도 유지되는지 본다.
H2O는 물체, 사람, 카메라가 모두 처음 보는 데이터다.

확인할 HOT3D 결과 (EXP-40, held-out 1,426클립):

| 행 | AUC | 95% 구간 |
| --- | --- | --- |
| R1 Muse Spark 단독 (같은 질문) | 0.501 | 0.469~0.529 |
| R2 체크리스트, 같은 가중치 (학습 없음) | 0.617 | 0.579~0.659 |
| R2 - R1 (체크리스트 효과) | +0.117 | 0.073~0.166 |
| 우리 점수 (12개 항목 로지스틱 회귀) | 0.624 | 0.589~0.661 |
| 우리 점수, 처음 보는 물체 | 0.571 | 0.534~0.609 |
| 물체 이름 기준선 | 0.771 | 0.744~0.792 |

- 가장 옮겨 가기 쉬운 주장은 "체크리스트(R2)가 Muse 단독(R1)보다 낫다"다.
- R1과 R2는 학습이 필요 없고 SAM 호출도 필요 없다.
- H2O에서는 물체 이름 기준선을 쓸 수 없다. 물체 8종 모두 처음 보는 물체라 기준선이 0.5가 된다.

숫자 1(접촉 시점)은 하지 않는다.
- H2O에는 촉각 장갑이 없다. 정답이 손-물체 거리로 바뀌어 장갑 결과와 비교할 수 없다.
- verb 라벨(grab, place)은 동작 구간이지 접촉 경계가 아니다.

## 2. H2O에서 확인한 사실

출처:
- 아래 README 원문 (사용자가 받아 붙인 것)
- ETH Research Collection: https://www.research-collection.ethz.ch/entities/researchdata/e629726a-b4df-4e4e-8474-9eb44eef98e2
- 프로젝트 페이지: https://taeinkwon.com/projects/h2o/

확인한 것:
- 라이선스: CC BY-NC 4.0 (ETH 페이지 기준).
- 파일: ETH 페이지에서 바로 받을 수 있는 것은 README.md(4.78 KB)뿐이다. 데이터는 "Access the files" 링크(libdrive.ethz.ch)에 있다. 파일 목록과 크기는 페이지에 없다.
- 물체 메쉬: 소개글에 "object meshes and scene point clouds"를 준다고 적혀 있다. 실제 파일은 받아서 확인해야 한다.
- 카메라: cam4가 머리 착용 카메라(1인칭)이고 해상도 1280x720이다. 카메라 보정값이 fx, fy, cx, cy 4개라 어안이 아닐 것으로 보인다. 받아서 확인해야 한다.
- 분할: test는 subject4 (h1, h2, k1, k2, o1, o2).

확인하지 않은 것:
- 프레임 수, 실제 렌즈 형태, tar 파일 크기.
- hand_pose 관절 21개의 순서 (MediaPipe 순서인지).
- 좌표계가 정확히 무엇인지 (cam_to_hand, cam_to_obj).
- annotate 0인 프레임이 얼마나 많은지.

## 3. 계획 (순서대로)

### 1단계: 무료 확인 (기한 하루, 실패하면 접는다)

1. 받을 파일을 정한다: subject4의 cam4 `rgb`, `hand_pose`(또는 `hand_pose_MANO`), `obj_pose_RT`, `cam_intrinsics.txt`, 물체 메쉬.
   - 시작 전에 사용자에게 파일 이름, 출처, 크기를 보여주고 승인을 받는다.
   - libdrive에서 바로 받을 수 없으면 사용자가 받는다.
   - tar가 카메라 5개를 모두 담고 있으면 크기가 클 수 있다. WSL 여유 공간은 약 900 GB다.
2. 좌표와 관절 순서를 확인한다. 손 관절과 물체 메쉬를 cam4 영상에 투영해서 눈으로 본다 (HOT3D EXP-08, EXP-09와 같은 방식).
3. 클립을 찾는다. HOT3D 규칙(EXP-11)을 그대로 쓴다:
   - 손별로 손가락 끝 2개 이상이 물체 표면 15 mm 이내면 접촉.
   - 0.5초 이하 끊김은 합친다.
   - 0.3초 이상인 것만 남긴다.
   - 앞뒤로 1초씩 붙인다.
   - 물체가 3 cm 이상 움직여야 한다.
   - HOT3D의 QA 90% 조건은 H2O에 없다. annotate 플래그로 바꿀지 1단계에서 정하고 기록한다.
4. copyable v2 라벨을 HOT3D와 같은 정의로 계산한다:
   - position 리타게팅, 손목 이동 2 cm 이내 탐색.
   - 잡기: Allegro 엄지 끝 + 다른 손가락 끝 1개가 표면 1.5 cm 이내.
   - 도달: 닿은 사람 손가락마다 Allegro 손가락 끝이 2.0 cm 이내.
   - 충돌 없음: Allegro 링크 메쉬 점이 물체 안 5 mm를 넘지 않음.
5. 멈출 조건:
   - copyable 비율이 20~80%를 벗어난다. 물체 8종이 대부분 상자나 병(책, 에스프레소 상자, 로션, 스프레이, 우유, 코코아, 과자, 카푸치노)이라 한쪽으로 쏠릴 수 있다.
   - 좌표가 맞지 않는다.
   - 메쉬가 없다.
   - 이 경우에는 멈추고 한계로만 기록한다.
6. 클립 수와 유료 비용을 계산한다.

### 2단계: 측정 계획에 규칙 먼저 쓰기

`docs/MEASUREMENT_PLAN.md`에 H2O 절을 추가한다. 계산 전에 쓴다.
- 클립 선택 규칙. 예산 안에서 몇 개를 뽑을지와 seed (기존 seed는 20261026).
- 행:
  - R1 Muse 단독
  - R2 체크리스트 같은 가중치
  - (SAM까지 쓰면) HOT3D 튜닝 400개로 배운 우리 점수를 그대로 적용
- 지표: AUC, 시퀀스 단위 부트스트랩 95% 구간, R2 - R1 짝지은 부트스트랩.
- 합격 기준 예시: R2 - R1 구간이 0 위면 "체크리스트 효과가 옮겨 간다".
- 채점은 한 번만 한다.

### 3단계: 유료 (사용자 승인 필수)

- Muse 체크리스트와 단독 호출. 프롬프트는 HOT3D와 같은 파일을 쓴다 (버전 1):
  - `scripts/prompts/number2_checklist.txt`
  - `scripts/prompts/number2_baseline.txt`
- 프레임: 클립당 720 px 6장 (HOT3D와 같음).
- 비용 (HOT3D 실측 단가, 호출당 약 $0.0097):
  - 클립당 2호출이면 약 $0.02. 남은 크레딧 $6.73이면 약 300클립.
  - SAM까지 쓰면 클립당 약 $0.035, 약 190클립. SAM 단가는 $0.20 / 1,000프레임, 10fps, 최소 50프레임 과금.
  - 크레딧을 넘으면 사용자 돈이다 (결정 D4). 단계마다 비용을 먼저 알리고 승인받는다.
- 300클립이면 AUC 구간이 대략 ±0.07로 넓다 (어림).

### 4단계: 기록

- 영어 기록 `docs/experiments/exp43_h2o_external.md`를 쓴다. 설정, 수치, 비용, 그림, 결론. 그림은 `docs/experiments/img/`.
- `docs/experiments/README.md` 표에 한 줄 추가한다.
- PROGRESS, README, 보고서 페이지 반영은 결과가 나온 뒤 따로 한다.

## 4. 다시 쓸 코드 (HOT3D용, H2O에 맞게 고쳐야 함)

| 파일 | 쓰임 |
| --- | --- |
| `scripts/hot3d_hands.py` | 손 랜드마크 21개. H2O는 관절이 이미 있어 순서만 맞추면 된다 |
| `scripts/hot3d_calib_check.py` | 투영으로 좌표 확인 |
| `scripts/hot3d_clips.py` | 손가락 끝-물체 거리로 접촉 구간과 클립 |
| `scripts/hot3d_copyable_v2.py` | copyable v2 라벨 (정답) |
| `scripts/hot3d_number2_inputs.py` | Muse 프레임 추출 (`frames`) |
| `scripts/muse_hot3d.py` | Muse 체크리스트/단독 호출. `--send` 없으면 비용만, 응답은 클립별 캐시 |
| `scripts/number2_score.py` | R1, R2, R3 계산 |
| `scripts/hot3d_local_checks.py`, `scripts/number2_sam_checks.py` | 12개 항목 전체를 쓸 때만 |

기존 HOT3D 결과 파일은 덮어쓰지 않는다. H2O 출력은 `out/h2o/`처럼 따로 둔다.

## 5. 지켜야 할 규칙 (CLAUDE.md에서)

- Meta API 호출은 사용자 승인 없이 하지 않는다.
- `.env` 값은 출력하지 않는다.
- 결정 트리 금지. 통합 점수는 로지스틱 회귀.
- 영어 산출물은 쉬운 영어로 쓴다. em-dash, 축약형, 세미콜론 금지.
- 규칙은 계산 전에 측정 계획에 쓴다. held-out은 한 번만 채점한다.
- 커밋과 푸시는 보고서 세션이 한다.
- H2O 영상과 프레임은 저장소에 넣지 않는다 (CC BY-NC 4.0, 재배포 범위 미확인).

## 6. 위험

- 마감은 2026-11-04 정오 ET다. 다운로드 크기와 좌표 맞추기에 시간이 걸리면 제출 준비가 밀린다. 1단계는 하루 안에 끝내고, 안 되면 접는다.
- 클립 수가 적으면 구간이 넓어 결론이 "판단 불가"가 될 수 있다. 그것도 그대로 보고한다.

## 7. H2O README 원문 (사용자 제공)

````
# H2O
For more detailed information, please visit our official website (https://taeinkwon.com/projects/h2o/).

# How to merge tar.part files?
```cat subject1_v1_1.tar.gz.parta* >subject1_v1_1.tar.gz```

# Visualization code (H2OPlayer)
Please check the visualization code in the following link: https://github.com/taeinkwon/h2oplayer.

# Dataset Structure

<pre>
.
├── h1
│   ├── 0
│   │   │── cam0
│   │   │   ├── rgb
│   │   │   ├── depth
│   │   │   ├── cam_pose
│   │   │   ├── hand_pose
│   │   │   ├── hand_pose_MANO
│   │   │   ├── obj_pose
│   │   │   ├── obj_pose_RT
│   │   │   ├── action_label (only in cam4)
│   │   │   ├── rgb256 (only in cam4)
│   │   │   ├── verb_label
│   │   │   └── cam_intrinsics.txt
│   │   ├── cam1
│   │   ├── cam2
│   │   ├── cam3
│   │   └── cam4
│   ├── 1
│   ├── 2
│   ├── 3
│   └── ...
├── h2
├── k1
├── k2
└── ...
</pre>

cam0 ~ cam3 are fixed cameras. cam4 is an head-mounted camera (egocentric view). <br>
train_sequences = ['subject1/h1', 'subject1/h2', 'subject1/k1', 'subject1/k2', 'subject1/o1', 'subject1/o2', 'subject2/h1', 'subject2/h2', 'subject2/k1',
'subject2/k2', 'subject2/o1', 'subject2/o2', 'subject3/h1', 'subject3/h2', 'subject3/k1'] (subject 1,2,3) <br>
val_sequences = ['subject3/k2', 'subject3/o1', 'subject3/o2'] (subject 3)<br>
test_sequences  = ['subject4/h1', 'subject4/h2', 'subject4/k1', 'subject4/k2', 'subject4/o1', 'subject4/o2'] (subject4)<br>


### rgb
1280 * 720 resolution rgb images

### rgb256
455 * 256 resolution resized rgb images

### depth
1280 * 720 resolution depth images

### cam_pose
cam_to_world rotion matrix <br>

16 numbers, 4x4 camera matrix
### hand_pose
cam_to_hand <br>
1 (whether annotate or not, 0: not annotate 1: annotate) + 21 * 3 (x, y, z in order) + 1 + 21 * 3 (right hand) <br>
First 64 numbers belong to the left hand. Next 64 numbers belong to the right hand
### hand_pose_MANO
1 (whether annotate or not, 0: not annotate 1: annotate) + 3 translation values + 48 pose values + 10 shape values + 1 + 3 + 48 + 10 (right hand) <br>
First 59 numbers belong to the left hand. Next 59 numbers belong to the right hand
### obj_pose
cam_to_obj <br>
1 (object class) + 21 * 3 (x, y, z in order) <br>
21 numbers : 1 center, 8 corners, 12 mid edge point. <br>
0 background (no object) <br>
1 book <br>
2 espresso <br>
3 lotion <br>
4 spray <br>
5 milk <br>
6 cocoa <br>
7 chips <br>
8 capuccino <br>

<img id="image_canv" src="obj_point_order.jpg"/>

### object_pose_RT
1 (object class) + 16 numbers, 4x4 camera matrix

### verb_label
0 background (no verb) <br>
1 grab <br>
2 place <br>
3 open <br>
4 close <br>
5 pour <br>
6 take out <br>
7 put in <br>
8 apply <br>
9 read <br>
10 spray <br>
11 squeeze <br>

### action_label
Combination of a noun (object class) and a verb (verb label)

0 background <br>
1 grab book <br>
2 grab espresso <br>
3 grab lotion <br>
4 grab spray <br>
5 grab milk <br>
6 grab cocoa <br>
7 grab chips <br>
8 grab cappuccino <br>
9 place book <br>
10 place espresso <br>
11 place lotion <br>
12 place spray <br>
13 place milk <br>
14 place cocoa <br>
15 place chips <br>
16 place cappuccino <br>
17 open lotion <br>
18 open milk <br>
19 open chips <br>
20 close lotion <br>
21 close milk <br>
22 close chips <br>
23 pour milk <br>
24 take out espresso <br>
25 take out cocoa <br>
26 take out chips <br>
27 take out cappuccino <br>
28 put in espresso <br>
29 put in cocoa <br>
30 put in cappuccino <br>
31 apply lotion <br>
32 apply spray <br>
33 read book <br>
34 read espresso <br>
35 spray spray <br>
36 squeeze lotion <br>

## For Actions
[Train set file](action_labels/action_train.txt) <br>
[Validation set file](action_labels/action_val.txt) <br>
[Test set file](action_labels/action_test.txt) <br>

## For Poses
[Train set file](pose_lists/pose_test.txt) <br>
[Validation set file](pose_lists/pose_train.txt) <br>
[Test set file](pose_lists/pose_test.txt) <br>

### cam_instrinsics.txt
six numbers : fx, fy, cx, cy, width, height

# Citations
If you find any usefulness in this dataset, please consider citing:
```
@InProceedings{Kwon_2021_ICCV,
    author    = {Kwon, Taein and Tekin, Bugra and St\"uhmer, Jan and Bogo, Federica and Pollefeys, Marc},
    title     = {H2O: Two Hands Manipulating Objects for First Person Interaction Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {10138-10148}
}
```
````
