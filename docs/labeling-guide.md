# 철근 겹침이음 — Roboflow 라벨링

**클래스 1개**: `lap_splice` · **박스**: 축정렬 바운딩박스

단축키와 메뉴 이름은 Roboflow 공식 문서에서 확인한 것이다 (2026-08-25).

---

## 규칙 — 이게 전부다

> **이음부처럼 보이면 박스를 친다.**
> 철근 두 가닥이 나란히 겹쳐 있는 부분을 감싼다.

애매하면 그냥 판단하고 넘어간다. 경계가 몇 cm 어긋나도 상관없다 —
`mAP50` 으로 평가하므로 IoU 0.5 만 넘으면 정답이다.

**안 보이는 건 안 그린다.** 결속선 종류·이음 길이·철근 지름은 신경 쓰지 않는다.

---

# 순서

## 1. 프로젝트 만들기

1. [app.roboflow.com](https://app.roboflow.com) → **Create New Project**
2. 채울 것:

| 칸 | 값 |
|---|---|
| Project Name | `rebar-splice` |
| Project Type | **Object Detection** |
| Annotation Group | `splice` |

> 무료 플랜은 데이터셋이 공개된다. 이 프로젝트는 공개 전제라 상관없다.

---

## 2. 사진 올리기

**Upload** → 사진을 끌어다 놓는다 → **Save and Continue**

올린 사진은 **Annotate** 탭에 배치(batch)로 쌓인다.

### 올리기 전에 파일명부터

파일명 앞에 촬영 장소를 붙인다 — `siteA_001.jpg`, `siteB_014.jpg`.

나중에 학습/검증을 장소 단위로 자를 때(6번) 이 이름으로 걸러낸다.
`IMG_1234.jpg` 로 올리면 어느 사진이 어느 현장인지 알 수 없다.

```
Get-ChildItem *.jpg | ForEach-Object { Rename-Item $_ "siteA_$($_.Name)" }
```

**해상도는 줄이지 않는다.** 아이폰 원본 그대로.

---

## 3. 편집기 열기

**Annotate** 탭 → 배치를 누른다 → **Start Annotating**
(또는 사진을 직접 클릭)

---

## 4. 박스 그리기

1. **`B`** 를 누른다 → 바운딩박스 도구. 십자선이 나온다
2. 이음부를 **드래그**해서 감싼다
3. 박스를 놓으면 **클래스 선택창**이 바로 뜬다
4. 첫 박스에서만 `lap_splice` 를 타이핑해 새 클래스로 만든다 → **`Enter`**
5. 두 번째부터는 그 클래스가 후보에 뜨므로 `Enter` 만 누르면 된다
6. 한 사진에 이음이 여러 개면 2~5 를 반복한다
7. **`→`** (오른쪽 화살표) 로 다음 사진

저장 버튼은 없다. 박스는 자동 저장된다.

### 단축키 (공식 문서 확인)

| 키 | 동작 |
|---|---|
| **`B`** | 바운딩박스 도구 |
| **`D`** | 드래그/선택 도구 — **이미 그린 박스를 고칠 때** |
| **`N`** | **Mark Null** — 이음 없는 사진 처리 (5번) |
| **`Enter`** | 클래스 선택창에서 선택 확정 |
| **`→`** / **`←`** | 다음 / 이전 사진 |
| **`Backspace`** | 선택한 박스 삭제 |
| **`Ctrl`**+**`Z`** | 되돌리기 |
| **`Ctrl`**+**`Shift`**+**`Z`** | 다시 실행 |
| **`0`** | 화면에 맞춤 · **`+`** / **`-`** 확대·축소 |
| **`Esc`** | 편집기 나가기 |

> `D` 는 "다음 사진"이 아니라 **드래그/선택 도구**다. `A` 도 다음/이전이 아니라
> 리뷰 모드의 승인 키다. 사진 이동은 **화살표 키**뿐이다.

### 어떻게 그리나

겹쳐 있는 **두 가닥이 나란한 구간 전체**를 감싼다. 겹침이 시작되는 곳부터
끝나는 곳까지. 박스끼리 겹쳐도 된다. 벽체 수직철근이면 세로로 긴 박스가
되는 게 정상이다.

**한 장에 10~20초.** 100장이면 30분~1시간이면 끝난다.
한 장에서 1분씩 고민하고 있으면 규칙이 잘못된 것이다.

---

## 5. 이음 없는 사진 — `N` 으로 명시 처리

그냥 넘기지 말고 **`N`** 을 눌러 **Mark Null** 로 표시한다.
"검토했고, 여기엔 대상이 없다" 를 명시하는 것이다.

**이 사진들을 삭제하면 안 된다.** 데이터셋에 그대로 남아야 한다.

전체의 **20~30%** 를 이음 없는 사진으로 채운다.

### 왜 필요한가

이음 있는 사진만 학습시키면 모델이 배우는 규칙은 이렇게 된다:

> "철근이 보인다 → 이음이다"

배근 사진 전부에 박스를 치게 되고 검출기로 쓸모가 없어진다.
**철근은 있는데 이음은 없는** 상태를 보여줘야 둘을 구분한다.

---

## 6. 학습/검증 나누기 — 무작위 분할 금지

같은 이음부를 찍은 사진이 학습과 검증에 나뉘어 들어가면, 모델이 검증셋의
그 이음부를 **이미 본 것**이다. 점수가 가짜로 높게 나오고 현장에서 무너진다.

```
❌ 무작위 80:20
✅ 현장 A·B = 학습  /  현장 C = 검증
```

### 하는 법

1. **Dataset** 탭으로 간다
2. 검색창에 `siteC` 를 쳐서 그 장소 사진만 거른다
3. 전체 선택 → 스플릿을 **Valid** 로 지정
   (Train/Test Split 아래 **Edit Splits** → 지정 후 **Save Splits**)
4. 나머지는 Train 에 둔다

### 버전 생성 때 주의

7번에서 **Train/Test Split** 단계가 나오면 반드시
**"Move as few as possible"** 을 고른다.

**"Random shuffle" 을 고르면 방금 나눈 장소 분할이 전부 헝클어진다.**

---

## 7. 버전 만들기 (Generate)

**Versions** → **Generate New Version**

| 단계 | 설정 |
|---|---|
| Train/Test Split | **Move as few as possible** (6번 참고) |
| Preprocessing → Auto-Orient | **켠다** — 아이폰 EXIF 회전 문제를 없앤다 |
| Preprocessing → Resize | 640×640 이든 끄든 상관없다 |
| **Augmentation** | **전부 끈다. 배수도 1x.** |

**Augmentation 을 끄는 이유**: `train.py` 가 이미 증강한다
(회전·명암·좌우반전·모자이크·가림). 여기서 또 걸면 이중으로 걸려 오히려 나빠진다.

→ **Create** / **Generate**

---

## 8. 내보내기

**Dataset** 탭 → 검색창 오른쪽 **Export** 버튼
→ 포맷 **YOLOv8** → **Download zip to computer**

> **"YOLO26 을 쓰는데 왜 YOLOv8?"**
> YOLOv8 은 모델 버전이 아니라 **데이터셋 형식** 이름이다. Ultralytics 는
> v5 부터 v26 까지 라벨 형식이 같다. 목록에 `yolov11` 이 있어도 나오는 파일은 같다.

압축을 `data/` 에 푼다:

```
data/
  data.yaml
  train/images/   train/labels/
  valid/images/   valid/labels/
```

`data.yaml` 은 손대지 않아도 된다.

라벨 파일은 한 줄에 박스 하나, 값은 0~1 정규화:

```
0 0.512340 0.433210 0.087500 0.245800
↑ 클래스   ↑ 중심x   ↑ 중심y   ↑ 폭     ↑ 높이
```

Null 로 표시한 사진의 `.txt` 는 **빈 파일**이거나 아예 없다. 정상이다.

---

## 9. 검사

```
.venv\Scripts\python scripts\check_dataset.py data\data.yaml
```

`이상 없음` 이면 → **`training-guide.md`** 로.

---

## 어겼을 때 100장이 통째로 날아가는 셋

1. **이음 없는 사진 20~30%** — `N` 으로 표시해 포함 (빼는 게 아니다)
2. **서로 다른 이음부 30개소 이상** — 같은 자리 연사 30장은 1장으로 친다
   섞을 것: 거리(0.3m/1m/2m) · 각도 · 조명(맑음/그늘/역광) · 배근 밀도 · 녹·흙
3. **학습/검증을 촬영 장소 단위로** — 무작위 분할 금지

---

## 100장으로 판단할 것

정확도를 재는 게 아니다. 100장으로는 못 잰다. 볼 것은 셋:

1. 학습 손실이 내려가는가
2. 검증셋에서 뭐라도 잡는가
3. 오검출이 **어디서** 나는가 — 패턴이 보이면 그때 규칙을 추가한다

`mAP50` **0.3~0.5** 면 정상. **0 에 가까우면** 라벨 정의를 다시 본다.

---

*참고: 철근 이음부 공개 데이터셋은 존재하지 않는다(Roboflow Universe · Kaggle ·
HuggingFace · AI-Hub 전수 검색, 2026-08). 처음부터 만들어야 한다.*

*출처: [Use Roboflow Annotate](https://docs.roboflow.com/datasets/annotate/annotate/use-roboflow-annotate) ·
[Keyboard Shortcuts](https://docs.roboflow.com/datasets/annotate/annotate/use-roboflow-annotate/keyboard-shortcuts) ·
[Create a Dataset Version](https://docs.roboflow.com/datasets/versions/dataset-versions/create-a-dataset-version) ·
[Download a Dataset](https://docs.roboflow.com/datasets/create-and-upload/download-a-dataset)*
