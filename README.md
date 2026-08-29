# RebarSplice

철근 **겹침이음(lap splice)** 을 사진에서 자동 검출하는 **웹앱**과 그 학습 파이프라인.

건설현장 배근 검측용. 사진을 올리면 브라우저 안에서 검출이 돌고 박스를 그려 준다.
실시간 영상이 아니라 **찍어 둔 사진**을 처리한다.

---

## 지금 상태

**파일럿 학습 완료 (2026-08-28).** 실데이터 첫 학습 결과 — yolo26s, 200 epoch, 23분(RTX 2060 SUPER):

| 클래스 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **lap_splice** | 0.833 | 0.719 | **0.787** | 0.62 |
| rebar | 0.791 | 0.649 | 0.749 | 0.528 |

검증셋은 학습에 안 쓴 영상 2개(IMG_0329·IMG_2410, 54타일). 파일럿 목표(0.3~0.5)를
크게 넘겼다. 웹앱에서 같은 타일을 돌려 torch 와 겹침이음 개수가 일치함을 확인했다.

| | |
|---|---|
| 데이터 | 현장 영상 6개 → 프레임 190 → **타일 309장** (720²·1080²), 2026-08-28 라벨 완료 |
| 클래스 | **2개** — `lap_splice`(1,068) + `rebar`(2,542). 폴리곤 라벨 (detect 학습이 자동으로 박스 변환) |
| 분할 | **영상 단위** — 같은 영상의 프레임은 사실상 중복이라 무작위 분할 금지. `scripts/prepare_dataset.py` |
| 모델 | **yolo26s**, 입력 640 |
| 배포 | 웹앱(`web/`) — 브라우저에서 추론(ONNX Runtime Web), 서버 없음 |

**철근 이음부 공개 데이터셋은 존재하지 않는다.** Roboflow Universe · Kaggle ·
HuggingFace · AI-Hub 를 전수 검색한 결과 0건이다(2026-08 확인). 처음부터 만들어야 한다.

---

## 순서

상세 절차는 두 문서에 있다:

| 문서 | 내용 |
|---|---|
| **[docs/labeling-guide.md](docs/labeling-guide.md)** | Roboflow 계정부터 내보내기까지 — 라벨링 전 과정 |
| **[docs/training-guide.md](docs/training-guide.md)** | 설치부터 `.onnx` 까지 — 학습 전 과정 (소요 시간 실측) |

아래는 요약이다.

### 1. 사진을 모은다

`docs/labeling-guide.md` 참고. 요점 하나만 옮기면:

> **같은 이음부를 100번 찍으면 100장이 아니라 1장이다.**
> 서로 다른 이음부를 30개소 이상, 거리·각도·조명을 섞어서.

파일명 앞에 촬영 장소를 붙인다 — `siteA_001.jpg`, `siteB_014.jpg`.
검사 스크립트가 이걸로 학습/검증 누수를 잡는다.

### 2. 라벨한다

**Roboflow.** 규칙은 한 줄이다:

> 이음부처럼 보이면 박스를 친다.

**이음이 없는 사진을 20~30% 섞어 박스 0개로 넣는다.** 안 넣으면 모델이
"철근이 보이면 이음"으로 학습한다 — 이 도메인 최빈 실패다.

**업로드는 원본 해상도 그대로.** 미리 줄이지 마라 — 나중에 다른 입력 크기로
다시 뽑을 수 있어야 한다. Roboflow 의 Preprocessing 에서 Resize 640 이면 충분하다.

**Roboflow 의 Augmentation 은 끈다.** 학습 스크립트가 이미 증강하므로 이중으로 걸리면
오히려 나빠진다.

내보내기는 **YOLOv8 포맷**을 고른다. 이건 모델 버전이 아니라 **데이터셋 형식** 이름이다 —
Ultralytics 는 v5 부터 v26 까지 라벨 형식이 같으므로 YOLOv8 로 내보내도 YOLO26 학습에
그대로 쓴다. (v11 / v12 옵션이 있어도 나오는 파일은 동일하다.)

압축을 `data/` 에 푼다. **`data.yaml` 은 고치지 않아도 된다** — 안에 `../train/images`
처럼 `../` 로 시작하는 경로가 들어 있지만 Ultralytics 가 알아서 처리한다
(`ultralytics/data/utils.py:606` — 경로가 없으면 `../` 를 떼고 다시 찾는다. 2026-08-25 실측 확인).

### 2.5 실데이터는 prepare_dataset 으로 나눈다

라벨 데이터가 영상에서 뽑은 타일(`IMG_0328_..._frame119_...jpg`)이고 전부 train/ 에
들어 있다면 — 지금 실데이터가 그렇다 — **영상 단위**로 분할해야 한다:

```bash
.venv\Scripts\python scripts\prepare_dataset.py "D:/Projects/LH/Detection/rebar lapping.yolov11"
```

`data/train`·`data/valid`·`data/dataset.yaml` 이 만들어진다. 같은 영상의 프레임이
학습·검증에 나뉘면 검증 점수가 가짜로 높아지므로 무작위 분할은 안 된다.

### 3. 데이터를 검사한다

```bash
.venv\Scripts\python scripts\check_dataset.py data\dataset.yaml
```

네거티브 비율 · 촬영 장소 수 · **학습/검증 장소 누수** 를 본다.
문제가 있으면 고치고 다시 돌린다. 학습은 그다음이다.

### 4. 학습한다

실측: 100장 · yolo26s · 640px · RTX 2060 SUPER 기준 **30 epoch 105초**,
기본값 200 epoch 이면 **약 12분**.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python -m pip install -r requirements.txt

.venv\Scripts\python scripts\train.py
```

**GPU 를 쓰는지 먼저 확인하라.** 안 쓰면 100장에도 몇 시간이 걸린다:

```bash
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

파일럿에서는 `n` 과 `s` 를 둘 다 돌려 비교해 볼 만하다 — 몇 분이면 끝난다:

```bash
.venv\Scripts\python scripts\train.py --model yolo26n.pt --name splice_n
.venv\Scripts\python scripts\train.py --model yolo26s.pt --name splice_s
```

### 5. 결과를 눈으로 본다

```bash
.venv\Scripts\python scripts\predict_test.py runs\splice\weights\best.pt data\valid\images --conf 0.15
```

**숫자보다 이게 중요하다.** 어디서 틀리는지 — 다발철근? 스터럽 교차? 역광? —
패턴이 보이면 라벨 규칙에 한 줄 추가하고 다음 라운드에 반영한다.

### 6. ONNX 로 바꾼다

```bash
.venv\Scripts\python scripts\export_onnx.py runs\splice\weights\best.pt --imgsz 640
```

**Windows 에서 그냥 된다** — Core ML 과 달리 CI 가 필요 없다(실측 확인).

---

## 모델 선택 — 실측 근거

640px, CPU, ONNX Runtime 기준으로 직접 쟀다:

| 모델 | 파일 | 단일스레드 | 4스레드 | COCO mAP50-95 |
|---|---|---|---|---|
| yolo26n | 10MB | 83ms | 47ms | 40.9 |
| **yolo26s** | **38MB** | **254ms** | **126ms** | **48.6** ← 기본 |
| yolo26m | 82MB | 827ms | 396ms | 53.1 |

폰 브라우저의 WASM 은 여기서 2~5배 느리므로 **`s` 기준 폰에서 0.5~1.3초**다.
사진을 올리고 결과를 보는 방식에는 충분하다.

**실시간이 아니라서 `s` 를 쓸 수 있다.** 30fps 를 맞출 이유가 없으니 그 여유를
정확도에 쓴다. `m` 은 82MB 라 첫 로딩이 부담스럽고 폰에서 2~4초까지 간다.

---

## 웹앱 (`web/`) — 구현됨

```bash
python scripts/serve_web.py            # http://localhost:8377
```

`python -m http.server` 를 쓰면 안 된다 — Windows 레지스트리의 .js=text/plain 때문에
ES 모듈 로딩이 거부된다(실측). `serve_web.py` 가 MIME 과 COOP/COEP(멀티스레드용)를
챙긴다. 배포는 `web/vercel.json` 이 같은 헤더를 준다.

- 좌표 수학(레터박스·타일 계획·병합)은 `web/infer.js` — 순수 함수, `node --test web/infer.test.mjs` 로 12케이스 검증
- **타일 추론**: 학습 데이터가 720~1080px 타일이라, 큰 사진은 960px 타일(겹침 25%)로
  잘라 돌리고 겹침 구간 중복은 클래스별 IoU 0.5 로 병합한다. 4K 사진을 통째로 640 에
  넣으면 철근이 학습 분포보다 가늘어져 못 잡는다
- 신뢰도 슬라이더는 **다시 추론하지 않는다** — 0.05 까지 받아 두고 그리기만 거른다
- 모델 파일은 `web/models/best.onnx` (gitignore — 재학습마다 수십 MB 를 git 에 쌓지
  않는다. 배포 시 로컬 파일이 그대로 올라간다)

### 브라우저에서 도는 이유

| | |
|---|---|
| 서버 | 없음. Vercel 정적 호스팅 |
| 비용 | 0원, 영구 |
| 사진 | **기기 밖으로 안 나감** |
| 기기 | 폰·태블릿·PC 같은 주소 |

**후처리가 간단하다.** YOLO26 은 NMS-free(end2end) 라 출력이 이미 최종 박스다 —
보통 브라우저 추론에서 제일 골치 아픈 JS NMS 구현이 통째로 없다:

```
입력  images   [1, 3, 640, 640]  float32   (0~1 정규화, RGB, NCHW)
출력  output0  [1, 300, 6]       float32
      한 행 = [x1, y1, x2, y2, confidence, class]
```

브라우저가 할 일은 셋뿐이다: 사진을 640 으로 레터박스 리사이즈 → `conf` 로 거르기 →
좌표를 원본 크기로 되돌리기.

**아이폰 사진(12MP)은 반드시 640 으로 줄여서 넣는다.** 원본 그대로 텐서를 만들면
iOS Safari 가 메모리로 죽는다. 모델 입력이 640 이라 정확도 손실은 없다.

---

## 설계 판단 — 왜 이렇게 했나

**바운딩박스, 키포인트 아님.**
v1 에서 이음 **길이 측정** 을 뺐다. 박스로는 길이를 못 재지만 위치만 찾는 데는 충분하고
라벨이 3~5배 빠르다. 길이가 필요해지면 그때 폴리곤/키포인트로 재라벨한다.

**`mAP50` 만 본다.**
경계가 애매한 대상은 사람끼리도 IoU 일치도 상한이 약 0.88 이다. `mAP50-95` 는 박스가
얼마나 딱 맞는지를 벌주는데, 그건 이 문제에서 의미가 없다.

**재현율이 정확도보다 중요하다.**
검측에서 오검출은 사람이 화면에서 지우면 되지만, 미검출은 **검사를 안 한 것** 이다.
임계값을 낮게 잡아 과검출 쪽으로 기울이고 사람이 거르는 구조가 맞다.

**LiDAR 로 길이를 재는 건 안 된다 (검토했음).**
iPhone LiDAR 는 프레임당 실제 레이저 점이 576 개뿐이고 256×192 는 업샘플 결과다.
0.5m 에서 점 간격이 약 21mm 라 D25 철근(25.4mm) 지름에 표본이 1개 남짓이다.
논문 저자들이 명시한 검출 한계가 약 5cm 인데 철근은 1~3.2cm 다.
→ 겹침을 기하로 분리하는 것은 성립하지 않는다. 영상으로 찾는 게 맞다.

**AGPL 은 문제없다.**
Ultralytics YOLO 는 AGPL-3.0 이고 이 저장소는 공개다.

---

## 구조

```
scripts/
  prepare_dataset.py  Roboflow 내보내기를 영상 단위로 분할 -> data/
  check_dataset.py    학습 전 위생 검사 — 네거티브·분할 누수
  train.py            Ultralytics 전이학습 (기본 yolo26s)
  predict_test.py     결과를 눈으로 확인
  export_onnx.py      .pt -> .onnx (웹앱용)
  serve_web.py        web/ 개발 서버 (MIME + COOP/COEP)
docs/
  labeling-guide.md   Roboflow 라벨링 절차
  training-guide.md   설치부터 .onnx 까지 (소요 시간 실측)
data/                 데이터셋 (gitignore — 현장 사진은 커밋하지 않는다)
web/
  index.html, app.js  업로드 -> 타일 추론 -> 박스 표시
  infer.js(.test.mjs) 좌표 수학 — 단위테스트 12케이스
  vendor/ort/         ONNX Runtime Web 1.22 (vendored)
  models/best.onnx    학습된 모델 (gitignore)
```

## 라이선스

코드는 MIT. 단 학습에 Ultralytics(AGPL-3.0)를 쓰므로, 그 코드로 만든 **가중치를
배포할 때는 AGPL 이 따라온다.**
