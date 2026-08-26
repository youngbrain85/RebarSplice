# 사진 학습 — 순서대로

사진을 모은 뒤부터 웹앱에 넣을 `.onnx` 가 나올 때까지의 전 과정.

이 문서의 명령과 소요 시간은 **실제로 돌려서 확인한 것**이다 (2026-08-25,
RTX 2060 SUPER / 합성 데이터 100장).

---

## 0. 한 번만 하는 준비

### 0-1. 파이썬 환경

```
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python -m pip install -r requirements.txt
```

**`--index-url` 을 빼먹으면 CPU 버전 torch 가 깔린다.** 그러면 100장에도 몇 시간이
걸린다. 반드시 확인한다:

```
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`True NVIDIA GeForce RTX 2060 SUPER` 처럼 나와야 한다. `False` 면 CPU 로 도는 것이니
torch 를 지우고 다시 깐다.

### 0-2. 이 저장소의 GPU

이 PC 는 **RTX 2060 SUPER (8GB)** 다. 아래 기준값은 전부 이 GPU 실측이다.

---

## 1. 사진을 모은다

`labeling-guide.md` 의 「데이터 구성」이 실제 규칙이고, 요약하면 셋이다:

1. 이음 **없는** 사진을 20~30% 섞는다
2. 서로 다른 이음부를 30개소 이상 — 같은 곳 연사 30장은 1장으로 친다
3. 파일명 앞에 촬영 장소를 붙인다 — `siteA_001.jpg`, `siteB_014.jpg`

**3번을 꼭 지켜라.** `check_dataset.py` 가 이 이름으로 학습/검증 누수를 잡는다.
아이폰 기본 이름(`IMG_1234.jpg`)이면 그 검사를 통째로 건너뛴다.

이름 일괄 변경 (PowerShell, 폴더 안 사진 전부에 `siteA_` 접두어):

```
Get-ChildItem *.jpg | ForEach-Object { Rename-Item $_ "siteA_$($_.Name)" }
```

---

## 2. 라벨한다

→ `labeling-guide.md` 로. 끝나면 3번으로 돌아온다.

---

## 3. Roboflow 내보내기를 푼다

Roboflow → Versions → Download → **YOLOv8** → zip.

압축을 `data/` 에 푼다. 이런 모양이 된다:

```
data/
  data.yaml
  train/images/siteA_001.jpg      train/labels/siteA_001.txt
  valid/images/siteC_001.jpg      valid/labels/siteC_001.txt
  test/...                        (있을 수도, 없을 수도)
```

**`data.yaml` 은 손대지 않아도 된다.** 안에 `train: ../train/images` 처럼 `../` 로
시작하는 경로가 들어 있는데, Ultralytics 가 경로를 못 찾으면 `../` 를 떼고 다시
찾는다 (`ultralytics/data/utils.py:606`). 실제로 그 상태로 학습이 되는 것을 확인했다.

> 이전 문서에 "고치지 않으면 학습이 실패한다"고 적혀 있었는데 **틀린 내용이었다.**
> 2026-08-25 에 원본 yaml 그대로 학습을 돌려 성공하는 것을 확인하고 정정했다.

---

## 4. 데이터를 검사한다 — 학습보다 먼저

```
.venv\Scripts\python scripts\check_dataset.py data\data.yaml
```

이렇게 나오면 통과다:

```
[train]  D:\Projects\RebarSplice\data\train\images
         사진 80장  ·  박스 있는 사진 60장  ·  박스 121개
         네거티브 20장 (25%)  ·  장소 2곳 {'siteA': 40, 'siteB': 40}

[val]    D:\Projects\RebarSplice\data\valid\images
         사진 20장  ·  박스 있는 사진 15장  ·  박스 32개
         네거티브 5장 (25%)  ·  장소 1곳 {'siteC': 20}

  이상 없음.
```

**맨 윗줄의 경로를 확인하라.** 검사기가 실제로 어느 폴더를 읽었는지 찍어 준다.
엉뚱한 곳이면 사진 수가 0으로 나온다.

`[문제]` 가 나오면 **고치고 다시 돌린다.** 학습은 그다음이다. 대표적인 셋:

| 나오는 말 | 뜻 | 조치 |
|---|---|---|
| 네거티브가 N% 뿐이다 | 이음 없는 사진이 모자람 | 배근 사진을 박스 0개로 더 넣는다 |
| 학습과 검증에 같은 장소가 있다 | 누수 — **검증 점수가 가짜가 된다** | Roboflow 에서 장소 단위로 다시 나눈다 |
| train 경로가 없다 | 압축을 엉뚱한 데 풀었다 | 폴더 구조 확인 |

---

## 5. 학습한다

```
.venv\Scripts\python scripts\train.py --data data\data.yaml
```

기본값은 `yolo26s` · 640px · 200 epoch · batch 8 이다. 100장 규모를 전제로 잡았다.

### 얼마나 걸리나 (실측)

100장 / yolo26s / 640px / batch 8 / RTX 2060 SUPER:

| | |
|---|---|
| 30 epoch | **105초** (epoch 당 약 3.5초) |
| 200 epoch (기본값) | **약 12분** |

조기종료(`patience=50`)가 있어서 보통 200 을 다 채우기 전에 끝난다.

GPU 메모리는 약 1.2GB 를 쓴다. 8GB 라 `--batch 16` 까지는 여유가 있다.

### 화면에서 볼 것

```
      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      12/200      1.23G      1.421      1.832    0.00812         28        640
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all         20         32      0.512      0.438      0.447      0.221
```

- **`box_loss` / `cls_loss` 가 내려가는가** — 안 내려가면 라벨이 잘못됐다는 뜻이다
- **`mAP50`** — 이것만 본다. `mAP50-95` 는 무시한다 (아래 참고)
- `R`(재현율)이 `P`(정밀도)보다 중요하다 — 검측에서는 놓치는 게 더 나쁘다

### 자주 쓰는 변형

```
:: n 과 s 를 비교해 본다 (몇 분이면 끝난다)
.venv\Scripts\python scripts\train.py --data data\data.yaml --model yolo26n.pt --name splice_n
.venv\Scripts\python scripts\train.py --data data\data.yaml --model yolo26s.pt --name splice_s

:: 중간에 끊겼을 때 이어서 (last.pt 사용)
.venv\Scripts\python scripts\train.py --data data\data.yaml --model runs\splice\weights\last.pt

:: GPU 메모리가 모자라면
.venv\Scripts\python scripts\train.py --data data\data.yaml --batch 4
```

---

## 6. 결과를 읽는다

산출물은 `runs/splice/` 에 있다:

| 파일 | 무엇 |
|---|---|
| `weights/best.pt` | **mAP 가 가장 좋았던 시점**의 가중치 — 이걸 쓴다 |
| `weights/last.pt` | 마지막 epoch. 학습을 이어갈 때만 쓴다 |
| `results.csv` | epoch 별 손실·mAP 전부 |
| `results.png` | 위 내용을 그래프로 |
| `val_batch0_pred.jpg` | 검증셋 예측을 그려 놓은 것 |
| `confusion_matrix.png` | 무엇을 무엇으로 착각했는지 |

### mAP50 을 어떻게 해석하나

**100장으로는 정확도를 재는 게 아니다.** "될 일인가"를 보는 것이다.

| mAP50 | 뜻 | 다음 |
|---|---|---|
| 0.3 ~ 0.5 | 100장 기준 **정상** | 사진을 늘린다 |
| 0.6 이상 | 좋다 — 다만 **누수를 의심**하라 | `check_dataset` 의 장소 검사 재확인 |
| 0.1 미만 | 라벨 정의가 흔들린다 | 라벨을 다시 본다. 사진을 늘려도 안 오른다 |

`mAP50-95` 는 **보지 않는다.** 박스가 얼마나 딱 맞는지를 벌주는 지표인데,
이음부는 경계가 원래 애매해서 사람끼리도 IoU 일치도가 0.9 를 못 넘는다.

---

## 7. 눈으로 확인한다 — 여기가 제일 중요하다

```
.venv\Scripts\python scripts\predict_test.py runs\splice\weights\best.pt data\valid\images --conf 0.15
```

`--conf` 를 **낮게** 잡는다. 무엇을 놓치는지 보려는 것이다.

```
사진 20장 · 검출 37개 · 아무것도 못 잡은 사진 5장
```

결과 이미지는 `runs/pred/` 에 저장된다. **반드시 열어 봐라.** 숫자로는 안 보이는
것을 본다:

- 이음이 아닌데 이음이라 하는 것은 무엇인가 — 다발철근? 스터럽 교차? 그림자?
- 어떤 상황에서 통째로 놓치는가 — 역광? 빽빽한 배근? 먼 거리?

**패턴이 보이면 그게 다음 라운드의 라벨 규칙이다.** 이게 사진을 100장 더 모으는
것보다 값질 때가 많다.

---

## 8. 웹앱용 ONNX 로 바꾼다

```
.venv\Scripts\python scripts\export_onnx.py runs\splice\weights\best.pt --imgsz 640
```

`models/best.onnx` 가 나온다 (yolo26s 기준 **38MB**). Windows 에서 그냥 된다.

끝나면 실제 입출력 형태를 찍어 준다:

```
입력  images     [1, 3, 640, 640]  tensor(float)
출력  output0    [1, 300, 6]       tensor(float)
      한 행 = [x1, y1, x2, y2, confidence, class]
```

`--imgsz` 는 **학습 때와 같은 값**을 써야 한다.

---

## 9. 다음 라운드

파일럿 100장이 끝나면 순서는 이렇게 돈다:

```
7번에서 본 오류 패턴  ->  라벨 규칙 한 줄 추가  ->  사진 보강  ->  4번으로
```

사진을 무작정 늘리기보다 **틀리는 유형을 골라서** 모으는 편이 훨씬 빠르다.
역광에서 다 놓치면 역광 사진을, 다발철근을 오검출하면 다발철근 사진을
네거티브로 넣는다.

---

## 문제가 생기면

| 증상 | 원인 | 조치 |
|---|---|---|
| 학습이 아주 느리다 | CPU 로 돌고 있다 | 0-1 의 `cuda.is_available()` 확인 |
| `CUDA out of memory` | 배치가 크다 | `--batch 4` |
| mAP 가 0 에서 안 움직인다 | 라벨이 안 읽힌다 | `check_dataset` 의 박스 개수 확인 |
| 검증 점수만 이상하게 높다 | 장소 누수 | `check_dataset` 의 장소 검사 |
| `dataset.yaml 이 없다` | `--data` 를 안 줬다 | `--data data\data.yaml` |
