# 학습 실행 기록 · 재현 근거

파일럿 보고서 표 4의 다섯 실행에 대한 원본 산출물과 확정 하이퍼파라미터.
실행 폴더별 원본은 [`docs/runs/`](runs/) 에 있다.

환경: 단일 RTX 2060 SUPER 8GB · Ultralytics 8.4.129 · torch 2.5.1+cu121 · Python 3.9.12 · Windows

## 1. 실행 목록

| 실행 | 모델 | 입력 | batch | 총 에폭 | best 에폭 | 학습시간 | lap_splice mAP50 |
|---|---|---|---|---|---|---|---|
| [`splice_s`](runs/splice_s/) | yolo26s | 640 | 8 | 200 | 167 | 23.3분 | 0.787 |
| [`splice_m`](runs/splice_m/) | yolo26m | 640 | 6 | 200 | 177 | 44.0분 | 0.838 |
| **[`splice_l`](runs/splice_l/)** | **yolo26l** | **640** | **4** | **200** | **179** | **57.1분** | **0.861 ← 배포** |
| [`splice_m960`](runs/splice_m960/) | yolo26m | 960 | 3 | 200 | 195 | 88.1분 | 0.841 |
| [`splice_l960`](runs/splice_l960/) | yolo26l | 960 | 2 | 200 | 189 | 110.3분 | 0.857 |

batch 는 8GB VRAM 에 맞춰 모델·해상도별로 내린 값이다(성능 비교 목적의 변수가 아니라
메모리 제약). 유효 배치는 `nbs=64` 로 정규화되므로 옵티마이저가 보는 배치는 동일하다.

## 2. 확정 하이퍼파라미터

다섯 실행은 `model`·`batch`·`imgsz`·`name` 네 가지만 다르고 **나머지는 전부 같다**
(`diff docs/runs/splice_l/args.yaml docs/runs/<다른실행>/args.yaml` 로 확인 가능).
아래는 배포 모델 `splice_l` 기준이며 원본은 [`docs/runs/splice_l/args.yaml`](runs/splice_l/args.yaml).

| 항목 | 값 |
|---|---|
| 사전학습 가중치 | `yolo26l.pt` (COCO 사전학습), `pretrained: true` |
| epochs / patience | 200 / 50 |
| batch / imgsz | 4 / 640 |
| optimizer | `auto` → **실제 AdamW** (아래 참조) |
| lr0 / lrf | **0.001667** / 0.01 (선형 감쇠) |
| momentum | 0.9 |
| weight_decay | 0.0005 |
| warmup | epochs 3.0 · momentum 0.8 · **bias_lr 0.0** |
| nbs (유효 배치) | 64 |
| loss 가중치 | box 7.5 · cls 0.5 · dfl 1.5 |
| 기타 | amp true · seed 0 · deterministic true · cos_lr false · close_mosaic 10 · single_cls false · rect false |

### `optimizer: auto` 가 실제로 무엇으로 풀렸는가

`args.yaml` 에 기록된 `optimizer: auto` · `lr0: 0.01` · `momentum: 0.937` 은 **설정값이지
실제 사용값이 아니다.** Ultralytics 는 학습 시작 시 이 값들을 무시하고 다시 정한다
(`ultralytics/engine/trainer.py:1105`):

```
iterations = ceil(len(train_set) / max(batch, nbs)) * epochs
           = ceil(255 / max(4, 64)) * 200 = 4 * 200 = 800
800 <= 10000  ->  AdamW,  lr = round(0.002 * 5 / (4 + nc), 6) = round(0.002 * 5 / 6, 6) = 0.001667
                  momentum = 0.9,  warmup_bias_lr = 0.0
```

따라서 **다섯 실행 모두 AdamW(lr0=0.001667, momentum=0.9)** 다. batch 가 2~8 로 달라도
`max(batch, 64) = 64` 라 iterations 가 800 으로 같기 때문이다.

이 값은 로그가 아니라 데이터로도 확인된다. `results.csv` 의 마지막 에폭 학습률이
`2.49217e-05` 인데, 선형 감쇠식 `lr0 x ((1 - 199/200) x (1 - lrf) + lrf)` 에
lr0=0.001667 을 넣으면 `0.001667 x 0.01495 = 2.492e-05` 로 일치한다.
lr0 이 0.01 이었다면 `1.495e-04` 가 찍혔어야 한다.

### 증강 최종값

| 항목 | 값 | 항목 | 값 |
|---|---|---|---|
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 | mosaic | 1.0 (마지막 10 에폭 해제) |
| degrees (회전) | 15.0 | mixup | 0.1 |
| translate / scale | 0.1 / 0.5 | cutmix / copy_paste | 0.0 / 0.0 |
| shear / perspective | 0.0 / 0.0 | auto_augment | randaugment |
| **flipud (상하반전)** | **0.0 — 사용 안 함** | erasing | 0.2 |
| fliplr (좌우반전) | 0.5 | bgr | 0.0 |

상하반전을 끈 이유는 배근 사진의 상하가 의미를 갖기 때문이다(겹침이음 배치가 중력
방향과 무관하지 않다). 좌우반전은 유지했다.

## 3. 조기종료

`patience: 50` 으로 설정했으나 **다섯 실행 모두 200 에폭을 완주했고 조기종료는 발동하지
않았다.** 각 `results.csv` 의 행 수가 200 이다. `best.pt` 는 조기종료가 아니라 fitness
(`0.1 x mAP50 + 0.9 x mAP50-95`) 최댓값 에폭에서 저장된 것으로, 배포 모델은 **179 에폭**이다.

## 4. 클래스별 성능 (배포 모델 재검증)

```bash
python -c "from ultralytics import YOLO; YOLO('runs/splice_l/weights/best.pt').val(data='data/dataset.yaml', imgsz=640, split='val')"
```

검증셋 54 타일 / 528 인스턴스 (lap_splice 132, rebar 396):

| 클래스 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **lap_splice** | 0.868 | 0.818 | **0.861** | 0.698 |
| rebar | 0.857 | 0.678 | 0.796 | 0.576 |
| all | 0.862 | 0.748 | 0.828 | 0.637 |

보고서 표 4 의 lap_splice 정밀도는 0.872 로 적혀 있으나 재검증값은 0.868 이다
(mAP50 0.861, 재현율 0.818 은 일치). 보고서 값은 학습 종료 시점 검증 출력에서 옮긴
것이고 위 표는 `best.pt` 를 독립 실행해 다시 잰 값이므로, **위 표를 기준값으로 삼는다.**

## 5. 이 폴더에 없는 것

- **`weights/best.pt` · `last.pt`** — 실행당 20~53MB 라 git 에 넣지 않았다. 배포 모델
  (`splice_l/weights/best.pt`, 53MB)을 포함해 필요한 것을 요청하면 별도로 전달한다.
- **`train_batch*.jpg` · `val_batch*.jpg` · `labels.jpg`** — 현장 배근 사진이 그대로
  들어 있어 공개하지 않는다. 같은 이유로 `data/` 도 저장소에 없다.
