# RebarSplice

철근 **겹침이음(lap splice)** 을 카메라 화면에서 자동 검출하는 iOS 앱과 그 학습 파이프라인.

건설현장 배근 검측 용도. 이 저장소는 **학습·변환 쪽**이고, iOS 앱은 모델이 나온 뒤에 붙인다.

---

## 지금 상태

파일럿 단계다. 사진 100장으로 **"이게 될 일인가"** 를 먼저 판단한다.

| | |
|---|---|
| 클래스 | `lap_splice` 1개 |
| 라벨 | 축정렬 바운딩박스 |
| 목표 | mAP50 **0.3~0.5** (100장 기준). 0 에 가까우면 라벨 정의를 다시 본다 |
| 배포 | TestFlight 전용 (App Store 안 올림) |

**철근 이음부 공개 데이터셋은 존재하지 않는다.** Roboflow Universe · Kaggle ·
HuggingFace · AI-Hub 를 전수 검색한 결과 0건이다(2026-08 확인). 처음부터 만들어야 한다.

---

## 순서

### 1. 사진을 모은다

`docs/labeling-guide.md` 의 촬영 가이드를 따른다. 요점 하나만 옮기면:

> **같은 이음부를 100번 찍으면 100장이 아니라 1장이다.**
> 서로 다른 이음부를 30개소 이상, 거리·각도·조명을 섞어서.

파일명 앞에 촬영 장소를 붙인다 — `siteA_001.jpg`, `siteB_014.jpg`.
검사 스크립트가 이걸로 학습/검증 누수를 잡는다.

### 2. 라벨한다

**Roboflow** 가 가장 빠르다. 규칙은 한 줄이다:

> 이음부처럼 보이면 박스를 친다.

**이음이 없는 사진을 20~30% 섞어 박스 0개로 넣는다.** 안 넣으면 모델이
"철근이 보이면 이음"으로 학습한다 — 이 도메인 최빈 실패다.

내보내기는 **YOLOv8 포맷**. 압축을 `data/` 에 풀고 `dataset.yaml` 을 맞춘다
(`data/dataset.yaml.example` 참고).

### 3. 데이터를 검사한다

```bash
python scripts/check_dataset.py data/dataset.yaml
```

네거티브 비율 · 촬영 장소 수 · **학습/검증 장소 누수** 를 본다.
문제가 있으면 고치고 다시 돌린다. 학습은 그다음이다.

### 4. 학습한다

```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\python -m pip install -r requirements.txt

.venv\Scripts\python scripts/train.py
```

100장 규모에 맞춰 기본값을 잡아 뒀다 — 긴 epoch + 조기종료, 강한 증강, 작은 배치.

**GPU 를 쓰는지 먼저 확인하라.** 안 쓰면 100장에도 몇 시간이 걸린다:

```bash
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 5. 결과를 눈으로 본다

```bash
python scripts/predict_test.py runs/splice/weights/best.pt data/images/val --conf 0.15
```

**숫자보다 이게 중요하다.** 어디서 틀리는지 — 다발철근? 스터럽 교차? 역광? —
패턴이 보이면 라벨 규칙에 한 줄 추가하고 다음 라운드에 반영한다.

### 6. Core ML 로 바꾼다

**★ Windows 에서는 안 된다.** Ultralytics 가 `assert not WINDOWS` 로 막는다(실측 확인).
이 저장소는 공개라 **GitHub Actions 의 Linux 러너가 무료**이므로 거기서 돌린다.

```bash
copy runs\splice\weights\best.pt models\best.pt
git add models/best.pt && git commit -m "weights" && git push

gh workflow run coreml.yml -f weights=models/best.pt -f imgsz=640
gh run download <run-id>          # .mlpackage 를 아티팩트로
```

맥이나 WSL 이 있으면 로컬에서 그대로 돌려도 된다:

```bash
python scripts/export_coreml.py models/best.pt --imgsz 640
python scripts/export_coreml.py models/best.pt --imgsz 640 --int8   # 비교용
```

변환 후 **Xcode 에서 .mlpackage 를 열고 Performance 탭**을 확인한다.
CPU 로 떨어지는 레이어가 있으면 ARKit 과 같이 못 쓴다.

**YOLO26 은 NMS-free 다.** 변환 시 `nms=True` 가 자동으로 꺼지는데 정상이다 —
모델이 이미 최종 박스를 내므로 Swift 쪽에서 NMS 를 따로 구현할 필요가 없다.

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
Ultralytics YOLO 는 AGPL-3.0 이고 이 저장소는 공개다. App Store 심사와 GPL 계열의
충돌은 알려진 마찰 지점이지만 **TestFlight 배포에는 해당 없다.**

---

## 구조

```
scripts/
  check_dataset.py    학습 전 위생 검사 — 네거티브 비율·장소 누수
  train.py            Ultralytics 전이학습
  predict_test.py     결과를 눈으로 확인
  export_coreml.py    .pt -> .mlpackage
docs/
  labeling-guide.md   라벨링·촬영 규칙
data/                 데이터셋 (gitignore — 현장 사진은 커밋하지 않는다)
models/               변환 산출물 (gitignore — 용량이 크다)
```

## 라이선스

코드는 MIT. 단 학습에 Ultralytics(AGPL-3.0)를 쓰므로, 그 코드로 만든 **가중치를
배포할 때는 AGPL 이 따라온다.**
