# RebarSplice

건설현장 배근 사진에서 철근 **겹침이음(lap splice)** 을 자동 검출하는 웹앱.

**▶ https://rebarsplice.vercel.app** — 사진을 올리면 브라우저 안에서 검출한다.
서버 전송 없음(추론은 ONNX Runtime Web, 기기 안에서 실행).

| | |
|---|---|
| 모델 | YOLO26-l, 입력 640 — lap_splice **mAP50 0.861** (P 0.872 / R 0.818) |
| 데이터셋 | [Roboflow Universe — rebar-lapping](https://universe.roboflow.com/hee-jun-yang-endorphiny/rebar-lapping/dataset/dataset) (타일 309장, `lap_splice` + `rebar` 2클래스) |
| 상세 | [테스트 보고서 (PDF)](docs/test-report-2026-08-29.pdf) |

---

## 실행

### 웹앱 (로컬)

```bash
python scripts/serve_web.py        # → http://localhost:8377
```

모델 파일 `web/models/best.onnx` 는 저장소에 없다(용량). 아래 학습 절차로 직접
만들거나, 배포된 웹앱을 그대로 쓰면 된다.

### 학습 → 모델 만들기

요구사항: Python 3.9+, NVIDIA GPU 권장(CPU도 되지만 느리다).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/python -m pip install -r requirements.txt

# 1. 데이터셋(위 링크)을 YOLOv8 포맷으로 받아 영상 단위로 학습/검증 분할
.venv/Scripts/python scripts/prepare_dataset.py <Roboflow 내보내기 폴더>

# 2. 검사 → 학습 → 확인 → 변환
.venv/Scripts/python scripts/check_dataset.py data/dataset.yaml
.venv/Scripts/python scripts/train.py --data data/dataset.yaml --model yolo26l.pt
.venv/Scripts/python scripts/predict_test.py runs/splice/weights/best.pt data/valid/images --conf 0.15
.venv/Scripts/python scripts/export_onnx.py runs/splice/weights/best.pt --imgsz 640

# 3. 웹앱에 넣기
cp models/best.onnx web/models/best.onnx
```

참고: RTX 2060 SUPER 기준 학습 약 1시간(l 모델, 200 epoch).

### 웹앱 배포

```bash
cd web && npx vercel deploy --prod --yes
```

### 테스트

```bash
node --test web/infer.test.mjs     # 추론 좌표 수학 12케이스
```

---

## 구조

```
scripts/   데이터 분할·검사·학습·ONNX 변환·개발서버
web/       웹앱 (index.html · app.js · infer.js + 단위테스트)
docs/      상세 문서 · 테스트 보고서
```

## 라이선스

코드는 MIT. 학습에 Ultralytics(AGPL-3.0)를 쓰므로 그 코드로 만든 **가중치를
배포할 때는 AGPL 이 따라온다.** 데이터셋은 CC BY 4.0.
