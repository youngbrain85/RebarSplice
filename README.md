# RebarSplice

A web app that automatically detects rebar **lap splices** in photos of rebar placement taken on construction sites.

**▶ https://rebarsplice.vercel.app** — upload a photo and detection runs inside the browser.
Nothing is sent to a server (inference runs on the device with ONNX Runtime Web).

| | |
|---|---|
| Model | YOLO26-l, 640 input — lap_splice **mAP50 0.861** (P 0.868 / R 0.818) |
| Dataset | [Roboflow Universe — rebar-lapping](https://universe.roboflow.com/hee-jun-yang-endorphiny/rebar-lapping/dataset/dataset) (309 tiles, two classes: `lap_splice` + `rebar`) |
| Details | [Test report (PDF)](docs/test-report-2026-08-29.pdf) |
| Training record | [Run outputs and final hyperparameters](docs/training-runs.md) |

---

## Running

### Web app (local)

```bash
python scripts/serve_web.py        # → http://localhost:8377
```

The model file `web/models/best.onnx` is not in the repository (file size). Build it yourself with the
training procedure below, or use the deployed web app as it is.

### Training → building the model

Requirements: Python 3.9+, an NVIDIA GPU is recommended (a CPU also works, but slowly).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/python -m pip install -r requirements.txt

# 1. Download the dataset (link above) in YOLOv8 format and split it into training/validation sets by source video
.venv/Scripts/python scripts/prepare_dataset.py <Roboflow export folder>

# 2. Check → train → verify → convert
.venv/Scripts/python scripts/check_dataset.py data/dataset.yaml
.venv/Scripts/python scripts/train.py --data data/dataset.yaml --model yolo26l.pt
.venv/Scripts/python scripts/predict_test.py runs/splice/weights/best.pt data/valid/images --conf 0.15
.venv/Scripts/python scripts/export_onnx.py runs/splice/weights/best.pt --imgsz 640

# 3. Add it to the web app
cp models/best.onnx web/models/best.onnx
```

Note: training takes about 1 hour on an RTX 2060 SUPER (l model, 200 epochs).

### Deploying the web app

```bash
cd web && npx vercel deploy --prod --yes
```

### Tests

```bash
node --test web/infer.test.mjs     # 12 cases for the inference coordinate math
```

---

## Structure

```
scripts/   Dataset splitting, checking, training, ONNX conversion, development server
web/       Web app (index.html · app.js · infer.js + unit tests)
docs/      Detailed documents · test report
```

## License

The code is MIT-licensed. Training uses Ultralytics (AGPL-3.0), so **distributing weights produced with that
code carries the AGPL with it.** The dataset is CC BY 4.0.
