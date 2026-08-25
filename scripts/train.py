#!/usr/bin/env python3
"""철근 겹침이음 검출기 학습 — Ultralytics 전이학습.

사용:
    python scripts/train.py                        # 기본값으로
    python scripts/train.py --model yolo11n.pt     # 다른 가중치로
    python scripts/train.py --epochs 300           # 더 길게

100장 규모를 전제로 기본값을 잡았다. 큰 데이터셋과는 설정이 다르다:
  - epochs 를 길게(200) + patience 로 조기종료   소량이라 한 epoch 이 짧다
  - 증강을 강하게                                 현장 조건(역광·녹·먼지) 대응
  - batch 를 작게(8)                              100장에서 큰 배치는 스텝 수를 죽인다
  - freeze 없음                                   도메인이 COCO 와 멀어 전체를 푼다

학습 전에 `check_dataset.py` 를 먼저 돌려라. 네거티브 비율과 장소 누수를 잡아준다.
"""
from __future__ import annotations

import sys

# Windows 콘솔 기본 인코딩(cp949)은 em-dash 같은 문자를 못 찍고 UnicodeEncodeError 로 죽는다.
# 출력 인코딩을 UTF-8 로 고정한다. (Python 3.7+)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "dataset.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 모델: YOLO26 이 2026-01 릴리스로 가장 최신이고 NMS 가 없어 Core ML 변환이 단순하다.
    # 다만 가중치 파일명이 버전마다 달라질 수 있으므로 실패하면 --model 로 바꿔라
    # (대안: yolo11n.pt, yolov8n.pt — 셋 다 Ultralytics 가 자동 다운로드한다).
    p.add_argument("--model", default="yolo26n.pt", help="사전학습 가중치 (기본 yolo26n.pt)")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="dataset.yaml 경로")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50, help="이 epoch 동안 개선 없으면 중단")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="'0' / 'cpu' / 'mps'. 비우면 자동")
    p.add_argument("--name", default="splice", help="runs/detect/<name> 에 저장")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    a = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics 가 없다:  pip install -r requirements.txt")

    data = Path(a.data)
    if not data.is_file():
        raise SystemExit(f"dataset.yaml 이 없다: {data}\ndata/dataset.yaml.example 을 복사해서 만들어라")

    print(f"모델   : {a.model}")
    print(f"데이터 : {data}")
    print(f"설정   : epochs={a.epochs} imgsz={a.imgsz} batch={a.batch}\n")

    model = YOLO(a.model)

    results = model.train(
        data=str(data),
        epochs=a.epochs,
        patience=a.patience,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        seed=a.seed,
        project=str(ROOT / "runs"),
        name=a.name,
        exist_ok=True,

        # --- 증강: 현장 조건에 맞춰 강하게 ---
        hsv_h=0.015,   # 색조는 조금만 — 녹슨 철근의 적갈색이 단서다
        hsv_s=0.7,     # 채도는 크게 — 흐림/맑음/그늘 차이
        hsv_v=0.4,     # 명도 — 역광 대응
        degrees=15.0,  # 회전 — 기기를 기울여 찍는다
        translate=0.1,
        scale=0.5,     # 거리 변화 (0.3m ~ 2m)
        shear=0.0,
        perspective=0.0,
        flipud=0.0,    # 상하반전 금지 — 중력 방향이 배근에서 의미를 갖는다
        fliplr=0.5,    # 좌우반전은 괜찮다
        mosaic=1.0,    # 소량 데이터에서 특히 효과가 크다
        close_mosaic=10,  # 마지막 10 epoch 은 모자이크를 꺼서 실제 분포로 마무리
        mixup=0.1,
        erasing=0.2,   # 가림 대응 — 배근이 서로 가리는 상황이 흔하다
    )

    print("\n" + "=" * 60)
    print(f"저장 위치: {ROOT / 'runs' / a.name}")
    print("\n  best.pt 로 Core ML 변환:")
    print(f"    python scripts/export_coreml.py runs/{a.name}/weights/best.pt")
    print("\n  검증셋에서 눈으로 확인:")
    print(f"    python scripts/predict_test.py runs/{a.name}/weights/best.pt <이미지폴더>")
    print("\n★ mAP50 을 본다. mAP50-95 는 보지 않는다.")
    print("  100장 기준 mAP50 0.3~0.5 면 정상이다. 0 에 가까우면 라벨 정의를 다시 본다.")

    return results


if __name__ == "__main__":
    main()
