#!/usr/bin/env python3
"""학습 결과를 눈으로 확인한다 — 숫자보다 이게 중요하다.

사용:
    python scripts/predict_test.py runs/splice/weights/best.pt data/images/val
    python scripts/predict_test.py best.pt some.jpg --conf 0.15

100장 파일럿에서 mAP 숫자만 보면 판단을 그르친다. 실제로 봐야 할 것은 **어디서 틀리는가**다:

  - 다발철근을 이음으로 잡는가?
  - 스터럽 교차를 이음으로 잡는가?
  - 배근이 빽빽한 곳에서 다 놓치는가?

패턴이 보이면 그때 라벨 규칙에 항목을 추가한다. 그게 다음 100장을 훨씬 값지게 만든다.

**conf 를 낮게 잡는 이유**: 검측에서는 놓치는 것이 잘못 잡는 것보다 나쁘다. 오검출은
사람이 화면에서 지우면 되지만 미검출은 검사를 안 한 것이다. 기본값을 0.25 로 두었지만
파일럿에서는 0.1~0.15 로 낮춰 **무엇을 놓치는지** 보는 편이 낫다.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights")
    p.add_argument("source", help="이미지 파일 또는 폴더")
    p.add_argument("--conf", type=float, default=0.25, help="신뢰도 하한 (파일럿에서는 0.1~0.15 권장)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--name", default="pred")
    return p.parse_args()


def main() -> None:
    a = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics 가 없다:  pip install -r requirements.txt")

    w, src = Path(a.weights), Path(a.source)
    if not w.is_file():
        raise SystemExit(f"가중치가 없다: {w}")
    if not src.exists():
        raise SystemExit(f"입력이 없다: {src}")

    print(f"가중치: {w}\n입력  : {src}\nconf  : {a.conf}\n")

    model = YOLO(str(w))
    results = model.predict(
        source=str(src),
        conf=a.conf,
        imgsz=a.imgsz,
        save=True,
        project=str(ROOT / "runs"),
        name=a.name,
        exist_ok=True,
    )

    total = sum(len(r.boxes) for r in results)
    empty = sum(1 for r in results if len(r.boxes) == 0)

    print("\n" + "=" * 60)
    print(f"사진 {len(results)}장 · 검출 {total}개 · 아무것도 못 잡은 사진 {empty}장")
    if results:
        print(f"사진당 평균 {total / len(results):.1f}개")
    print(f"\n결과 이미지: {ROOT / 'runs' / a.name}")
    print("""
★ 이미지를 직접 열어 보라. 숫자로는 안 보이는 것을 본다:
   - 무엇을 이음이 아닌데 이음이라 하는가  (다발철근? 교차? 그림자?)
   - 어떤 상황에서 통째로 놓치는가        (역광? 빽빽한 배근? 먼 거리?)
   패턴이 보이면 라벨 규칙에 한 줄 추가하고 다음 라운드에 반영한다.
""")


if __name__ == "__main__":
    main()
