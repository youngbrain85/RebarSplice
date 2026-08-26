#!/usr/bin/env python3
"""학습된 가중치 -> ONNX. 웹앱(ONNX Runtime Web)이 이 파일을 브라우저에서 돌린다.

사용:
    python scripts/export_onnx.py runs/splice/weights/best.pt
    python scripts/export_onnx.py runs/splice/weights/best.pt --imgsz 640

Core ML 과 달리 **Windows 에서 그냥 된다**(실측 확인). CI 가 필요 없다.

## 인자를 이렇게 잡은 이유

**opset 12**: ONNX Runtime Web 이 넓게 지원하는 하한이다. 더 올리면 최신 브라우저에서만
돌 수 있다. 이 모델에는 12 로 충분하다.

**dynamic=False**: 입력 크기를 640 으로 고정한다. 웹에서 동적 shape 는 실행 계획을 매번
다시 짜게 만들어 느려지고, 어차피 브라우저가 사진을 640 으로 리사이즈해 넣는다.

**quantize=32**: fp32 로 둔다. WASM 은 fp16 연산을 제대로 못 쓰고, 양자화는 정확도를 깎는다.

**양자화 안 함**: int8 로 줄이면 파일이 작아지지만 정확도가 떨어진다. 이 프로젝트는
정확도가 우선이고, 파일 크기는 첫 로딩 한 번만 부담이므로(이후 캐시) 원본을 쓴다.

**simplify 기본 꺼짐**: onnxslim 이 이 환경에서 세그폴트를 낸다(실측). ONNX 파일은
쓰인 뒤에 죽어서 파일은 남지만 스크립트가 끝을 못 본다. 그래프가 조금 커질 뿐 동작에는
지장이 없다. 다른 환경에서 되면 `--simplify` 로 켜라.

## 출력 형태 — 웹앱 후처리가 왜 간단한가

YOLO26 은 **NMS-free(end2end)** 다. 출력이 이미 최종 박스라 JS 로 NMS 를 짤 필요가 없다:

    입력  images   [1, 3, 640, 640]  float32   (0~1 정규화, RGB, NCHW)
    출력  output0  [1, 300, 6]       float32
          한 행 = [x1, y1, x2, y2, confidence, class]   좌표는 640 기준 픽셀

브라우저가 할 일은 (1) 사진을 640 으로 레터박스 리사이즈, (2) conf 로 거르기,
(3) 좌표를 원본 사진 크기로 되돌리기 — 이 셋뿐이다.
"""
from __future__ import annotations

import sys

# Windows 콘솔 기본 인코딩(cp949)은 em-dash 같은 문자를 못 찍고 UnicodeEncodeError 로 죽는다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="학습된 .pt (보통 runs/<name>/weights/best.pt)")
    p.add_argument("--imgsz", type=int, default=640, help="학습 때와 같은 값을 써라")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset (기본 12 — ORT Web 호환)")
    p.add_argument("--name", default=None, help="출력 파일명 (기본: 가중치 이름)")
    # onnxslim 은 이 환경(win / py3.9 / onnx 1.19 / onnxslim 0.1.96)에서 **세그폴트**를 낸다.
    # ONNX 파일은 쓰인 뒤에 죽어서 파일은 남지만 스크립트가 끝을 못 본다. 기본으로 끈다.
    # 그래프가 조금 커지지만 동작에는 지장이 없다.
    p.add_argument("--simplify", action="store_true",
                   help="onnxslim 으로 그래프 정리 (이 환경에서는 세그폴트 — 기본 꺼짐)")
    return p.parse_args()


def main() -> None:
    a = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics 가 없다:  pip install -r requirements.txt")

    w = Path(a.weights)
    if not w.is_file():
        import re
        if not re.fullmatch(r"yolo[0-9v]+[nsmlx](-[a-z]+)?\.pt", w.name):
            raise SystemExit(f"가중치가 없다: {w}")
        print(f"(사전학습 가중치로 본다: {w.name})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"변환: {w}")
    print(f"설정: imgsz={a.imgsz}  opset={a.opset}  dynamic=False  fp32  simplify={a.simplify}\n")

    model = YOLO(str(w))
    out = Path(model.export(
        format="onnx",
        imgsz=a.imgsz,
        opset=a.opset,
        dynamic=False,
        quantize=32,   # fp32 유지. WASM 은 fp16 을 제대로 못 쓴다
        simplify=a.simplify,
    ))

    dest = OUT_DIR / f"{a.name or w.stem}.onnx"
    if dest.exists():
        dest.unlink()
    shutil.move(str(out), str(dest))
    size_mb = dest.stat().st_size / 1e6

    print("\n" + "=" * 60)
    print(f"완료: {dest}  ({size_mb:.1f} MB)")

    # 실제 입출력 형태를 찍어 준다 — 웹앱 후처리를 짤 때 이 값이 그대로 필요하다
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
        print()
        for i in s.get_inputs():
            print(f"  입력  {i.name:10} {i.shape}  {i.type}")
        for o in s.get_outputs():
            print(f"  출력  {o.name:10} {o.shape}  {o.type}")
        print("\n  출력 한 행 = [x1, y1, x2, y2, confidence, class]  (좌표는 imgsz 기준 픽셀)")
        print("  YOLO26 은 NMS-free 라 이미 최종 박스다 — JS 에서 NMS 를 짤 필요가 없다.")
    except ImportError:
        print("\n  (onnxruntime 이 없어 입출력 형태는 못 찍었다)")

    print(f"""
다음:
  1. 웹앱이 읽을 위치로 복사한다
       copy models\\{dest.name} web\\public\\models\\
  2. 브라우저에서 ONNX Runtime Web 으로 부른다:

       import * as ort from 'onnxruntime-web';
       const session = await ort.InferenceSession.create('/models/{dest.name}', {{
         executionProviders: ['wasm'],
       }});
       const feeds = {{ images: new ort.Tensor('float32', chw, [1, 3, {a.imgsz}, {a.imgsz}]) }};
       const out = (await session.run(feeds)).output0.data;   // 300 x 6

  3. 사진은 브라우저에서 {a.imgsz} 로 레터박스 리사이즈해 넣는다.
     아이폰 사진은 12MP 라 원본 그대로 텐서를 만들면 iOS Safari 가 메모리로 죽는다.
""")


if __name__ == "__main__":
    main()
