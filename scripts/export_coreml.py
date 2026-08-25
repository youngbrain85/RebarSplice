#!/usr/bin/env python3
"""학습된 가중치 -> Core ML (.mlpackage) 변환.

★ **Windows 에서는 안 된다.** Ultralytics 가 `assert not WINDOWS` 로 막는다.
   macOS / Linux / WSL 에서 돌리거나, `.github/workflows/coreml.yml` 을 쓴다
   (공개 저장소라 Linux 러너가 무료다).

사용:
    python scripts/export_coreml.py runs/splice/weights/best.pt
    python scripts/export_coreml.py best.pt --int8        # 크기·속도 우선
    python scripts/export_coreml.py best.pt --imgsz 320   # 더 빠르게

## 왜 이 인자들인가

**nms**: NMS(중복 박스 제거)를 Core ML 그래프 안에 넣는다. 안 넣으면 Swift 쪽에서
직접 구현해야 하고, 좌표계 변환에서 실수가 잘 난다. 넣는 쪽이 압도적으로 낫다.
단 YOLO26 계열은 애초에 NMS-free(end2end) 라 Ultralytics 가 경고와 함께 자동으로 끈다:
`WARNING 'nms=True' is not available for end2end models. Forcing 'nms=False'.`
그 경우 Swift 쪽에서 NMS 를 따로 할 필요가 없다 — 모델이 이미 최종 박스를 낸다.

**imgsz**: 학습 때와 같은 값을 써야 한다. 다르면 정확도가 떨어진다.

**양자화**: fp16 이 기본이다. int8 은 크기가 절반이고 Neural Engine 에서 더 빠르지만
정확도가 떨어질 수 있다 — 두 개 다 만들어 기기에서 비교하는 걸 권한다.

## 변환 후 반드시 할 것

Xcode 에서 .mlpackage 를 열고 **Performance 탭**에서 레이어별 실행 유닛을 확인하라.
CPU 로 떨어지는 레이어가 있으면 추론이 몇 배 느려진다. ARKit 이 이미 GPU 를 쓰는
상황에서는 치명적이다.
"""
from __future__ import annotations

import sys

# Windows 콘솔 기본 인코딩(cp949)은 em-dash 같은 문자를 못 찍고 UnicodeEncodeError 로 죽는다.
# 출력 인코딩을 UTF-8 로 고정한다. (Python 3.7+)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import platform
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", help="학습된 .pt (보통 runs/<name>/weights/best.pt)")
    p.add_argument("--imgsz", type=int, default=640, help="학습 때와 같은 값을 써라")
    p.add_argument("--int8", action="store_true", help="INT8 양자화 (기본은 FP16)")
    p.add_argument("--no-nms", action="store_true", help="NMS 를 그래프에 넣지 않는다")
    p.add_argument("--name", default=None, help="출력 파일명 (기본: 가중치 이름)")
    return p.parse_args()


def main() -> None:
    a = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics 가 없다:  pip install -r requirements.txt")

    w = Path(a.weights)
    if not w.is_file():
        raise SystemExit(f"가중치가 없다: {w}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(w))

    kwargs = dict(format="coreml", imgsz=a.imgsz, quantize=8 if a.int8 else 16)
    if not a.no_nms:
        kwargs["nms"] = True

    print(f"변환: {w}")
    print(f"설정: imgsz={a.imgsz}  양자화={'INT8' if a.int8 else 'FP16'}  nms={not a.no_nms}\n")

    if platform.system() == "Windows":
        raise SystemExit(
            "Core ML 변환은 Windows 에서 안 된다 — Ultralytics 가 assert 로 막는다.\n\n"
            "이 저장소는 공개라 GitHub Actions 의 Linux 러너가 무료다. 그걸 쓴다:\n\n"
            "  1) 가중치를 models/ 로 복사하고 커밋\n"
            "       copy runs\\splice\\weights\\best.pt models\\best.pt\n"
            "       git add models/best.pt && git commit -m weights && git push\n\n"
            "  2) 변환 워크플로 실행\n"
            "       gh workflow run coreml.yml -f weights=models/best.pt -f imgsz=640\n\n"
            "  3) 끝나면 .mlpackage 를 아티팩트로 내려받기\n"
            "       gh run download <run-id>\n\n"
            "(맥이나 WSL 이 있으면 거기서 이 스크립트를 그대로 돌려도 된다)"
        )

    # YOLO26 같은 end2end 모델은 애초에 NMS 가 없다. Ultralytics 가 nms=True 를
    # 경고와 함께 자동으로 끄므로 우리가 예외 처리할 필요는 없다:
    #   WARNING 'nms=True' is not available for end2end models. Forcing 'nms=False'.
    # 그 경우 Swift 쪽에서 NMS 를 따로 할 필요도 없다 — 모델이 최종 박스를 낸다.
    out = model.export(**kwargs)

    out = Path(out)
    stem = a.name or w.stem
    suffix = "_int8" if a.int8 else "_fp16"
    dest = OUT_DIR / f"{stem}{suffix}{out.suffix}"

    if dest.exists():
        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
    shutil.move(str(out), str(dest))

    size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6 if dest.is_dir() \
        else dest.stat().st_size / 1e6

    print("\n" + "=" * 60)
    print(f"완료: {dest}  ({size_mb:.1f} MB)")
    print("""
다음:
  1. .mlpackage 를 Xcode 프로젝트에 끌어다 놓는다
  2. Xcode 에서 열어 **Performance 탭**을 확인한다
     -> CPU 로 떨어지는 레이어가 있으면 ARKit 과 같이 못 쓴다
  3. Vision 으로 부른다:

       let model = try VNCoreMLModel(for: SpliceDetector(configuration: .init()).model)
       let request = VNCoreMLRequest(model: model) { req, _ in
           guard let results = req.results as? [VNRecognizedObjectObservation] else { return }
           // results[i].boundingBox 는 정규화 좌표(좌하단 원점)다.
           // ARKit 프레임 좌표계와 다르니 변환에 주의할 것.
       }
       request.imageCropAndScaleOption = .scaleFill

  4. ARKit 과 같이 쓸 때는 매 프레임 추론하지 마라 — 10Hz 정도로 제한하고
     백그라운드 큐에서 돌린다. ARSession 프레임을 굶기면 트래킹이 흔들린다.
""")


if __name__ == "__main__":
    main()
