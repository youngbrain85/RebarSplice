#!/usr/bin/env python3
"""Roboflow 내보내기(전부 train/에 든 것)를 **영상 단위**로 학습/검증 분할한다.

사용:
    python scripts/prepare_dataset.py "D:/Projects/LH/Detection/rebar lapping.yolov11"
    python scripts/prepare_dataset.py <내보내기 폴더> --val IMG_0329 IMG_2410

왜 이 스크립트가 필요한가
------------------------
실데이터(2026-08-28)는 영상 6개에서 프레임을 뽑아 타일로 자른 것이다:
    IMG_0328_001_frame119_x1080_y2430_jpg.rf.<hash>.jpg
    └ 영상 ID   └ 프레임    └ 타일 위치

**같은 영상의 프레임들은 사실상 같은 사진이다.** 무작위로 나누면 검증셋이
학습셋과 같은 장면을 보게 되어 점수가 가짜로 높게 나온다. 그래서 분할 단위는
타일도 프레임도 아닌 **영상**이다.

기본 검증셋은 IMG_0329 + IMG_2410 이다 — 두 촬영 세션(03xx / 24xx)에서
하나씩 뽑아 54/309 타일(약 17%)이고, 두 클래스가 모두 들어 있다.

라벨은 폴리곤(YOLO seg 형식)이어도 그대로 복사한다. Ultralytics 는 detect
학습에서 폴리곤을 자동으로 축정렬 박스로 바꿔 쓴다(실측 확인).
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def video_of(name: str) -> str:
    """IMG_0328_001_frame119_... -> IMG_0328. 못 읽으면 파일명 앞 12자."""
    m = re.match(r"([A-Za-z]+_\d+)", name)
    return m.group(1) if m else name[:12]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="Roboflow 내보내기 폴더 (train/images, train/labels, data.yaml 포함)")
    p.add_argument("--val", nargs="+", default=["IMG_0329", "IMG_2410"],
                   help="검증셋으로 보낼 영상 ID (기본: IMG_0329 IMG_2410)")
    p.add_argument("--out", default=str(OUT), help=f"출력 폴더 (기본 {OUT})")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    src = Path(a.src)
    src_imgs = src / "train" / "images"
    src_labs = src / "train" / "labels"
    if not src_imgs.is_dir():
        raise SystemExit(f"train/images 가 없다: {src_imgs}")

    # 클래스 이름은 원본 data.yaml 에서 그대로 가져온다
    import yaml
    cfg = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]  # ['lap_splice', 'rebar']

    out = Path(a.out)
    imgs = sorted(p for p in src_imgs.iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise SystemExit(f"이미지가 없다: {src_imgs}")

    vids = Counter(video_of(p.name) for p in imgs)
    val_set = set(a.val)
    unknown = val_set - set(vids)
    if unknown:
        raise SystemExit(f"--val 에 준 영상이 데이터에 없다: {sorted(unknown)}\n있는 영상: {sorted(vids)}")

    print(f"원본: {src}")
    print(f"영상 {len(vids)}개, 타일 {len(imgs)}장: " + ", ".join(f"{v}({c})" for v, c in sorted(vids.items())))
    print(f"검증셋 영상: {sorted(val_set)}\n")

    # 출력 폴더를 새로 만든다 (기존 분할 잔재가 섞이면 안 된다)
    for split in ("train", "valid"):
        d = out / split
        if d.exists():
            shutil.rmtree(d)
        (d / "images").mkdir(parents=True)
        (d / "labels").mkdir(parents=True)

    n = Counter()
    boxes = Counter()
    for img in imgs:
        split = "valid" if video_of(img.name) in val_set else "train"
        shutil.copy2(img, out / split / "images" / img.name)
        lab = src_labs / (img.stem + ".txt")
        if lab.is_file():
            shutil.copy2(lab, out / split / "labels" / lab.name)
            for line in lab.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    boxes[(split, int(line.split()[0]))] += 1
        n[split] += 1

    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        "# prepare_dataset.py 가 생성. 손으로 고치지 말고 스크립트를 다시 돌려라.\n"
        f"path: {out.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(names)),
        encoding="utf-8",
    )

    print(f"[train] {n['train']}장 · " + " · ".join(f"{names[c]} {boxes[('train', c)]}" for c in range(len(names))))
    print(f"[valid] {n['valid']}장 · " + " · ".join(f"{names[c]} {boxes[('valid', c)]}" for c in range(len(names))))
    print(f"\n생성: {yaml_path}")
    print(f"\n다음:\n  python scripts/check_dataset.py {yaml_path.relative_to(ROOT) if yaml_path.is_relative_to(ROOT) else yaml_path}")
    print("  python scripts/train.py")


if __name__ == "__main__":
    main()
