#!/usr/bin/env python3
"""데이터셋 위생 검사 — 학습 전에 반드시 돌린다.

라벨을 아무리 잘 그려도 아래 셋을 어기면 100장이 통째로 쓸모없어진다.
그래서 조언으로 두지 않고 검사로 만들었다.

  1. 네거티브(이음 없는) 사진 비율      권장 20~30%
  2. 서로 다른 촬영 장소 수              권장 3곳 이상 (검증 분리에 필요)
  3. 학습/검증 분할이 장소 단위인가       같은 장소가 양쪽에 있으면 검증 점수가 가짜다

사용:
    python scripts/check_dataset.py data/dataset.yaml

촬영 장소는 이미지 **파일명의 첫 토큰**으로 판정한다(`siteA_001.jpg` -> `siteA`).
구분자는 `_` 또는 `-`. 파일명에 장소가 없으면 경고만 내고 3번 검사를 건너뛴다.
"""
from __future__ import annotations

import sys

# Windows 콘솔 기본 인코딩(cp949)은 em-dash 같은 문자를 못 찍고 UnicodeEncodeError 로 죽는다.
# 출력 인코딩을 UTF-8 로 고정한다. (Python 3.7+)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml 이 필요하다:  pip install pyyaml")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_path_for(img: Path) -> Path:
    """Ultralytics 규약: .../images/... -> .../labels/....txt"""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        return img.with_suffix(".txt")
    return Path(*parts).with_suffix(".txt")


def session_of(img: Path) -> str | None:
    """파일명 첫 토큰을 촬영 장소로 본다. 없으면 None."""
    stem = img.stem
    for sep in ("_", "-"):
        if sep in stem:
            head = stem.split(sep)[0]
            # 순수 숫자면 장소명이 아니라 일련번호로 본다
            return None if head.isdigit() else head
    return None


def scan(split_dir: Path) -> tuple[list[Path], int, int, Counter]:
    """(이미지들, 박스 있는 장수, 박스 총수, 장소별 장수)"""
    imgs = sorted(p for p in split_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    positives = 0
    boxes = 0
    sessions: Counter = Counter()
    for img in imgs:
        lp = label_path_for(img)
        n = 0
        if lp.is_file():
            n = sum(1 for line in lp.read_text(encoding="utf-8").splitlines() if line.strip())
        if n:
            positives += 1
            boxes += n
        s = session_of(img)
        if s:
            sessions[s] += 1
    return imgs, positives, boxes, sessions


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cfg_path = Path(sys.argv[1]).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    root = Path(cfg.get("path", cfg_path.parent))
    if not root.is_absolute():
        root = (cfg_path.parent / root).resolve()

    problems: list[str] = []
    warnings: list[str] = []
    all_sessions: dict[str, Counter] = {}

    print(f"데이터셋: {cfg_path}")
    print(f"클래스: {cfg.get('names')}\n")

    total_imgs = total_pos = total_boxes = 0

    for split in ("train", "val"):
        rel = cfg.get(split)
        if not rel:
            problems.append(f"dataset.yaml 에 '{split}' 항목이 없다")
            continue
        d = root / rel
        if not d.is_dir():
            problems.append(f"{split} 경로가 없다: {d}")
            continue

        imgs, pos, boxes, sessions = scan(d)
        neg = len(imgs) - pos
        ratio = (neg / len(imgs) * 100) if imgs else 0
        total_imgs += len(imgs)
        total_pos += pos
        total_boxes += boxes
        all_sessions[split] = sessions

        print(f"[{split}]  사진 {len(imgs)}장  ·  박스 있는 사진 {pos}장  ·  박스 {boxes}개")
        print(f"         네거티브 {neg}장 ({ratio:.0f}%)  ·  장소 {len(sessions)}곳 {dict(sessions)}")
        if imgs and boxes:
            print(f"         사진당 평균 박스 {boxes / max(pos, 1):.1f}개")
        print()

        if not imgs:
            problems.append(f"{split} 에 이미지가 없다")

    # --- 1. 네거티브 비율 (전체 기준) ---
    if total_imgs:
        neg_all = total_imgs - total_pos
        r = neg_all / total_imgs * 100
        if r < 10:
            problems.append(
                f"네거티브가 {r:.0f}% 뿐이다 (권장 20~30%). "
                "이음 없는 배근 사진을 박스 0개로 넣어라 — 안 넣으면 모델이 '철근=이음'으로 학습한다"
            )
        elif r < 20:
            warnings.append(f"네거티브 {r:.0f}% — 권장 20~30% 보다 적다")
        elif r > 50:
            warnings.append(f"네거티브가 {r:.0f}% 로 과하다 — 포지티브가 부족해 학습이 어려울 수 있다")

    # --- 2. 장소 수 ---
    known = set().union(*(set(c) for c in all_sessions.values())) if all_sessions else set()
    if not known:
        warnings.append(
            "파일명에서 촬영 장소를 못 읽었다. `siteA_001.jpg` 처럼 앞에 장소를 붙이면 "
            "학습/검증 누수를 검사할 수 있다 (지금은 3번 검사를 건너뛴다)"
        )
    elif len(known) < 3:
        warnings.append(f"촬영 장소가 {len(known)}곳 뿐이다 — 3곳 이상이면 검증 분리가 깔끔하다")

    # --- 3. 학습/검증 장소 누수 ---
    tr = set(all_sessions.get("train", {}))
    va = set(all_sessions.get("val", {}))
    if tr and va:
        leak = tr & va
        if leak:
            problems.append(
                f"학습과 검증에 같은 장소가 있다: {sorted(leak)}. "
                "같은 이음부를 찍은 사진이 양쪽에 들어가면 검증 점수가 가짜로 높게 나온다 — "
                "장소 단위로 나눠라"
            )

    # --- 결과 ---
    print("=" * 60)
    print(f"합계: 사진 {total_imgs}장 · 박스 {total_boxes}개")
    if total_imgs and total_imgs < 50:
        warnings.append(f"사진이 {total_imgs}장 뿐이다 — 100장은 모아야 파일럿 판단이 된다")
    print()

    for w in warnings:
        print(f"  [경고] {w}")
    for p in problems:
        print(f"  [문제] {p}")

    if not warnings and not problems:
        print("  이상 없음.")
    print()

    if problems:
        print("문제를 고치고 다시 돌려라. 학습은 그다음이다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
