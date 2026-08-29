// 추론 전후처리 — 순수 함수만. 브라우저와 node 양쪽에서 돈다 (테스트: infer.test.mjs).
//
// 왜 타일인가: 학습 데이터가 원본 프레임을 720~1080px 타일로 잘라 라벨한 것이라,
// 4K 사진을 통째로 640 에 욱여넣으면 철근이 학습 분포보다 훨씬 가늘어져 못 잡는다.
// 추론도 같은 크기로 잘라 돌리고(겹침 25%), 겹침 구간의 중복 박스는 IoU 로 합친다.

/** 원본(w,h)을 dst 정사각형에 비율 유지로 넣을 때의 배치. */
export function letterbox(srcW, srcH, dst) {
  const scale = Math.min(dst / srcW, dst / srcH);
  const drawW = Math.round(srcW * scale);
  const drawH = Math.round(srcH * scale);
  return { scale, dx: (dst - drawW) / 2, dy: (dst - drawH) / 2, drawW, drawH };
}

/** 모델 좌표(dst 기준) -> 원본 좌표. tileX/Y 는 타일의 원본 내 원점. */
export function mapToSource(box, lb, tileX, tileY) {
  return {
    x1: (box.x1 - lb.dx) / lb.scale + tileX,
    y1: (box.y1 - lb.dy) / lb.scale + tileY,
    x2: (box.x2 - lb.dx) / lb.scale + tileX,
    y2: (box.y2 - lb.dy) / lb.scale + tileY,
  };
}

/**
 * 이미지를 tile×tile 로 자르는 계획. 이미지가 타일의 1.4배 이하면 통째로 한 장.
 * 각 방향에서 마지막 타일은 끝에 붙여서(잘리는 구간 없이) 전체를 덮는다.
 */
export function planTiles(w, h, tile, overlap) {
  if (w <= tile * 1.4 && h <= tile * 1.4) return [{ x: 0, y: 0, w, h }];
  const stride = Math.round(tile * (1 - overlap));
  const starts = (len) => {
    if (len <= tile) return [0];
    const out = [];
    for (let p = 0; p + tile < len; p += stride) out.push(p);
    out.push(len - tile);
    return [...new Set(out)];
  };
  const tiles = [];
  for (const y of starts(h))
    for (const x of starts(w))
      tiles.push({ x, y, w: Math.min(tile, w), h: Math.min(tile, h) });
  return tiles;
}

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  if (!inter) return 0;
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return inter / ua;
}

/** 타일 겹침 구간의 중복 검출을 합친다 — 클래스별 greedy NMS. */
export function mergeDetections(dets, iouThr) {
  const sorted = [...dets].sort((a, b) => b.conf - a.conf);
  const kept = [];
  for (const d of sorted) {
    if (!kept.some((k) => k.cls === d.cls && iou(k, d) > iouThr)) kept.push(d);
  }
  return kept;
}

/**
 * YOLO26 end2end 출력 [rows,6] (x1,y1,x2,y2,conf,cls — dst 픽셀 기준) 을
 * conf 로 거르고 원본 좌표로 되돌린다. srcW/H 를 주면 경계로 자른다.
 */
export function parseOutput(data, rows, confThr, lb, tileX, tileY, srcW, srcH) {
  const dets = [];
  for (let i = 0; i < rows; i++) {
    const o = i * 6;
    const conf = data[o + 4];
    if (conf < confThr) continue;
    const b = mapToSource({ x1: data[o], y1: data[o + 1], x2: data[o + 2], y2: data[o + 3] }, lb, tileX, tileY);
    if (srcW != null) {
      b.x1 = Math.max(tileX, Math.min(b.x1, srcW));
      b.x2 = Math.max(0, Math.min(b.x2, srcW));
      b.y1 = Math.max(tileY, Math.min(b.y1, srcH));
      b.y2 = Math.max(0, Math.min(b.y2, srcH));
    }
    dets.push({ ...b, conf, cls: data[o + 5] });
  }
  return dets;
}
