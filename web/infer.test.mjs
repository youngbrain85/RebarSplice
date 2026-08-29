// infer.js 의 순수 함수 검증.  실행:  node --test web/
import { test } from "node:test";
import assert from "node:assert/strict";
import { letterbox, mapToSource, planTiles, iou, mergeDetections, parseOutput } from "./infer.js";

// ── letterbox ────────────────────────────────────────────────
test("letterbox: 가로가 긴 이미지는 세로에 패딩이 붙는다", () => {
  const lb = letterbox(1280, 720, 640);
  assert.equal(lb.scale, 0.5);           // 1280 -> 640
  assert.equal(lb.dx, 0);
  assert.equal(lb.dy, (640 - 360) / 2);  // 위아래 140씩
  assert.equal(lb.drawW, 640);
  assert.equal(lb.drawH, 360);
});

test("letterbox: 정사각형이면 패딩이 없다", () => {
  const lb = letterbox(1080, 1080, 640);
  assert.ok(Math.abs(lb.scale - 640 / 1080) < 1e-9);
  assert.equal(lb.dx, 0);
  assert.equal(lb.dy, 0);
});

test("letterbox: 모델 입력보다 작은 이미지는 확대된다", () => {
  const lb = letterbox(320, 320, 640);
  assert.equal(lb.scale, 2);
});

// ── mapToSource ──────────────────────────────────────────────
test("mapToSource: 640 좌표를 원본 좌표로 되돌린다", () => {
  const lb = letterbox(1280, 720, 640);  // scale .5, dy 140
  // 640 공간의 (0,140)-(640,500) == 원본 전체
  const b = mapToSource({ x1: 0, y1: 140, x2: 640, y2: 500 }, lb, 0, 0);
  assert.deepEqual([b.x1, b.y1, b.x2, b.y2], [0, 0, 1280, 720]);
});

test("mapToSource: 타일 원점이 더해진다", () => {
  const lb = letterbox(960, 960, 640);
  const b = mapToSource({ x1: 0, y1: 0, x2: 640, y2: 640 }, lb, 1000, 2000);
  assert.deepEqual([b.x1, b.y1, b.x2, b.y2], [1000, 2000, 1960, 2960]);
});

// ── planTiles ────────────────────────────────────────────────
test("planTiles: 타일보다 작은 이미지는 한 장 그대로", () => {
  const tiles = planTiles(800, 600, 960, 0.25);
  assert.equal(tiles.length, 1);
  assert.deepEqual(tiles[0], { x: 0, y: 0, w: 800, h: 600 });
});

test("planTiles: 큰 이미지는 겹치며 전체를 덮는다", () => {
  const W = 3840, H = 2160, T = 960;
  const tiles = planTiles(W, H, T, 0.25);
  assert.ok(tiles.length > 1);
  for (const t of tiles) {
    assert.ok(t.x >= 0 && t.y >= 0 && t.x + t.w <= W && t.y + t.h <= H, JSON.stringify(t));
    assert.equal(t.w, T); assert.equal(t.h, T);
  }
  // 오른쪽·아래 끝이 덮인다
  assert.ok(tiles.some((t) => t.x + t.w === W));
  assert.ok(tiles.some((t) => t.y + t.h === H));
  // 겹침: 이웃 타일과 25% 겹친다 (stride = 720)
  const xs = [...new Set(tiles.map((t) => t.x))].sort((a, b) => a - b);
  assert.ok(xs[1] - xs[0] <= T * 0.75 + 1e-9);
});

test("planTiles: 한 변만 타일보다 크면 그 방향으로만 자른다", () => {
  const tiles = planTiles(2000, 700, 960, 0.25);
  assert.ok(tiles.every((t) => t.y === 0 && t.h === 700));
  assert.ok(tiles.length >= 2);
});

// ── iou / mergeDetections ────────────────────────────────────
test("iou: 동일 박스는 1, 떨어진 박스는 0", () => {
  const a = { x1: 0, y1: 0, x2: 10, y2: 10 };
  assert.equal(iou(a, a), 1);
  assert.equal(iou(a, { x1: 20, y1: 20, x2: 30, y2: 30 }), 0);
});

test("mergeDetections: 겹치는 같은 클래스는 conf 높은 쪽만 남는다", () => {
  const dets = [
    { x1: 0, y1: 0, x2: 100, y2: 100, conf: 0.9, cls: 0 },
    { x1: 5, y1: 5, x2: 105, y2: 105, conf: 0.7, cls: 0 },   // 위와 중복
    { x1: 5, y1: 5, x2: 105, y2: 105, conf: 0.8, cls: 1 },   // 클래스 다름 -> 유지
    { x1: 300, y1: 300, x2: 400, y2: 400, conf: 0.6, cls: 0 }, // 떨어짐 -> 유지
  ];
  const out = mergeDetections(dets, 0.5);
  assert.equal(out.length, 3);
  assert.ok(out.some((d) => d.conf === 0.9));
  assert.ok(!out.some((d) => d.conf === 0.7));
});

// ── parseOutput ──────────────────────────────────────────────
test("parseOutput: conf 미달은 거르고 좌표를 원본으로 되돌린다", () => {
  // [1,300,6] 흉내: 두 행만 채운다
  const data = new Float32Array(300 * 6);
  data.set([100, 100, 200, 200, 0.9, 0], 0);   // lap_splice, conf .9
  data.set([10, 10, 20, 20, 0.05, 1], 6);      // conf 낮음 -> 걸러짐
  const lb = letterbox(1080, 1080, 640);
  const dets = parseOutput(data, 300, 0.25, lb, 0, 0);
  assert.equal(dets.length, 1);
  assert.equal(dets[0].cls, 0);
  const s = 640 / 1080;
  assert.ok(Math.abs(dets[0].x1 - 100 / s) < 1e-3);
});

test("parseOutput: 박스를 원본 경계 안으로 자른다", () => {
  const data = new Float32Array(300 * 6);
  data.set([-5, -5, 700, 700, 0.9, 0], 0);     // 640 밖으로 나간 박스
  const lb = letterbox(640, 640, 640);
  const dets = parseOutput(data, 300, 0.25, lb, 0, 0, 640, 640);
  assert.equal(dets[0].x1, 0);
  assert.equal(dets[0].x2, 640);
});
