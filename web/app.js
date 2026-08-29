// UI 배선 + 추론 오케스트레이션. 좌표 수학은 전부 infer.js (테스트됨).
import { letterbox, planTiles, mergeDetections, parseOutput } from "./infer.js";

const MODEL_URL = "./models/best.onnx";
const INPUT = 640;          // export_onnx.py 의 imgsz 와 같아야 한다
const ROWS = 300;           // YOLO26 end2end 출력 [1,300,6]
const TILE = 960;           // 학습 타일이 720~1080px — 그 가운데
const OVERLAP = 0.25;
const FLOOR = 0.05;         // 추론은 이 밑까지 받아 두고, 슬라이더는 그리기만 거른다
const CLASSES = [
  { name: "겹침이음", color: "#ff7a1a" },   // 0 lap_splice
  { name: "철근",     color: "#4aa3ff" },   // 1 rebar
];

const $ = (id) => document.getElementById(id);
const statusEl = $("status");

// ── 모델 ─────────────────────────────────────────────────────
// wasmPaths 를 상대경로로 주면 ort 스크립트 위치 기준으로 풀려
// /vendor/ort/vendor/ort/ 처럼 중복된다(실측 404). 페이지 기준 절대 URL 로 준다.
ort.env.wasm.wasmPaths = new URL("vendor/ort/", document.baseURI).href;
// 멀티스레드는 교차출처 격리(COOP/COEP)가 있어야만 돈다. 없으면 1로 — 경고만 시끄럽다.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let session = null;
const ready = (async () => {
  try {
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
    statusEl.textContent = "모델 준비됨";
  } catch (e) {
    statusEl.textContent = "모델 로딩 실패 — models/best.onnx 확인";
    console.error(e);
    throw e;
  }
})();

// ── 상태 ─────────────────────────────────────────────────────
let bitmap = null;      // 원본 이미지
let detections = [];    // 병합 끝난 전체 검출 (FLOOR 기준)

// ── 업로드 ───────────────────────────────────────────────────
const drop = $("drop");
$("file").addEventListener("change", (e) => e.target.files[0] && handleFile(e.target.files[0]));
for (const ev of ["dragover", "dragleave", "drop"]) {
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.toggle("armed", ev === "dragover");
    if (ev === "drop" && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
}

async function handleFile(file) {
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    statusEl.textContent = "이미지를 읽지 못했다";
    return;
  }
  $("stage").style.display = "block";
  $("controls").style.display = "flex";
  const canvas = $("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d").drawImage(bitmap, 0, 0);
  await runDetection();
}

// ── 추론 ─────────────────────────────────────────────────────
const tileCanvas = document.createElement("canvas");
tileCanvas.width = tileCanvas.height = INPUT;
const tileCtx = tileCanvas.getContext("2d", { willReadFrequently: true });

function tileToTensor(x, y, w, h) {
  const lb = letterbox(w, h, INPUT);
  tileCtx.fillStyle = "#727272";                    // Ultralytics 패딩 회색(114)
  tileCtx.fillRect(0, 0, INPUT, INPUT);
  tileCtx.drawImage(bitmap, x, y, w, h, lb.dx, lb.dy, lb.drawW, lb.drawH);
  const { data } = tileCtx.getImageData(0, 0, INPUT, INPUT);
  const n = INPUT * INPUT;
  const chw = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {                     // RGBA -> RGB, 0~1, CHW
    chw[i] = data[i * 4] / 255;
    chw[n + i] = data[i * 4 + 1] / 255;
    chw[2 * n + i] = data[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", chw, [1, 3, INPUT, INPUT]), lb };
}

async function runDetection() {
  await ready;
  const t0 = performance.now();
  const tiles = planTiles(bitmap.width, bitmap.height, TILE, OVERLAP);
  const prog = $("progress");
  prog.style.display = "flex";

  const all = [];
  for (let i = 0; i < tiles.length; i++) {
    const t = tiles[i];
    $("progressText").textContent = `검출 중 ${i + 1}/${tiles.length}`;
    $("bar").firstElementChild.style.width = `${((i + 1) / tiles.length) * 100}%`;
    await new Promise((r) => setTimeout(r));       // 진행 표시가 그려질 틈
    const { tensor, lb } = tileToTensor(t.x, t.y, t.w, t.h);
    const out = await session.run({ images: tensor });
    all.push(...parseOutput(out.output0.data, ROWS, FLOOR, lb, t.x, t.y, bitmap.width, bitmap.height));
  }
  detections = mergeDetections(all, 0.5);
  window.__dets = detections;   // 디버그·검증용 (콘솔에서 분포 확인)
  prog.style.display = "none";
  const ms = Math.round(performance.now() - t0);
  statusEl.textContent = `${tiles.length}개 타일 · ${ms}ms`;
  draw();
}

// ── 그리기 ───────────────────────────────────────────────────
function draw() {
  if (!bitmap) return;
  const conf = parseFloat($("conf").value);
  const showRebar = $("showRebar").checked;
  const ctx = $("canvas").getContext("2d");
  ctx.drawImage(bitmap, 0, 0);

  const lw = Math.max(2, Math.round(bitmap.width / 400));
  ctx.font = `${Math.max(12, lw * 6)}px "SF Mono", Consolas, monospace`;
  ctx.textBaseline = "top";

  const counts = [0, 0];
  // rebar 를 먼저 그려서 겹침이음 박스가 항상 위에 오게 한다
  for (const pass of [1, 0]) {
    for (const d of detections) {
      if (d.cls !== pass || d.conf < conf) continue;
      if (d.cls === 1 && !showRebar) { counts[1]++; continue; }
      counts[d.cls]++;
      const c = CLASSES[d.cls];
      ctx.strokeStyle = c.color;
      ctx.lineWidth = d.cls === 0 ? lw : Math.max(1, lw - 1);
      ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
      if (d.cls === 0) {
        const label = `${c.name} ${Math.round(d.conf * 100)}%`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = c.color;
        ctx.fillRect(d.x1, Math.max(0, d.y1 - lw * 9), tw + lw * 4, lw * 9);
        ctx.fillStyle = "#14161a";
        ctx.fillText(label, d.x1 + lw * 2, Math.max(0, d.y1 - lw * 8));
      }
    }
  }
  $("counts").innerHTML = `겹침이음 <b>${counts[0]}</b> · 철근 <span>${counts[1]}</span>`;
}

$("conf").addEventListener("input", () => { $("confVal").textContent = $("conf").value; draw(); });
$("showRebar").addEventListener("change", draw);
