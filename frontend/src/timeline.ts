import { S, CAT_COLOR, InsertFit, Phrase, applyChanges, barBeatAt, beginInteraction, call, clipDuration, clipTrim, clips, endInteraction, formatTime, isDoublePress, isTrimmed, markGenerating, phraseOf, refresh, selectedClip, toast, warnLockedDrag, Clip, Join } from "./store";
import { openFormation } from "./cast";
import { FIT_LABELS, openInsertMenu, openPhraseMenu, type MenuAction } from "./insertMenu";
import { poseAt } from "./stage";
import { seek, togglePlay } from "./transport";

const L = 124;
const Y = { ov: 50, rul: 102, sec: 126, flow: 158, wav: 190, ph: 266, pos: 372 };
const H = { ov: 52, rul: 24, sec: 32, flow: 32, wav: 76, ph: 106, pos: 42 };
const LOCK_BUTTON_MIN_WIDTH = 50;
const lockIcon = (closed: boolean) => `<svg width="11" height="13" viewBox="0 0 13 15"><rect y="6" width="13" height="9" rx="2" fill="currentColor"/><path d="${closed ? "M3 6V4a3.500 3.500 0 017 0v2" : "M3 6V4a3.500 3.500 0 016.700-1.400"}" stroke="currentColor" stroke-width="2" fill="none"/></svg>`;
const lanes = () => document.getElementById("lanes")!;

const areaWidth = () => Math.max(200, lanes().clientWidth - L - 16);

function viewRange(): [number, number] {
  const song = S.song!;
  const bars = [...song.bars, song.duration];
  const first = Math.max(0, Math.min(S.viewStart, bars.length - 2));
  const last = Math.min(first + S.viewBars, bars.length - 1);
  return [bars[first], bars[last]];
}
const X = (t: number) => { const [a, b] = viewRange(); return L + ((t - a) / (b - a)) * areaWidth(); };
const T = (px: number) => { const [a, b] = viewRange(); return a + ((px - L) / areaWidth()) * (b - a); };

export function snapTime(t: number, free: boolean): number {
  const song = S.song!;
  if (free || S.snap === "free") { return Math.round(t * 1000) / 1000; }
  const grid = S.snap === "bar" ? song.bars : song.beats;
  return grid.reduce((best, value) => (Math.abs(value - t) < Math.abs(best - t) ? value : best), grid[0]);
}

const lane = (top: number, height: number, label: string, content: string, extra = "") =>
  `<div class="lane" style="top:${top}px;height:${height}px;${extra}border-bottom:1px solid var(--line)">${label ? `<div class="lab">${label}</div>` : ""}${content}</div>`;

function overview(): string {
  const song = S.song!;
  const width = areaWidth();
  const OX = (t: number) => (t / song.duration) * width;
  let out = `<svg id="ovSvg" style="position:absolute;left:${L}px;top:0;z-index:5;cursor:pointer" width="${width + 12}" height="52">`;
  const step = song.duration > 240 ? 60 : 30;
  for (let t = 0; t <= song.duration; t += step) {
    const x = OX(t);
    out += `<line x1="${x}" x2="${x}" y1="10" y2="19" stroke="#8b8e9a"/><text x="${x + 4}" y="13" font-family="Bahnschrift" font-weight="600" font-size="13.500" fill="#b4b6c0">${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}</text>`;
  }
  song.sections.forEach((section, index) => {
    const x = OX(section.start), w = OX(section.end) - x;
    out += `<rect x="${x + 0.5}" y="19" width="${Math.max(1, w - 1)}" height="28" rx="2" fill="${section.kind === "chorus" ? "#7a3d4b" : index % 2 ? "#3a3e4b" : "#31353f"}"/>`;
    if (w > 34) { out += `<text x="${x + w / 2}" y="38" text-anchor="middle" font-family="Yu Gothic UI" font-weight="700" font-size="13.500" fill="#f0eee8">${section.label}</text>`; }
  });
  const [a, b] = viewRange();
  out += `<rect id="ovWindow" x="${OX(a)}" y="18" width="${OX(b) - OX(a)}" height="30" rx="3" fill="#ff8a70" fill-opacity=".14" stroke="#ff8a70" stroke-width="2" style="cursor:grab"/>`;
  out += `<g id="ovKnob"><line x1="0" x2="0" y1="16" y2="50" stroke="#fff" stroke-width="2"/><circle cx="0" cy="33" r="9" fill="#fff" stroke="#ff8a70" stroke-width="3"/></g></svg>`;
  return out;
}

function phrases(): string {
  const composed = S.composed;
  const list = clips();
  const joins = composed?.joins ?? [];
  let out = "";
  const [a, b] = viewRange();
  list.forEach((clip) => {
    if (clip.end < a || clip.start > b) { return; }
    const item = phraseOf(clip.phraseId);
    if (!item) { return; }
    const left = X(clip.start), width = X(clip.end) - left;
    const selected = clip.id === S.selected;
    const speed = clipDuration(clip, item) / (clip.end - clip.start);
    const trimmed = isTrimmed(clip, item);
    const report = composed?.clips[clip.id];
    let inner = "";
    if (selected && report) {
      const points = report.curve.map((v, i) => `${(84 + (i / (report.curve.length - 1)) * Math.max(10, width - 104)).toFixed(1)},${(32 - ((v - 0.6) / 1.1) * 30).toFixed(1)}`).join(" ");
      inner = width > 170 ? `<span class="curve-label">局所速度</span><svg style="position:absolute;left:0;top:60px" width="${width}" height="34"><polyline points="${points}" fill="none" stroke="#14151a" stroke-width="2.200" stroke-linejoin="round"/></svg>` : "";
    }
    const lockButton = width < LOCK_BUTTON_MIN_WIDTH ? "" : `<button class="clip-lock ${clip.locked ? "on" : ""}" data-lock="${clip.id}" title="${clip.locked ? "ロックを解除（L）" : "ロックする（L）"}">${lockIcon(clip.locked)}</button>`;
    const handleTitle = "ドラッグで伸縮 / Shift+ドラッグで途中カット";
    out += `<div class="bk ph ${selected ? "sel" : ""} ${clip.locked ? "locked" : ""} ${trimmed ? "trimmed" : ""}" data-clip="${clip.id}" style="left:${left}px;width:${width}px;background:${CAT_COLOR[item.category] ?? "var(--groove)"}">
      ${selected && !clip.locked ? `<i class="hd" style="left:3px" title="${handleTitle}"></i><i class="hd" style="right:3px" title="${handleTitle}"></i>` : ""}${lockButton}
      <div class="a">${escapeHtml(item.name)}</div><div class="b">×${speed.toFixed(2)}${trimmed ? " ・ カット" : ""}${selected ? ` ・ ${barsOf(clip).toFixed(2)}小節` : ""}</div>${inner}</div>`;
  });
  for (const join of joins) {
    if (join.end < a || join.start > b || join.status === "overlap" || join.status === "cut") { continue; }
    const left = X(join.start), width = X(join.end) - left;
    if (width < 1) { continue; }
    const warn = join.status === "fast" || join.status === "short" || join.status === "engine-error";
    const generating = join.status === "pending";
    const title = { ok: "MotionBricks 接続", fast: "移動が速め", short: "短すぎるため生成していません", "engine-error": "MotionBricks に接続できません", pending: "MotionBricks で生成しています" }[join.status] ?? "";
    out += `<svg class="conn-b ${warn ? "w" : ""} ${generating ? "gen" : ""}" data-join="${join.from}>${join.to}" style="left:${left}px;width:${width}px" width="${width}" height="78"><title>${title} ${join.distance} m / ${join.seconds.toFixed(2)} 秒 — ドラッグで曲に合うモーションを自動複数割り当て</title><rect width="${width}" height="78" fill="url(#brickP)"/></svg>`;
    if (generating) { out += `<div class="conn-gen" style="left:${left + width / 2}px">生成中</div>`; }
  }
  return out;
}

const barsOf = (clip: Clip) => (clip.end - clip.start) / (S.song?.barSeconds ?? 1);
const escapeHtml = (text: string) => text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));

function accentsOverlay(): string {
  const song = S.song!;
  const composed = S.composed;
  const [a, b] = viewRange();
  const yS = Y.wav + 13, yM = Y.ph + 5 + 11;
  const used = new Set<number>();
  const dia = (x: number, y: number, r: number, fill: string, stroke: string, extra = "") => `<path d="M${x} ${y - r}L${x + r} ${y}L${x} ${y + r}L${x - r} ${y}Z" fill="${fill}" stroke="${stroke}" stroke-width="1.500" ${extra}/>`;
  // A wider, unseen target under a diamond of the selected phrase, so it can be grabbed and put on a sound.
  const grip = (x: number, source: number, pinned: boolean) => `<circle cx="${x}" cy="${yM}" r="11" fill="transparent" data-accent="${source}" style="pointer-events:all;cursor:ew-resize"><title>${pinned ? "手動で合わせたアクセント。ドラッグで移動、ダブルクリックで解除" : "ドラッグして音の ◆ に合わせます（Alt で吸着なし）"}</title></circle>`;
  let body = "";
  for (const beat of song.beats) {
    if (beat < a || beat > b) { continue; }
    const isBar = song.bars.some((bar) => Math.abs(bar - beat) < 0.03);
    body += `<line x1="${X(beat)}" x2="${X(beat)}" y1="${Y.wav + 69}" y2="${Y.wav + 75}" stroke="#fff" stroke-opacity="${isBar ? 0.8 : 0.42}" stroke-width="${isBar ? 2 : 1}"/>`;
  }
  let marks = "";
  if (composed) {
    for (const clip of clips()) {
      const report = composed.clips[clip.id];
      if (!report || clip.end < a || clip.start > b) { continue; }
      const selected = clip.id === S.selected;
      let lines = "", diamonds = "";
      for (const accent of report.accents) {
        if (accent.t < a || accent.t > b) { continue; }
        const x = X(accent.t);
        if (accent.matched && accent.sound !== null) {
          used.add(accent.sound);
          lines += `<line x1="${X(accent.sound)}" x2="${x}" y1="${yS + 6}" y2="${yM - 6}" stroke="#fff" stroke-opacity="${selected ? 0.95 : 0.24}" stroke-width="${selected ? 1.6 : 1}"/>`;
        } else if (selected && accent.sound !== null) {
          lines += `<line x1="${X(accent.sound)}" x2="${x}" y1="${yS + 6}" y2="${yM - 6}" stroke="#f3bb55" stroke-width="1.800" stroke-dasharray="4 4"/>`;
        }
        const missed = selected && !accent.matched;
        if (selected && !clip.locked) { diamonds += grip(x, accent.source, accent.pinned); }
        diamonds += dia(x, yM, selected ? 6.5 : 5.5, accent.pinned ? "#ff8a70" : missed ? "#f3bb55" : selected ? "#fff" : "#14151a", selected || accent.pinned ? "#14151a" : "#ffffff");
      }
      body += `<g data-accents="${clip.id}">${lines}</g>`;
      marks += `<g data-accents="${clip.id}">${diamonds}</g>`;
    }
  }
  for (const accent of song.accents) {
    if (accent.t < a || accent.t > b) { continue; }
    const on = used.has(accent.t) || accent.kime;
    body += dia(X(accent.t), yS, on ? 7 : 5.5, "#fff", "#121318", `opacity="${on ? 1 : 0.5}"`);
  }
  return `<svg style="position:absolute;left:0;top:0;pointer-events:none;z-index:5" width="${lanes().clientWidth}" height="414">${body}${marks}</svg>`;
}

function grid(): string {
  const song = S.song!;
  const [a, b] = viewRange();
  const showBeats = (b - a) / areaWidth() < 0.06;
  let out = `<svg style="position:absolute;left:0;top:${Y.rul}px;pointer-events:none;z-index:1" width="${lanes().clientWidth}" height="312">`;
  song.bars.forEach((bar, index) => {
    if (bar >= a && bar <= b) { out += `<line x1="${X(bar)}" x2="${X(bar)}" y1="0" y2="312" stroke="#fff" stroke-opacity="${index % 2 === 0 ? 0.2 : 0.11}"/>`; }
  });
  if (showBeats) { for (const beat of song.beats) { if (beat >= a && beat <= b) { out += `<line x1="${X(beat)}" x2="${X(beat)}" y1="0" y2="312" stroke="#fff" stroke-opacity=".04"/>`; } } }
  return out + "</svg>";
}

export function renderTimeline() {
  const song = S.song;
  const host = lanes();
  document.getElementById("tlEmpty")!.hidden = !!song && clips().length > 0;
  if (!song) { host.innerHTML = ""; return; }
  const [a, b] = viewRange();
  const list = clips();
  const rulerMarks = song.bars.map((bar, index) => (bar >= a && bar < b ? `<span style="left:${X(bar)}px">${index + 1}</span>` : "")).join("");
  const sections = song.sections.filter((s) => s.end > a && s.start < b).map((s) => {
    const left = X(Math.max(s.start, a)), width = X(Math.min(s.end, b)) - left;
    return `<div class="bk sec" style="top:2px;left:${left + 1}px;width:${Math.max(2, width - 3)}px;${s.kind === "chorus" ? "background:#57333b;color:#ffc9c0" : ""}">${s.label}</div>`;
  }).join("");
  const flow = song.sections.filter((s) => s.end > a && s.start < b).map((s) => {
    const left = X(Math.max(s.start, a)), width = X(Math.min(s.end, b)) - left;
    return `<div class="bk flow" style="top:2px;left:${left + 2}px;width:${Math.max(2, width - 5)}px">${escapeHtml(S.project!.flow[s.kind]?.text ?? "")}</div>`;
  }).join("");
  const positions = list.map((clip, index) => {
    if (clip.end < a || clip.start > b) { return ""; }
    return `<div class="bk pos ${clip.id === S.selected ? "sel" : ""}" data-pos="${clip.id}" style="top:6px;left:${X(clip.start) + 4}px"><b>${index + 1}</b>${clip.x.toFixed(1)}, ${clip.z.toFixed(1)}</div>`;
  }).join("");
  const formations = S.project?.formations ?? [];
  const flags = formations.map((formation, index) => (formation.time >= a - 1e-3 && formation.time < b
    ? `<div class="form-flag" data-formation="${formation.time}" style="left:${X(formation.time)}px" title="フォーメーション。クリックで編集">${escapeHtml(formation.name || `隊形 ${index + 1}`)}</div>` : "")).join("");
  host.innerHTML =
    lane(Y.ov, H.ov, "曲全体", overview()) +
    lane(Y.rul, H.rul, "", `<div class="rul">${rulerMarks}</div>${flags}`) +
    lane(Y.sec, H.sec, "曲の構成", sections) +
    lane(Y.flow, H.flow, "流れの辞書", flow) +
    lane(Y.wav, H.wav, "音・アクセント", `<canvas id="waveCanvas" style="position:absolute;left:${L}px;top:0" width="${areaWidth() * 2}" height="${H.wav * 2}"></canvas>`) +
    lane(Y.ph, H.ph, "フレーズ", phrases(), "background:#16181e;") +
    lane(Y.pos, H.pos, "ダンス位置", positions) +
    grid() + accentsOverlay() +
    `<svg id="playhead" style="position:absolute;top:${Y.rul}px;z-index:6;pointer-events:none" width="18" height="312"><line x1="9" x2="9" y1="0" y2="312" stroke="transparent" stroke-width="12" style="pointer-events:stroke;cursor:ew-resize"/><path d="M1 0h16v9l-8 9-8-9z" fill="#fff" style="pointer-events:all;cursor:ew-resize"/><line x1="9" x2="9" y1="14" y2="312" stroke="#fff" stroke-width="2"/></svg>`;
  drawWave();
  document.getElementById("viewLabel")!.textContent = `${S.viewStart + 1}–${Math.min(S.viewStart + S.viewBars, song.bars.length)}`;
  updatePlayhead();
}

function drawWave() {
  const song = S.song!;
  const canvas = document.getElementById("waveCanvas") as HTMLCanvasElement | null;
  if (!canvas) { return; }
  canvas.style.width = `${areaWidth()}px`;
  canvas.style.height = `${H.wav}px`;
  const c = canvas.getContext("2d")!;
  const [a, b] = viewRange();
  const width = canvas.width, height = canvas.height;
  const toIndex = (t: number) => (t / song.duration) * song.wave.length;
  const columns = Math.floor(width / 6);
  const middle = height * 0.59;
  for (let i = 0; i < columns; i++) {
    const t0 = a + (i / columns) * (b - a), t1 = a + ((i + 1) / columns) * (b - a);
    let peak = 0;
    for (let k = Math.floor(toIndex(t0)); k <= Math.ceil(toIndex(t1)) && k < song.wave.length; k++) { peak = Math.max(peak, song.wave[k] ?? 0); }
    const h = Math.max(2, peak * height * 0.38);
    c.fillStyle = t0 < S.playhead ? "#aeb4c6" : "#666c7c";
    c.fillRect((i / columns) * width, middle - h, 4, h * 2);
  }
  c.strokeStyle = "#ff8a70"; c.lineWidth = 5; c.lineJoin = "round"; c.beginPath();
  const spans = [...song.bars, song.duration];
  let begun = false;
  song.energyPerBar.forEach((level, index) => {
    const t = (spans[index] + spans[index + 1]) / 2;
    if (t < a - 3 || t > b + 3) { return; }
    const x = ((t - a) / (b - a)) * width, y = height * 0.94 - level * height * 0.62;
    if (begun) { c.lineTo(x, y); } else { c.moveTo(x, y); begun = true; }
  });
  c.stroke();
}

export function updatePlayhead() {
  const song = S.song;
  if (!song) { return; }
  const head = document.getElementById("playhead") as unknown as SVGSVGElement | null;
  const [a, b] = viewRange();
  if (head) {
    const visible = S.playhead >= a && S.playhead <= b;
    head.style.display = visible ? "block" : "none";
    head.style.left = `${X(S.playhead) - 9}px`;
  }
  const knob = document.getElementById("ovKnob");
  if (knob) { knob.setAttribute("transform", `translate(${(S.playhead / song.duration) * areaWidth()},0)`); }
  poseAt(S.playhead);
}

export function followPlayhead() {
  const song = S.song;
  if (!song) { return; }
  const [a, b] = viewRange();
  if (S.playhead >= b || S.playhead < a) {
    let index = 0;
    song.bars.forEach((bar, i) => { if (bar <= S.playhead) { index = i; } });
    S.viewStart = Math.max(0, Math.min(index, song.bars.length - S.viewBars));
    renderTimeline();
  }
}

async function patchClip(id: string, changes: Record<string, unknown>) {
  try { await applyChanges(await call("PATCH", `/api/clips/${id}`, changes)); }
  catch (error) { toast((error as Error).message, "error"); await applyChanges(await call("GET", "/api/state")); }
}

export function toggleLock(id: string) {
  const clip = clips().find((c) => c.id === id);
  if (clip) { void patchClip(id, { locked: !clip.locked }); }
}

const EDGE_PIXELS = 10;

/** Which part of a phrase block the pointer is on: its edges stretch it, the rest moves it. */
function clipZone(element: HTMLElement, clientX: number): "left" | "right" | "move" {
  const rect = element.getBoundingClientRect();
  return clientX - rect.left < EDGE_PIXELS ? "left" : rect.right - clientX < EDGE_PIXELS ? "right" : "move";
}

/** Keeps one cursor for the whole drag, even when the pointer leaves the element that started it. */
export function holdCursor(name: "drag-move" | "drag-resize" | null) {
  document.body.classList.remove("drag-move", "drag-resize");
  if (name) { document.body.classList.add(name); }
}

function startClipDrag(event: MouseEvent, element: HTMLElement) {
  const clip = clips().find((c) => c.id === element.dataset.clip)!;
  if (clip.locked) {
    S.selected = clip.id;
    refresh();
    warnLockedDrag(event);
    return;
  }
  const item = phraseOf(clip.phraseId)!;
  const mode = clipZone(element, event.clientX);
  S.selected = clip.id;
  const original = { start: clip.start, end: clip.end, trimStart: clip.trimStart, trimEnd: clip.trimEnd };
  const [origTrimStart, origTrimEnd] = clipTrim(clip, item);
  const origSpeed = (origTrimEnd - origTrimStart) / (original.end - original.start);
  const origin = event.clientX;
  const others = clips().filter((c) => c.id !== clip.id);
  const previous = others.filter((c) => c.start < original.start).reduce<Clip | null>((best, c) => (!best || c.start > best.start ? c : best), null);
  const following = others.filter((c) => c.start > original.start).reduce<Clip | null>((best, c) => (!best || c.start < best.start ? c : best), null);
  const lowest = previous ? previous.end + 0.02 : 0;
  const highest = following ? following.start - 0.02 : S.song!.duration;
  const entering = S.composed?.joins.find((join) => join.to === clip.id);
  const leaving = S.composed?.joins.find((join) => join.from === clip.id);
  const joinTimes = { end: entering?.end, start: leaving?.start };
  const inLanes = <T extends Element = HTMLElement>(selector: string) => lanes().querySelector<T>(selector);  // the timeline is redrawn after the press, so old references go stale
  let moved = false;
  const placeJoin = (join: Join) => {
    const brick = inLanes<SVGSVGElement>(`[data-join="${join.from}>${join.to}"]`);
    if (!brick) { return; }
    const left = X(join.start), width = Math.max(0, X(join.end) - left);
    brick.style.left = `${left}px`;
    brick.style.width = `${width}px`;
    brick.setAttribute("width", String(width));
    brick.querySelector("rect")?.setAttribute("width", String(width));
  };
  const paint = (trimming = false) => {
    const block = inLanes(`[data-clip="${clip.id}"]`);
    if (block) {
      block.style.left = `${X(clip.start)}px`;
      block.style.width = `${X(clip.end) - X(clip.start)}px`;
      const label = block.querySelector(".b");
      const [t0, t1] = clipTrim(clip, item);
      const speed = (t1 - t0) / (clip.end - clip.start);
      if (label) {
        label.textContent = trimming
          ? `カット ${t0.toFixed(2)}–${t1.toFixed(2)}秒 ・ ×${speed.toFixed(2)}`
          : `×${speed.toFixed(2)}${isTrimmed(clip, item) ? " ・ カット" : ""} ・ ${barsOf(clip).toFixed(2)}小節`;
      }
    }
    if (entering) { entering.end = clip.start; placeJoin(entering); }
    if (leaving) { leaving.start = clip.end; placeJoin(leaving); }
    const chip = inLanes(`[data-pos="${clip.id}"]`);
    if (chip) { chip.style.left = `${X(clip.start) + 4}px`; }
    lanes().querySelectorAll<SVGGElement>(`[data-accents="${clip.id}"]`).forEach((group) => (group.style.display = "none"));  // they are redrawn from the new timing
  };
  const onMove = (e: MouseEvent) => {
    const [a, b] = viewRange();
    const delta = ((e.clientX - origin) / areaWidth()) * (b - a);
    const nearOrigin = Math.abs(e.clientX - origin) <= 2;
    if (!nearOrigin) { moved = true; }
    if (!moved) { return; }
    holdCursor(mode === "move" ? "drag-move" : "drag-resize");
    let trimming = false;
    if (nearOrigin) {
      clip.start = original.start; clip.end = original.end;
      clip.trimStart = original.trimStart; clip.trimEnd = original.trimEnd;
    } else if (mode === "move") {
      const length = original.end - original.start;
      let start = snapTime(original.start + delta, e.altKey);
      start = Math.max(lowest, Math.min(highest - length, start));
      clip.start = start; clip.end = start + length;
    } else if (mode === "left" && e.shiftKey) {
      trimming = true;
      const minStart = Math.max(lowest, original.start - origTrimStart / origSpeed);
      const maxStart = Math.min(original.end - 0.3, original.start + (origTrimEnd - 0.3 - origTrimStart) / origSpeed);
      clip.start = Math.max(minStart, Math.min(maxStart, snapTime(original.start + delta, e.altKey)));
      clip.end = original.end;
      clip.trimStart = round(Math.max(0, origTrimStart + (clip.start - original.start) * origSpeed));
      clip.trimEnd = round(origTrimEnd);
    } else if (mode === "right" && e.shiftKey) {
      trimming = true;
      const minEnd = Math.max(original.start + 0.3, original.end - (origTrimEnd - origTrimStart - 0.3) / origSpeed);
      const maxEnd = Math.min(highest, original.end + (item.duration - origTrimEnd) / origSpeed);
      clip.end = Math.max(minEnd, Math.min(maxEnd, snapTime(original.end + delta, e.altKey)));
      clip.start = original.start;
      clip.trimStart = round(origTrimStart);
      clip.trimEnd = round(Math.min(item.duration, origTrimEnd + (clip.end - original.end) * origSpeed));
    } else if (mode === "left") {
      clip.trimStart = original.trimStart; clip.trimEnd = original.trimEnd;
      clip.start = Math.max(lowest, Math.min(original.end - 0.3, snapTime(original.start + delta, e.altKey)));
      clip.end = original.end;
    } else {
      clip.trimStart = original.trimStart; clip.trimEnd = original.trimEnd;
      clip.end = Math.min(highest, Math.max(original.start + 0.3, snapTime(original.end + delta, e.altKey)));
      clip.start = original.start;
    }
    paint(trimming);
  };
  const onUp = async () => {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    holdCursor(null);
    const unchanged = clip.start === original.start && clip.end === original.end && clip.trimStart === original.trimStart && clip.trimEnd === original.trimEnd;
    if (!moved || unchanged) {
      clip.start = original.start; clip.end = original.end;
      clip.trimStart = original.trimStart; clip.trimEnd = original.trimEnd;
      if (entering) { entering.end = joinTimes.end!; }
      if (leaving) { leaving.start = joinTimes.start!; }
      endInteraction();
      refresh();
      return;
    }
    endInteraction(true);
    markGenerating(clip.id);
    const report = S.composed?.clips[clip.id];
    if (report) { report.accents = []; }
    const trimChanged = clip.trimStart !== original.trimStart || clip.trimEnd !== original.trimEnd;
    refresh();
    await patchClip(clip.id, {
      start: round(clip.start),
      end: round(clip.end),
      speedMode: "manual",
      ...(trimChanged ? { trimStart: clip.trimStart ?? null, trimEnd: clip.trimEnd ?? null } : {}),
    });
  };
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  beginInteraction();
  refresh();
}

const round = (value: number) => Math.round(value * 1000) / 1000;

/** Drag a motion accent of the selected phrase onto a sound. It stays pinned there; a double click lets it go. */
function startAccentDrag(event: MouseEvent, mark: HTMLElement) {
  const clip = selectedClip();
  const source = Number(mark.dataset.accent);
  if (!clip || clip.locked || !Number.isFinite(source)) { return; }
  const song = S.song!;
  const pins = clip.pins ?? [];
  const mine = pins.find((pin) => Math.abs(pin.source - source) < 1e-3);
  const others = pins.filter((pin) => pin !== mine);
  if (isDoublePress(`accent:${clip.id}:${source}`)) {
    if (mine) { void patchClip(clip.id, { pins: others }); }
    return;
  }
  // It has to stay between the pins on either side, and inside the phrase.
  const lower = Math.max(clip.start, ...others.filter((pin) => pin.source < source).map((pin) => clip.start + pin.at)) + 0.04;
  const upper = Math.min(clip.end, ...others.filter((pin) => pin.source > source).map((pin) => clip.start + pin.at)) - 0.04;
  const origin = event.clientX;
  let time: number | null = null;
  let ghost: HTMLElement | null = null;
  const onMove = (e: MouseEvent) => {
    if (time === null && Math.abs(e.clientX - origin) < 3) { return; }
    const host = lanes();
    let t = T(e.clientX - host.getBoundingClientRect().left);
    let onSound = false;
    if (!e.altKey && song.accents.length) {
      const near = song.accents.reduce((best, accent) => (Math.abs(accent.t - t) < Math.abs(best - t) ? accent.t : best), song.accents[0].t);
      if (Math.abs(X(near) - X(t)) <= 12 && near > lower && near < upper) { t = near; onSound = true; }
    }
    time = Math.max(lower, Math.min(upper, t));
    if (!ghost || !ghost.isConnected) {  // the timeline may have been redrawn under the drag
      ghost = document.createElement("div");
      ghost.className = "accent-ghost";
      ghost.style.top = `${Y.wav + 6}px`;
      ghost.style.height = `${Y.ph + 22 - Y.wav - 6}px`;
      host.appendChild(ghost);
      holdCursor("drag-resize");
    }
    ghost.style.left = `${X(time)}px`;
    ghost.classList.toggle("on-sound", onSound);
    ghost.dataset.label = onSound ? "音に合わせる" : `${(time - clip.start).toFixed(2)} 秒`;
  };
  const onUp = async () => {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    holdCursor(null);
    ghost?.remove();
    if (time === null) { endInteraction(); return; }
    endInteraction(true);
    markGenerating(clip.id);
    refresh();
    await patchClip(clip.id, { pins: [...others, { source, at: round(time - clip.start) }] });
  };
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  beginInteraction();
}

export function initTimeline() {
  const host = lanes();
  new ResizeObserver(() => renderTimeline()).observe(host);
  host.addEventListener("mousedown", (event) => {
    if (!S.song || event.button !== 0) { return; }
    const target = event.target as HTMLElement;
    if (target.closest("#playhead")) { startScrub(); event.preventDefault(); return; }
    const accent = target.closest<HTMLElement>("[data-accent]");
    if (accent) { startAccentDrag(event, accent); event.preventDefault(); return; }
    const flag = target.closest<HTMLElement>("[data-formation]");
    if (flag) { openFormation(Number(flag.dataset.formation)); event.preventDefault(); return; }
    const lockButton = target.closest<HTMLElement>("[data-lock]");
    if (lockButton) { toggleLock(lockButton.dataset.lock!); event.preventDefault(); return; }
    const block = target.closest<HTMLElement>("[data-clip]");
    if (block) { startClipDrag(event, block); event.preventDefault(); return; }
    const chip = target.closest<HTMLElement>("[data-pos]");
    if (chip) {
      if (isDoublePress(chip.dataset.pos!)) { jumpToClip(chip.dataset.pos!); } else { S.selected = chip.dataset.pos!; refresh(); }
      return;
    }
    if (target.closest("#ovSvg")) { startOverview(event); event.preventDefault(); return; }
    const rect = host.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (x < L) { return; }
    if (y >= Y.ph && y <= Y.ph + H.ph && startGapFillDrag(event, x)) {
      event.preventDefault();
      return;
    }
    S.selected = null;
    seek(snapTime(T(x), event.altKey));
    refresh();
    startScrub();
    event.preventDefault();
  });
  host.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    if (!S.song || (event.target as HTMLElement).closest("#ovSvg")) { return; }
    const x = event.clientX - host.getBoundingClientRect().left;
    if (x < L) { return; }
    const t = Math.max(0, Math.min(S.song.duration, T(x)));
    const at = snapTime(t, event.altKey);
    const insWhere = exactInsertLabel(at);
    const picked = selectedClip();
    if (picked) {
      const name = phraseOf(picked.phraseId)?.name ?? "";
      openPhraseMenu(event.clientX, event.clientY, [
        { title: "モーションを差し替え", where: `${clips().indexOf(picked) + 1} 番「${name}」を差し替え`, choose: (phraseId) => replaceClip(picked.id, phraseId), switchLabel: "挿入へ切替", at: picked.start, replaceClip: picked },
        { title: "モーションを挿入", where: insWhere, choose: (phraseId) => insertPhrase(phraseId, at, undefined, true), switchLabel: "差し替えに戻す", at },
      ], clipActions(picked));
      return;
    }
    openInsertMenu(event.clientX, event.clientY, insWhere, (phraseId) => insertPhrase(phraseId, at, undefined, true), at, clipActions());
  });
  host.addEventListener("mousemove", (event) => {
    const target = event.target as HTMLElement;
    const block = target.closest<HTMLElement>("[data-clip]");
    if (block) { block.style.cursor = block.classList.contains("locked") ? "default" : clipZone(block, event.clientX) === "move" ? "grab" : "ew-resize"; }
  });
  host.addEventListener("wheel", (event) => {
    if (!S.song) { return; }
    event.preventDefault();
    if (event.ctrlKey) {
      S.viewBars = Math.max(4, Math.min(S.song.bars.length, S.viewBars + (event.deltaY > 0 ? 4 : -4)));
    } else {
      S.viewStart = Math.max(0, Math.min(S.song.bars.length - S.viewBars, S.viewStart + (event.deltaY > 0 ? 2 : -2)));
    }
    renderTimeline();
  }, { passive: false });
  const clearDropGhost = () => { document.getElementById("dropGhost")?.remove(); };
  window.addEventListener("dragend", clearDropGhost);
  host.addEventListener("dragleave", (event) => {
    if (!host.contains(event.relatedTarget as Node)) { clearDropGhost(); }
  });
  host.addEventListener("dragover", (event) => {
    if (!S.song || (event.target as HTMLElement).closest("#ovSvg")) { clearDropGhost(); return; }
    const x = event.clientX - host.getBoundingClientRect().left;
    if (x < L) { clearDropGhost(); return; }
    event.preventDefault();
    if (event.dataTransfer) { event.dataTransfer.dropEffect = "copy"; }
    const at = Math.max(0, Math.min(S.song.duration - 0.3, snapTime(T(x), event.altKey)));
    const item = S.draggingPhrase ? phraseOf(S.draggingPhrase) : undefined;
    const left = X(at);
    const estLen = previewInsertLength(item, at);
    const end = Math.min(S.song.duration, at + estLen);
    const width = Math.max(18, X(end) - left);
    let ghost = document.getElementById("dropGhost");
    if (!ghost || !ghost.isConnected) {
      ghost = document.createElement("div");
      ghost.id = "dropGhost";
      ghost.className = "drop-ghost";
      ghost.innerHTML = `<div class="drop-line"></div><div class="drop-box"></div><div class="drop-tag"></div>`;
      host.appendChild(ghost);
    }
    const line = ghost.querySelector<HTMLElement>(".drop-line")!;
    const box = ghost.querySelector<HTMLElement>(".drop-box")!;
    const tag = ghost.querySelector<HTMLElement>(".drop-tag")!;
    line.style.left = `${left}px`;
    line.style.top = `${Y.rul}px`;
    line.style.height = `${Y.pos + H.pos - Y.rul}px`;
    box.style.left = `${left}px`;
    box.style.top = `${Y.ph + 14}px`;
    box.style.width = `${width}px`;
    box.style.borderColor = item ? (CAT_COLOR[item.category] ?? "var(--accent)") : "var(--accent)";
    box.textContent = item?.name ?? "";
    const bb = barBeatAt(at);
    tag.style.left = `${Math.min(left + 6, host.clientWidth - 260)}px`;
    tag.style.top = `${Y.ph - 18}px`;
    tag.textContent = `${bb ? `${bb.bar}小節 ${bb.beat}拍` : formatTime(at)}${item ? ` に「${item.name}」（${FIT_LABELS[S.insertFit]}）` : " に挿入"}`;
  });
  host.addEventListener("drop", async (event) => {
    event.preventDefault();
    clearDropGhost();
    const phraseId = event.dataTransfer?.getData("text/phrase") || S.draggingPhrase;
    if (!phraseId || !S.song || (event.target as HTMLElement).closest("#ovSvg")) { return; }
    const x = event.clientX - host.getBoundingClientRect().left;
    if (x < L) { return; }
    const at = Math.max(0, Math.min(S.song.duration - 0.3, snapTime(T(x), event.altKey)));
    await insertPhrase(phraseId, at, undefined, true);
  });
}

/** Drag the white bar. Playback pauses while it moves and resumes where the bar is let go. */
function startScrub() {
  const song = S.song!;
  const host = lanes();
  const wasPlaying = S.playing;
  if (wasPlaying) { togglePlay(); }
  holdCursor("drag-resize");
  const onMove = (event: MouseEvent) => {
    const rect = host.getBoundingClientRect();
    const [a, b] = viewRange();
    const x = Math.max(L, Math.min(L + areaWidth(), event.clientX - rect.left));
    const t = Math.max(a, Math.min(b, T(x)));
    seek(Math.min(song.duration - 0.01, event.shiftKey ? snapTime(t, false) : t));
  };
  const onUp = () => {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    holdCursor(null);
    renderTimeline();
    if (wasPlaying) { togglePlay(); }
  };
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
}

/** Select a phrase, put the white bar on its start, and scroll the timeline so it is in view. */
export function jumpToClip(id: string) {
  const clip = clips().find((c) => c.id === id);
  const song = S.song;
  if (!clip || !song) { return; }
  S.selected = id;
  seek(clip.start);
  const [a, b] = viewRange();
  if (clip.start < a || clip.end > b) {
    let index = 0;
    song.bars.forEach((bar, i) => { if (bar <= clip.start + 1e-3) { index = i; } });
    S.viewStart = Math.max(0, Math.min(index - 2, song.bars.length - S.viewBars));
  }
  refresh();
}

export const clipAt = (t: number) => clips().find((clip) => t >= clip.start && t < clip.end);

/** Says where a phrase requested at this time will go. A phrase is never placed on top of another; it takes the nearer side. */
export function insertLabel(t: number): string {
  const inside = clipAt(t);
  const name = inside && phraseOf(inside.phraseId)?.name;
  if (inside && name) { return `「${name}」の${t - inside.start < (inside.end - inside.start) / 2 ? "前" : "後ろ"}に挿入`; }
  const at = barBeatAt(t);
  return at ? `${at.bar}小節 ${at.beat}拍に挿入` : "ここに挿入";
}

export function exactInsertLabel(t: number): string {
  const at = barBeatAt(t);
  return at ? `${at.bar}小節 ${at.beat}拍（${formatTime(t)}）に挿入` : `${formatTime(t)} に挿入`;
}

function previewInsertLength(item: Phrase | undefined, at: number): number {
  const song = S.song;
  const raw = item?.duration ?? 3.0;
  if (!song || S.insertFit === "raw") { return raw; }
  const barSec = song.barSeconds || 2.0;
  const minJoin = Math.ceil(0.6 / (barSec / 4) - 1e-6) * (barSec / 4);
  if (S.insertFit === "1" || S.insertFit === "2" || S.insertFit === "4") {
    const slot = Number(S.insertFit) * barSec;
    return Math.max(0.35, Math.min(slot - minJoin, raw / 0.55));
  }
  const next = clips().find((c) => c.start > at + 1e-3);
  if (next) {
    const avail = next.start - minJoin - at;
    if (avail >= Math.max(0.4, raw / 1.45)) {
      return Math.min(avail, raw / 0.75);
    }
    if (avail >= Math.max(1.5, barSec * 0.9) && (item?.bars ?? 2) === 2) {
      return avail;
    }
  }
  const nom = item?.bars ?? 2;
  const slot = nom === 1 && barSec - minJoin < raw / 1.55 ? 2 * barSec : nom * barSec;
  return Math.max(0.4, Math.min(slot - minJoin, raw / 0.78));
}

export async function fillGapRange(start: number, end: number) {
  try {
    const data = await call("POST", "/api/clips/fill-gap", { start: round(start), end: round(end) });
    await applyChanges(data);
    if (data.added?.length) {
      S.selected = data.added[0];
      refresh();
    }
    toast(`補間区間に曲に合ったモーションを ${data.count} 個割り当てました（Ctrl+Z で戻せます）`);
  } catch (error) {
    toast((error as Error).message, "error");
  }
}

function estimateGapMotionCount(start: number, end: number, prevClip: Clip | null, nextClip: Clip | null): number {
  const song = S.song;
  if (!song) { return 0; }
  const baseJoin = 0.64;
  const totalDist = prevClip && nextClip ? Math.hypot(nextClip.x - prevClip.x, nextClip.z - prevClip.z) : 0;
  const playable = (count: number) => {
    const stepJoin = Math.max(baseJoin, (prevClip && nextClip ? totalDist / (count + 1) : 0) / 1.4);
    const ear = prevClip ? Math.max(start, prevClip.end + stepJoin) : start;
    const lat = nextClip ? Math.min(end, nextClip.start - stepJoin) : end;
    return { ear, lat, sum: (lat - ear) - (count - 1) * stepJoin };
  };
  if (playable(1).sum < 0.85) { return 0; }
  const p2 = playable(2);
  if (p2.sum < 2.30) { return 1; }
  const barSec = song.barSeconds || 2.0;
  const targetSec = Math.min(2.9, Math.max(2.0, 1.35 * barSec));
  const desired = Math.max(2, Math.round(((p2.lat - p2.ear) + baseJoin) / (targetSec + baseJoin)));
  let count = 2;
  for (let k = 2; k <= desired + 2; k++) {
    const pk = playable(k);
    if (pk.sum / k >= 1.15 && k <= desired) { count = k; }
    else if (pk.sum / k < 1.15) { break; }
  }
  return count;
}

/** Drag across a clip-free interpolation/gap interval in the phrase lane to auto-assign multiple matching motions. */
function startGapFillDrag(event: MouseEvent, startX: number): boolean {
  const song = S.song;
  if (!song) { return false; }
  const list = clips();
  const joinEl = (event.target as HTMLElement).closest<SVGElement>("[data-join]");
  let prevClip: Clip | null = null;
  let nextClip: Clip | null = null;
  if (joinEl?.dataset.join) {
    const [fromId, toId] = joinEl.dataset.join.split(">");
    prevClip = list.find((c) => c.id === fromId) ?? null;
    nextClip = list.find((c) => c.id === toId) ?? null;
  } else {
    const t0 = Math.max(0, Math.min(song.duration, T(startX)));
    if (list.some((c) => t0 > c.start + 0.04 && t0 < c.end - 0.04)) { return false; }
    prevClip = list.filter((c) => c.end <= t0 + 0.05).at(-1) ?? null;
    nextClip = list.find((c) => c.start >= t0 - 0.05) ?? null;
  }
  const gapStart = prevClip ? prevClip.end : 0;
  const gapEnd = nextClip ? nextClip.start : song.duration;
  if (gapEnd - gapStart < 0.2) { return false; }

  const originClientX = event.clientX;
  const originTime = Math.max(gapStart, Math.min(gapEnd, T(startX)));
  let dragging = false;
  let crossedClip = false;
  let selStart = originTime;
  let selEnd = originTime;
  let estCount = 0;

  const onMove = (e: MouseEvent) => {
    if (!dragging && Math.abs(e.clientX - originClientX) < 6) { return; }
    if (!dragging) {
      dragging = true;
      beginInteraction();
      holdCursor("drag-resize");
    }
    const host = lanes();
    const rect = host.getBoundingClientRect();
    const curX = Math.max(L, Math.min(L + areaWidth(), e.clientX - rect.left));
    const curTime = Math.max(0, Math.min(song.duration, T(curX)));
    const rawMin = Math.min(originTime, curTime);
    const rawMax = Math.max(originTime, curTime);

    const overLeft = prevClip !== null && (X(gapStart) - X(rawMin) > 28) && rawMin < (prevClip.start + prevClip.end) * 0.5;
    const overRight = nextClip !== null && (X(rawMax) - X(gapEnd) > 28) && rawMax > (nextClip.start + nextClip.end) * 0.5;
    crossedClip = overLeft || overRight;

    if (!crossedClip) {
      selStart = X(rawMin) - X(gapStart) <= 16 ? gapStart : Math.max(gapStart, snapTime(rawMin, e.altKey));
      selEnd = X(gapEnd) - X(rawMax) <= 16 ? gapEnd : Math.min(gapEnd, snapTime(rawMax, e.altKey));
      if (selEnd < selStart) {
        const mid = 0.5 * (gapStart + gapEnd);
        selStart = mid;
        selEnd = mid;
      }
      estCount = S.library.length ? estimateGapMotionCount(selStart, selEnd, prevClip, nextClip) : 0;
    }

    let ghost = document.getElementById("gapFillGhost");
    if (!ghost || !ghost.isConnected) {
      ghost = document.createElement("div");
      ghost.id = "gapFillGhost";
      ghost.className = "gap-fill-drag";
      ghost.innerHTML = `<div class="gap-fill-box"></div><div class="gap-fill-tag"></div>`;
      host.appendChild(ghost);
    }
    const box = ghost.querySelector<HTMLElement>(".gap-fill-box")!;
    const tag = ghost.querySelector<HTMLElement>(".gap-fill-tag")!;
    const drawA = crossedClip ? rawMin : selStart;
    const drawB = crossedClip ? rawMax : selEnd;
    const left = Math.max(L, X(drawA));
    const width = Math.max(6, X(drawB) - left);
    box.style.left = `${left}px`;
    box.style.top = `${Y.ph + 8}px`;
    box.style.width = `${width}px`;
    box.classList.toggle("invalid", crossedClip || !S.library.length);
    box.classList.toggle("short", !crossedClip && !!S.library.length && estCount === 0);

    tag.style.left = `${Math.max(L + 4, Math.min(left, host.clientWidth - 340))}px`;
    tag.style.top = `${Y.ph - 22}px`;
    tag.classList.toggle("invalid", crossedClip || !S.library.length);
    if (!S.library.length) {
      tag.textContent = "先にモーションを読み込んでください";
    } else if (crossedClip) {
      tag.textContent = "補間区間のみドラッグしてください（既存フレーズを含んでいます）";
    } else if (estCount === 0) {
      tag.textContent = `補間区間 ${(selEnd - selStart).toFixed(1)}秒 — 短すぎるため追加できません（約2.2秒以上）`;
    } else {
      tag.textContent = `補間区間 ${(selEnd - selStart).toFixed(1)}秒 → 曲に合う約 ${estCount} モーションを自動割り当て`;
    }
  };

  const onUp = async (e: MouseEvent) => {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    holdCursor(null);
    document.getElementById("gapFillGhost")?.remove();
    if (!dragging) {
      S.selected = null;
      seek(snapTime(originTime, e.altKey));
      refresh();
      return;
    }
    if (!S.library.length) {
      endInteraction();
      toast("割り当てに使えるフレーズがありません。先にモーションを読み込んでください", "error");
      return;
    }
    if (crossedClip) {
      endInteraction();
      toast("既存のフレーズをまたいでいるためキャンセルしました。補間区間のみをドラッグしてください", "error");
      return;
    }
    if (estCount === 0) {
      endInteraction();
      toast("この補間区間は短すぎるため、モーションを追加できません（約 2.2 秒以上の補間区間をドラッグしてください）", "error");
      return;
    }
    endInteraction(true);
    await fillGapRange(selStart, selEnd);
  };

  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  return true;
}

export async function insertPhrase(phraseId: string, start: number, place?: { x: number; z: number }, exact = false, fitMode: InsertFit = S.insertFit) {
  const fitPayload = fitMode === "raw"
    ? { fit: "raw" }
    : fitMode === "auto"
      ? { fit: "auto" }
      : { fit: "bars", bars: Number(fitMode) };
  try {
    const data = await call("POST", "/api/clips", { phraseId, start: round(start), exact, ...place, ...fitPayload });
    await applyChanges(data);
    if (data.added) { jumpToClip(data.added); }
    if (data.dropped) { toast(`曲の長さに入らなくなった ${data.dropped} 個のフレーズを外しました（Ctrl+Z で戻せます）`); }
    if (data.adjusted) { toast(`前後のフレーズに合わせて ${data.adjusted} 個のフレーズの長さを調整しました（Ctrl+Z で戻せます）`); }
  } catch (error) { toast((error as Error).message, "error"); }
}

/** Ctrl+C: remember the selected phrase as it is now. The copy can be pasted any number of times, for another dancer too, and after the phrase itself is gone. */
/** Stand a phrase where the one before it ends, so the join between them has no distance to cover. */
export async function alignClip(clipId: string, facing: boolean) {
  const clip = clips().find((c) => c.id === clipId);
  if (!clip) { return; }
  const from = { x: clip.x, z: clip.z };
  try {
    await applyChanges(await call("POST", `/api/clips/${clipId}/align`, { facing }));
    const now = clips().find((c) => c.id === clipId);
    const moved = now ? Math.hypot(now.x - from.x, now.z - from.z) : 0;
    toast(`前のフレーズの終わりの位置${facing ? "と向き" : ""}に合わせました（${moved.toFixed(2)} m 移動。Ctrl+Z で戻せます）`);
  } catch (error) {
    toast((error as Error).message, "error");
  }
}

export async function deleteClip(clipId: string) {
  try { await applyChanges(await call("DELETE", `/api/clips/${clipId}`)); }
  catch (error) { toast((error as Error).message, "error"); }
}

/** The commands of the right-click menu: for the selected phrase, or for an empty spot. */
export function clipActions(clip?: Clip): MenuAction[] {
  const paste: MenuAction = { label: "貼り付け", run: () => void pasteCopiedClip(), disabled: !S.clipboard, title: clip ? "コピーしたフレーズを、このフレーズの次に貼り付けます（Ctrl+V）" : "コピーしたフレーズを、白いバーの位置に貼り付けます（Ctrl+V）" };
  if (!clip) { return S.clipboard ? [paste] : []; }
  const first = clips()[0]?.id === clip.id;
  const fixed = !!clip.locked;
  const why = first ? "前にフレーズがありません" : fixed ? "ロック中です" : "";
  return [
    { label: "前の終わりに位置を合わせる", run: () => void alignClip(clip.id, false), disabled: first || fixed, title: why || "このフレーズの始まりを、前のフレーズが終わる位置へ動かします。つなぎで足が滑るのを抑えます" },
    { label: "位置と向きを合わせる", run: () => void alignClip(clip.id, true), disabled: first || fixed, title: why || "位置に加えて、体の向きも前のフレーズの終わりに合わせます" },
    { label: "コピー", run: () => void copySelectedClip(), title: "Ctrl+C" },
    paste,
    { label: clip.locked ? "ロックを解除" : "ロック", run: () => toggleLock(clip.id) },
    { label: "削除", run: () => void deleteClip(clip.id), disabled: fixed, danger: true, title: fixed ? "ロック中です" : "Delete" },
  ];
}

export function copySelectedClip(): boolean {
  const clip = selectedClip();
  if (!clip) { return false; }
  S.clipboard = JSON.parse(JSON.stringify(clip)) as Clip;
  toast(`「${phraseOf(clip.phraseId)?.name ?? "フレーズ"}」をコピーしました。Ctrl+V で、選択中のフレーズの次か白いバーの位置に貼り付けます`);
  return true;
}

let pasting = false;

/** Ctrl+V: paste the copied phrase after the selected phrase, or at the white bar when none is selected. The pasted phrase becomes the selected one, so pasting again adds the next copy after it. */
export async function pasteCopiedClip() {
  const copied = S.clipboard;
  if (!copied) { toast("先にタイムラインのフレーズを選んで、Ctrl+C でコピーしてください"); return; }
  if (!S.song || pasting) { return; }
  const after = selectedClip();
  pasting = true;
  try {
    const data = await call("POST", "/api/clips/paste", { clip: copied, start: round(after ? after.end : S.playhead) });
    await applyChanges(data);
    if (data.added) { jumpToClip(data.added); }
    const notes = [data.dropped ? `曲の長さに入らなくなった ${data.dropped} 個を外しました` : "", data.adjusted ? `${data.adjusted} 個の長さを調整しました` : ""].filter(Boolean);
    toast(`「${phraseOf(copied.phraseId)?.name ?? "フレーズ"}」を貼り付けました${notes.length ? `。${notes.join("。")}` : ""}（Ctrl+Z で戻せます）`);
  } catch (error) {
    toast((error as Error).message, "error");
  } finally {
    pasting = false;
  }
}

export async function replaceClip(clipId: string, phraseId: string) {
  const clip = clips().find((c) => c.id === clipId);
  const next = phraseOf(phraseId);
  if (!clip || !next) { return; }
  markGenerating(clip.id);
  refresh();
  try {
    await applyChanges(await call("PATCH", `/api/clips/${clip.id}`, { phraseId, speedMode: "auto" }));
    S.selected = clip.id;
    refresh();
    toast(`「${next.name}」に差し替えました（Ctrl+Z で戻せます）`);
  } catch (error) {
    toast((error as Error).message, "error");
    await applyChanges(await call("GET", "/api/state"));
  }
}

export async function cutClip(clipId: string, at: number, mode: "split" | "before" | "after" = "split") {
  try {
    const data = await call("POST", `/api/clips/${clipId}/cut`, { at: round(at), mode });
    if (data.selected) { S.selected = data.selected; }
    await applyChanges(data);
    const msg = mode === "split" ? "白いバーの位置で分割しました" : mode === "before" ? "白いバーより前をカットしました" : "白いバーより後ろをカットしました";
    toast(`${msg}（Ctrl+Z で戻せます）`);
  } catch (error) {
    toast((error as Error).message, "error");
  }
}

export async function splitAtPlayhead() {
  const sel = selectedClip();
  const insideSel = sel && S.playhead > sel.start + 0.05 && S.playhead < sel.end - 0.05 ? sel : undefined;
  const target = insideSel ?? clips().find((c) => S.playhead > c.start + 0.05 && S.playhead < c.end - 0.05);
  if (!target) {
    toast("分割したいフレーズの内側（端から 0.3 秒以上）に白いバーを置いてください", "error");
    return;
  }
  await cutClip(target.id, S.playhead, "split");
}

function startOverview(event: MouseEvent) {
  const song = S.song!;
  const svg = document.getElementById("ovSvg")!;
  const rect = svg.getBoundingClientRect();
  const width = areaWidth();
  const onWindow = (event.target as Element).id === "ovWindow";
  const barIndexAt = (t: number) => { let index = 0; song.bars.forEach((bar, i) => { if (bar <= t) { index = i; } }); return index; };
  const place = (e: MouseEvent) => {
    const t = Math.max(0, Math.min(song.duration - 0.01, ((e.clientX - rect.left) / width) * song.duration));
    seek(t);
    const [a, b] = viewRange();
    if (onWindow || t < a || t > b) {
      const center = barIndexAt(t) - Math.floor(S.viewBars / 2);
      S.viewStart = Math.max(0, Math.min(song.bars.length - S.viewBars, center));
      renderTimeline();
    } else { updatePlayhead(); }
  };
  place(event);
  const onUp = () => { window.removeEventListener("mousemove", place); window.removeEventListener("mouseup", onUp); };
  window.addEventListener("mousemove", place);
  window.addEventListener("mouseup", onUp);
}

export const clipAtSelection = selectedClip;
