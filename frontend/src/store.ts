export interface Phrase {
  id: string; name: string; duration: number; bars: number; usage: "auto" | "manual"; sourceBpm: number;
  category: string; energy: number; travel: number; turnDegrees: number; accents: number[];
  thumb?: Record<string, [number, number]>; tags: string[]; description: string; pack: string;
}
export interface Pin { source: number; at: number; }
export interface Clip {
  id: string; phraseId: string; start: number; end: number; x: number; z: number; facing: number;
  locked: boolean; speedMode: string; match: number; pins?: Pin[]; trimStart?: number; trimEnd?: number;
}
export interface Section { startBar: number; endBar: number; kind: string; label: string; start: number; end: number; energy: number; repeatOf: number; }
export interface Accent { t: number; strength: number; kind: string; onBeat: boolean; kime: boolean; }
export interface Song {
  name: string; path: string; duration: number; tempo: number; meter: number; barSeconds: number; beats: number[]; bars: number[];
  sections: Section[]; energyPerBar: number[]; accents: Accent[]; wave: number[]; engines: Record<string, string>; tempoFactor: number;
}
export interface Join { from: string; to: string; start: number; end: number; seconds: number; distance: number; speed: number; status: string; maxJointSpeed?: number; via?: [number, number]; }
export interface ClipAccent { t: number; sound: number | null; matched: boolean; source: number; pinned: boolean; }
export interface ClipReport { speed: number; localSpeed: [number, number]; accents: ClipAccent[]; matched: number; candidates: number; curve: number[]; }
export interface Composed { clips: Record<string, ClipReport>; joins: Join[]; generated: number; reused: number; engineError: string | null; seconds: number; version: number; frames: number; fps: number; }
export interface FlowRule { categories: string[]; energy: number; move: number; text: string; }
export interface Dancer { id: string; name: string; clips: Clip[]; }
export interface Spot { x: number; z: number; facing?: number | null; }
export interface Formation { id: string; time: number; name: string; spots: Record<string, Spot>; }
export interface ModelRef { path: string; name: string; mapping: Record<string, string>; }
export interface Project {
  songPath: string | null; tempoFactor: number | null; match: number; seed: number; snap: string; flow: Record<string, FlowRule>; dancers: Dancer[];
  active?: string; formations?: Formation[]; models?: Record<string, ModelRef>; arrangeFilter?: string[];
}
export interface CastMember { id: string; name: string; clips: number; version: number; }
export interface CloseCall { a: string; b: string; start: number; end: number; distance: number; }
export interface Motion { version: number; data: Float32Array; }

export type InsertFit = "auto" | "1" | "2" | "4" | "raw";

export const S = {
  library: [] as Phrase[], categories: {} as Record<string, string>, song: null as Song | null, project: null as Project | null,
  composed: null as Composed | null, jointNames: [] as string[], preview: null as Float32Array | null, previewVersion: -1,
  selected: null as string | null, selectedPhrase: null as string | null, playhead: 0, playing: false,
  viewStart: 0, viewBars: 16, catFilter: "all", search: "", semantic: null as string[] | null, snap: "beat", audioVersion: "",
  history: { undo: 0, redo: 0 }, leftTab: "library", interacting: false, draggingPhrase: null as string | null,
  insertFit: "auto" as InsertFit,
  uiHidden: false,
  clipboard: null as Clip | null,  // the phrase copied with Ctrl+C, as it was at that moment
  cast: [] as CastMember[], collisions: [] as CloseCall[], modelBones: [] as string[],
  previews: {} as Record<string, Motion>,  // joint positions per dancer, for the mannequins
  turns: {} as Record<string, Motion>,  // bone turns per dancer, fetched only for dancers shown with a model
  wantsTurns: new Set<string>(),
};

/** One colour per dancer, in order. The first keeps the mannequin's own white. */
export const DANCER_COLORS = ["#f1f0ec", "#7fc8ff", "#ffb86b", "#b5e48c", "#f7a8d8", "#c9b6ff", "#ffe066", "#8ce0d0", "#ff9d9d", "#a8b8ff", "#d6c19a", "#9fe3ff"];
export const dancers = (): Dancer[] => S.project?.dancers ?? [];
export const activeDancer = (): Dancer | undefined => dancers().find((d) => d.id === S.project?.active) ?? dancers()[0];
export const dancerColor = (id: string) => DANCER_COLORS[Math.max(0, dancers().findIndex((d) => d.id === id)) % DANCER_COLORS.length];

let lastPress = { id: "", time: 0 };

/** True when this press repeats the last one quickly. For elements that are redrawn between clicks, where dblclick never fires. */
export function isDoublePress(id: string): boolean {
  const now = performance.now();
  const repeated = lastPress.id === id && now - lastPress.time < 350;
  lastPress = repeated ? { id: "", time: 0 } : { id, time: now };
  return repeated;
}

export const clips = (): Clip[] => activeDancer()?.clips ?? [];
export const phraseOf = (id: string): Phrase | undefined => S.library.find((p) => p.id === id);
export const selectedClip = (): Clip | undefined => clips().find((c) => c.id === S.selected);

export function clipTrim(clip: Clip, item: Phrase): [number, number] {
  const start = Math.max(0, clip.trimStart ?? 0);
  const end = Math.min(item.duration, clip.trimEnd ?? item.duration);
  return [start, Math.max(start + 0.01, end)];
}

export function clipDuration(clip: Clip, item: Phrase): number {
  const [start, end] = clipTrim(clip, item);
  return end - start;
}

export function isTrimmed(clip: Clip, item: Phrase): boolean {
  const [start, end] = clipTrim(clip, item);
  return start > 1e-3 || end < item.duration - 1e-3;
}

type Listener = () => void;
const listeners: Listener[] = [];
export const subscribe = (fn: Listener) => listeners.push(fn);
export const refresh = () => listeners.forEach((fn) => fn());
const playheadListeners: Listener[] = [];
export const onPlayhead = (fn: Listener) => playheadListeners.push(fn);
export const notifyPlayhead = () => playheadListeners.forEach((fn) => fn());

export class ApiError extends Error {}

let editsInFlight = 0;
let editSerial = 0;

export async function call<T = any>(method: string, path: string, body?: unknown): Promise<T> {
  const edits = method !== "GET";
  if (edits) { editSerial++; editsInFlight++; }
  let response: Response;
  try {
    response = await fetch(path, { method, headers: body === undefined ? undefined : { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) });
  } catch {
    throw new ApiError("アプリのサーバーに接続できません。アプリを起動し直してください");
  } finally {
    if (edits) { editsInFlight--; }
  }
  if (!response.ok) {
    let message = `エラー ${response.status}`;
    try { const detail = await response.json(); message = detail.error ?? detail.detail ?? message; } catch { /* keep the status text */ }
    throw new ApiError(message);
  }
  return response.json() as Promise<T>;
}

export function toast(text: string, kind: "info" | "error" = "info") {
  const box = document.getElementById("toast")!;
  box.textContent = text;
  box.className = kind;
  box.hidden = false;
  window.clearTimeout((box as any)._timer);
  (box as any)._timer = window.setTimeout(() => (box.hidden = true), kind === "error" ? 8000 : 5000);
}

/** A press on a locked phrase only selects it. Says why nothing follows the pointer if it is then dragged. */
export function warnLockedDrag(press: { clientX: number; clientY: number }, released?: () => void) {
  const move = (event: PointerEvent) => {
    if (Math.hypot(event.clientX - press.clientX, event.clientY - press.clientY) < 4) { return; }
    toast("ロック中のフレーズは動かせません。動かすにはロックを解除してください");
    window.removeEventListener("pointermove", move);
  };
  const up = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
    released?.();
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
}

export async function withBusy<T>(text: string, work: () => Promise<T>): Promise<T | undefined> {
  const busy = document.getElementById("busy")!;
  document.getElementById("busyText")!.textContent = text;
  busy.hidden = false;
  try {
    return await work();
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error), "error");
    return undefined;
  } finally {
    busy.hidden = true;
  }
}

type Changes = { project?: Project; composed?: Composed | null; history?: { undo: number; redo: number }; cast?: CastMember[]; collisions?: CloseCall[] };

let held: Changes | null = null;

/** While something is being dragged, answers from the server wait, so the screen does not change under the pointer. */
export function beginInteraction() {
  S.interacting = true;
  editSerial++;
}

/** Pass `discard` when an edit is about to be sent: its answer carries everything newer than what was held. */
export function endInteraction(discard = false) {
  S.interacting = false;
  const data = held;
  held = null;
  if (data && !discard) { void applyChanges(data); }
}

/** The joins touching this phrase are about to be remade. Shown at once, before the server answers. */
export function markGenerating(id: string) {
  for (const join of S.composed?.joins ?? []) {
    if ((join.from === id || join.to === id) && join.status !== "overlap" && join.status !== "cut") { join.status = "pending"; }
  }
}

const joinsPending = () => S.composed?.joins.some((join) => join.status === "pending") ?? false;
const pause = (milliseconds: number) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));
let watching = false;

/** Picks up the finished motion once MotionBricks has made the joins that are marked as generating. */
async function watchJoins() {
  if (watching) { return; }
  watching = true;
  try {
    while (joinsPending()) {
      while (S.interacting || editsInFlight) { await pause(50); }
      const asked = editSerial;
      const data = await call<Changes>("GET", `/api/joins/wait?version=${S.composed?.version ?? -1}`);
      if (asked === editSerial && !S.interacting) { await applyChanges(data); }
    }
  } catch { /* the next action reports a lost connection */ }
  finally { watching = false; }
}

export async function applyChanges(data: Changes) {
  if (S.interacting) { held = { ...held, ...data }; return; }
  if (data.project) { S.project = data.project; }
  if (data.history) { S.history = data.history; }
  if ("composed" in data) { S.composed = data.composed ?? null; }
  if (data.cast) { S.cast = data.cast; }
  if (data.collisions) { S.collisions = data.collisions; }
  if (S.selected && !selectedClip()) { S.selected = null; }
  await loadPreview();
  refresh();
  void watchJoins();
}

export async function applyState(state: any) {
  S.library = state.library; S.categories = state.categories; S.song = state.song; S.project = state.project;
  S.composed = state.composed; S.jointNames = state.previewJoints?.names ?? [];
  S.cast = state.cast ?? []; S.collisions = state.collisions ?? []; S.modelBones = state.modelBones ?? [];
  S.history = state.history ?? S.history;
  if (S.project) { S.snap = S.project.snap ?? "beat"; }
  if (S.selected && !selectedClip()) { S.selected = null; }
  await loadPreview();
  refresh();
  void watchJoins();
}

async function fetchMotion(kind: "preview" | "turns", id: string): Promise<Float32Array | null> {
  try {
    const response = await fetch(`/api/${kind}.bin?dancer=${encodeURIComponent(id)}`);
    return response.ok ? new Float32Array(await response.arrayBuffer()) : null;
  } catch { return null; }
}

/** Fetches the motion of every dancer whose version changed. Bone turns are fetched only for dancers shown with a model. */
export async function loadPreview() {
  const live = new Set(S.cast.filter((member) => member.version > 0).map((member) => member.id));
  for (const store of [S.previews, S.turns]) {
    for (const id of Object.keys(store)) { if (!live.has(id)) { delete store[id]; } }
  }
  await Promise.all(S.cast.filter((member) => live.has(member.id)).flatMap((member) => {
    const jobs: Promise<void>[] = [];
    if (S.previews[member.id]?.version !== member.version) {
      jobs.push(fetchMotion("preview", member.id).then((data) => { if (data) { S.previews[member.id] = { version: member.version, data }; } }));
    }
    if ((S.modelBones.length > 0 || S.wantsTurns.has(member.id)) && S.turns[member.id]?.version !== member.version) {
      jobs.push(fetchMotion("turns", member.id).then((data) => { if (data) { S.turns[member.id] = { version: member.version, data }; } }));
    }
    return jobs;
  }));
  S.preview = S.previews[activeDancer()?.id ?? ""]?.data ?? null;
  S.previewVersion = S.composed?.version ?? -1;
}

export const formatTime = (t: number) => {
  const minutes = Math.floor(t / 60);
  const seconds = t - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(2).padStart(5, "0")}`;
};

export function barBeatAt(t: number): { bar: number; beat: number } | null {
  const song = S.song;
  if (!song) { return null; }
  let index = 0;
  for (let i = 0; i < song.bars.length; i++) { if (song.bars[i] <= t + 1e-3) { index = i; } else { break; } }
  const beats = song.beats.filter((b) => b >= song.bars[index] - 1e-3 && b <= t + 1e-3).length;
  return { bar: index + 1, beat: Math.max(1, beats) };
}

export const CAT_COLOR: Record<string, string> = { pose: "var(--pose)", groove: "var(--groove)", wave: "var(--wave)", power: "var(--power)", foot: "var(--foot)" };
