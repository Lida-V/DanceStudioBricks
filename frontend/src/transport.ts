import { S, barBeatAt, formatTime, notifyPlayhead, toast } from "./store";
import { followPlayhead, updatePlayhead } from "./timeline";

const audio = () => document.getElementById("audio") as HTMLAudioElement;
const PLAY = '<path d="M2 0l13 8-13 8z"/>';
const PAUSE = '<rect x="2" width="4" height="16"/><rect x="10" width="4" height="16"/>';

function showClock() {
  document.getElementById("timecode")!.textContent = formatTime(S.playhead);
  const position = barBeatAt(S.playhead);
  document.getElementById("barBeat")!.textContent = position ? `${position.bar}小節 ${position.beat}拍` : "--";
  notifyPlayhead();
}

export function seek(seconds: number) {
  S.playhead = Math.max(0, seconds);
  const element = audio();
  if (element.src && Math.abs(element.currentTime - S.playhead) > 0.02) { element.currentTime = S.playhead; }
  showClock();
  updatePlayhead();
}

async function explainPlayFailure(element: HTMLAudioElement, error: DOMException): Promise<string> {
  if (error.name === "NotAllowedError") { return "ブラウザーに再生を止められました。もう一度再生を押してください"; }
  try {
    await fetch("/api/state");
  } catch {
    return "アプリのサーバーに接続できないため、再生できません。アプリを起動し直してください";
  }
  return element.error?.code === 4 ? "この音声ファイルの形式は再生できません。WAV か MP3 を読み込んでください" : "音声を読み込めませんでした。もう一度再生を押してください";
}

export function togglePlay() {
  const element = audio();
  if (!S.song) { toast("先に曲を読み込んでください", "error"); return; }
  if (!element.paused) { element.pause(); return; }
  if (element.error) { element.load(); }  // a failed load stays failed until the element is told to try again
  element.play().catch(async (error: DOMException) => {
    if (error.name !== "AbortError") { toast(await explainPlayFailure(element, error), "error"); }
  });
}

export function initTransport() {
  const element = audio();
  element.addEventListener("loadedmetadata", () => { if (Math.abs(element.currentTime - S.playhead) > 0.05) { element.currentTime = S.playhead; } });
  element.addEventListener("play", () => {
    S.playing = true;
    document.getElementById("playIcon")!.innerHTML = PAUSE;
    const uiPlay = document.getElementById("btnUiPlay");
    if (uiPlay) { uiPlay.textContent = "一時停止"; }
  });
  element.addEventListener("pause", () => {
    S.playing = false;
    document.getElementById("playIcon")!.innerHTML = PLAY;
    const uiPlay = document.getElementById("btnUiPlay");
    if (uiPlay) { uiPlay.textContent = "再生"; }
  });
  document.getElementById("btnPlay")!.addEventListener("click", togglePlay);
  const jump = (direction: 1 | -1) => {
    const song = S.song;
    if (!song) { return; }
    const starts = song.sections.map((s) => s.start);
    const next = direction > 0 ? starts.find((t) => t > S.playhead + 0.05) : [...starts].reverse().find((t) => t < S.playhead - 0.5);
    seek(next ?? (direction > 0 ? song.duration - 0.05 : 0));
    followPlayhead();
  };
  document.getElementById("btnPrev")!.addEventListener("click", () => jump(-1));
  document.getElementById("btnNext")!.addEventListener("click", () => jump(1));
  const frame = () => {
    if (S.playing) {
      S.playhead = element.currentTime;
      showClock();
      followPlayhead();
      updatePlayhead();
    }
    requestAnimationFrame(frame);
  };
  frame();
  showClock();
}

export function syncAudio() {
  const key = S.song ? `${S.song.path}` : "";
  if (key === S.audioVersion) { return; }
  S.audioVersion = key;
  const element = audio();
  element.pause();
  element.src = key ? `/api/audio?v=${encodeURIComponent(key)}` : "";
  S.playhead = 0;
  showClock();
}
