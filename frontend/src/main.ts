import "./base.css";
import "./app.css";
import { S, CAT_COLOR, InsertFit, applyChanges, applyState, call, clips, onPlayhead, phraseOf, refresh, selectedClip, subscribe, toast, withBusy } from "./store";
import { initCast } from "./cast";
import { FIT_LABELS } from "./insertMenu";
import { initStage, isEditingStage, syncStageUiHidden } from "./stage";
import { copySelectedClip, initTimeline, insertPhrase, pasteCopiedClip, renderTimeline, replaceClip, splitAtPlayhead, toggleLock } from "./timeline";
import { initTransport, seek, syncAudio, togglePlay } from "./transport";
import { initInstaller } from "./installer";
import { bindOrder, clearSearchQuery, importMotion, loadSong, openArrangeFilter, openExport, openFlow, openPhrase, renderInspector, renderLibrary, renderOrder, semanticSearch, toggleSearchTag, updateInspectorCutState, updateOrderNow } from "./panels";

function renderHeader() {
  const song = S.song;
  document.getElementById("songName")!.textContent = song ? song.name : "曲が未選択です";
  document.getElementById("songSub")!.textContent = song ? `${Math.floor(song.duration / 60)}:${String(Math.round(song.duration % 60)).padStart(2, "0")} ・ 解析済み` : "「曲を読み込む」から音声を選びます";
  document.getElementById("tempo")!.textContent = song ? song.tempo.toFixed(1) : "--";
  document.getElementById("meter")!.textContent = song ? `${song.meter}/4` : "--";
  document.getElementById("barCount")!.textContent = song ? String(song.bars.length) : "--";
  document.getElementById("tempoFix")!.hidden = !song;
  document.querySelectorAll<HTMLElement>("#snapSeg span").forEach((el) => el.classList.toggle("on", el.dataset.snap === S.snap));
  const fitBtn = document.getElementById("btnInsertFit");
  if (fitBtn) {
    fitBtn.textContent = `長さ: ${FIT_LABELS[S.insertFit]}`;
    fitBtn.classList.toggle("chip-accent", S.insertFit !== "raw");
  }
  const afBtn = document.getElementById("btnArrangeFilter");
  if (afBtn) {
    const tags = S.project?.arrangeFilter ?? [];
    if (!tags.length) {
      afBtn.textContent = "対象: すべて ▾";
      afBtn.classList.remove("filter-active");
      afBtn.title = "自動配置や入れ替えで使うジャンル・雰囲気・動きタグを絞り込みます";
    } else {
      const shortLabel = tags.length <= 2 ? tags.join("・") : `${tags.slice(0, 2).join("・")} +${tags.length - 2}`;
      afBtn.textContent = `対象: ${shortLabel} ▾`;
      afBtn.classList.add("filter-active");
      afBtn.title = `現在の自動配置対象: ${tags.join("・")}（クリックで変更・解除）`;
    }
  }
  const uiPlayBtn = document.getElementById("btnUiPlay");
  if (uiPlayBtn) {
    uiPlayBtn.textContent = S.playing ? "一時停止" : "再生";
  }
  const clip = selectedClip();
  const card = document.getElementById("nowCard")!;
  const item = clip && phraseOf(clip.phraseId);
  card.hidden = !item;
  if (clip && item) {
    document.getElementById("nowName")!.textContent = item.name;
    const index = clips().indexOf(clip);
    document.getElementById("nowSub")!.textContent = `位置 ${index + 1} ・ ${clip.start.toFixed(1)} 秒から`;
  }
}

function renderLegend() {
  const entries = Object.entries(S.categories).map(([key, label]) => `<span><i style="background:${CAT_COLOR[key]}"></i>${label}</span>`).join("");
  document.getElementById("legend")!.innerHTML = `${entries}<span><svg width="22" height="13"><rect width="22" height="13" fill="url(#brickP)"/></svg>接続</span>
    <span><svg width="14" height="14"><path d="M7 1L13 7L7 13L1 7Z" fill="#fff"/></svg>音</span>
    <span><svg width="14" height="14"><path d="M7 1L13 7L7 13L1 7Z" fill="#14151a" stroke="#fff" stroke-width="1.500"/></svg>振り</span>`;
}

async function main() {
  initStage();
  initTimeline();
  initTransport();
  initCast();
  subscribe(renderHeader);
  subscribe(renderLibrary);
  subscribe(renderOrder);
  onPlayhead(updateOrderNow);
  onPlayhead(updateInspectorCutState);
  bindOrder();
  document.getElementById("leftTabs")!.addEventListener("click", (event) => {
    const tab = (event.target as HTMLElement).closest<HTMLElement>("[data-tab]")?.dataset.tab;
    if (!tab) { return; }
    S.leftTab = tab;
    document.querySelectorAll<HTMLElement>("#leftTabs [data-tab]").forEach((el) => el.classList.toggle("on", el.dataset.tab === tab));
    document.getElementById("tabLibrary")!.hidden = tab !== "library";
    document.getElementById("tabOrder")!.hidden = tab !== "order";
    renderOrder();
  });
  subscribe(renderInspector);
  subscribe(renderTimeline);
  subscribe(renderLegend);
  subscribe(syncAudio);
  document.getElementById("btnSong")!.onclick = () => loadSong();
  document.getElementById("btnImport")!.onclick = () => importMotion();
  document.getElementById("btnFlow")!.onclick = () => { if (S.project) { openFlow(); } };
  document.getElementById("btnExport")!.onclick = openExport;
  document.getElementById("tempoHalf")!.onclick = () => loadSong((S.song?.tempoFactor ?? 1) === 2 ? 1 : 0.5);
  document.getElementById("tempoDouble")!.onclick = () => loadSong((S.song?.tempoFactor ?? 1) === 0.5 ? 1 : 2);
  const run = async (label: string, path: string, body: unknown = {}) => {
    const data = await withBusy(label, () => call("POST", path, body));
    if (data) { await applyChanges(data); }
  };
  document.getElementById("btnArrangeFilter")!.onclick = () => openArrangeFilter();
  document.getElementById("btnArrange")!.onclick = () => {
    const tags = S.project?.arrangeFilter ?? [];
    const prefix = tags.length ? `「${tags.join("・")}」で` : "";
    void run(`${prefix}自動配置して MotionBricks でつないでいます`, "/api/arrange");
  };
  document.getElementById("btnShuffle")!.onclick = () => {
    const tags = S.project?.arrangeFilter ?? [];
    const prefix = tags.length ? `「${tags.join("・")}」で` : "";
    void run(`${prefix}入れ替えています`, "/api/shuffle");
  };
  document.getElementById("btnRangeArrange")!.onclick = () => {
    const song = S.song;
    if (!song) { return; }
    const bars = [...song.bars, song.duration];
    const end = bars[Math.min(S.viewStart + S.viewBars, bars.length - 1)];
    const tags = S.project?.arrangeFilter ?? [];
    const prefix = tags.length ? `「${tags.join("・")}」で` : "";
    void run(`${prefix}表示範囲を自動配置しています`, "/api/arrange", { start: bars[S.viewStart], end });
  };

  let uiFadeTimer = 0;
  const restoreBar = document.getElementById("uiRestoreBar")!;
  const bumpRestoreBar = () => {
    if (!S.uiHidden) { return; }
    restoreBar.classList.remove("idle");
    window.clearTimeout(uiFadeTimer);
    uiFadeTimer = window.setTimeout(() => {
      if (S.uiHidden) { restoreBar.classList.add("idle"); }
    }, 2000);
  };
  const setUiHidden = (hidden: boolean) => {
    S.uiHidden = hidden;
    document.body.classList.toggle("ui-hidden", hidden);
    document.getElementById("insertMenu")?.remove();
    restoreBar.hidden = !hidden;
    window.clearTimeout(uiFadeTimer);
    if (hidden) {
      bumpRestoreBar();
    } else {
      restoreBar.classList.remove("idle");
    }
    syncStageUiHidden();
  };
  document.getElementById("btnHideUi")!.onclick = () => setUiHidden(true);
  document.getElementById("btnShowUi")!.onclick = () => setUiHidden(false);
  document.getElementById("btnUiPlay")!.onclick = () => { togglePlay(); bumpRestoreBar(); };
  document.getElementById("stage")!.addEventListener("mousemove", bumpRestoreBar);
  document.getElementById("btnInsert")!.onclick = () => {
    if (!S.selectedPhrase) { toast("先にライブラリのフレーズを選んでください", "error"); return; }
    insertPhrase(S.selectedPhrase, S.playhead);
  };
  const fitCycle: InsertFit[] = ["auto", "1", "2", "4", "raw"];
  document.getElementById("btnInsertFit")!.onclick = () => {
    const next = fitCycle[(fitCycle.indexOf(S.insertFit) + 1) % fitCycle.length];
    S.insertFit = next;
    refresh();
    toast(`挿入時の長さを「${FIT_LABELS[next]}」に切り替えました（右クリックメニューからも選べます）`);
  };
  document.getElementById("btnSplit")!.onclick = () => { void splitAtPlayhead(); };
  document.getElementById("snapSeg")!.addEventListener("click", (event) => {
    const snap = (event.target as HTMLElement).dataset.snap;
    if (snap) { S.snap = snap; refresh(); }
  });
  const grid = document.getElementById("grid")!;
  grid.addEventListener("click", (event) => {
    const card = (event.target as HTMLElement).closest<HTMLElement>("[data-phrase]");
    if (card) { S.selectedPhrase = card.dataset.phrase!; refresh(); }
  });
  grid.addEventListener("dblclick", (event) => {
    const card = (event.target as HTMLElement).closest<HTMLElement>("[data-phrase]");
    if (card) { openPhrase(card.dataset.phrase!); }
  });
  grid.addEventListener("dragstart", (event) => {
    const card = (event.target as HTMLElement).closest<HTMLElement>("[data-phrase]");
    if (card) {
      S.draggingPhrase = card.dataset.phrase!;
      event.dataTransfer?.setData("text/phrase", card.dataset.phrase!);
    }
  });
  grid.addEventListener("dragend", () => { S.draggingPhrase = null; });
  grid.addEventListener("contextmenu", (event) => {
    const card = (event.target as HTMLElement).closest<HTMLElement>("[data-phrase]");
    if (!card) { return; }
    event.preventDefault();
    const picked = selectedClip();
    if (picked) {
      void replaceClip(picked.id, card.dataset.phrase!);
    } else {
      toast("タイムラインのフレーズを選んでから右クリックすると差し替えます");
    }
  });
  const catChips = document.getElementById("catChips")!;
  catChips.addEventListener("click", (event) => {
    const chip = (event.target as HTMLElement).closest<HTMLElement>("[data-cat]");
    const cat = chip?.dataset.cat;
    if (cat) {
      S.catFilter = cat;
      refresh();
      catChips.querySelector<HTMLElement>(`[data-cat="${cat}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
    }
  });
  const tagChips = document.getElementById("tagChips")!;
  tagChips.addEventListener("click", (event) => {
    const clearEl = (event.target as HTMLElement).closest<HTMLElement>("[data-clear-search]");
    if (clearEl) {
      clearSearchQuery();
      return;
    }
    const chip = (event.target as HTMLElement).closest<HTMLElement>("[data-tag]");
    const tag = chip?.dataset.tag;
    if (tag) {
      toggleSearchTag(tag);
    }
  });
  for (const strip of [catChips, tagChips, document.getElementById("leftTabs")!]) {
    strip.addEventListener("wheel", (event) => {
      if (strip.scrollWidth > strip.clientWidth && Math.abs(event.deltaY) > Math.abs(event.deltaX)) {
        event.preventDefault();
        strip.scrollLeft += event.deltaY;
      }
    }, { passive: false });
  }
  const search = document.getElementById("search") as HTMLInputElement;
  search.oninput = () => { S.search = search.value; S.semantic = null; renderLibrary(); };
  search.onkeydown = (event) => { if (event.key === "Enter") { semanticSearch(search.value); } };
  let historyBusy = false;
  const stepHistory = async (direction: "undo" | "redo") => {
    if (historyBusy || isEditingStage() || S.interacting) { return; }
    if (S.history[direction] === 0) { toast(direction === "undo" ? "これ以上は戻せません" : "やり直す操作はありません"); return; }
    historyBusy = true;
    try {
      await applyState(await call("POST", `/api/${direction}`));
    } catch (error) {
      toast((error as Error).message, "error");
    } finally {
      historyBusy = false;
    }
  };
  // Typing keeps the browser's own text shortcuts; sliders still let the keys through.
  const isTyping = (element: HTMLElement) => element.isContentEditable || element.tagName === "TEXTAREA" || element.tagName === "SELECT"
    || (element.tagName === "INPUT" && !["range", "checkbox"].includes((element as HTMLInputElement).type));
  window.addEventListener("keydown", async (event) => {
    const overlay = !document.getElementById("modal")!.hidden || !document.getElementById("busy")!.hidden;
    if (overlay || isTyping(event.target as HTMLElement)) { return; }
    const key = event.key.toLowerCase();
    if (event.key === "Escape" && S.uiHidden) {
      event.preventDefault();
      setUiHidden(false);
      return;
    }
    if (key === "h" && !event.ctrlKey && !event.metaKey && !event.altKey && !event.repeat) {
      event.preventDefault();
      setUiHidden(!S.uiHidden);
      return;
    }
    if (event.code === "Space") {
      event.preventDefault();
      if (event.repeat) { return; }
      (document.activeElement as HTMLElement | null)?.blur();  // otherwise the focused button would also click
      togglePlay();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && (key === "z" || key === "y")) {
      event.preventDefault();
      await stepHistory(key === "y" || event.shiftKey ? "redo" : "undo");
      return;
    }
    if ((event.ctrlKey || event.metaKey) && !event.altKey && !event.shiftKey && (key === "c" || key === "v")) {
      if (historyBusy || isEditingStage() || S.interacting) { return; }
      if (key === "c") {
        if (window.getSelection()?.toString()) { return; }  // text on the page is selected: the browser copies that
        event.preventDefault();
        if (!copySelectedClip()) { toast("コピーするフレーズを、タイムラインで選んでください"); }
      } else {
        event.preventDefault();
        if (!event.repeat) { await pasteCopiedClip(); }
      }
      return;
    }
    if (key === "l" && S.selected && !event.ctrlKey && !event.metaKey && !event.altKey && !event.repeat) {
      toggleLock(S.selected);
      return;
    }
    if (key === "s" && !event.ctrlKey && !event.metaKey && !event.altKey && !event.repeat) {
      event.preventDefault();
      await splitAtPlayhead();
      return;
    }
    if ((event.key === "Delete" || event.key === "Backspace") && S.selected) {
      try { await applyChanges(await call("DELETE", `/api/clips/${S.selected}`)); }
      catch (error) { toast((error as Error).message, "error"); }
    }
  });
  await applyState(await call("GET", "/api/state"));
  seek(0);
  await initInstaller();
}

main().catch((error) => toast(String(error), "error"));
