import { S, CAT_COLOR, Phrase, applyChanges, applyState, barBeatAt, beginInteraction, call, clipDuration, clipTrim, clips, endInteraction, formatTime, isDoublePress, isTrimmed, phraseOf, refresh, selectedClip, toast, warnLockedDrag, withBusy } from "./store";
import { collisionList } from "./cast";
import { ACTION_TAG_OPTIONS, GENRE_TAG_OPTIONS, MOOD_TAG_OPTIONS, activeQueryNormTokens, cleanFilterTags, filterByFeeling, filterPhrasesByTags, normalizeToken } from "./feelingSearch";
import { openInsertMenu, openPhraseMenu } from "./insertMenu";
import { exportModel, hasModel } from "./stage";
import { cutClip, exactInsertLabel, fillGapRange, holdCursor, insertPhrase, jumpToClip, replaceClip, snapTime } from "./timeline";

const LIMBS = [
  ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head"], ["Chest", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"],
  ["Chest", "RightShoulder", "RightArm", "RightForeArm", "RightHand"], ["Hips", "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase"], ["Hips", "RightLeg", "RightShin", "RightFoot", "RightToeBase"],
];
export const escapeHtml = (text: string) => text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));

export function thumb(item: Phrase, width = 130, height = 76): string {
  const color = CAT_COLOR[item.category] ?? "var(--groove)";
  const points = item.thumb;
  if (!points) { return ""; }
  const scale = 34, cx = 65, base = 72;
  const px = (name: string) => { const p = points[name]; return `${(cx + p[0] * scale).toFixed(1)},${(base - p[1] * scale).toFixed(1)}`; };
  const head = points.Head;
  let body = LIMBS.map((chain) => `<polyline points="${chain.filter((n) => points[n]).map(px).join(" ")}" fill="none" stroke="${color}" stroke-width="3.800" stroke-linecap="round" stroke-linejoin="round"/>`).join("");
  if (head) { body += `<circle cx="${cx + head[0] * scale}" cy="${base - head[1] * scale - 4}" r="4.800" fill="${color}"/>`; }
  return `<svg width="${width}" height="${height}" viewBox="0 0 130 76">${body}</svg>`;
}

function filtered(): Phrase[] {
  const text = S.search.trim();
  let items = S.library.filter((item) => (S.catFilter === "all" || item.category === S.catFilter));
  if (S.semantic) {
    const order = new Map(S.semantic.map((id, index) => [id, index]));
    items = items.filter((item) => order.has(item.id)).sort((a, b) => order.get(a.id)! - order.get(b.id)!);
  } else if (text) {
    items = filterByFeeling(items, text, S.categories);
  }
  return items;
}

const QUICK_SEARCH_TAGS = [
  "キック", "足上げ", "ターン", "ジャンプ", "ハート", "ピース", "指差し", "クラップ",
  "ウェーブ", "スクワット", "ステップ", "ウォーク", "キメ",
  "K-POP", "アイドル系", "J-POP", "ヒップホップ", "ハウス", "バラード",
];

export function toggleSearchTag(tag: string) {
  const input = document.getElementById("search") as HTMLInputElement;
  const targetNorm = normalizeToken(tag);
  const parts = S.search.trim() ? S.search.trim().split(/[\s　、，,・]+/).filter(Boolean) : [];
  const exists = parts.some((p) => normalizeToken(p) === targetNorm);
  const next = exists ? parts.filter((p) => normalizeToken(p) !== targetNorm) : [...parts, tag];
  S.search = next.join(" ");
  input.value = S.search;
  S.semantic = null;
  renderLibrary();
}

export function clearSearchQuery() {
  const input = document.getElementById("search") as HTMLInputElement;
  S.search = "";
  input.value = "";
  S.semantic = null;
  renderLibrary();
}

export function renderLibrary() {
  document.getElementById("libCount")!.textContent = String(S.library.length);
  const chips = [["all", "すべて"], ...Object.entries(S.categories)];
  const catChips = document.getElementById("catChips")!;
  const prevScroll = catChips.scrollLeft;
  catChips.innerHTML = chips.map(([key, label]) => `<span class="chip ${S.catFilter === key ? "on" : ""}" data-cat="${key}">${label}</span>`).join("");
  catChips.scrollLeft = prevScroll;

  const tagChips = document.getElementById("tagChips");
  if (tagChips) {
    const prevTagScroll = tagChips.scrollLeft;
    const activeNorms = activeQueryNormTokens(S.search);
    const clearChip = S.search.trim()
      ? `<span class="chip tag-chip tag-clear" data-clear-search="1" title="検索キーワードを解除します">× 解除</span>`
      : "";
    tagChips.innerHTML = clearChip + QUICK_SEARCH_TAGS.map(
      (tag) => `<span class="chip tag-chip ${activeNorms.has(normalizeToken(tag)) ? "on" : ""}" data-tag="${escapeHtml(tag)}">#${escapeHtml(tag)}</span>`
    ).join("");
    tagChips.scrollLeft = prevTagScroll;
  }

  const items = filtered();
  document.getElementById("grid")!.innerHTML = items.length ? items.map((item) => {
    const tagSummary = item.tags?.length ? `\nタグ: ${item.tags.join("・")}` : "";
    const descSummary = item.description ? `\n${item.description}` : "";
    return `
    <div class="card ${S.selectedPhrase === item.id ? "sel" : ""}" draggable="true" data-phrase="${item.id}" title="${escapeHtml(`${item.name}${descSummary}${tagSummary}\n（ドラッグで挿入 / 選択中右クリックで差し替え / ダブルクリックで編集）`)}">
      <div class="thumb"><i class="tag" style="background:${CAT_COLOR[item.category] ?? "var(--groove)"}"></i>${thumb(item)}<span class="bars">${item.bars}小節</span>${item.usage === "manual" ? '<span class="manual">任意</span>' : ""}</div>
      <div class="name">${escapeHtml(item.name)}</div>
      <div class="sub"><span>${S.categories[item.category] ?? ""}</span><span class="en">${[1, 2, 3, 4, 5].map((i) => `<i class="${i <= item.energy ? "f" : ""}"></i>`).join("")}</span></div>
    </div>`;
  }).join("") : `<div class="grid-empty">${S.library.length ? "該当するフレーズがありません" : "BONES 基準の SOMA BVH を読み込むと、ここに表示されます"}</div>`;
}

// ---- Dance order tab

const LOCK_SMALL = '<svg class="order-lock" width="10" height="12" viewBox="0 0 13 15"><rect y="6" width="13" height="9" rx="2" fill="#ffb59f"/><path d="M3 6V4a3.500 3.500 0 017 0v2" stroke="#ffb59f" stroke-width="2" fill="none"/></svg>';
let nowId: string | null = null;
let shownSelection: string | null = null;

export function updateOrderNow() {
  if (S.leftTab !== "order") { return; }
  const t = S.playhead;
  const id = clips().find((clip) => t >= clip.start - 1e-3 && t <= clip.end + 1e-3)?.id ?? null;
  if (id === nowId) { return; }
  nowId = id;
  const host = document.getElementById("orderList")!;
  host.querySelector(".order-row.current")?.classList.remove("current");
  const row = id ? host.querySelector<HTMLElement>(`[data-order="${id}"]`) : null;
  row?.classList.add("current");
  if (row && S.playing) { row.scrollIntoView({ block: "nearest" }); }
}

export function renderOrder() {
  document.getElementById("orderCount")!.textContent = String(clips().length);
  if (S.leftTab !== "order") { return; }
  const host = document.getElementById("orderList")!;
  const song = S.song;
  const list = clips();
  if (!song || !list.length) {
    host.innerHTML = '<div class="grid-empty">「自動配置」を押すか、フレーズをタイムラインに置くと、踊る順番がここに並びます</div>';
    return;
  }
  const joins = S.composed?.joins ?? [];
  let html = "";
  let lastSection = -2;
  list.forEach((clip, index) => {
    const item = phraseOf(clip.phraseId);
    if (!item) { return; }
    const sectionIndex = song.sections.findIndex((s) => clip.start >= s.start - 1e-3 && clip.start < s.end);
    if (sectionIndex !== lastSection) {
      lastSection = sectionIndex;
      const section = song.sections[sectionIndex];
      if (section) { html += `<div class="order-section ${section.kind === "chorus" ? "hot" : ""}"><b>${section.label}</b><span>${section.startBar + 1}–${section.endBar} 小節</span></div>`; }
    }
    const at = barBeatAt(clip.start);
    const trimmed = isTrimmed(clip, item);
    html += `<div class="order-row ${clip.id === S.selected ? "sel" : ""}" data-order="${clip.id}" title="クリックで選択 / 右クリックで差し替え / ダブルクリックでこの位置へ移動 / ドラッグで並び替え">
      <b class="order-no">${index + 1}</b>
      <span class="order-thumb">${thumb(item, 58, 42)}</span>
      <div class="order-main">
        <div class="order-name"><i style="background:${CAT_COLOR[item.category] ?? "var(--groove)"}"></i><span>${escapeHtml(item.name)}</span>${clip.locked ? LOCK_SMALL : ""}</div>
        <div class="order-sub">${at ? `${at.bar}小節 ${at.beat}拍` : ""} ・ ${formatTime(clip.start)} ・ ×${(clipDuration(clip, item) / (clip.end - clip.start)).toFixed(2)}${trimmed ? " ・ カット" : ""}</div>
        <div class="order-sub dim">位置 ${clip.x.toFixed(1)}, ${clip.z.toFixed(1)} ・ 向き ${Math.round(clip.facing)}°</div>
      </div></div>`;
    const join = joins.find((j) => j.from === clip.id);
    if (join && join.status === "cut") {
      html += `<div class="order-join">✂ 分割位置（連続）</div>`;
    } else if (join && join.status !== "overlap") {
      const warn = join.status === "fast" || join.status === "short" || join.status === "engine-error";
      const note = { fast: " ・ 速め", short: " ・ 短すぎ", "engine-error": " ・ 未接続", pending: " ・ 生成中" }[join.status] ?? "";
      html += `<div class="order-join ${warn ? "warn" : ""} ${join.status === "pending" ? "gen" : ""}"><svg width="16" height="9"><rect width="16" height="9" fill="url(#brickP)"/></svg>つなぎ ${join.seconds.toFixed(1)} 秒 ・ ${join.distance.toFixed(1)} m${note}</div>`;
    }
  });
  host.innerHTML = html;
  nowId = null;
  updateOrderNow();
  if (S.selected !== shownSelection) {
    shownSelection = S.selected;
    host.querySelector(".order-row.sel")?.scrollIntoView({ block: "nearest" });
  }
}

function startOrderDrag(host: HTMLElement, id: string, press: MouseEvent) {
  let line: HTMLElement | null = null;
  let ghost: HTMLElement | null = null;
  let target = 0;
  let pointerY = press.clientY;
  let grabbedAt = 0;
  let scroller = 0;
  const otherRows = () => [...host.querySelectorAll<HTMLElement>("[data-order]")].filter((row) => row.dataset.order !== id);
  const aim = () => {
    const rows = otherRows();
    target = rows.findIndex((row) => { const box = row.getBoundingClientRect(); return pointerY < box.top + box.height / 2; });
    if (target < 0) { target = rows.length; }
    const anchor = rows[Math.min(target, rows.length - 1)];
    if (!anchor || !line) { return; }
    const box = anchor.getBoundingClientRect();
    line.style.top = `${(target < rows.length ? box.top - 4 : box.bottom + 2) - host.getBoundingClientRect().top + host.scrollTop}px`;
  };
  const scrollNearEdges = () => {
    const box = host.getBoundingClientRect();
    const speed = pointerY < box.top + 40 ? -12 : pointerY > box.bottom - 40 ? 12 : 0;
    if (speed) { host.scrollTop += speed; aim(); }
  };
  const lift = () => {
    const source = host.querySelector<HTMLElement>(`[data-order="${id}"]`);
    if (!source) { return; }
    beginInteraction();
    holdCursor("drag-move");
    const box = source.getBoundingClientRect();
    grabbedAt = press.clientY - box.top;
    ghost = source.cloneNode(true) as HTMLElement;
    ghost.removeAttribute("data-order");
    ghost.classList.add("order-ghost");
    ghost.style.left = `${box.left}px`;
    ghost.style.width = `${box.width}px`;
    document.body.appendChild(ghost);
    source.classList.add("dragging");
    line = document.createElement("div");
    line.className = "order-drop";
    host.appendChild(line);
    scroller = window.setInterval(scrollNearEdges, 30);
  };
  const move = (event: MouseEvent) => {
    pointerY = event.clientY;
    if (!line && Math.abs(event.clientY - press.clientY) >= 5) { lift(); }
    if (!line || !ghost) { return; }
    ghost.style.top = `${event.clientY - grabbedAt}px`;
    aim();
  };
  const up = async () => {
    window.removeEventListener("mousemove", move);
    window.removeEventListener("mouseup", up);
    if (!line) { return; }
    window.clearInterval(scroller);
    ghost?.remove();
    line.remove();
    holdCursor(null);
    const current = clips().map((clip) => clip.id);
    const rest = current.filter((other) => other !== id);
    const order = [...rest.slice(0, target), id, ...rest.slice(target)];
    if (order.every((value, index) => value === current[index])) {
      host.querySelector(".order-row.dragging")?.classList.remove("dragging");
      endInteraction();
      return;
    }
    endInteraction(true);
    try { await applyChanges(await call("POST", "/api/clips/reorder", { order })); }
    catch (error) { toast((error as Error).message, "error"); await applyState(await call("GET", "/api/state")); }
  };
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up);
}

export function bindOrder() {
  const host = document.getElementById("orderList")!;
  host.addEventListener("mousedown", (event) => {
    if (event.button !== 0) { return; }
    const row = (event.target as HTMLElement).closest<HTMLElement>("[data-order]");
    if (!row) { return; }
    event.preventDefault();
    const id = row.dataset.order!;
    if (isDoublePress(id)) { jumpToClip(id); return; }
    S.selected = id;
    refresh();
    if (clips().find((clip) => clip.id === id)?.locked) { warnLockedDrag(event); return; }
    startOrderDrag(host, id, event);
  });
  host.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    if (!S.song) { return; }
    const list = clips();
    const row = (event.target as HTMLElement).closest<HTMLElement>("[data-order]");
    const rowClip = row ? list.find((c) => c.id === row.dataset.order) : undefined;
    const at = snapTime(S.playhead, event.altKey);
    const insAt = rowClip ? rowClip.end : at;
    const insWhere = rowClip ? `${list.indexOf(rowClip) + 1} 番の次に挿入` : exactInsertLabel(at);
    const insChoose = rowClip ? (phraseId: string) => insertPhrase(phraseId, rowClip.end) : (phraseId: string) => insertPhrase(phraseId, at, undefined, true);
    const picked = selectedClip();
    if (picked) {
      const name = phraseOf(picked.phraseId)?.name ?? "";
      openPhraseMenu(event.clientX, event.clientY, [
        { title: "モーションを差し替え", where: `${list.indexOf(picked) + 1} 番「${name}」を差し替え`, choose: (phraseId) => replaceClip(picked.id, phraseId), switchLabel: "挿入へ切替", at: picked.start, replaceClip: picked },
        { title: "モーションを挿入", where: insWhere, choose: insChoose, switchLabel: "差し替えに戻す", at: insAt },
      ]);
      return;
    }
    openInsertMenu(event.clientX, event.clientY, insWhere, insChoose, insAt);
  });
}

let candidateKey = "";
let candidateData: { phraseId: string; score: number }[] = [];

async function loadCandidates(id: string) {
  const key = `${id}:${S.composed?.version}:${(S.project?.arrangeFilter ?? []).join(",")}`;
  if (key === candidateKey) { return; }
  candidateKey = key;
  candidateData = [];
  try {
    candidateData = (await call("GET", `/api/clips/${id}/candidates`)).candidates;
    if (candidateKey === key) { renderInspector(); }
  } catch { /* candidates are optional */ }
}

const status = (join: any) => {
  if (!join) { return ""; }
  const label = ({ ok: ["ok", "● 良好"], cut: ["ok", "● 連続"], fast: ["w", "▲ 速め"], short: ["w", "▲ 短すぎ"], overlap: ["w", "▲ 重なり"], "engine-error": ["w", "▲ 未接続"], pending: ["gen", "生成中"] } as Record<string, [string, string]>)[join.status] ?? ["w", join.status];
  return `<span class="badge ${label[0]}">${label[1]}</span>`;
};

export function updateInspectorCutState() {
  const sel = selectedClip();
  const insideSel = sel && !sel.locked && S.playhead >= sel.start + 0.3 && S.playhead <= sel.end - 0.3 ? sel : undefined;
  const anyTarget = insideSel ?? clips().find((c) => !c.locked && S.playhead >= c.start + 0.3 && S.playhead <= c.end - 0.3);
  const splitBtn = document.getElementById("btnSplit");
  if (splitBtn) { splitBtn.classList.toggle("chip-accent", !!anyTarget); }
  const hint = document.getElementById("cutHint");
  if (!sel || !hint) { return; }
  const canCut = !sel.locked && S.playhead >= sel.start + 0.3 && S.playhead <= sel.end - 0.3;
  for (const id of ["cutBefore", "cutSplit", "cutAfter"]) {
    const btn = document.getElementById(id) as HTMLButtonElement | null;
    if (btn) { btn.disabled = !canCut; }
  }
  hint.textContent = sel.locked
    ? "ロック中のためカットできません"
    : canCut
      ? `白いバー（+${(S.playhead - sel.start).toFixed(2)} 秒）の位置でカット・分割できます`
      : "白いバーをこのフレーズの内側（端から 0.3 秒以上）に置くとカット・分割できます";
}

export function renderInspector() {
  const host = document.getElementById("inspector")!;
  const clip = selectedClip();
  const item = clip && phraseOf(clip.phraseId);
  if (!clip || !item) {
    const composed = S.composed;
    host.classList.remove("is-locked");
    host.innerHTML = `<div class="insp-empty"><div class="big">フレーズを選択してください</div>
      <p>タイムラインのフレーズを選ぶと、入れ替え・カット・速度・音はめ・位置を編集できます。補間区間を左右にドラッグすると、曲に合う複数のモーションで自動的に埋められます。</p>
      ${composed ? `<p class="stat">接続 ${composed.joins.length} 件　再生成 ${composed.generated} ・ 再利用 ${composed.reused}<br>計算 ${composed.seconds.toFixed(2)} 秒</p>` : ""}
      ${composed?.engineError ? `<p class="err">${escapeHtml(composed.engineError)}</p><button class="btn" data-open-installer>環境の導入を開く</button>` : ""}
      ${collisionList()}
      ${S.song ? `<p class="stat">拍: ${S.song.engines.beats} ・ 区間: ${S.song.engines.sections}</p>` : ""}</div>`;
    updateInspectorCutState();
    return;
  }
  void loadCandidates(clip.id);
  const report = S.composed?.clips[clip.id];
  const joins = S.composed?.joins ?? [];
  const before = joins.find((j) => j.to === clip.id);
  const after = joins.find((j) => j.from === clip.id);
  const length = clip.end - clip.start;
  const [trimStart, trimEnd] = clipTrim(clip, item);
  const span = trimEnd - trimStart;
  const trimmed = isTrimmed(clip, item);
  const speed = span / length;
  const position = (clamp(speed, 0.5, 2.0) - 0.5) / 1.5;
  const start = (S.song!.bars.filter((b) => b <= clip.start + 1e-3).length).toString();
  const joinRow = (label: string, join: any) => join
    ? `<div class="conn"><span>${label}</span><span><b class="n">${join.distance.toFixed(1)} m</b> ／ <b class="n">${join.seconds.toFixed(1)} 秒</b>${join.via ? " ・ 迂回" : ""}</span>${status(join)}${join.status === "cut" || join.status === "overlap" ? "<span></span>" : `<span class="conn-acts">${join.seconds >= 2.15 ? `<button class="mini-btn" data-fill-join="${join.from}>${join.to}" title="この補間区間に曲に合ったモーションを自動で割り当てます（タイムライン上で補間区間をドラッグしても実行できます）">埋める</button>` : ""}<button class="mini-btn" data-options="${join.from}" title="長さの違うつなぎを作って比べます">長さ</button></span>`}</div>`
    : `<div class="conn"><span>${label}</span><span class="muted">なし</span><span></span><span></span></div>`;
  const pins = clip.pins ?? [];
  host.innerHTML = `
  <div class="sect"><div class="ttl">
    <svg width="10" height="44"><rect width="6" height="44" rx="3" fill="${CAT_COLOR[item.category]}"/></svg>
    <div class="ttl-main"><div class="nm">${escapeHtml(item.name)}</div><div class="ds">${S.categories[item.category]} ・ ${start}小節目から ・ 素材 ${item.bars}小節${trimmed ? "（カット中）" : ""}</div></div>
    <button class="lock ${clip.locked ? "" : "off"}" id="toggleLock" title="ロックすると移動、伸縮、入れ替え、削除を防ぎます。前に挿入しても時刻は変わりません（L）">${clip.locked ? "ロック中" : "ロックする"}</button>
  </div></div>
  <div class="sect"><div class="st">入れ替え候補<small>右クリックでも差し替え可</small></div>
    ${candidateData.slice(0, 2).map((c) => { const p = phraseOf(c.phraseId); return p ? `<div class="cand" data-swap="${c.phraseId}"><i class="dot" style="background:${CAT_COLOR[p.category]}"></i>${escapeHtml(p.name)}<span class="m">${Math.round(c.score * 100)}%</span><span class="sw">入替</span></div>` : ""; }).join("") || '<div class="muted small-pad">候補を計算中です</div>'}
  </div>
  <div class="sect"><div class="st">長さと速度<div class="seg" id="speedSeg"><span class="${clip.speedMode === "auto" ? "on" : ""}" data-mode="auto">自動</span><span class="${clip.speedMode !== "auto" ? "on" : ""}" data-mode="manual">手動</span></div></div>
    <div class="slider"><div class="band" style="left:20%;width:36.700%"></div><div class="tr"></div><div class="fi" style="width:${position * 100}%"></div><div class="kn" style="left:${position * 100}%"></div><input type="range" id="speedRange" min="0.5" max="2" step="0.01" value="${speed}"></div>
    <div class="row"><span>再生速度 <b id="speedValue">×${speed.toFixed(2)}</b></span><span>配置 <b id="lengthValue">${(length / S.song!.barSeconds).toFixed(2)}</b> 小節</span></div>
    <div class="row" style="margin-top:2px"><span>使用範囲 <b>${span.toFixed(2)}</b> 秒 → <b id="secondsValue">${length.toFixed(2)}</b> 秒</span><span>元 <b>${item.sourceBpm.toFixed(0)}</b> BPM</span></div>
  </div>
  <div class="sect"><div class="st">使う範囲・カット${trimmed ? `<button class="mini-btn" id="resetTrim" type="button">全長（${item.duration.toFixed(2)}秒）に戻す</button>` : `<small>全長 ${item.duration.toFixed(2)} 秒 ・ Shift+端ドラッグも可</small>`}</div>
    <div class="fields two-col">
      <label class="f"><span class="fl">開始 秒</span><input type="number" step="0.1" min="0" max="${Math.max(0, trimEnd - 0.3).toFixed(2)}" id="trimStartInput" value="${trimStart.toFixed(2)}"></label>
      <label class="f"><span class="fl">終了 秒</span><input type="number" step="0.1" min="${Math.min(item.duration, trimStart + 0.3).toFixed(2)}" max="${item.duration.toFixed(2)}" id="trimEndInput" value="${trimEnd.toFixed(2)}"></label>
    </div>
    <div class="cut-actions">
      <button class="mini-btn" id="cutBefore" type="button" title="白いバーより前の部分を切り落とします">前をカット</button>
      <button class="mini-btn cut-pri" id="cutSplit" type="button" title="白いバーの位置で前半・後半の 2 つに分割します（S）">✂ 白いバーで分割</button>
      <button class="mini-btn" id="cutAfter" type="button" title="白いバーより後ろの部分を切り落とします">後ろをカット</button>
    </div>
    <div class="muted cut-hint" id="cutHint"></div>
  </div>
  <div class="sect"><div class="st">音はめ<small>一致 <b class="n" style="color:var(--text)">${report ? `${report.matched} / ${report.candidates}` : "--"}</b></small></div>
    <div class="slider"><div class="tr"></div><div class="fi" style="width:${clip.match}%"></div><div class="kn" style="left:${clip.match}%"></div><input type="range" id="matchRange" min="0" max="100" step="1" value="${clip.match}"></div>
    <div class="row"><span>強さ <b id="matchValue">${clip.match}</b></span><span title="×0.70〜1.60 が目安です。手動で合わせたアクセントはこの範囲を超えても固定します">局所速度 <b class="${report && (report.localSpeed[0] < 0.695 || report.localSpeed[1] > 1.605) ? "warn-text" : ""}">${report ? `×${report.localSpeed[0].toFixed(2)}〜${report.localSpeed[1].toFixed(2)}` : "--"}</b></span></div>
    <div class="row pin-row">${pins.length ? `<span>手動で合わせた <b>${pins.length}</b> 点</span><button class="mini-btn" id="clearPins">すべて解除</button>` : '<span class="muted">振りの ◇ を音の ◆ へドラッグすると、その音に固定します</span>'}</div>
  </div>
  <div class="sect"><div class="st">ダンス位置<small>3D ・ 上面図でドラッグ可</small></div>
    <div class="fields">
      <label class="f"><span class="fl">左右</span><input type="number" step="0.1" data-field="x" value="${clip.x}"></label>
      <label class="f"><span class="fl">奥行</span><input type="number" step="0.1" data-field="z" value="${clip.z}"></label>
      <label class="f"><span class="fl">向き</span><input type="number" step="5" data-field="facing" value="${clip.facing}"></label>
    </div>
  </div>
  <div class="sect"><div class="st">MotionBricks 接続<small>更新 ${S.composed?.generated ?? 0} ・ 再利用 ${S.composed?.reused ?? 0}</small></div>${joinRow("前から", before)}${joinRow("次へ", after)}</div>`;
  bindInspector(clip.id, item);
  host.classList.toggle("is-locked", clip.locked);
  host.querySelectorAll<HTMLInputElement>("input").forEach((input) => (input.disabled = clip.locked));
  updateInspectorCutState();
}

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

function bindInspector(id: string, item: Phrase) {
  const clip = selectedClip()!;
  const [trimStart, trimEnd] = clipTrim(clip, item);
  const span = trimEnd - trimStart;
  const curSpeed = span / (clip.end - clip.start);
  const patch = async (changes: Record<string, unknown>) => {
    try { await applyChanges(await call("PATCH", `/api/clips/${id}`, changes)); }
    catch (error) { toast((error as Error).message, "error"); }
  };
  document.getElementById("toggleLock")!.onclick = () => patch({ locked: !clip.locked });
  const clearPins = document.getElementById("clearPins");
  if (clearPins) { clearPins.onclick = () => patch({ pins: [] }); }
  document.querySelectorAll<HTMLElement>("#inspector [data-options]").forEach((el) => (el.onclick = () => openJoinOptions(el.dataset.options!)));
  document.querySelectorAll<HTMLElement>("#inspector [data-fill-join]").forEach((el) => (el.onclick = () => {
    const [fromId, toId] = (el.dataset.fillJoin ?? "").split(">");
    const list = clips();
    const fromClip = list.find((c) => c.id === fromId);
    const toClip = list.find((c) => c.id === toId);
    if (fromClip && toClip) { void fillGapRange(fromClip.end, toClip.start); }
  }));
  document.querySelectorAll<HTMLElement>("[data-swap]").forEach((el) => (el.onclick = () => replaceClip(id, el.dataset.swap!)));
  document.querySelectorAll<HTMLElement>("#speedSeg span").forEach((el) => (el.onclick = () => patch({ speedMode: el.dataset.mode })));
  const others = clips().filter((c) => c.id !== clip.id);
  const previousEnd = others.filter((c) => c.start < clip.start).reduce((best, c) => (best === null || c.end > best ? c.end : best), null as number | null);
  const following = others.filter((c) => c.start > clip.start).reduce((best, c) => (best === null || c.start < best ? c.start : best), null as number | null);
  const lowest = previousEnd === null ? 0 : previousEnd + 0.02;
  const highest = following === null ? S.song!.duration : following - 0.02;
  const endFor = (speed: number) => Math.min(clip.start + span / speed, highest);
  const speedRange = document.getElementById("speedRange") as HTMLInputElement;
  speedRange.oninput = () => {
    const speed = Number(speedRange.value);
    const end = endFor(speed);
    const actual = span / (end - clip.start);
    document.getElementById("speedValue")!.textContent = `×${actual.toFixed(2)}`;
    document.getElementById("lengthValue")!.textContent = ((end - clip.start) / S.song!.barSeconds).toFixed(2);
    document.getElementById("secondsValue")!.textContent = (end - clip.start).toFixed(2);
    speedRange.parentElement!.querySelector<HTMLElement>(".kn")!.style.left = `${((speed - 0.5) / 1.5) * 100}%`;
    speedRange.parentElement!.querySelector<HTMLElement>(".fi")!.style.width = `${((speed - 0.5) / 1.5) * 100}%`;
  };
  speedRange.onchange = () => patch({ end: Math.round(endFor(Number(speedRange.value)) * 1000) / 1000, speedMode: "manual" });
  const trimStartInput = document.getElementById("trimStartInput") as HTMLInputElement;
  const trimEndInput = document.getElementById("trimEndInput") as HTMLInputElement;
  trimStartInput.onchange = () => {
    const nextStart = clamp(Number(trimStartInput.value), 0, trimEnd - 0.3);
    if (!Number.isFinite(nextStart)) { return; }
    const newStart = clamp(clip.start + (nextStart - trimStart) / curSpeed, lowest, clip.end - 0.3);
    void patch({ start: Math.round(newStart * 1000) / 1000, trimStart: Math.round(nextStart * 1000) / 1000, trimEnd: Math.round(trimEnd * 1000) / 1000, speedMode: "manual" });
  };
  trimEndInput.onchange = () => {
    const nextEnd = clamp(Number(trimEndInput.value), trimStart + 0.3, item.duration);
    if (!Number.isFinite(nextEnd)) { return; }
    const newEnd = clamp(clip.start + (nextEnd - trimStart) / curSpeed, clip.start + 0.3, highest);
    void patch({ end: Math.round(newEnd * 1000) / 1000, trimStart: Math.round(trimStart * 1000) / 1000, trimEnd: Math.round(nextEnd * 1000) / 1000, speedMode: "manual" });
  };
  const resetTrim = document.getElementById("resetTrim");
  if (resetTrim) {
    resetTrim.onclick = () => {
      const newStart = clamp(clip.start - trimStart / curSpeed, lowest, clip.end - 0.3);
      const newEnd = clamp(newStart + item.duration / curSpeed, newStart + 0.3, highest);
      void patch({ start: Math.round(newStart * 1000) / 1000, end: Math.round(newEnd * 1000) / 1000, trimStart: null, trimEnd: null, speedMode: "manual" });
    };
  }
  document.getElementById("cutBefore")!.onclick = () => cutClip(id, S.playhead, "before");
  document.getElementById("cutSplit")!.onclick = () => cutClip(id, S.playhead, "split");
  document.getElementById("cutAfter")!.onclick = () => cutClip(id, S.playhead, "after");
  const matchRange = document.getElementById("matchRange") as HTMLInputElement;
  matchRange.oninput = () => {
    document.getElementById("matchValue")!.textContent = matchRange.value;
    matchRange.parentElement!.querySelector<HTMLElement>(".kn")!.style.left = `${matchRange.value}%`;
    matchRange.parentElement!.querySelector<HTMLElement>(".fi")!.style.width = `${matchRange.value}%`;
  };
  matchRange.onchange = () => patch({ match: Number(matchRange.value) });
  document.querySelectorAll<HTMLInputElement>("[data-field]").forEach((input) => (input.onchange = () => {
    const value = Number(input.value);
    if (Number.isFinite(value)) { void patch({ [input.dataset.field!]: value }); }
  }));
  document.querySelectorAll<HTMLElement>("#inspector .cand").forEach((el) => el.style.cursor = "pointer");
}

// ---- Library interactions and dialogs

export const bridge = () => (window as any).dsb as { pick: (kind: string) => Promise<string[]>; reveal?: (path: string) => void } | undefined;

export async function ask(title: string, hint: string, action: "file-motion" | "file-song" | "file-model" | "file-pmx" | "text"): Promise<string[] | null> {
  const native = bridge();
  if (native && action !== "text") {
    const picked = await native.pick(action);
    return picked && picked.length ? picked : null;
  }
  return new Promise((resolve) => {
    const modal = document.getElementById("modal")!;
    modal.hidden = false;
    modal.innerHTML = `<div class="dialog"><h3>${title}</h3><p>${hint}</p><input id="pathInput" spellcheck="false" placeholder="D:\\..."><div class="dialog-actions"><button class="btn" id="cancelPath">キャンセル</button><button class="btn pri" id="okPath">読み込む</button></div></div>`;
    const input = document.getElementById("pathInput") as HTMLInputElement;
    input.focus();
    const close = (value: string[] | null) => { modal.hidden = true; modal.innerHTML = ""; resolve(value); };
    document.getElementById("cancelPath")!.onclick = () => close(null);
    document.getElementById("okPath")!.onclick = () => close(input.value.trim() ? [input.value.trim().replace(/^"|"$/g, "")] : null);
    input.onkeydown = (e) => { if (e.key === "Enter") { document.getElementById("okPath")!.click(); } if (e.key === "Escape") { close(null); } };
  });
}

export async function importMotion() {
  const paths = await ask("モーションデータを読み込み", "BONES 基準の SOMA BVH ファイル、または BVH が入ったフォルダのパスを入力します。", "file-motion");
  if (!paths) { return; }
  const result = await withBusy("モーションを解析しています", () => call("POST", "/api/library/import", { paths }));
  if (!result) { return; }
  await applyState(result.state);
  if (result.rejected.length) {
    showMessage("読み込めなかったファイル", result.rejected.map((r: any) => `<li><b>${escapeHtml(r.path.split(/[\\/]/).pop())}</b><br>${escapeHtml(r.reason)}</li>`).join(""));
  } else { toast(`${result.added.length} 件のフレーズを追加しました`); }
}

export async function loadSong(factor?: number) {
  let path = S.song?.path;
  if (factor === undefined) {
    const picked = await ask("曲を読み込む", "WAV / FLAC / MP3 のパスを入力します。", "file-song");
    if (!picked) { return; }
    path = picked[0];
  }
  const state = await withBusy("曲を解析しています（数十秒かかります）", () => call("POST", "/api/song", { path, factor: factor ?? null }));
  if (state) {
    S.viewStart = 0;
    await applyState(state);
    if (state.restored) { toast(`この曲の前回の配置（${state.restored} フレーズ）を復元しました`); }
  }
}

export function showMessage(title: string, html: string, extra?: { label: string; run: () => void }) {
  const modal = document.getElementById("modal")!;
  modal.hidden = false;
  modal.innerHTML = `<div class="dialog"><h3>${title}</h3><ul class="msg">${html}</ul><div class="dialog-actions">${extra ? `<button class="btn" id="extraMsg">${extra.label}</button>` : ""}<button class="btn pri" id="closeMsg">閉じる</button></div></div>`;
  document.getElementById("closeMsg")!.onclick = () => { modal.hidden = true; modal.innerHTML = ""; };
  if (extra) { document.getElementById("extraMsg")!.onclick = extra.run; }
}

export const closeModal = () => { const modal = document.getElementById("modal")!; modal.hidden = true; modal.innerHTML = ""; };

/** Makes the join after a phrase at several lengths and shows how each came out, so one can be taken. */
export async function openJoinOptions(fromId: string) {
  const data = await withBusy("長さの違うつなぎを作って比べています", () => call("GET", `/api/joins/${fromId}/options`));
  if (!data) { return; }
  const from = clips().find((c) => c.id === data.from);
  const name = (from && phraseOf(from.phraseId)?.name) ?? "";
  const warn = (on: boolean) => (on ? ' <span class="badge w inline">▲</span>' : "");
  const rows = data.options.map((o: any) => `<tr class="${o.current ? "current" : ""}">
    <td><b class="n">${o.seconds.toFixed(2)}</b> 秒 ・ ${o.bars.toFixed(2)} 小節</td>
    <td>×${o.clipSpeed.toFixed(2)}${warn(o.flags.includes("speed"))}</td>
    <td>${o.moveSpeed.toFixed(2)} m/秒${warn(o.flags.includes("fast"))}</td>
    <td>${o.maxJointSpeed.toFixed(2)} m/秒</td>
    <td>${(o.minHeight * 100).toFixed(1)} cm</td>
    <td>${o.best ? '<span class="badge ok inline">● 最良</span>' : ""}${o.current ? '<span class="muted">現在</span>' : ""}</td>
    <td>${o.current ? "" : `<button class="mini-btn" data-take="${o.end}" ${data.locked ? "disabled" : ""}>採用</button>`}</td></tr>`).join("");
  const modal = document.getElementById("modal")!;
  modal.hidden = false;
  modal.innerHTML = `<div class="dialog wide"><h3>つなぎの長さの候補</h3>
    <p>長さを変えると、前のフレーズ「${escapeHtml(name)}」の終わりの時刻と速度が変わります。次のフレーズは動きません。「最良」は、警告のない候補のうち最大関節速度が最も低いものです。${data.locked ? "<br>前のフレーズがロック中のため、採用できません。" : ""}</p>
    <table class="edit options"><thead><tr><th>つなぎの長さ</th><th>前のフレーズ</th><th>移動速度</th><th>最大関節速度</th><th>最も低い関節</th><th></th><th></th></tr></thead><tbody>${rows}</tbody></table>
    <div class="dialog-actions"><button class="btn pri" id="closeOptions">閉じる</button></div></div>`;
  document.getElementById("closeOptions")!.onclick = closeModal;
  modal.querySelectorAll<HTMLButtonElement>("[data-take]").forEach((button) => (button.onclick = async () => {
    closeModal();
    try { await applyChanges(await call("PATCH", `/api/clips/${data.from}`, { end: Number(button.dataset.take), speedMode: "manual" })); toast("つなぎの長さを変えました（Ctrl+Z で戻せます）"); }
    catch (error) { toast((error as Error).message, "error"); }
  }));
}

export function openExport() {
  const modal = document.getElementById("modal")!;
  if (!S.composed) { toast("書き出すモーションがまだありません", "error"); return; }
  const several = S.cast.filter((member) => member.version > 0).length > 1;
  modal.hidden = false;
  modal.innerHTML = `<div class="dialog"><h3>書き出し</h3>
    <p>現在のタイムライン全体を 30fps で書き出します。保存先は data/exports です。${several ? "ダンサーごとに 1 ファイルになります。" : ""}</p>
    <div class="export-kinds">
      <button class="btn pri" data-format="bvh"><b>BVH</b><small>SOMA 骨格</small></button>
      <button class="btn" data-format="fbx"><b>FBX</b><small>SOMA 骨格</small></button>
      <button class="btn" data-format="vrma"><b>VRMA</b><small>VRM モデル用</small></button>
      <button class="btn" data-format="vmd"><b>VMD</b><small>MMD 用</small></button>
      ${hasModel() ? '<button class="btn" data-format="glb"><b>GLB</b><small>表示中のモデル＋動き</small></button>' : ""}
    </div>
    <label class="export-pmx">BVH / FBX の基準の姿勢（取り込み先のモデルに合わせます）<span class="path-row">
      <select id="restKind"><option value="t">T ポーズ</option><option value="a">A ポーズ</option><option value="bones">SOMA のまま（BONES 基準）</option></select>
      <span class="rest-arm" id="restArmBox">腕の角度 <input id="restArm" type="number" min="5" max="85" step="1" value="${Number(localStorage.getItem("dsb.restArm")) || 45}"> °</span></span></label>
    <label class="export-pmx">VMD を合わせる MMD モデル（PMX、任意）<span class="path-row"><input id="pmxPath" spellcheck="false" placeholder="指定しないと標準的な体型と A ポーズを仮定します" value="${escapeHtml(localStorage.getItem("dsb.pmx") ?? "")}"><button class="mini-btn" id="pickPmx">選ぶ</button></span></label>
    <div class="dialog-actions"><button class="btn" id="cancelExport">閉じる</button></div></div>`;
  document.getElementById("cancelExport")!.onclick = closeModal;
  const pmxInput = document.getElementById("pmxPath") as HTMLInputElement;
  const restKind = document.getElementById("restKind") as HTMLSelectElement;
  const restArm = document.getElementById("restArm") as HTMLInputElement;
  restKind.value = localStorage.getItem("dsb.rest") ?? "t";
  const showArm = () => { document.getElementById("restArmBox")!.hidden = restKind.value !== "a"; };
  restKind.onchange = showArm;
  showArm();
  document.getElementById("pickPmx")!.onclick = async () => {
    const picked = bridge() ? await bridge()!.pick("file-pmx") : null;
    if (picked?.length) { pmxInput.value = picked[0]; } else if (!bridge()) { pmxInput.focus(); }
  };
  modal.querySelectorAll<HTMLElement>("[data-format]").forEach((el) => (el.onclick = async () => {
    const format = el.dataset.format!;
    const pmx = pmxInput.value.trim().replace(/^"|"$/g, "");
    localStorage.setItem("dsb.pmx", pmx);
    const rest = restKind.value, armDegrees = Math.min(85, Math.max(5, Number(restArm.value) || 45));
    localStorage.setItem("dsb.rest", rest);
    localStorage.setItem("dsb.restArm", String(armDegrees));
    closeModal();
    const result = await withBusy(`${format.toUpperCase()} を書き出しています`, () => (format === "glb" ? exportModel() : call("POST", "/api/export", { format, pmx: format === "vmd" && pmx ? pmx : null, rest, armDegrees })));
    if (!result) { return; }
    const list = result.files.map((file: any) => `<li>${escapeHtml(file.path)}<br>${file.dancer ? `${escapeHtml(file.dancer)} ・ ` : ""}${(file.bytes / 1048576).toFixed(1)} MB</li>`).join("");
    const reveal = bridge()?.reveal;
    showMessage("書き出しました", list, reveal ? { label: "フォルダを開く", run: () => reveal(result.path) } : undefined);
  }));
}

export function openPhrase(id: string) {
  const item = phraseOf(id);
  if (!item) { return; }
  const modal = document.getElementById("modal")!;
  const provenance = (item as any).provenance ?? {};
  const close = () => { modal.hidden = true; modal.innerHTML = ""; };
  modal.hidden = false;
  modal.innerHTML = `<div class="dialog"><h3>フレーズの編集</h3>
    <div class="form">
      <label>名前<input id="pName" value="${escapeHtml(item.name)}"></label>
      <label>カテゴリ<select id="pCategory">${Object.entries(S.categories).map(([k, v]) => `<option value="${k}" ${k === item.category ? "selected" : ""}>${v}</option>`).join("")}</select></label>
      <label>強さ（1〜5）<input id="pEnergy" type="number" min="1" max="5" value="${item.energy}"></label>
      <label>用途<select id="pUsage"><option value="auto" ${item.usage === "auto" ? "selected" : ""}>自動配置に使う</option><option value="manual" ${item.usage === "manual" ? "selected" : ""}>任意追加のみ</option></select></label>
      <label class="full">タグ（カンマ区切り）<input id="pTags" value="${escapeHtml(item.tags.join(", "))}"></label>
      <label class="full">説明（意味検索に使います）<input id="pDescription" value="${escapeHtml(item.description)}"></label>
      <label class="full">振りのアクセント（秒、カンマ区切り。全長 ${item.duration.toFixed(2)} 秒）<input id="pAccents" value="${item.accents.join(", ")}"></label>
      <label>権利者<input id="pOwner" value="${escapeHtml(provenance.owner ?? "")}"></label>
      <label>ライセンス<input id="pLicense" value="${escapeHtml(provenance.license ?? "")}"></label>
      <label class="full">クレジット表記<input id="pCredit" value="${escapeHtml(provenance.credit ?? "")}"></label>
    </div>
    <div class="dialog-actions"><button class="btn danger" id="pDelete">削除</button><div class="grow"></div><button class="btn" id="pCancel">キャンセル</button><button class="btn pri" id="pSave">保存</button></div></div>`;
  const value = (name: string) => (document.getElementById(name) as HTMLInputElement).value;
  document.getElementById("pCancel")!.onclick = close;
  document.getElementById("pDelete")!.onclick = async () => {
    close();
    const state = await withBusy("削除しています", () => call("DELETE", `/api/library/${id}`));
    if (state) { await applyState(state); }
  };
  document.getElementById("pSave")!.onclick = async () => {
    const list = (text: string) => text.split(/[,、]/).map((part) => part.trim()).filter(Boolean);
    const accents = list(value("pAccents")).map(Number);
    if (accents.some((v) => !Number.isFinite(v))) { toast("アクセントは数値で入力してください", "error"); return; }
    const body = { name: value("pName").trim() || item.name, category: value("pCategory"), energy: Number(value("pEnergy")), usage: value("pUsage"), tags: list(value("pTags")),
      description: value("pDescription"), accents, provenance: { ...provenance, owner: value("pOwner"), license: value("pLicense"), credit: value("pCredit") } };
    close();
    const saved = await withBusy("保存しています", async () => { await call("PATCH", `/api/library/${id}`, body); await applyState(await call("GET", "/api/state")); return true; });
    if (saved) { toast("フレーズを更新しました"); }
  };
}

const KIND_NAMES: Record<string, string> = { intro: "イントロ", verse: "Aメロ", build: "Bメロ", chorus: "サビ", interlude: "間奏", outro: "アウトロ" };

export function openFlow() {
  const modal = document.getElementById("modal")!;
  const project = S.project!;
  const catOptions = (selected: string) => Object.entries(S.categories).map(([k, v]) => `<option value="${k}" ${k === selected ? "selected" : ""}>${v}</option>`).join("");
  const flowRows = Object.entries(project.flow).map(([kind, rule]) => `<tr data-kind="${kind}"><th>${KIND_NAMES[kind]}</th>
    <td><input data-k="text" value="${escapeHtml(rule.text)}"></td>
    ${[0, 1, 2].map((i) => `<td><select data-c="${i}">${catOptions(rule.categories[i] ?? "groove")}</select></td>`).join("")}
    <td><input data-k="energy" type="number" min="1" max="5" value="${rule.energy}"></td><td><input data-k="move" type="number" min="0" max="5" step="0.1" value="${rule.move}"></td></tr>`).join("");
  const song = S.song;
  const sectionRows = song ? song.sections.map((s, i) => `<tr data-index="${i}"><td><input type="number" min="1" data-start value="${s.startBar + 1}"></td><td><select data-kind>${Object.entries(KIND_NAMES).map(([k, v]) => `<option value="${k}" ${k === s.kind ? "selected" : ""}>${v}</option>`).join("")}</select></td><td class="muted">${Math.floor(s.start / 60)}:${String(Math.floor(s.start % 60)).padStart(2, "0")}</td><td><button class="mini-btn" data-remove="${i}">削除</button></td></tr>`).join("") : "";
  modal.hidden = false;
  modal.innerHTML = `<div class="dialog wide"><h3>流れの辞書を編集</h3>
    <table class="edit"><thead><tr><th></th><th>流れの説明</th><th>第1候補</th><th>第2候補</th><th>第3候補</th><th>強さ</th><th>移動 m</th></tr></thead><tbody id="flowBody">${flowRows}</tbody></table>
    ${song ? `<h3 class="second">曲の構成の修正</h3><table class="edit narrow"><thead><tr><th>開始小節</th><th>種類</th><th>時刻</th><th></th></tr></thead><tbody id="sectionBody">${sectionRows}</tbody></table><button class="mini-btn add" id="addSection">＋ 区間を追加</button>` : ""}
    <div class="dialog-actions"><button class="btn" id="cancelFlow">キャンセル</button><button class="btn pri" id="saveFlow">保存</button></div></div>`;
  const close = () => { modal.hidden = true; modal.innerHTML = ""; };
  document.getElementById("cancelFlow")!.onclick = close;
  const sectionBody = document.getElementById("sectionBody");
  sectionBody?.addEventListener("click", (event) => { const row = (event.target as HTMLElement).closest("[data-remove]"); if (row) { row.closest("tr")!.remove(); } });
  const addButton = document.getElementById("addSection");
  if (addButton) {
    addButton.onclick = () => {
      const row = document.createElement("tr");
      row.innerHTML = `<td><input type="number" min="1" data-start value="${(song!.sections.at(-1)?.startBar ?? 0) + 5}"></td><td><select data-kind>${Object.entries(KIND_NAMES).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select></td><td class="muted"></td><td><button class="mini-btn" data-remove="new">削除</button></td>`;
      sectionBody!.appendChild(row);
    };
  }
  document.getElementById("saveFlow")!.onclick = async () => {
    const flow: Record<string, any> = {};
    modal.querySelectorAll<HTMLElement>("#flowBody tr").forEach((row) => {
      const get = (key: string) => row.querySelector<HTMLInputElement>(`[data-k="${key}"]`)!.value;
      const categories = [...new Set([0, 1, 2].map((i) => row.querySelector<HTMLSelectElement>(`[data-c="${i}"]`)!.value))];
      flow[row.dataset.kind!] = { text: get("text"), categories, energy: Number(get("energy")), move: Number(get("move")) };
    });
    close();
    const saved = await withBusy("保存しています", async () => {
      await applyChanges(await call("PUT", "/api/flow", flow));
      if (!song || !sectionBody) { return true; }
      const rows = [...sectionBody.querySelectorAll("tr")].map((row) => ({ start: Number(row.querySelector<HTMLInputElement>("[data-start]")!.value) - 1, kind: row.querySelector<HTMLSelectElement>("[data-kind]")!.value })).sort((a, b) => a.start - b.start);
      const sections = rows.map((row, i) => ({ startBar: row.start, endBar: rows[i + 1]?.start ?? song.bars.length, kind: row.kind }));
      await applyState(await call("PUT", "/api/song/sections", { sections }));
      return true;
    });
    if (saved) { toast("流れの辞書を保存しました"); }
  };
}

export async function semanticSearch(query: string) {
  if (!query.trim()) { S.semantic = null; refresh(); return; }
  const result = await withBusy("意味検索を実行しています", () => call("POST", "/api/library/search", { query, limit: 24 }));
  if (result) { S.semantic = result.results.map((r: any) => r.id); refresh(); }
}

export function openArrangeFilter() {
  if (!S.project) { return; }
  const modal = document.getElementById("modal")!;
  const presetAll = new Set<string>([
    ...GENRE_TAG_OPTIONS,
    ...MOOD_TAG_OPTIONS,
    ...ACTION_TAG_OPTIONS,
  ].map((t) => normalizeToken(t)));

  const initial = S.project.arrangeFilter ?? [];
  const selected = new Set<string>();
  const customInit: string[] = [];
  for (const tag of initial) {
    const norm = normalizeToken(tag);
    const matchedPreset = [...GENRE_TAG_OPTIONS, ...MOOD_TAG_OPTIONS, ...ACTION_TAG_OPTIONS].find((p) => normalizeToken(p) === norm);
    if (matchedPreset && presetAll.has(norm)) {
      selected.add(matchedPreset);
    } else if (tag.trim()) {
      customInit.push(tag.trim());
    }
  }

  const renderGroup = (list: readonly string[]) =>
    list.map((tag) => `<button type="button" class="chip filter-chip ${selected.has(tag) ? "on" : ""}" data-ftag="${escapeHtml(tag)}">${escapeHtml(tag)}</button>`).join("");

  modal.hidden = false;
  modal.innerHTML = `<div class="dialog wide arrange-filter-dialog">
    <h3>自動配置・入れ替えの対象（ジャンル・タグ絞り込み）</h3>
    <p>選んだジャンルや動きタグに一致するモーションだけを使って「自動配置」「ロック以外を入れ替え」「表示範囲だけ自動配置」「補間区間の自動埋め」を行います。同じ欄の中では OR、別欄（例: K-POP × キック）は掛け合わせ（AND）で絞り込みます。</p>
    <div class="af-group">
      <div class="af-title">ジャンル</div>
      <div class="af-chips">${renderGroup(GENRE_TAG_OPTIONS)}</div>
    </div>
    <div class="af-group">
      <div class="af-title">雰囲気</div>
      <div class="af-chips">${renderGroup(MOOD_TAG_OPTIONS)}</div>
    </div>
    <div class="af-group">
      <div class="af-title">動き・特徴・構成タグ</div>
      <div class="af-chips">${renderGroup(ACTION_TAG_OPTIONS)}</div>
    </div>
    <label class="af-custom">その他のタグ・キーワード（カンマ区切り、任意）
      <input id="afCustomInput" spellcheck="false" placeholder="例: 蹴り, 猫の手, ボックスステップ" value="${escapeHtml(customInit.join(", "))}">
    </label>
    <div class="af-preview" id="afPreview"></div>
    <div class="dialog-actions">
      <button class="btn" id="afClear" type="button">すべて解除</button>
      <div class="grow"></div>
      <button class="btn" id="afCancel" type="button">キャンセル</button>
      <button class="btn" id="afSaveOnly" type="button">条件だけ保存</button>
      <button class="btn pri" id="afRunArrange" type="button">${S.song ? "この条件で自動配置" : "保存する"}</button>
    </div>
  </div>`;

  const customInput = document.getElementById("afCustomInput") as HTMLInputElement;
  const previewEl = document.getElementById("afPreview")!;
  const saveBtn = document.getElementById("afSaveOnly") as HTMLButtonElement;
  const runBtn = document.getElementById("afRunArrange") as HTMLButtonElement;

  const collectTags = (): string[] => {
    const customParts = customInput.value.split(/[,、，]+/).map((s) => s.trim()).filter(Boolean);
    return cleanFilterTags([...selected, ...customParts]);
  };

  const updatePreview = () => {
    const tags = collectTags();
    const matched = filterPhrasesByTags(S.library, tags, S.categories);
    const empty = tags.length > 0 && matched.length === 0;
    saveBtn.disabled = empty;
    runBtn.disabled = empty;
    if (!tags.length) {
      previewEl.innerHTML = `<span class="af-count ok">対象: <b>全 ${S.library.length} 件</b>（絞り込みなし）</span>`;
      return;
    }
    if (empty) {
      previewEl.innerHTML = `<span class="af-count warn">条件（${escapeHtml(tags.join("・"))}）に一致するフレーズが 0 件です。条件を緩めてください。</span>`;
      return;
    }
    const sampleNames = matched.slice(0, 6).map((m) => escapeHtml(m.name)).join("、");
    const more = matched.length > 6 ? ` ほか ${matched.length - 6} 件` : "";
    previewEl.innerHTML = `<span class="af-count ok">該当フレーズ: <b>${matched.length}</b> 件 / 全 ${S.library.length} 件</span><small class="af-samples">${sampleNames}${more}</small>`;
  };

  modal.querySelectorAll<HTMLButtonElement>("[data-ftag]").forEach((btn) => {
    btn.onclick = () => {
      const tag = btn.dataset.ftag!;
      if (selected.has(tag)) {
        selected.delete(tag);
        btn.classList.remove("on");
      } else {
        selected.add(tag);
        btn.classList.add("on");
      }
      updatePreview();
    };
  });
  customInput.oninput = updatePreview;

  document.getElementById("afClear")!.onclick = () => {
    selected.clear();
    customInput.value = "";
    modal.querySelectorAll<HTMLButtonElement>("[data-ftag]").forEach((btn) => btn.classList.remove("on"));
    updatePreview();
  };
  document.getElementById("afCancel")!.onclick = closeModal;

  saveBtn.onclick = async () => {
    const tags = collectTags();
    closeModal();
    const data = await withBusy("対象条件を保存しています", () => call("PUT", "/api/arrange-filter", { tags }));
    if (data) {
      await applyChanges(data);
      toast(tags.length ? `自動配置の対象を「${tags.join("・")}」に設定しました` : "自動配置の絞り込みを解除しました");
    }
  };

  runBtn.onclick = async () => {
    const tags = collectTags();
    closeModal();
    if (!S.song) {
      const data = await withBusy("対象条件を保存しています", () => call("PUT", "/api/arrange-filter", { tags }));
      if (data) {
        await applyChanges(data);
        toast(tags.length ? `対象「${tags.join("・")}」を保存しました。曲を読み込んで「自動配置」を押すと適用されます` : "自動配置の絞り込みを解除しました");
      }
      return;
    }
    const label = tags.length
      ? `「${tags.join("・")}」で自動配置して MotionBricks でつないでいます`
      : "自動配置して MotionBricks でつないでいます";
    const data = await withBusy(label, () => call("POST", "/api/arrange", { tags }));
    if (data) { await applyChanges(data); }
  };

  updatePreview();
}
