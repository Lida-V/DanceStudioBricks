import { Clip, InsertFit, Phrase, S, CAT_COLOR, refresh } from "./store";
import { SpotMatch, rankPhrasesForSpot, scoreFeelingItem } from "./feelingSearch";
import { escapeHtml, thumb } from "./panels";

let closeOpenMenu: (() => void) | null = null;

export const FIT_LABELS: Record<InsertFit, string> = {
  auto: "自動調整",
  "1": "1小節",
  "2": "2小節",
  "4": "4小節",
  raw: "原寸",
};

const FIT_OPTIONS: [InsertFit, string, string][] = [
  ["auto", "自動調整", "空き隙間や曲のテンポ・小節に合わせて長さを自動調整します"],
  ["1", "1小節", "1小節枠に収まるよう速度・長さを調整して挿入します"],
  ["2", "2小節", "2小節枠に収まるよう速度・長さを調整して挿入します"],
  ["4", "4小節", "4小節枠に収まるよう速度・長さを調整して挿入します"],
  ["raw", "原寸", "モーション本来の秒数のまま挿入します"],
];

/** One command in a right-click menu. */
export interface MenuAction {
  label: string;
  run: () => void;
  title?: string;
  disabled?: boolean;
  danger?: boolean;
}

const actionButtons = (actions: MenuAction[]) => actions.map((action, index) =>
  `<button type="button" class="menu-action ${action.danger ? "danger" : ""}" data-action="${index}" ${action.disabled ? "disabled" : ""} title="${escapeHtml(action.title ?? "")}">${escapeHtml(action.label)}</button>`).join("");

/** A short list of commands at the pointer (for things that are not a choice of phrase). */
export function openActionMenu(clientX: number, clientY: number, title: string, actions: MenuAction[]) {
  closeOpenMenu?.();
  const box = document.createElement("div");
  box.id = "actionMenu";
  box.innerHTML = `<div class="menu-title">${escapeHtml(title)}</div>${actionButtons(actions)}`;
  document.body.appendChild(box);
  const close = () => {
    box.remove();
    window.removeEventListener("mousedown", onOutside, true);
    window.removeEventListener("keydown", onKey, true);
    closeOpenMenu = null;
  };
  const onOutside = (event: MouseEvent) => { if (!box.contains(event.target as Node)) { close(); } };
  const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") { close(); } };
  box.addEventListener("click", (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-action]");
    if (!button || button.disabled) { return; }
    close();
    actions[Number(button.dataset.action)].run();
  });
  box.addEventListener("contextmenu", (event) => event.preventDefault());
  box.style.left = `${Math.max(8, Math.min(clientX, window.innerWidth - box.offsetWidth - 8))}px`;
  box.style.top = `${Math.max(8, Math.min(clientY, window.innerHeight - box.offsetHeight - 8))}px`;
  window.addEventListener("mousedown", onOutside, true);
  window.addEventListener("keydown", onKey, true);
  closeOpenMenu = close;
}

export interface MenuMode {
  title: string;
  where: string;
  choose: (phraseId: string) => void;
  switchLabel?: string;
  at?: number;
  replaceClip?: Clip;
}

/** The list of library phrases that opens on a right click, ranked by musical match at the target spot. */
export function openPhraseMenu(clientX: number, clientY: number, modes: [MenuMode] | [MenuMode, MenuMode], actions: MenuAction[] = []) {
  closeOpenMenu?.();
  let modeIndex = 0;
  const box = document.createElement("div");
  box.id = "insertMenu";
  box.innerHTML = `<div class="ins-head"><div class="ins-title-row"><b id="insTitle"></b>${modes.length > 1 ? '<button class="mini-btn ins-switch" id="insSwitch" type="button"></button>' : ""}</div><span id="insWhere"></span></div>
    ${actions.length ? `<div class="ins-actions" id="insActions">${actionButtons(actions)}</div>` : ""}
    <div class="ins-fit-bar" id="insFitBar"><small>長さ</small><div class="ins-fit-chips" id="insFitChips"></div></div>
    <div class="search"><input placeholder="名前・「キック」「楽しそう」「かわいい」等で絞り込み" spellcheck="false"></div>
    <div class="ins-list"></div>`;
  document.body.appendChild(box);

  const titleEl = box.querySelector<HTMLElement>("#insTitle")!;
  const whereEl = box.querySelector<HTMLElement>("#insWhere")!;
  const switchBtn = box.querySelector<HTMLButtonElement>("#insSwitch");
  const fitBar = box.querySelector<HTMLElement>("#insFitBar")!;
  const fitChips = box.querySelector<HTMLElement>("#insFitChips")!;
  const input = box.querySelector("input")!;
  const list = box.querySelector<HTMLElement>(".ins-list")!;

  let spotMap = new Map<string, SpotMatch>();
  let shown: Phrase[] = [];
  let active = 0;

  const paintFitChips = () => {
    const cur = modes[modeIndex];
    fitBar.hidden = !!cur.replaceClip;
    if (cur.replaceClip) { return; }
    const prevScroll = fitChips.scrollLeft;
    fitChips.innerHTML = FIT_OPTIONS.map(
      ([key, label, tip]) => `<button type="button" class="chip ins-fit-chip ${S.insertFit === key ? "on" : ""}" data-fit="${key}" title="${tip}">${label}</button>`
    ).join("");
    fitChips.scrollLeft = prevScroll;
  };

  const recomputeSpotMatches = () => {
    const cur = modes[modeIndex];
    const targetAt = cur.replaceClip ? cur.replaceClip.start : (cur.at ?? S.playhead);
    spotMap = S.song ? rankPhrasesForSpot(S.library, targetAt, { replaceClip: cur.replaceClip }) : new Map();
  };

  const highlight = () => {
    list.querySelectorAll<HTMLElement>("[data-index]").forEach((row) => row.classList.toggle("on", Number(row.dataset.index) === active));
    list.querySelector(".ins-item.on")?.scrollIntoView({ block: "nearest" });
  };

  const paint = () => {
    const text = input.value.trim();
    const cur = modes[modeIndex];
    let entries = S.library.map((item) => {
      const feel = text ? scoreFeelingItem(item, text, S.categories) : 1;
      const spot = spotMap.get(item.id);
      const spotNorm = spot ? (spot.percent - 50) / 50 : 0.5;
      const combined = text ? 0.65 * feel + 0.35 * spotNorm : (spot?.score ?? 0);
      return { item, feel, spot, combined };
    });
    if (text) {
      entries = entries.filter((e) => e.feel > 0.05);
    }
    if (cur.replaceClip) {
      entries = entries.filter((e) => e.item.id !== cur.replaceClip!.phraseId);
    }
    entries.sort((a, b) => b.combined - a.combined);
    shown = entries.map((e) => e.item);
    active = Math.max(0, Math.min(active, shown.length - 1));

    list.innerHTML = entries.length
      ? entries.map(({ item, spot }, index) => {
          const isTop = spot && spot.percent >= 78;
          const badge = spot
            ? `<span class="ins-badge ${isTop ? "hot" : ""}">${isTop ? "★ " : ""}${spot.percent}%</span>`
            : "";
          const reason = spot ? ` ・ <em class="ins-reason">${escapeHtml(spot.reason)}</em>` : "";
          return `<div class="ins-item" data-index="${index}"><span class="ins-thumb">${thumb(item, 46, 34)}</span>
      <div class="ins-main"><div class="ins-name"><i style="background:${CAT_COLOR[item.category] ?? "var(--groove)"}"></i><span>${escapeHtml(item.name)}</span>${badge}</div>
      <div class="ins-sub">${S.categories[item.category] ?? ""} ・ ${item.bars}小節（${item.duration.toFixed(1)}s）${reason}</div></div></div>`;
        }).join("")
      : `<div class="ins-empty">${S.library.length ? "該当するフレーズがありません" : "ライブラリにフレーズがありません"}</div>`;
    highlight();
  };

  const syncHead = () => {
    const cur = modes[modeIndex];
    titleEl.textContent = cur.title;
    whereEl.textContent = cur.where;
    if (switchBtn) { switchBtn.textContent = cur.switchLabel ?? ""; }
    paintFitChips();
    recomputeSpotMatches();
    active = 0;
    paint();
  };

  if (switchBtn) {
    switchBtn.onclick = () => {
      modeIndex = modeIndex === 0 ? 1 : 0;
      syncHead();
      input.focus();
    };
  }

  fitChips.addEventListener("click", (event) => {
    const btn = (event.target as HTMLElement).closest<HTMLElement>("[data-fit]");
    if (!btn) { return; }
    S.insertFit = btn.dataset.fit as InsertFit;
    paintFitChips();
    btn.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
    refresh();
    input.focus();
  });
  fitChips.addEventListener("wheel", (event) => {
    if (Math.abs(event.deltaY) > Math.abs(event.deltaX)) {
      event.preventDefault();
      fitChips.scrollLeft += event.deltaY;
    }
  }, { passive: false });

  const close = () => {
    box.remove();
    window.removeEventListener("mousedown", onOutside, true);
    closeOpenMenu = null;
  };
  const onOutside = (event: MouseEvent) => { if (!box.contains(event.target as Node)) { close(); } };
  const pickIndex = (index: number) => {
    const item = shown[index];
    if (!item) { return; }
    const action = modes[modeIndex].choose;
    close();
    action(item.id);
  };

  input.addEventListener("input", () => { active = 0; paint(); });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") { close(); }
    else if (event.key === "Enter") { pickIndex(active); }
    else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      active = Math.max(0, Math.min(shown.length - 1, active + (event.key === "ArrowDown" ? 1 : -1)));
      highlight();
    }
  });
  list.addEventListener("mousemove", (event) => {
    const row = (event.target as HTMLElement).closest<HTMLElement>("[data-index]");
    if (row && Number(row.dataset.index) !== active) { active = Number(row.dataset.index); highlight(); }
  });
  list.addEventListener("click", (event) => {
    const row = (event.target as HTMLElement).closest<HTMLElement>("[data-index]");
    if (row) { pickIndex(Number(row.dataset.index)); }
  });
  box.querySelector("#insActions")?.addEventListener("click", (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-action]");
    if (!button || button.disabled) { return; }
    close();
    actions[Number(button.dataset.action)].run();
  });
  box.addEventListener("contextmenu", (event) => event.preventDefault());

  syncHead();
  box.style.left = `${Math.max(8, Math.min(clientX, window.innerWidth - box.offsetWidth - 8))}px`;
  box.style.top = `${Math.max(8, Math.min(clientY, window.innerHeight - box.offsetHeight - 8))}px`;
  window.addEventListener("mousedown", onOutside, true);
  closeOpenMenu = close;
  input.focus();
}

export function openInsertMenu(clientX: number, clientY: number, where: string, choose: (phraseId: string) => void, at?: number, actions: MenuAction[] = []) {
  openPhraseMenu(clientX, clientY, [{ title: "モーションを挿入", where, choose, at }], actions);
}
