import { S, activeDancer, applyChanges, barBeatAt, call, dancerColor, dancers, formatTime, subscribe, toast, withBusy, type Formation, type Spot } from "./store";
import { MAIN_BONES, boneLabel } from "./model";
import { openActionMenu } from "./insertMenu";
import { bridge, closeModal, escapeHtml } from "./panels";
import { isModelLoading, onModelChange, rigOf } from "./stage";
import { followPlayhead } from "./timeline";
import { seek } from "./transport";

const modal = () => document.getElementById("modal")!;
const nameOf = (id: string) => dancers().find((dancer) => dancer.id === id)?.name ?? id;

async function send(method: string, path: string, body?: unknown): Promise<any | undefined> {
  try {
    const data = await call(method, path, body);
    await applyChanges(data);
    return data;
  } catch (error) {
    toast((error as Error).message, "error");
    return undefined;
  }
}

// ---- The strip of dancers on the stage

function renderCast() {
  const active = activeDancer()?.id ?? "";
  const chips = dancers().map((dancer) => `<button class="cast-chip ${dancer.id === active ? "on" : ""} ${dancer.hidden ? "hidden-dancer" : ""}" data-dancer="${dancer.id}" title="${dancer.hidden ? "非表示中です。" : ""}クリックで編集の対象にします。右クリックで表示の切り替えと削除"><i style="background:${dancerColor(dancer.id)}"></i>${escapeHtml(dancer.name)}</button>`).join("");
  const close = S.collisions.length ? `<button class="cast-chip warn" id="castWarn" title="ダンサー同士が近づきすぎる場面があります。クリックでその場面へ">▲ 接近 ${S.collisions.length}</button>` : "";
  document.getElementById("castBar")!.innerHTML = `${chips}<button class="cast-chip add" id="castAdd" title="ダンサーを追加">＋</button><button class="cast-chip" id="castFormation" title="全員の立ち位置を決めます">フォーメーション</button>${close}`;
  const model = S.project?.models?.[active];
  const rig = rigOf(active);
  document.getElementById("modelName")!.textContent = !model ? "標準マネキン" : isModelLoading(active) ? "読み込み中…" : rig && !rig.ready ? `${model.name}（骨の対応が必要）` : model.name;
}

/** The moments two dancers come too close, as rows for the right panel. Clicking one goes there. */
export function collisionList(): string {
  if (!S.collisions.length) { return ""; }
  const rows = S.collisions.slice(0, 8).map((close) => {
    const at = barBeatAt(close.start);
    return `<div class="close-call" data-seek="${close.start}"><span>▲ ${escapeHtml(nameOf(close.a))} と ${escapeHtml(nameOf(close.b))}</span><span>${at ? `${at.bar}小節` : formatTime(close.start)} ・ ${close.distance.toFixed(2)} m</span></div>`;
  }).join("");
  return `<div class="close-calls"><div class="st">ダンサーの接近<small>${S.collisions.length} 件 ・ クリックで移動</small></div>${rows}</div>`;
}

function goTo(time: number) {
  seek(time);
  followPlayhead();
}

async function setHidden(ids: string[], hidden: boolean) {
  for (const id of ids) { await send("PATCH", `/api/dancers/${id}`, { hidden }); }
}

/** Right click on a dancer: show or hide it on the stage, or remove it. */
function openDancerMenu(dancerId: string, clientX: number, clientY: number) {
  const dancer = dancers().find((d) => d.id === dancerId);
  if (!dancer) { return; }
  const others = dancers().filter((d) => d.id !== dancerId);
  const alone = others.length === 0;
  openActionMenu(clientX, clientY, dancer.name, [
    { label: dancer.hidden ? "表示する" : "非表示にする", run: () => void setHidden([dancerId], !dancer.hidden), title: "ステージに出すかどうかだけを切り替えます。振り、接近の検出、書き出しは変わりません" },
    { label: "このダンサーだけ表示", run: async () => { await setHidden(others.filter((d) => !d.hidden).map((d) => d.id), true); if (dancer.hidden) { await setHidden([dancerId], false); } }, disabled: alone },
    { label: "全員を表示", run: () => void setHidden(dancers().filter((d) => d.hidden).map((d) => d.id), false), disabled: !dancers().some((d) => d.hidden) },
    { label: "削除", danger: true, disabled: alone, title: alone ? "最後の 1 人は削除できません" : "このダンサーと、その振りを削除します",
      run: async () => { const name = dancer.name; if (await send("DELETE", `/api/dancers/${dancerId}`)) { toast(`「${name}」を削除しました（Ctrl+Z で戻せます）`); } } },
  ]);
}

function openAddDancer() {
  const source = activeDancer();
  modal().hidden = false;
  modal().innerHTML = `<div class="dialog"><h3>ダンサーを追加</h3>
    <p>追加したダンサーが編集の対象になります。振りをコピーすると、同じフレーズが横に ${1.5} m ずれた位置に入ります。あとから「フォーメーション」で立ち位置を決められます。</p>
    <div class="dialog-actions"><button class="btn" id="addCancel">キャンセル</button><button class="btn" id="addEmpty">空のダンサー</button>
    ${source?.clips.length ? `<button class="btn pri" id="addCopy">「${escapeHtml(source.name)}」の振りをコピー</button>` : ""}</div></div>`;
  document.getElementById("addCancel")!.onclick = closeModal;
  const add = async (copyFrom: string | null) => {
    closeModal();
    S.selected = null;
    if (await send("POST", "/api/dancers", { copyFrom })) { toast("ダンサーを追加しました（Ctrl+Z で戻せます）"); }
  };
  document.getElementById("addEmpty")!.onclick = () => add(null);
  const copy = document.getElementById("addCopy");
  if (copy) { copy.onclick = () => add(source!.id); }
}

// ---- Formations

const PRESETS: [string, string][] = [["line", "横一列"], ["rows", "二列"], ["v", "V字"], ["wedge", "逆V字"], ["triangle", "三角"], ["diamond", "ひし形"], ["circle", "円"], ["diagonal", "斜め"]];
const CENTRE_Z = 3;
const range = (count: number) => Array.from({ length: count }, (_, index) => index);
const clampSpot = (x: number, z: number): [number, number] => [Math.round(Math.max(-4, Math.min(4, x)) * 10) / 10, Math.round(Math.max(0, Math.min(6, z)) * 10) / 10];

/** Floor places (x, z) for `count` dancers `gap` metres apart. The audience is at high z, and the first dancer takes the leading place. */
function presetSpots(kind: string, count: number, gap: number): [number, number][] {
  const centred = (index: number, of: number) => index - (of - 1) / 2;
  if (kind === "rows") {
    const front = Math.ceil(count / 2);
    return range(count).map((i) => (i < front ? [centred(i, front) * gap, CENTRE_Z + gap / 2] : [centred(i - front, count - front) * gap, CENTRE_Z - gap / 2]));
  }
  if (kind === "v" || kind === "wedge") {
    const step = gap * 0.75, deepest = Math.ceil((count - 1) / 2);
    const toward = kind === "v" ? 1 : -1;  // V: the point at the back, opening to the audience. Wedge: the point in front.
    return range(count).map((i) => {
      const rank = Math.ceil(i / 2);
      return [(i % 2 ? -1 : 1) * rank * step, CENTRE_Z + toward * (rank - deepest / 2) * step];
    });
  }
  if (kind === "triangle") {
    const rows: number[] = [];
    for (let left = count, size = 1; left > 0; left -= size, size++) { rows.push(Math.min(size, left)); }
    const spots: [number, number][] = [];
    rows.forEach((size, row) => range(size).forEach((i) => spots.push([centred(i, size) * gap, CENTRE_Z + ((rows.length - 1) / 2 - row) * gap * 0.85])));
    return spots;
  }
  if (kind === "diamond") {
    const reach = gap * Math.max(1, count / 4);
    const corners: [number, number][] = [[0, reach], [reach, 0], [0, -reach], [-reach, 0]];
    return range(count).map((i) => {
      const along = (i / count) * 4, edge = Math.floor(along), part = along - edge;
      const from = corners[edge % 4], to = corners[(edge + 1) % 4];
      return [from[0] + (to[0] - from[0]) * part, CENTRE_Z + from[1] + (to[1] - from[1]) * part];
    });
  }
  if (kind === "circle") {
    const reach = count > 1 ? Math.max((gap * count) / (2 * Math.PI), gap / 2) : 0;
    return range(count).map((i) => [reach * Math.sin((i / count) * 2 * Math.PI), CENTRE_Z + reach * Math.cos((i / count) * 2 * Math.PI)]);
  }
  if (kind === "diagonal") { return range(count).map((i) => [centred(i, count) * gap * 0.8, CENTRE_Z + centred(i, count) * gap * 0.5]); }
  return range(count).map((i) => [centred(i, count) * gap, CENTRE_Z]);
}

const FX = (x: number) => 30 + (x + 4) * 55;
const FZ = (z: number) => 26 + z * 55;

/** The formation screen: every dancer's place from one bar until the next formation. */
export function openFormation(time?: number) {
  const song = S.song;
  if (!song) { toast("先に曲を読み込んでください", "error"); return; }
  const barOf = (t: number) => { let index = 0; song.bars.forEach((bar, i) => { if (bar <= t + 1e-3) { index = i; } }); return index; };
  const saved = (bar: number): Formation | undefined => (S.project?.formations ?? []).find((formation) => Math.abs(formation.time - song.bars[bar]) < 1e-3);
  /** Where everyone stands at this bar: the formation saved there, else where their phrases already put them. */
  const standing = (bar: number): Record<string, Spot> => {
    const at = song.bars[bar], kept = saved(bar)?.spots ?? {};
    const spots: Record<string, Spot> = {};
    dancers().forEach((dancer, index) => {
      const clip = dancer.clips.find((c) => c.end > at) ?? dancer.clips[dancer.clips.length - 1];
      spots[dancer.id] = kept[dancer.id] ? { ...kept[dancer.id] } : clip ? { x: clip.x, z: clip.z, facing: null } : { x: clampSpot((index - (dancers().length - 1) / 2) * 1.5, CENTRE_Z)[0], z: CENTRE_Z, facing: null };
    });
    return spots;
  };
  let bar = barOf(time ?? S.playhead);
  let spots = standing(bar);
  let name = saved(bar)?.name ?? "";
  let gap = 1.5;

  const drawStage = () => {
    let svg = `<rect x="${FX(-4)}" y="${FZ(0)}" width="${8 * 55}" height="${6 * 55}" rx="4" fill="#ffffff08" stroke="#ff8a70" stroke-opacity=".5" stroke-dasharray="6 5"/>`;
    for (let x = -3; x <= 3; x++) { svg += `<line x1="${FX(x)}" x2="${FX(x)}" y1="${FZ(0)}" y2="${FZ(6)}" stroke="#fff" stroke-opacity="${x === 0 ? 0.16 : 0.06}"/>`; }
    for (let z = 1; z <= 5; z++) { svg += `<line x1="${FX(-4)}" x2="${FX(4)}" y1="${FZ(z)}" y2="${FZ(z)}" stroke="#fff" stroke-opacity="${z === 3 ? 0.16 : 0.06}"/>`; }
    dancers().forEach((dancer, index) => {
      const spot = spots[dancer.id];
      if (!spot) { return; }
      svg += `<g class="form-dot" data-id="${dancer.id}"><circle cx="${FX(spot.x)}" cy="${FZ(spot.z)}" r="17" fill="${dancerColor(dancer.id)}" stroke="#14151a" stroke-width="2"/><text x="${FX(spot.x)}" y="${FZ(spot.z) + 6}" text-anchor="middle" font-family="Bahnschrift" font-weight="700" font-size="17" fill="#111">${index + 1}</text></g>`;
    });
    document.getElementById("formSvg")!.innerHTML = svg;
    dancers().forEach((dancer) => {
      const cell = document.querySelector<HTMLElement>(`[data-place="${dancer.id}"]`);
      if (cell && spots[dancer.id]) { cell.textContent = `${spots[dancer.id].x.toFixed(1)}, ${spots[dancer.id].z.toFixed(1)}`; }
    });
  };

  const draw = () => {
    const existing = saved(bar);
    const others = (S.project?.formations ?? []).map((formation, index) => `<span class="chip ${formation === existing ? "on" : ""}" data-jump="${barOf(formation.time)}">${barOf(formation.time) + 1}小節 ${escapeHtml(formation.name || `隊形 ${index + 1}`)}</span>`).join("");
    const rows = dancers().map((dancer, index) => `<tr data-row="${dancer.id}"><td><i class="cast-dot" style="background:${dancerColor(dancer.id)}">${index + 1}</i></td>
      <td><input data-rename value="${escapeHtml(dancer.name)}" maxlength="24"></td><td class="n" data-place="${dancer.id}"></td>
      <td><input data-facing type="number" step="15" placeholder="そのまま" value="${spots[dancer.id]?.facing ?? ""}"></td>
      <td>${dancers().length > 1 ? '<button class="mini-btn" data-remove>削除</button>' : ""}</td></tr>`).join("");
    modal().hidden = false;
    modal().innerHTML = `<div class="dialog wide formation"><h3>フォーメーション</h3>
      <p>ここで決めた位置に、各ダンサーのフレーズを置きます。対象は指定した小節から次のフォーメーションまでで、ロック中のフレーズは動きません。移動の途中でぶつかる経路は、後ろのダンサーが自動でよけます。</p>
      <div class="form-grid">
        <div class="form-stage"><svg id="formSvg" width="500" height="386"></svg><div class="form-front">▼ 客席側</div></div>
        <div class="form-side">
          <div class="form-line"><label>開始<input id="formBar" type="number" min="1" max="${song.bars.length}" value="${bar + 1}"></label><span>小節から</span><label class="grow">名前<input id="formName" maxlength="24" placeholder="例: サビ" value="${escapeHtml(name)}"></label></div>
          ${others ? `<div class="form-chips">${others}</div>` : ""}
          <div class="form-presets">${PRESETS.map(([key, label]) => `<button class="mini-btn" data-preset="${key}">${label}</button>`).join("")}</div>
          <div class="form-line"><label class="grow"><span class="plain">プリセットの間隔 <b id="formGapValue">${gap.toFixed(1)}</b> m</span><input id="formGap" type="range" min="0.8" max="3" step="0.1" value="${gap}"></label></div>
          <table class="edit cast-table"><thead><tr><th></th><th>名前</th><th>位置</th><th>向き °</th><th></th></tr></thead><tbody>${rows}</tbody></table>
          ${dancers().length > 1 ? `<button class="mini-btn" id="formSync" title="フレーズと時刻を全員同じにします。位置は今の間隔を保ちます">「${escapeHtml(activeDancer()?.name ?? "")}」の振りを全員にそろえる</button>` : ""}
        </div>
      </div>
      <div class="dialog-actions">${existing ? '<button class="btn danger" id="formDelete">このフォーメーションを削除</button>' : ""}<div class="grow"></div><button class="btn" id="formClose">閉じる</button><button class="btn pri" id="formApply">${bar + 1} 小節から適用</button></div></div>`;
    drawStage();
    bind();
  };

  const moveTo = (next: number) => {
    bar = Math.max(0, Math.min(song.bars.length - 1, next));
    spots = standing(bar);
    name = saved(bar)?.name ?? "";
    draw();
  };

  const bind = () => {
    const svg = document.getElementById("formSvg") as unknown as SVGSVGElement;
    svg.onmousedown = (event) => {
      const dot = (event.target as Element).closest(".form-dot") as SVGGElement | null;
      if (!dot) { return; }
      event.preventDefault();
      const id = dot.dataset.id!;
      const box = svg.getBoundingClientRect();
      const move = (e: MouseEvent) => {
        const grid = e.shiftKey ? 2 : 10;  // Shift: half-metre steps
        const x = Math.round(((e.clientX - box.left - 30) / 55 - 4) * grid) / grid, z = Math.round(((e.clientY - box.top - 26) / 55) * grid) / grid;
        const [cx, cz] = clampSpot(x, z);
        spots[id] = { ...spots[id], x: cx, z: cz };
        drawStage();
      };
      const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    };
    (document.getElementById("formBar") as HTMLInputElement).onchange = (event) => moveTo(Number((event.target as HTMLInputElement).value) - 1);
    (document.getElementById("formName") as HTMLInputElement).oninput = (event) => { name = (event.target as HTMLInputElement).value; };
    const gapInput = document.getElementById("formGap") as HTMLInputElement;
    gapInput.oninput = () => { gap = Number(gapInput.value); document.getElementById("formGapValue")!.textContent = gap.toFixed(1); };
    modal().querySelectorAll<HTMLElement>("[data-jump]").forEach((chip) => (chip.onclick = () => moveTo(Number(chip.dataset.jump))));
    modal().querySelectorAll<HTMLElement>("[data-preset]").forEach((button) => (button.onclick = () => {
      presetSpots(button.dataset.preset!, dancers().length, gap).forEach(([x, z], index) => {
        const id = dancers()[index].id, [cx, cz] = clampSpot(x, z);
        spots[id] = { ...spots[id], x: cx, z: cz };
      });
      drawStage();
    }));
    modal().querySelectorAll<HTMLTableRowElement>("[data-row]").forEach((row) => {
      const id = row.dataset.row!;
      row.querySelector<HTMLInputElement>("[data-rename]")!.onchange = async (event) => { await send("PATCH", `/api/dancers/${id}`, { name: (event.target as HTMLInputElement).value }); };
      row.querySelector<HTMLInputElement>("[data-facing]")!.onchange = (event) => {
        const value = (event.target as HTMLInputElement).value;
        spots[id] = { ...spots[id], facing: value === "" || !Number.isFinite(Number(value)) ? null : Number(value) };
      };
      const remove = row.querySelector<HTMLElement>("[data-remove]");
      if (remove) {
        remove.onclick = async () => {
          if (await send("DELETE", `/api/dancers/${id}`)) { delete spots[id]; toast(`「${nameOf(id)}」を削除しました（Ctrl+Z で戻せます）`); draw(); }
        };
      }
    });
    const sync = document.getElementById("formSync");
    if (sync) {
      sync.onclick = async () => {
        closeModal();
        const done = await withBusy("振りをそろえています", () => call("POST", "/api/dancers/sync"));
        if (done) { await applyChanges(done); toast("全員の振りをそろえました（Ctrl+Z で戻せます）"); }
      };
    }
    document.getElementById("formClose")!.onclick = closeModal;
    document.getElementById("formApply")!.onclick = async () => {
      closeModal();
      const done = await withBusy("フォーメーションを適用しています", () => call("PUT", "/api/formations", { time: song.bars[bar], spots, name }));
      if (done) { await applyChanges(done); goTo(song.bars[bar]); toast("フォーメーションを適用しました（Ctrl+Z で戻せます）"); }
    };
    const remove = document.getElementById("formDelete");
    if (remove) {
      remove.onclick = async () => {
        const target = saved(bar);
        closeModal();
        if (target && await send("DELETE", `/api/formations/${target.id}`)) { toast("フォーメーションを削除しました（Ctrl+Z で戻せます）"); }
      };
    }
  };
  draw();
}

// ---- Models

let awaitingModel: string | null = null;  // the dancer whose model was just chosen, to offer bone matching if it needs it

async function pickPath(kind: "file-model" | "file-pmx", input: HTMLInputElement) {
  const native = bridge();
  if (!native) { input.focus(); return; }
  const picked = await native.pick(kind);
  if (picked?.length) { input.value = picked[0]; }
}

export function openModel() {
  const dancer = activeDancer();
  if (!dancer) { return; }
  const current = S.project?.models?.[dancer.id];
  const rig = rigOf(dancer.id);
  const note = !current ? '<p class="model-note">現在は指先まで動く標準マネキン（Quaternius UAL・CC0）を表示しています。</p>'
    : rig && !rig.ready ? `<p class="model-note warn">骨の対応が足りません（${rig.missing.map(boneLabel).join("、")}）。「骨の対応」で指定してください。</p>`
    : rig?.vrm ? '<p class="model-note">VRM のヒューマノイド定義で動かしています。</p>' : rig ? '<p class="model-note">骨の名前から自動で対応付けました。動きがおかしい部位は「骨の対応」で直せます。</p>' : "";
  modal().hidden = false;
  modal().innerHTML = `<div class="dialog"><h3>モデル</h3>
    <p>VRM / GLB / FBX の人型モデルを、標準マネキンの代わりに踊らせます。T ポーズと A ポーズのどちらのモデルでも読み込めます。大きさは動きの体格に合わせます。配置やつなぎは変わりません。</p>
    <label class="export-pmx">「${escapeHtml(dancer.name)}」のモデル<span class="path-row"><input id="modelPath" spellcheck="false" placeholder="D:\\...\\model.vrm" value="${escapeHtml(current?.path ?? "")}"><button class="mini-btn" id="pickModel">選ぶ</button></span></label>
    ${dancers().length > 1 ? '<label class="check"><input type="checkbox" id="modelAll"> 全員に同じモデルを使う</label>' : ""}${note}
    <div class="dialog-actions">${current ? '<button class="btn" id="modelClear">標準マネキンに戻す</button>' : ""}<div class="grow"></div>
    ${current && rig && !rig.vrm ? '<button class="btn" id="modelBones">骨の対応</button>' : ""}<button class="btn" id="modelClose">閉じる</button><button class="btn pri" id="modelLoad">読み込む</button></div></div>`;
  const input = document.getElementById("modelPath") as HTMLInputElement;
  const everyone = () => (document.getElementById("modelAll") as HTMLInputElement | null)?.checked ?? false;
  document.getElementById("pickModel")!.onclick = () => pickPath("file-model", input);
  document.getElementById("modelClose")!.onclick = closeModal;
  const clear = document.getElementById("modelClear");
  if (clear) { clear.onclick = async () => { const all = everyone(); closeModal(); await send("PUT", `/api/dancers/${dancer.id}/model`, { path: null, everyone: all }); }; }
  const bones = document.getElementById("modelBones");
  if (bones) { bones.onclick = () => openBones(dancer.id); }
  document.getElementById("modelLoad")!.onclick = async () => {
    const path = input.value.trim().replace(/^"|"$/g, "");
    if (!path) { toast("モデルのファイルを指定してください", "error"); return; }
    const all = everyone();
    closeModal();
    awaitingModel = dancer.id;
    await send("PUT", `/api/dancers/${dancer.id}/model`, { path, mapping: path === current?.path ? current.mapping : {}, everyone: all });
  };
}

/** Match the model's bones to the body by hand, for a GLB or FBX whose names did not say enough. */
function openBones(dancerId: string) {
  const rig = rigOf(dancerId);
  const model = S.project?.models?.[dancerId];
  if (!rig || !model) { return; }
  const names = [...rig.boneNames].sort((a, b) => a.localeCompare(b));
  const row = (bone: string) => `<tr><th class="${rig.missing.includes(bone) ? "need" : ""}">${boneLabel(bone)}</th><td><select data-bone="${bone}"><option value="">（なし）</option>${names.map((name) => `<option ${rig.mapping[bone] === name ? "selected" : ""}>${escapeHtml(name)}</option>`).join("")}</select></td></tr>`;
  const half = Math.ceil(MAIN_BONES.length / 2);
  modal().hidden = false;
  modal().innerHTML = `<div class="dialog wide"><h3>骨の対応</h3>
    <p>体の各部位に、モデルの骨を割り当てます。名前から分かったものは選択済みです。腰、腕、脚、足首は必須です。指は名前が合うものだけ自動で動きます。</p>
    <div class="bones-grid"><table class="edit"><tbody>${MAIN_BONES.slice(0, half).map(row).join("")}</tbody></table><table class="edit"><tbody>${MAIN_BONES.slice(half).map(row).join("")}</tbody></table></div>
    <div class="dialog-actions"><button class="btn" id="bonesCancel">キャンセル</button><button class="btn pri" id="bonesSave">保存して読み込み直す</button></div></div>`;
  document.getElementById("bonesCancel")!.onclick = closeModal;
  document.getElementById("bonesSave")!.onclick = async () => {
    const mapping: Record<string, string> = {};
    modal().querySelectorAll<HTMLSelectElement>("[data-bone]").forEach((select) => { mapping[select.dataset.bone!] = select.value; });
    closeModal();
    awaitingModel = dancerId;
    await send("PUT", `/api/dancers/${dancerId}/model`, { path: model.path, mapping });
  };
}

export function initCast() {
  subscribe(renderCast);
  onModelChange(() => {
    renderCast();
    const id = awaitingModel;
    if (!id || isModelLoading(id)) { return; }
    awaitingModel = null;
    const rig = rigOf(id);
    if (rig && !rig.ready) { toast("このモデルは骨の名前だけでは対応付けできませんでした。骨を指定してください", "error"); openBones(id); }
    else if (rig && !rig.builtin) { toast(`モデルを読み込みました: ${S.project?.models?.[id]?.name ?? ""}`); }
  });
  document.getElementById("castBar")!.addEventListener("click", async (event) => {
    const target = event.target as HTMLElement;
    const chip = target.closest<HTMLElement>("[data-dancer]");
    if (chip) {
      if (chip.dataset.dancer !== activeDancer()?.id) { S.selected = null; await send("POST", `/api/dancers/${chip.dataset.dancer}/select`); }
      return;
    }
    if (target.closest("#castAdd")) { openAddDancer(); }
    else if (target.closest("#castFormation")) { openFormation(); }
    else if (target.closest("#castWarn") && S.collisions.length) { goTo(S.collisions[0].start); }
  });
  document.getElementById("castBar")!.addEventListener("contextmenu", (event) => {
    const chip = (event.target as HTMLElement).closest<HTMLElement>("[data-dancer]");
    if (!chip) { return; }
    event.preventDefault();
    openDancerMenu(chip.dataset.dancer!, event.clientX, event.clientY);
  });
  document.getElementById("modelPill")!.addEventListener("click", openModel);
  document.getElementById("inspector")!.addEventListener("click", (event) => {
    const row = (event.target as HTMLElement).closest<HTMLElement>("[data-seek]");
    if (row) { goTo(Number(row.dataset.seek)); }
  });
}
