import { applyState, call, toast } from "./store";
import { escapeHtml } from "./panels";

interface License { name: string; url: string; }
interface Item { id: string; label: string; purpose: string; required: boolean; gigabytes: number; licenses: License[]; installed: boolean; }
interface Job { state: "idle" | "running" | "done" | "failed" | "cancelled"; components?: Record<string, string>; current?: string | null; step?: string; detail?: string; done?: number; total?: number; error?: string | null; }
interface Status { defaultRoot: string; job: Job; components: Item[]; }
interface Requirement { id: string; label: string; ok: boolean; detail: string; }

const STATE_TEXT: Record<string, string> = { waiting: "待機中", running: "導入中", done: "完了", failed: "失敗" };
let status: Status | null = null;
let root = "";
let selected = new Set<string>();
let accepted = false;
let polling = false;
let checkSerial = 0;
let shownJob = "idle";

const modal = () => document.getElementById("modal")!;
const close = () => { modal().hidden = true; modal().innerHTML = ""; };
const missing = () => status?.components.filter((item) => !item.installed) ?? [];
const running = () => status?.job.state === "running";

function renderChip() {
  const chip = document.getElementById("envChip")!;
  const job = status?.job;
  chip.hidden = !running() && !missing().length;
  if (running() && job?.components) {
    const states = Object.values(job.components);
    chip.textContent = `環境を導入中 ${states.filter((s) => s === "done").length}/${states.length}`;
  } else {
    chip.textContent = `環境が未導入 ${missing().length} 件`;
  }
}

async function load() {
  status = await call<Status>("GET", "/api/components");
  renderChip();
}

function renderForm() {
  const items = status!.components;
  modal().hidden = false;
  modal().innerHTML = `<div class="dialog wide" id="envForm"><h3>必要な環境の導入</h3>
    <p>足りない環境を、この PC へ配布元から直接ダウンロードして導入します。あとで「環境が未導入」のボタンからも開けます。</p>
    <div class="env-list">${items.map((item) => `<label class="env-row${item.installed ? " is-done" : ""}">
      <input type="checkbox" data-env="${item.id}" ${item.installed ? "disabled" : selected.has(item.id) ? "checked" : ""}>
      <span class="env-name">${escapeHtml(item.label)}<small>${item.required ? "必須" : "任意"}</small></span>
      <span class="env-purpose">${escapeHtml(item.purpose)}</span>
      <span class="env-size">${item.installed ? "導入済み" : `約 ${item.gigabytes} GB`}</span></label>`).join("")}</div>
    <div class="form env-root"><label class="full">インストール先<input id="envRoot" spellcheck="false" value="${escapeHtml(root)}"></label></div>
    <ul class="env-checks" id="envChecks"></ul>
    <div id="envLicenses"></div>
    <p class="env-note">MotionBricks の初回の準備には GPU を使います。他のアプリが GPU を使っていると、数十分かかることがあります。</p>
    <div class="dialog-actions"><button class="btn" id="envLater">あとで</button><button class="btn pri" id="envStart" disabled>インストール</button></div></div>`;
  document.getElementById("envLater")!.onclick = close;
  document.getElementById("envStart")!.onclick = start;
  (document.getElementById("envRoot") as HTMLInputElement).oninput = (event) => { root = (event.target as HTMLInputElement).value.trim(); void refreshChecks(); };
  modal().querySelectorAll<HTMLInputElement>("[data-env]").forEach((box) => {
    box.onchange = () => {
      if (box.checked) { selected.add(box.dataset.env!); } else { selected.delete(box.dataset.env!); }
      accepted = false;
      void refreshChecks();
    };
  });
  void refreshChecks();
}

function renderLicenses() {
  const host = document.getElementById("envLicenses")!;
  const chosen = status!.components.filter((item) => selected.has(item.id));
  host.innerHTML = chosen.length ? `<h3 class="second">ライセンス</h3><ul class="env-licenses">${chosen.map((item) => item.licenses.map((license) =>
    `<li><b>${escapeHtml(item.label)}</b> ${escapeHtml(license.name)} <a href="${escapeHtml(license.url)}" target="_blank" rel="noreferrer">全文</a></li>`).join("")).join("")}</ul>
    <label class="env-accept"><input type="checkbox" id="envAccept" ${accepted ? "checked" : ""}>上記のライセンスを確認し、同意します</label>` : "";
  const accept = document.getElementById("envAccept") as HTMLInputElement | null;
  if (accept) { accept.onchange = () => { accepted = accept.checked; updateStart(lastOk); }; }
}

let lastOk = false;

function updateStart(ok: boolean) {
  lastOk = ok;
  const button = document.getElementById("envStart") as HTMLButtonElement | null;
  if (button) { button.disabled = !(ok && accepted && selected.size > 0); }
}

async function refreshChecks() {
  renderLicenses();
  updateStart(false);
  const host = document.getElementById("envChecks")!;
  if (!selected.size) { host.innerHTML = ""; return; }
  const serial = ++checkSerial;
  try {
    const result = await call<{ ok: boolean; items: Requirement[] }>("GET", `/api/components/check?root=${encodeURIComponent(root)}&ids=${[...selected].join(",")}`);
    if (serial !== checkSerial || !document.getElementById("envChecks")) { return; }
    host.innerHTML = result.items.map((item) => `<li class="${item.ok ? "ok" : "ng"}"><b>${item.ok ? "OK" : "要対応"}</b>${escapeHtml(item.label)}<small>${escapeHtml(item.detail)}</small></li>`).join("");
    updateStart(result.ok);
  } catch (error) {
    if (serial === checkSerial) { host.innerHTML = `<li class="ng"><b>要対応</b>${escapeHtml((error as Error).message)}</li>`; }
  }
}

async function start() {
  try {
    status = await call<Status>("POST", "/api/components/install", { root, ids: [...selected], accepted });
  } catch (error) {
    toast((error as Error).message, "error");
    return;
  }
  shownJob = "running";
  renderProgress();
  void poll();
}

function renderProgress() {
  modal().hidden = false;
  modal().innerHTML = `<div class="dialog wide" id="envProgress"><h3>環境の導入</h3><div id="envBody"></div><div class="dialog-actions" id="envActions"></div></div>`;
  updateProgress();
}

function updateProgress() {
  const body = document.getElementById("envBody");
  if (!body || !status) { return; }
  const job = status.job;
  const names = Object.fromEntries(status.components.map((item) => [item.id, item.label]));
  const percent = job.total ? Math.min(100, (job.done ?? 0) / job.total * 100) : null;
  body.innerHTML = `<ul class="env-steps">${Object.entries(job.components ?? {}).map(([id, state]) => `<li class="${state}"><b>${escapeHtml(names[id])}</b><span>${STATE_TEXT[state]}</span></li>`).join("")}</ul>
    ${job.state === "running" ? `<p class="env-step">${escapeHtml(job.step ?? "")}${percent === null ? "" : `<small>${((job.done ?? 0) / 1e9).toFixed(2)} / ${((job.total ?? 0) / 1e9).toFixed(2)} GB</small>`}</p>
      <div class="env-bar${percent === null ? " unknown" : ""}"><i style="width:${percent ?? 100}%"></i></div><p class="env-detail">${escapeHtml(job.detail ?? "")}</p>` : ""}
    ${job.state === "failed" ? `<p class="err">${escapeHtml(job.error ?? "導入に失敗しました")}</p><p class="env-detail">もう一度実行すると、途中から続けます。</p>` : ""}
    ${job.state === "cancelled" ? `<p>中止しました。もう一度実行すると、途中から続けます。</p>` : ""}
    ${job.state === "done" ? `<p>導入が完了しました。</p>` : ""}`;
  const actions = document.getElementById("envActions")!;
  const key = job.state;
  if (actions.dataset.state === key) { return; }
  actions.dataset.state = key;
  if (key === "running") {
    actions.innerHTML = `<button class="btn danger" id="envCancel">中止</button><div class="grow"></div><button class="btn pri" id="envHide">閉じて作業を続ける</button>`;
    document.getElementById("envCancel")!.onclick = () => void call("POST", "/api/components/cancel").catch(() => undefined);
    document.getElementById("envHide")!.onclick = close;
  } else if (key === "done") {
    actions.innerHTML = `<button class="btn pri" id="envClose">閉じる</button>`;
    document.getElementById("envClose")!.onclick = close;
  } else {
    actions.innerHTML = `<button class="btn" id="envClose">閉じる</button><button class="btn pri" id="envRetry">もう一度実行</button>`;
    document.getElementById("envClose")!.onclick = close;
    document.getElementById("envRetry")!.onclick = () => openForm();
  }
}

async function poll() {
  if (polling) { return; }
  polling = true;
  try {
    while (true) {
      await load();
      updateProgress();
      if (!running()) { break; }
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
    }
  } catch { /* the next action reports a lost connection */ }
  finally { polling = false; }
  if (status?.job.state === "done" && shownJob !== "done") {
    shownJob = "done";
    toast("環境の導入が完了しました");
    await applyState(await call("GET", "/api/state"));
  }
}

function openForm() {
  if (!status) { return; }
  root = root || status.defaultRoot;
  selected = new Set(missing().map((item) => item.id));
  accepted = false;
  renderForm();
}

async function open() {
  await load();
  if (running()) { renderProgress(); void poll(); } else { openForm(); }
}

export async function initInstaller() {
  document.getElementById("envChip")!.onclick = () => void open().catch((error) => toast(String(error), "error"));
  document.getElementById("inspector")!.addEventListener("click", (event) => {
    if ((event.target as HTMLElement).closest("[data-open-installer]")) { void open(); }
  });
  await load();
  if (running()) { shownJob = "running"; void poll(); } else if (missing().length) { openForm(); }
}
