// Desktop shell: starts the local Python backend, shows its page, and owns native file dialogs.
const { app, BrowserWindow, Menu, dialog, ipcMain, shell } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..");
const PORT = 8765;
const URL = `http://127.0.0.1:${PORT}/`;
// Native texts are Japanese on a Japanese system and English elsewhere. Only valid once the app is ready.
const say = (ja, en) => (app.getLocale().toLowerCase().startsWith("ja") ? ja : en);
let backend = null;
let window = null;

function isUp() {
  return new Promise((resolve) => {
    const request = http.get(`${URL}api/state`, (response) => { response.resume(); resolve(response.statusCode === 200); });
    request.on("error", () => resolve(false));
    request.setTimeout(1500, () => { request.destroy(); resolve(false); });
  });
}

async function startBackend() {
  if (await isUp()) { return; }
  const python = path.join(ROOT, "backend", ".venv", "Scripts", "python.exe");
  if (!fs.existsSync(python)) { throw new Error(say("バックエンドの環境が見つかりません: ", "The backend environment was not found: ") + python); }
  fs.mkdirSync(path.join(ROOT, "runtime"), { recursive: true });
  const log = fs.openSync(path.join(ROOT, "runtime", "backend.log"), "a");
  backend = spawn(python, ["-X", "utf8", "-W", "ignore", "-m", "uvicorn", "dsb.server:app", "--app-dir", path.join(ROOT, "backend"), "--host", "127.0.0.1", "--port", String(PORT)],
    { cwd: ROOT, stdio: ["ignore", log, log], windowsHide: true });
  for (let i = 0; i < 240; i++) {
    if (await isUp()) { return; }
    if (backend.exitCode !== null) { throw new Error(say("バックエンドが起動できませんでした。runtime/backend.log を確認してください", "The backend could not start. See runtime/backend.log")); }
    await new Promise((done) => setTimeout(done, 500));
  }
  throw new Error(say("バックエンドの起動がタイムアウトしました", "The backend timed out while starting"));
}

const pickers = () => ({
  "file-motion": { title: say("モーションデータ（BVH）を選択", "Choose motion data (BVH)"), many: true, filters: [{ name: "BVH", extensions: ["bvh"] }] },
  "file-song": { title: say("曲を選択", "Choose a song"), filters: [{ name: say("音声", "Audio"), extensions: ["wav", "flac", "mp3", "ogg", "aiff", "aif"] }] },
  "file-model": { title: say("モデル（VRM / GLB / FBX）を選択", "Choose a model (VRM / GLB / FBX)"), filters: [{ name: say("モデル", "Model"), extensions: ["vrm", "glb", "fbx"] }] },
  "file-pmx": { title: say("MMD モデル（PMX）を選択", "Choose an MMD model (PMX)"), filters: [{ name: "PMX", extensions: ["pmx"] }] },
});

ipcMain.handle("pick", async (_event, kind) => {
  const picker = pickers()[kind];
  if (!picker) { return []; }
  const result = await dialog.showOpenDialog(window, { title: picker.title, properties: picker.many ? ["openFile", "multiSelections"] : ["openFile"], filters: picker.filters });
  return result.canceled ? [] : result.filePaths;
});

// Shows an exported file in Explorer. Only files inside the app's own export folder are accepted.
ipcMain.handle("reveal", (_event, file) => {
  const exportsFolder = path.resolve(process.env.DSB_DATA || path.join(ROOT, "data"), "exports");
  const target = path.resolve(String(file));
  if (target.startsWith(exportsFolder + path.sep) && fs.existsSync(target)) { shell.showItemInFolder(target); }
});

app.whenReady().then(async () => {
  try {
    await startBackend();
  } catch (error) {
    dialog.showErrorBox("Dance Studio Bricks", String(error.message ?? error));
    app.quit();
    return;
  }
  Menu.setApplicationMenu(null);  // the default menu would take Ctrl+Z, Ctrl+Y and Ctrl+R before the page
  window = new BrowserWindow({ width: 1920, height: 1080, minWidth: 1280, minHeight: 760, backgroundColor: "#121318", title: "Dance Studio Bricks", autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, "preload.cjs"), contextIsolation: true, sandbox: true } });
  window.maximize();
  window.webContents.setWindowOpenHandler(({ url }) => {  // license links open in the user's browser
    if (url.startsWith("https://")) { shell.openExternal(url); }
    return { action: "deny" };
  });
  if (process.env.DSB_SMOKE) {
    window.webContents.once("did-finish-load", () => {
      setTimeout(async () => {
        const text = await window.webContents.executeJavaScript("document.title + ' | bridge=' + (typeof window.dsb) + ' | library=' + document.getElementById('libCount').textContent");
        console.log("SMOKE " + text);
        const played = await window.webContents.executeJavaScript(`(async () => {
          const a = document.getElementById("audio");
          if (!a.src) { return "no song"; }
          document.getElementById("btnPlay").click();
          await new Promise((done) => setTimeout(done, 1500));
          const result = { time: a.currentTime, paused: a.paused, error: a.error && a.error.message, clock: document.getElementById("timecode").textContent };
          a.pause();
          return JSON.stringify(result);
        })()`, true);
        console.log("PLAYBACK " + played);
        app.quit();
      }, 2500);
    });
  }
  await window.webContents.session.clearCache();  // a page kept from before an update would hide the new one
  await window.loadURL(URL);
});

app.on("window-all-closed", () => app.quit());
app.on("quit", () => { if (backend && backend.exitCode === null) { backend.kill(); } });
