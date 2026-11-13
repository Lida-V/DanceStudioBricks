const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("dsb", { pick: (kind) => ipcRenderer.invoke("pick", kind), reveal: (file) => ipcRenderer.invoke("reveal", file) });
