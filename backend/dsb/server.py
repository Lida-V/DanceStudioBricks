"""Local HTTP API for the app. Binds to 127.0.0.1 only."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .installer import PREPARE_TIMEOUT, InstallError, Installer
from .mb_client import EngineError
from .studio import ROOT, Studio, StudioError

app = FastAPI(title="Dance Studio Bricks")
studio = Studio(background=True, seed=ROOT / "motions")
installer = Installer(warm=lambda: studio.engine.start(timeout=PREPARE_TIMEOUT), abort_warm=studio.engine.abort_start, on_ready=studio.environment_changed)


@app.exception_handler(StudioError)
async def _studio_error(_: Request, error: StudioError):
    return JSONResponse({"error": str(error)}, status_code=400)


@app.exception_handler(EngineError)
async def _engine_error(_: Request, error: EngineError):
    return JSONResponse({"error": str(error)}, status_code=503)


@app.exception_handler(InstallError)
async def _install_error(_: Request, error: InstallError):
    return JSONResponse({"error": str(error)}, status_code=400)


class Install(BaseModel):
    root: str
    ids: list[str]
    accepted: bool = False


class Paths(BaseModel):
    paths: list[str]


class SongRequest(BaseModel):
    path: str
    factor: float | None = None


class ArrangeRequest(BaseModel):
    start: float | None = None
    end: float | None = None
    seed: int | None = None
    tags: list[str] | None = None


class ShuffleRequest(BaseModel):
    tags: list[str] | None = None


class ArrangeFilterRequest(BaseModel):
    tags: list[str] = []


class FillGapRequest(BaseModel):
    start: float
    end: float
    seed: int | None = None
    tags: list[str] | None = None


class AddClip(BaseModel):
    phraseId: str
    start: float
    end: float | None = None
    x: float | None = None
    z: float | None = None
    exact: bool = False
    fit: str | None = None
    bars: float | None = None


class PasteClip(BaseModel):
    clip: dict
    start: float


class CutClip(BaseModel):
    at: float
    mode: str = "split"


class Reorder(BaseModel):
    order: list[str]


class Search(BaseModel):
    query: str
    limit: int = 20


class Sections(BaseModel):
    sections: list[dict]


class Export(BaseModel):
    format: str
    path: str | None = None
    pmx: str | None = None
    rest: str = "bones"
    armDegrees: float = 45.0


class NewDancer(BaseModel):
    copyFrom: str | None = None


class Formation(BaseModel):
    time: float
    spots: dict
    name: str = ""


class Model(BaseModel):
    path: str | None = None
    mapping: dict | None = None
    everyone: bool = False


@app.get("/api/health")
def health():
    return {"ok": True, "app": "Dance Studio Bricks"}


@app.get("/api/state")
def state():
    return studio.state()


@app.post("/api/library/import")
def import_motion(body: Paths):
    result = studio.import_motion(body.paths)
    return {**result, "state": studio.state()}


@app.patch("/api/library/{phrase_id}")
def update_phrase(phrase_id: str, body: dict):
    return studio.update_phrase(phrase_id, body)


@app.delete("/api/library/{phrase_id}")
def delete_phrase(phrase_id: str):
    studio.delete_phrase(phrase_id)
    return studio.state()


@app.post("/api/library/search")
def search(body: Search):
    return {"results": studio.search(body.query, body.limit)}


@app.post("/api/song")
def load_song(body: SongRequest):
    return studio.load_song(body.path, body.factor)


@app.put("/api/song/sections")
def sections(body: Sections):
    return studio.edit_sections(body.sections)


@app.put("/api/flow")
def flow(body: dict):
    return studio.set_flow(body)


@app.put("/api/arrange-filter")
def set_arrange_filter(body: ArrangeFilterRequest):
    return studio.set_arrange_filter(body.tags)


@app.post("/api/arrange")
def arrange(body: ArrangeRequest | None = None):
    req = body or ArrangeRequest()
    return studio.arrange(req.start, req.end, req.seed, req.tags)


@app.post("/api/shuffle")
def shuffle(body: ShuffleRequest | None = None):
    return studio.shuffle(body.tags if body is not None else None)


@app.post("/api/clips")
def add_clip(body: AddClip):
    return studio.add_clip(body.phraseId, body.start, body.end, body.x, body.z, body.exact, body.fit, body.bars)


@app.post("/api/clips/fill-gap")
def fill_gap(body: FillGapRequest):
    return studio.fill_gap(body.start, body.end, body.seed, body.tags)


@app.post("/api/clips/paste")
def paste_clip(body: PasteClip):
    return studio.paste_clip(body.clip, body.start)


@app.post("/api/clips/reorder")
def reorder_clips(body: Reorder):
    return studio.reorder_clips(body.order)


@app.patch("/api/clips/{clip_id}")
def update_clip(clip_id: str, body: dict):
    return studio.update_clip(clip_id, body)


@app.post("/api/clips/{clip_id}/cut")
def cut_clip(clip_id: str, body: CutClip):
    return studio.cut_clip(clip_id, body.at, body.mode)


@app.delete("/api/clips/{clip_id}")
def delete_clip(clip_id: str):
    return studio.delete_clip(clip_id)


@app.post("/api/undo")
def undo():
    return studio.undo()


@app.post("/api/redo")
def redo():
    return studio.redo()


@app.get("/api/joins/wait")
def wait_for_joins(version: int | None = None):
    return studio.wait_for_joins(version)


@app.get("/api/clips/{clip_id}/candidates")
def candidates(clip_id: str):
    return {"candidates": studio.candidates(clip_id)}


@app.get("/api/joins/{clip_id}/options")
def join_options(clip_id: str):
    return studio.join_options(clip_id)


@app.post("/api/dancers")
def add_dancer(body: NewDancer):
    return studio.add_dancer(body.copyFrom)


@app.post("/api/dancers/sync")
def sync_dancers():
    return studio.sync_dancers()


@app.post("/api/dancers/{dancer_id}/select")
def select_dancer(dancer_id: str):
    return studio.select_dancer(dancer_id)


@app.patch("/api/dancers/{dancer_id}")
def update_dancer(dancer_id: str, body: dict):
    return studio.update_dancer(dancer_id, body)


@app.delete("/api/dancers/{dancer_id}")
def delete_dancer(dancer_id: str):
    return studio.delete_dancer(dancer_id)


@app.put("/api/dancers/{dancer_id}/model")
def set_model(dancer_id: str, body: Model):
    return studio.set_model(dancer_id, body.path, body.mapping, body.everyone)


@app.get("/api/dancers/{dancer_id}/model/{name:path}")
def model_file(dancer_id: str, name: str):
    return FileResponse(studio.model_file(dancer_id, name), headers={"Cache-Control": "no-store"})


@app.put("/api/formations")
def set_formation(body: Formation):
    return studio.set_formation(body.time, body.spots, body.name)


@app.delete("/api/formations/{formation_id}")
def delete_formation(formation_id: str):
    return studio.delete_formation(formation_id)


@app.get("/api/preview.bin")
def preview(dancer: str | None = None):
    return Response(studio.preview_bytes(dancer), media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


@app.get("/api/turns.bin")
def turns(dancer: str | None = None):
    return Response(studio.turn_bytes(dancer), media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


@app.get("/api/audio")
def audio():
    path = studio.project.get("songPath")
    if not path or not Path(path).is_file():
        raise HTTPException(404, "曲がありません")
    return FileResponse(path)


@app.post("/api/export")
def export(body: Export):
    return studio.export(body.format, body.path, body.pmx, body.rest, body.armDegrees)


@app.post("/api/export/upload")
async def export_upload(request: Request, name: str):
    return studio.save_export(name, await request.body())


@app.get("/api/components")
def components():
    return installer.status()


@app.get("/api/components/check")
def check_components(root: str, ids: str = ""):
    return installer.check(root, [item for item in ids.split(",") if item])


@app.post("/api/components/install")
def install_components(body: Install):
    return installer.start(body.root, body.ids, body.accepted)


@app.post("/api/components/cancel")
def cancel_install():
    installer.cancel()
    return installer.status()


@app.on_event("shutdown")
def _shutdown():
    installer.cancel()
    studio.close()


_dist = Path(os.environ.get("DSB_DIST", ROOT / "frontend" / "dist"))
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="ui")
