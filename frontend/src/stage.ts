import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { S, ApiError, DANCER_COLORS, activeDancer, call, applyChanges, applyState, beginInteraction, clips, dancers, endInteraction, isDoublePress, loadPreview, markGenerating, phraseOf, refresh, selectedClip, subscribe, toast, warnLockedDrag, type Clip, type ModelRef } from "./store";
import { openInsertMenu, openPhraseMenu } from "./insertMenu";
import { Rig, loadDefaultRig, loadRig } from "./model";
import { clipActions, exactInsertLabel, insertLabel, insertPhrase, jumpToClip, replaceClip, snapTime } from "./timeline";
import { seek } from "./transport";

const BONES: [string, string, number][] = [
  ["Hips", "Spine1", 0.115], ["Spine1", "Spine2", 0.115], ["Spine2", "Chest", 0.125], ["Chest", "Neck1", 0.075], ["Neck1", "Neck2", 0.05], ["Neck2", "Head", 0.05],
  ["Chest", "LeftShoulder", 0.06], ["LeftShoulder", "LeftArm", 0.06], ["LeftArm", "LeftForeArm", 0.048], ["LeftForeArm", "LeftHand", 0.04],
  ["Chest", "RightShoulder", 0.06], ["RightShoulder", "RightArm", 0.06], ["RightArm", "RightForeArm", 0.048], ["RightForeArm", "RightHand", 0.04],
  ["Hips", "LeftLeg", 0.08], ["LeftLeg", "LeftShin", 0.07], ["LeftShin", "LeftFoot", 0.052], ["LeftFoot", "LeftToeBase", 0.042],
  ["Hips", "RightLeg", 0.08], ["RightLeg", "RightShin", 0.07], ["RightShin", "RightFoot", 0.052], ["RightFoot", "RightToeBase", 0.042],
];
const UP = new THREE.Vector3(0, 1, 0);
const VIEWS: Record<string, { position: [number, number, number]; target: [number, number, number] }> = {
  "3d": { position: [4.2, 3.0, 11.0], target: [0, 1.0, 3.0] },
  front: { position: [0, 1.2, 13.0], target: [0, 1.0, 3.0] },
  top: { position: [0, 13.0, 3.01], target: [0, 0, 3.0] },
};

let renderer: THREE.WebGLRenderer;
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let controls: OrbitControls;
let markers = new THREE.Group();
let boundaryLine: THREE.Line;
let dropRing: THREE.Mesh;

const LIMB = new THREE.CylinderGeometry(1, 1, 1, 14);
const BALL = new THREE.SphereGeometry(1, 18, 14);
const pa = new THREE.Vector3(), pb = new THREE.Vector3(), dir = new THREE.Vector3();
let limbJoints: [number, number][] = [];  // BONES as indexes into the preview's joints
let special = { head: -1, hips: -1, leftFoot: -1, rightFoot: -1 };

/** World position of one joint at a fractional frame of a dancer's preview. */
function readJoint(data: Float32Array, count: number, frame: number, joint: number, out: THREE.Vector3): THREE.Vector3 {
  const low = Math.floor(frame), blend = frame - low;
  const i0 = (low * count + joint) * 3, i1 = ((low + 1) * count + joint) * 3;
  return out.set(data[i0] * (1 - blend) + data[i1] * blend, data[i0 + 1] * (1 - blend) + data[i1 + 1] * blend, data[i0 + 2] * (1 - blend) + data[i1 + 2] * blend);
}

/** The standard mannequin: one per dancer, drawn straight from joint positions. */
class Mannequin {
  readonly group = new THREE.Group();
  private readonly skin = new THREE.MeshStandardMaterial({ color: 0xf1f0ec, roughness: 0.55 });
  private readonly limbs: { mesh: THREE.Mesh; radius: number; ends: [THREE.Mesh, THREE.Mesh] }[] = [];
  private readonly head = new THREE.Mesh(BALL, this.skin);

  constructor() {
    for (const [, , radius] of BONES) {
      const mesh = new THREE.Mesh(LIMB, this.skin);
      const ends: [THREE.Mesh, THREE.Mesh] = [new THREE.Mesh(BALL, this.skin), new THREE.Mesh(BALL, this.skin)];
      ends.forEach((end) => end.scale.setScalar(radius));
      this.group.add(mesh, ...ends);
      this.limbs.push({ mesh, radius, ends });
    }
    this.head.scale.set(0.095, 0.115, 0.1);
    this.group.add(this.head);
    this.group.visible = false;
  }

  setColor(color: string) { this.skin.color.set(color); }

  pose(data: Float32Array, count: number, frame: number) {
    this.limbs.forEach((limb, index) => {
      const [a, b] = limbJoints[index] ?? [-1, -1];
      if (a < 0 || b < 0) { return; }
      readJoint(data, count, frame, a, pa); readJoint(data, count, frame, b, pb);
      dir.copy(pb).sub(pa);
      const length = dir.length();
      limb.mesh.position.copy(pa).addScaledVector(dir, 0.5);
      limb.mesh.scale.set(limb.radius, Math.max(1e-4, length), limb.radius);
      if (length > 1e-5) { limb.mesh.quaternion.setFromUnitVectors(UP, dir.divideScalar(length)); }
      limb.ends[0].position.copy(pa);
      limb.ends[1].position.copy(pb);
    });
    if (special.head >= 0) {
      readJoint(data, count, frame, special.head, pa);
      this.head.position.set(pa.x, pa.y + 0.075, pa.z);
    }
  }

  dispose() { this.group.removeFromParent(); this.skin.dispose(); }
}

type Actor = { body: Mannequin; rig: Rig | null; key: string; loading: boolean };
const actors = new Map<string, Actor>();
const modelListeners: (() => void)[] = [];
/** Called whenever a model finishes loading, fails, or is taken off. */
export const onModelChange = (fn: () => void) => modelListeners.push(fn);
export const rigOf = (dancerId: string): Rig | null => actors.get(dancerId)?.rig ?? null;
export const isModelLoading = (dancerId: string): boolean => actors.get(dancerId)?.loading ?? false;
/** True when some dancer is shown with a model that can be written out as GLB. A VRM cannot: its materials do not survive, and VRMA is the format for it. */
export const hasModel = (): boolean => [...actors.values()].some((actor) => actor.rig?.ready && !actor.rig.vrm);

async function loadModel(dancerId: string, actor: Actor, model: ModelRef | undefined, color: string) {
  actor.rig?.dispose();
  actor.rig = null;
  const key = actor.key;
  actor.loading = true;
  if (model) { modelListeners.forEach((fn) => fn()); }
  try {
    const rig = model
      ? await loadRig(`/api/dancers/${dancerId}/model/${encodeURIComponent(model.name)}?v=${encodeURIComponent(model.path)}`, model.mapping, S.modelBones)
      : await loadDefaultRig(S.modelBones, color);
    if (actor.key !== key || actors.get(dancerId) !== actor) { rig.dispose(); return; }  // another model was chosen meanwhile
    actor.rig = rig;
    if (rig.ready) {
      scene.add(rig.root);
      S.wantsTurns.add(dancerId);
      if (!S.turns[dancerId]) { await loadPreview(); }
    }
  } catch (error) {
    if (model) { toast(`モデルを読み込めませんでした: ${(error as Error).message}`, "error"); }
  } finally {
    if (actor.key === key) { actor.loading = false; }
  }
  poseAt(S.playhead);
  modelListeners.forEach((fn) => fn());
}

/** One actor per dancer, each with its colour and the model chosen for it. */
function syncActors() {
  const cast = dancers();
  for (const [id, actor] of actors) {
    if (!cast.some((dancer) => dancer.id === id)) {
      actor.body.dispose();
      actor.rig?.dispose();
      actors.delete(id);
      S.wantsTurns.delete(id);
    }
  }
  cast.forEach((dancer, index) => {
    const color = DANCER_COLORS[index % DANCER_COLORS.length];
    let actor = actors.get(dancer.id);
    if (!actor) {
      actor = { body: new Mannequin(), rig: null, key: "", loading: false };
      scene.add(actor.body.group);
      actors.set(dancer.id, actor);
    }
    actor.body.setColor(color);
    if (actor.rig?.builtin) { actor.rig.setTint(color); }
    if (!S.modelBones.length) { return; }
    const model = S.project?.models?.[dancer.id];
    const key = model ? `${model.path}|${JSON.stringify(model.mapping)}` : "__builtin__";
    if (key !== actor.key) {
      actor.key = key;
      void loadModel(dancer.id, actor, model, color);
    }
  });
}

/** What the stage moves while a phrase of the dancer being edited is dragged. */
function carried(): THREE.Object3D[] {
  const actor = actors.get(activeDancer()?.id ?? "");
  return actor ? [actor.body.group, ...(actor.rig ? [actor.rig.root] : [])] : [];
}
let dirty = true;
let markersStale = false;
let lastLayout = 0;

const numberTextures = new Map<string, THREE.CanvasTexture>();

function numberTexture(text: string, active: boolean): THREE.CanvasTexture {
  const key = `${text}:${active}`;
  let texture = numberTextures.get(key);
  if (!texture) {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 64;
    const c = canvas.getContext("2d")!;
    c.fillStyle = active ? "#ff8a70" : "#f0eee8";
    c.beginPath(); c.arc(32, 32, 30, 0, Math.PI * 2); c.fill();
    c.fillStyle = "#111";
    c.font = `700 ${text.length > 2 ? 28 : 36}px Bahnschrift, sans-serif`;
    c.textAlign = "center"; c.textBaseline = "middle";
    c.fillText(text, 32, 34);
    texture = new THREE.CanvasTexture(canvas);
    numberTextures.set(key, texture);
  }
  return texture;
}

function numberSprite(text: string, active: boolean): THREE.Sprite {
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: numberTexture(text, active), depthTest: false }));
  sprite.scale.setScalar(0.34);
  sprite.renderOrder = active ? 3 : 2;
  return sprite;
}

type Spot = { x: number; z: number };

/**
 * Where to draw each number so none cover another. Numbers closer than `reach` spread into a ring around their
 * group; `fixed` (the selected phrase) keeps its true place, because that is the one being grabbed.
 */
function spreadNumbers(points: Spot[], fixed: number, reach: number, gap: number): Spot[] {
  const group = points.map((_, index) => index);
  const find = (index: number): number => (group[index] === index ? index : (group[index] = find(group[index])));
  points.forEach((a, i) => {
    for (let j = 0; j < i; j++) {
      if (Math.hypot(a.x - points[j].x, a.z - points[j].z) < reach) { group[find(i)] = find(j); }
    }
  });
  const clusters = new Map<number, number[]>();
  points.forEach((_, index) => {
    const root = find(index);
    clusters.set(root, [...(clusters.get(root) ?? []), index]);
  });
  const out = points.map((point) => ({ ...point }));
  for (const members of clusters.values()) {
    if (members.length < 2) { continue; }
    const hasFixed = members.includes(fixed);
    const moving = members.filter((index) => index !== fixed);
    const center = hasFixed ? points[fixed] : { x: moving.reduce((sum, i) => sum + points[i].x, 0) / moving.length, z: moving.reduce((sum, i) => sum + points[i].z, 0) / moving.length };
    const radius = Math.max(reach * (hasFixed ? 1.15 : 0.8), (moving.length * gap) / (2 * Math.PI));
    moving.forEach((index, order) => {
      const angle = -Math.PI / 2 + (order / moving.length) * 2 * Math.PI;
      out[index] = { x: center.x + radius * Math.cos(angle), z: center.z + radius * Math.sin(angle) };
    });
  }
  // A ring can land on a neighbouring group, so push any numbers that still touch apart.
  const minimum = reach * 1.05;
  for (let pass = 0; pass < 40; pass++) {
    let pushed = false;
    for (let i = 0; i < out.length; i++) {
      for (let j = i + 1; j < out.length; j++) {
        const dx = out[j].x - out[i].x, dz = out[j].z - out[i].z;
        const distance = Math.hypot(dx, dz);
        if (distance >= minimum || distance < 1e-6) { continue; }
        const share = (minimum - distance) / (i === fixed || j === fixed ? 1 : 2) / distance;
        if (i !== fixed) { out[i].x -= dx * share; out[i].z -= dz * share; }
        if (j !== fixed) { out[j].x += dx * share; out[j].z += dz * share; }
        pushed = true;
      }
    }
    if (!pushed) { break; }
  }
  return out;
}

/** World positions for the 3D numbers, spread in screen space so they stay readable from any camera angle. */
function layoutNumbers(list: Clip[], fixed: number): THREE.Vector3[] {
  const size = renderer.getSize(new THREE.Vector2());
  camera.updateMatrixWorld();
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
  const forward = camera.getWorldDirection(new THREE.Vector3());
  const halfTan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const items = list.map((clip) => {
    const world = new THREE.Vector3(clip.x, 0.32, clip.z);
    const depth = Math.max(0.1, world.clone().sub(camera.position).dot(forward));
    const screen = world.clone().project(camera);
    return { world, perMetre: size.y / 2 / (depth * halfTan), x: ((screen.x + 1) / 2) * size.x, z: ((1 - screen.y) / 2) * size.y };
  });
  const diameter = 0.34 * (items.reduce((sum, item) => sum + item.perMetre, 0) / Math.max(1, items.length));
  const spots = spreadNumbers(items.map((item) => ({ x: item.x, z: item.z })), fixed, diameter * 0.95, diameter * 1.12);
  return items.map((item, index) => item.world.clone()
    .addScaledVector(right, (spots[index].x - item.x) / item.perMetre)
    .addScaledVector(up, -(spots[index].z - item.z) / item.perMetre));
}

const HANDLE_RADIUS = 0.8;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const floorHit = new THREE.Vector3();
let picks: THREE.Object3D[] = [];
let editing = false;

function disposeGroup(group: THREE.Object3D) {
  group.traverse((node) => {
    if ((node as THREE.Sprite).isSprite) { ((node as THREE.Sprite).material as THREE.Material).dispose(); return; }  // textures and sprite geometry are shared
    const item = node as THREE.Mesh;
    item.geometry?.dispose();
    const material = item.material as (THREE.Material & { map?: THREE.Texture }) | undefined;
    material?.map?.dispose();
    material?.dispose?.();
  });
}

const facingVector = (degrees: number) => new THREE.Vector3(Math.sin(THREE.MathUtils.degToRad(degrees)), 0, Math.cos(THREE.MathUtils.degToRad(degrees)));

function addGizmo(group: THREE.Group, clip: Clip) {
  const y = 0.02;
  const guide = new THREE.Mesh(new THREE.RingGeometry(HANDLE_RADIUS - 0.012, HANDLE_RADIUS + 0.012, 72), new THREE.MeshBasicMaterial({ color: 0xff8a70, transparent: true, opacity: 0.45, side: THREE.DoubleSide }));
  guide.rotation.x = -Math.PI / 2;
  guide.position.set(clip.x, y, clip.z);
  group.add(guide);
  const direction = facingVector(clip.facing);
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, HANDLE_RADIUS, 10), new THREE.MeshBasicMaterial({ color: 0xff8a70 }));
  shaft.position.set(clip.x, y, clip.z).addScaledVector(direction, HANDLE_RADIUS / 2);
  shaft.quaternion.setFromUnitVectors(UP, direction);
  group.add(shaft);
  const knob = new THREE.Mesh(new THREE.SphereGeometry(0.11, 18, 14), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  knob.position.set(clip.x, 0.11, clip.z).addScaledVector(direction, HANDLE_RADIUS);
  knob.userData = { kind: "rotate", id: clip.id };
  group.add(knob);
  picks.push(knob);
}

function rebuildMarkers() {
  scene.remove(markers);
  disposeGroup(markers);
  markers = new THREE.Group();
  picks = [];
  const list = clips();
  const picked = selectedClip();
  if (picked) { addGizmo(markers, picked); }
  // The other dancers' routes, faint and not to be grabbed.
  dancers().forEach((dancer, index) => {
    if (dancer.id === activeDancer()?.id || !dancer.clips.length || dancer.hidden) { return; }
    const color = new THREE.Color(DANCER_COLORS[index % DANCER_COLORS.length]);
    const route = dancer.clips.map((c) => new THREE.Vector3(c.x, 0.008, c.z));
    if (route.length > 1) { markers.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(route), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.28 }))); }
    for (const point of route) {
      const dot = new THREE.Mesh(new THREE.CircleGeometry(0.07, 16), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.45, side: THREE.DoubleSide }));
      dot.rotation.x = -Math.PI / 2;
      dot.position.copy(point);
      markers.add(dot);
    }
  });
  const numberSpots = layoutNumbers(list, picked ? list.indexOf(picked) : -1);
  if (list.length > 1) {
    const points = list.map((c) => new THREE.Vector3(c.x, 0.012, c.z));
    markers.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: 0xe9d9b0, transparent: true, opacity: 0.5 })));
  }
  list.forEach((c, index) => {
    const active = c === picked;
    const ring = new THREE.Mesh(new THREE.RingGeometry(active ? 0.2 : 0.14, active ? 0.26 : 0.18, 40), new THREE.MeshBasicMaterial({ color: active ? 0xff8a70 : 0xf0eee8, transparent: true, opacity: active ? 0.95 : 0.4, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(c.x, 0.014, c.z);
    markers.add(ring);
    const grip = new THREE.Mesh(new THREE.CircleGeometry(0.34, 20), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }));
    grip.rotation.x = -Math.PI / 2;
    grip.position.set(c.x, 0.03, c.z);
    grip.userData = { kind: "move", id: c.id };
    markers.add(grip);
    picks.push(grip);
    if (list.length <= 150) {
      const spot = numberSpots[index];
      const sprite = numberSprite(String(index + 1), active);
      sprite.position.copy(spot);
      sprite.userData = { kind: "move", id: c.id };
      markers.add(sprite);
      picks.push(sprite);
      if (spot.distanceToSquared(new THREE.Vector3(c.x, 0.32, c.z)) > 1e-6) {
        markers.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(c.x, 0.02, c.z), spot]), new THREE.LineBasicMaterial({ color: 0xf0eee8, transparent: true, opacity: 0.55 })));
      }
    }
  });
  markers.visible = !S.uiHidden;
  scene.add(markers);
  dirty = true;
}

export function initStage() {
  const host = document.getElementById("stage3d")!;
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  host.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1015);
  camera = new THREE.PerspectiveCamera(32, 1, 0.05, 80);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.addEventListener("change", () => { dirty = true; markersStale = true; });
  setView("3d");
  scene.add(new THREE.HemisphereLight(0xffffff, 0x30323c, 1.5));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(3, 8, 7);
  scene.add(key);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(16, 13), new THREE.MeshStandardMaterial({ color: 0x262933, roughness: 0.95 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, 0, 3);
  scene.add(floor);
  const grid = new THREE.GridHelper(16, 16, 0x454a57, 0x383c47);
  grid.position.set(0, 0.002, 3);
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.55;
  scene.add(grid);
  const edge = [[-4, 0], [4, 0], [4, 6], [-4, 6], [-4, 0]].map(([x, z]) => new THREE.Vector3(x, 0.01, z));
  boundaryLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(edge), new THREE.LineDashedMaterial({ color: 0xff8a70, dashSize: 0.22, gapSize: 0.16 }));
  boundaryLine.computeLineDistances();
  boundaryLine.visible = !S.uiHidden;
  scene.add(boundaryLine);
  dropRing = new THREE.Mesh(new THREE.RingGeometry(0.22, 0.31, 44), new THREE.MeshBasicMaterial({ color: 0xff8a70, transparent: true, opacity: 0.9, side: THREE.DoubleSide }));
  dropRing.rotation.x = -Math.PI / 2;
  dropRing.visible = false;
  scene.add(dropRing);
  new ResizeObserver(resize).observe(host);
  resize();
  subscribe(() => { resolveJoints(); syncActors(); rebuildMarkers(); drawMini(); poseAt(S.playhead); });
  document.getElementById("viewTabs")!.addEventListener("click", (event) => {
    const view = (event.target as HTMLElement).dataset.view;
    if (!view) { return; }
    document.querySelectorAll("#viewTabs span").forEach((el) => el.classList.toggle("on", (el as HTMLElement).dataset.view === view));
    setView(view);
  });
  bindMini();
  bindDirectEdit();
  const loop = () => {
    controls.update();
    if (markersStale && !editing && performance.now() - lastLayout > 80) {  // the number layout depends on the camera
      lastLayout = performance.now();
      markersStale = false;
      rebuildMarkers();
    }
    if (dirty) { renderer.render(scene, camera); dirty = false; }
    requestAnimationFrame(loop);
  };
  loop();
}

function setView(name: string) {
  const view = VIEWS[name];
  camera.position.set(...view.position);
  controls.target.set(...view.target);
  controls.update();
  dirty = true;
}

function resize() {
  const host = document.getElementById("stage3d")!;
  const stage = document.getElementById("stage")!;
  const width = stage.clientWidth, height = stage.clientHeight;
  renderer.setSize(width, height);
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
  void host;
  dirty = true;
  markersStale = true;
}

export function syncStageUiHidden() {
  if (markers) { markers.visible = !S.uiHidden; }
  if (boundaryLine) { boundaryLine.visible = !S.uiHidden; }
  if (dropRing) { dropRing.visible = false; }
  document.getElementById("stageDropHint")?.remove();
  if (renderer) {
    resize();
    requestAnimationFrame(resize);
  }
}

function resolveJoints() {
  const names = S.jointNames;
  limbJoints = BONES.map(([from, to]) => [names.indexOf(from), names.indexOf(to)]);
  special = { head: names.indexOf("Head"), hips: names.indexOf("Hips"), leftFoot: names.indexOf("LeftFoot"), rightFoot: names.indexOf("RightFoot") };
}

const FPS = 30;
const hipsAt = new THREE.Vector3();
const anklesAt: [THREE.Vector3, THREE.Vector3] = [new THREE.Vector3(), new THREE.Vector3()];

/** Puts one dancer in the pose at a fractional frame: on its model when one is ready, on the mannequin otherwise. */
function poseActor(dancerId: string, actor: Actor, frame: number | null) {
  const motion = S.previews[dancerId];
  const count = S.jointNames.length;
  const rig = actor.rig?.ready ? actor.rig : null;
  const turns = S.turns[dancerId];
  if (!motion || !count || dancers().find((dancer) => dancer.id === dancerId)?.hidden) {
    actor.body.group.visible = false;
    if (actor.rig) { actor.rig.root.visible = false; }
    return;
  }
  const frames = motion.data.length / (count * 3);
  const f = Math.min(frames - 1.001, Math.max(0, frame ?? 0));
  const onModel = !!rig && !!turns && special.hips >= 0;
  if (onModel) {
    const feet = special.leftFoot >= 0 && special.rightFoot >= 0;
    if (feet) { readJoint(motion.data, count, f, special.leftFoot, anklesAt[0]); readJoint(motion.data, count, f, special.rightFoot, anklesAt[1]); }
    rig!.pose(turns!.data, f, readJoint(motion.data, count, f, special.hips, hipsAt), feet ? anklesAt : null);
  } else {
    actor.body.pose(motion.data, count, f);
  }
  actor.body.group.visible = !onModel;
  if (actor.rig) { actor.rig.root.visible = onModel; }
}

export function poseAt(time: number) {
  for (const [id, actor] of actors) { poseActor(id, actor, time * (S.composed?.fps ?? FPS)); }
  dirty = true;
}

/**
 * Bakes each shown model's dance into a GLB (the model with its animation) and hands it to the app to keep with the
 * other exports. One file per dancer that has a model.
 */
export async function exportModel(): Promise<{ path: string; files: { path: string; dancer: string; bytes: number }[] }> {
  const { GLTFExporter } = await import("three/examples/jsm/exporters/GLTFExporter.js");
  const files: { path: string; dancer: string; bytes: number }[] = [];
  const count = S.jointNames.length;
  try {
    for (const dancer of dancers()) {
      const actor = actors.get(dancer.id);
      const motion = S.previews[dancer.id];
      if (!actor?.rig?.ready || actor.rig.vrm || !S.turns[dancer.id] || !motion || !count) { continue; }
      const frames = motion.data.length / (count * 3);
      const { hips, nodes } = actor.rig.driven();
      const times = new Float32Array(frames), places = new Float32Array(frames * 3);
      const turns = nodes.map(() => new Float32Array(frames * 4));
      for (let frame = 0; frame < frames; frame++) {
        poseActor(dancer.id, actor, Math.min(frame, frames - 1.001));
        times[frame] = frame / FPS;
        hips.position.toArray(places, frame * 3);
        nodes.forEach((node, index) => node.quaternion.toArray(turns[index], frame * 4));
      }
      const clip = new THREE.AnimationClip("dance", -1, [new THREE.VectorKeyframeTrack(`${hips.uuid}.position`, times, places),
        ...nodes.map((node, index) => new THREE.QuaternionKeyframeTrack(`${node.uuid}.quaternion`, times, turns[index]))]);
      const buffer = await new GLTFExporter().parseAsync(actor.rig.root, { binary: true, animations: [clip] }) as ArrayBuffer;
      const model = S.project?.models?.[dancer.id]?.name.replace(/\.[^.]+$/, "") ?? "mannequin";
      const response = await fetch(`/api/export/upload?name=${encodeURIComponent(`${model}-${dancer.name}.glb`)}`, { method: "POST", body: buffer });
      if (!response.ok) { throw new ApiError((await response.json().catch(() => ({}))).error ?? `エラー ${response.status}`); }
      files.push({ ...(await response.json()).files[0], dancer: dancer.name });
    }
  } finally {
    poseAt(S.playhead);
  }
  if (!files.length) { throw new ApiError("GLB に書き出せるモデル（GLB / FBX）を表示しているダンサーがいません"); }
  return { path: files[0].path, files };
}

// ---- Editing directly on the stage

function aim(event: MouseEvent) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
}

function pick(event: MouseEvent): { kind: "move" | "rotate"; id: string } | null {
  if (S.uiHidden) { return null; }
  aim(event);
  const hits = raycaster.intersectObjects(picks, false);
  const hit = hits.find((item) => item.object.userData.kind === "rotate") ?? hits.find((item) => (item.object as THREE.Sprite).isSprite) ?? hits[0];
  return hit ? (hit.object.userData as { kind: "move" | "rotate"; id: string }) : null;
}

function floorAt(event: MouseEvent): THREE.Vector3 | null {
  aim(event);
  return raycaster.ray.intersectPlane(floorPlane, floorHit);
}

function showFields(clip: Clip) {
  const write = (field: string, value: number) => {
    const input = document.querySelector<HTMLInputElement>(`[data-field="${field}"]`);
    if (input) { input.value = String(value); }
  };
  write("x", clip.x); write("z", clip.z); write("facing", clip.facing);
}

function bindDirectEdit() {
  const canvas = renderer.domElement;
  let rightPress = { x: 0, y: 0 };
  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    if (S.uiHidden || !S.song || Math.hypot(event.clientX - rightPress.x, event.clientY - rightPress.y) > 5) { return; }  // a right drag pans the camera
    const target = pick(event);
    const list = clips();
    const clip = target?.kind === "move" ? list.find((c) => c.id === target.id) : undefined;
    const spot = floorAt(event);
    const place = spot ? { x: Math.round(THREE.MathUtils.clamp(spot.x, -4, 4) * 20) / 20, z: Math.round(THREE.MathUtils.clamp(spot.z, 0, 6) * 20) / 20 } : undefined;
    const time = snapTime(S.playhead, event.altKey);
    const insAt = clip ? clip.end : time;
    const insWhere = clip ? `${list.indexOf(clip) + 1} 番の次に挿入` : `${exactInsertLabel(time)}${place ? ` ・ 位置 ${place.x.toFixed(1)}, ${place.z.toFixed(1)}` : ""}`;
    const insChoose = clip ? (phraseId: string) => insertPhrase(phraseId, clip.end) : (phraseId: string) => insertPhrase(phraseId, time, place, true);
    const picked = selectedClip();
    if (picked) {
      const name = phraseOf(picked.phraseId)?.name ?? "";
      openPhraseMenu(event.clientX, event.clientY, [
        { title: "モーションを差し替え", where: `${list.indexOf(picked) + 1} 番「${name}」を差し替え`, choose: (phraseId) => replaceClip(picked.id, phraseId), switchLabel: "挿入へ切替", at: picked.start, replaceClip: picked },
        { title: "モーションを挿入", where: insWhere, choose: insChoose, switchLabel: "差し替えに戻す", at: insAt },
      ], clipActions(picked));
      return;
    }
    openInsertMenu(event.clientX, event.clientY, insWhere, insChoose, insAt, clipActions());
  });
  const stageEl = document.getElementById("stage")!;
  const hideStageDrop = () => {
    if (dropRing?.visible) { dropRing.visible = false; dirty = true; }
    document.getElementById("stageDropHint")?.remove();
  };
  window.addEventListener("dragend", hideStageDrop);
  canvas.addEventListener("dragleave", hideStageDrop);
  canvas.addEventListener("dragover", (event) => {
    if (!S.song) { hideStageDrop(); return; }
    const spot = floorAt(event);
    if (!spot) { hideStageDrop(); return; }
    event.preventDefault();
    if (event.dataTransfer) { event.dataTransfer.dropEffect = "copy"; }
    const px = Math.round(THREE.MathUtils.clamp(spot.x, -4, 4) * 20) / 20;
    const pz = Math.round(THREE.MathUtils.clamp(spot.z, 0, 6) * 20) / 20;
    dropRing.position.set(px, 0.02, pz);
    dropRing.visible = true;
    dirty = true;
    const item = S.draggingPhrase ? phraseOf(S.draggingPhrase) : undefined;
    const at = snapTime(S.playhead, event.altKey);
    let hint = document.getElementById("stageDropHint");
    if (!hint) {
      hint = document.createElement("div");
      hint.id = "stageDropHint";
      hint.className = "stage-drop-hint";
      stageEl.appendChild(hint);
    }
    hint.textContent = `${exactInsertLabel(at)} ・ 位置 ${px.toFixed(1)}, ${pz.toFixed(1)}${item ? ` に「${item.name}」` : ""}`;
  });
  canvas.addEventListener("drop", async (event) => {
    event.preventDefault();
    hideStageDrop();
    const phraseId = event.dataTransfer?.getData("text/phrase") || S.draggingPhrase;
    if (!phraseId || !S.song) { return; }
    const spot = floorAt(event);
    const place = spot ? { x: Math.round(THREE.MathUtils.clamp(spot.x, -4, 4) * 20) / 20, z: Math.round(THREE.MathUtils.clamp(spot.z, 0, 6) * 20) / 20 } : undefined;
    await insertPhrase(phraseId, snapTime(S.playhead, event.altKey), place, true);
    void insertLabel;
  });
  canvas.addEventListener("dblclick", (event) => {
    const target = pick(event);
    if (target?.kind === "move") { jumpToClip(target.id); }
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!editing && event.buttons === 0) {
      const over = pick(event);
      const locked = over && clips().find((c) => c.id === over.id)?.locked;
      canvas.style.cursor = over ? (locked ? "not-allowed" : over.kind === "rotate" ? "ew-resize" : "move") : "";
    }
  });
  // Registered for the capture phase so it runs before the camera controls see the press.
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button === 2) { rightPress = { x: event.clientX, y: event.clientY }; }
    if (event.button !== 0) { return; }
    const target = pick(event);
    if (!target) { return; }
    const clip = clips().find((c) => c.id === target.id);
    if (!clip) { return; }
    controls.enabled = false;
    if (clip.locked) {
      S.selected = clip.id;
      refresh();
      warnLockedDrag(event, () => { controls.enabled = true; });
      return;
    }
    editing = true;
    canvas.setPointerCapture(event.pointerId);
    S.selected = clip.id;
    beginInteraction();
    refresh();
    const original = { x: clip.x, z: clip.z, facing: clip.facing };
    let changed = false;
    let started = false;
    const preview = () => {
      // Follow the pointer with the mannequin until the baked motion arrives.
      const shown = S.playhead >= clip.start && S.playhead <= clip.end;
      const yaw = THREE.MathUtils.degToRad(clip.facing - original.facing);
      const cos = Math.cos(yaw), sin = Math.sin(yaw);
      for (const body of carried()) {
        body.rotation.y = shown ? yaw : 0;
        if (shown) { body.position.set(clip.x - (original.x * cos + original.z * sin), 0, clip.z - (-original.x * sin + original.z * cos)); }
        else { body.position.set(0, 0, 0); }
      }
      if (rigOf(activeDancer()?.id ?? "")) { poseAt(S.playhead); }  // a model works out its own pose inside what carries it
    };
    const move = (e: PointerEvent) => {
      // A small dead zone, so a click or double click never nudges the phrase.
      if (!started && Math.hypot(e.clientX - event.clientX, e.clientY - event.clientY) < 4) { return; }
      if (!started) {
        started = true;
        if (!S.playing && (S.playhead < clip.start || S.playhead > clip.end)) { seek(clip.start); }  // show the phrase being edited
      }
      const spot = floorAt(e);
      if (!spot) { return; }
      if (target.kind === "move") {
        clip.x = Math.round(THREE.MathUtils.clamp(spot.x, -4, 4) * 20) / 20;
        clip.z = Math.round(THREE.MathUtils.clamp(spot.z, 0, 6) * 20) / 20;
      } else {
        const step = e.shiftKey ? 15 : 1;
        clip.facing = Math.round(THREE.MathUtils.radToDeg(Math.atan2(spot.x - clip.x, spot.z - clip.z)) / step) * step;
      }
      changed = changed || clip.x !== original.x || clip.z !== original.z || clip.facing !== original.facing;
      preview();
      rebuildMarkers();
      drawMini();
      showFields(clip);
      dirty = true;
    };
    const release = async () => {
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", release);
      canvas.removeEventListener("pointercancel", release);
      controls.enabled = true;
      editing = false;
      if (changed) {
        endInteraction(true);
        markGenerating(clip.id);
        refresh();
        try {
          await applyChanges(await call("PATCH", `/api/clips/${clip.id}`, target.kind === "move" ? { x: clip.x, z: clip.z } : { facing: clip.facing }));
        } catch (error) {
          toast((error as Error).message, "error");
          await applyState(await call("GET", "/api/state"));
        }
      } else {
        endInteraction();
      }
      for (const body of carried()) {
        body.position.set(0, 0, 0);
        body.rotation.y = 0;
      }
      dirty = true;
      seek(S.playhead);
    };
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
  }, { capture: true });
}

export const isEditingStage = () => editing;

// ---- Top-view map

const MX = (x: number) => 129 + x * 25;
const MZ = (z: number) => 22 + z * 18;
const UX = (px: number) => Math.max(-4, Math.min(4, (px - 129) / 25));
const UZ = (py: number) => Math.max(0, Math.min(6, (py - 22) / 18));

function drawMini() {
  const list = clips();
  const picked = selectedClip();
  let text = `<svg width="258" height="150" id="miniSvg"><rect x="${MX(-4)}" y="${MZ(0)}" width="200" height="108" rx="3" fill="#ffffff08" stroke="#ff8a70" stroke-opacity=".5" stroke-dasharray="5 4"/>`;
  dancers().forEach((dancer, index) => {
    if (dancer.id === activeDancer()?.id || !dancer.clips.length || dancer.hidden) { return; }
    const color = DANCER_COLORS[index % DANCER_COLORS.length];
    text += `<polyline points="${dancer.clips.map((c) => `${MX(c.x)},${MZ(c.z)}`).join(" ")}" fill="none" stroke="${color}" stroke-opacity=".3"/>`;
    text += dancer.clips.map((c) => `<circle cx="${MX(c.x)}" cy="${MZ(c.z)}" r="2.500" fill="${color}" fill-opacity=".5"/>`).join("");
  });
  if (list.length > 1) { text += `<polyline points="${list.map((c) => `${MX(c.x)},${MZ(c.z)}`).join(" ")}" fill="none" stroke="#e9d9b0" stroke-opacity=".6" stroke-width="1.500"/>`; }
  const spots = spreadNumbers(list.map((c) => ({ x: MX(c.x), z: MZ(c.z) })), picked ? list.indexOf(picked) : -1, 14, 15);
  list.forEach((c, index) => {
    const on = c === picked;
    const x = Math.max(9, Math.min(249, spots[index].x)), y = Math.max(9, Math.min(141, spots[index].z));
    if (x !== MX(c.x) || y !== MZ(c.z)) { text += `<line x1="${MX(c.x)}" y1="${MZ(c.z)}" x2="${x}" y2="${y}" stroke="#f0eee8" stroke-opacity=".5"/>`; }
    text += `<g class="mini-dot" data-id="${c.id}"><circle cx="${x}" cy="${y}" r="${on ? 10 : 6.5}" fill="${on ? "#ff8a70" : "#f0eee8"}"/><text x="${x}" y="${y + (on ? 4.5 : 3.5)}" text-anchor="middle" font-family="Bahnschrift" font-weight="700" font-size="${on ? 13 : 9.5}" fill="#111">${index + 1}</text></g>`;
  });
  document.getElementById("minimap")!.innerHTML = text + "</svg>";
}

function bindMini() {
  const host = document.getElementById("minimap")!;
  host.addEventListener("mousedown", (event) => {
    const dot = (event.target as Element).closest(".mini-dot") as SVGGElement | null;
    if (!dot || event.button !== 0) { return; }
    const id = dot.dataset.id!;
    if (isDoublePress(id)) { jumpToClip(id); return; }
    S.selected = id;
    const clip = clips().find((c) => c.id === id)!;
    if (clip.locked) { refresh(); warnLockedDrag(event); return; }
    beginInteraction();
    const rect = (host.querySelector("svg") as SVGSVGElement).getBoundingClientRect();
    let moved = false;
    const move = (e: MouseEvent) => {
      if (!moved && Math.hypot(e.clientX - event.clientX, e.clientY - event.clientY) < 3) { return; }
      moved = true;
      clip.x = Math.round(UX(e.clientX - rect.left) * 10) / 10;
      clip.z = Math.round(UZ(e.clientY - rect.top) * 10) / 10;
      drawMini();
      rebuildMarkers();
    };
    const up = async () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      if (!moved) { endInteraction(); applyChanges({}); return; }
      endInteraction(true);
      markGenerating(id);
      refresh();
      try { await applyChanges(await call("PATCH", `/api/clips/${id}`, { x: clip.x, z: clip.z })); }
      catch (error) { toast((error as Error).message, "error"); await applyState(await call("GET", "/api/state")); }
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  });
}
