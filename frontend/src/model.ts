import * as THREE from "three";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils, type VRM } from "@pixiv/three-vrm";

/**
 * A humanoid model that dances the SOMA motion.
 *
 * The motion arrives as how far each humanoid bone has turned, in world space, from the standard T-pose (facing +Z,
 * left arm along +X). The model is put into that same T-pose once, whatever pose it rests in, and from then on each
 * bone's world rotation is the turn times its T-pose rotation. Hips height is matched by scaling the model, and the
 * ankles are pulled onto the motion's ankles so feet neither slide nor sink when leg proportions differ.
 */

// The standard T-pose of the motion's body (metres, unit directions).
const SOMA_HIPS_HEIGHT = 1.0052;
const SOMA_ANKLE_HEIGHT = 0.0671;
const THIGH = new THREE.Vector3(0, -0.9998, -0.0186).normalize();
const SHIN = new THREE.Vector3(0, -0.9966, -0.0823).normalize();
const THUMB = new THREE.Vector3(0.8528, -0.3885, 0.3489).normalize();

/** Without these the motion cannot drive a model. */
export const REQUIRED = ["hips", "leftUpperArm", "leftLowerArm", "rightUpperArm", "rightLowerArm", "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"];
/** The bones offered for matching by hand. Fingers are matched by name only. */
export const MAIN_BONES = ["hips", "spine", "chest", "upperChest", "neck", "head",
  ...["left", "right"].flatMap((side) => ["Shoulder", "UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot", "Toes"].map((part) => side + part))];
export const BONE_LABELS: Record<string, string> = {
  hips: "腰", spine: "背骨", chest: "胸", upperChest: "胸の上", neck: "首", head: "頭",
  Shoulder: "肩", UpperArm: "上腕", LowerArm: "前腕", Hand: "手", UpperLeg: "太もも", LowerLeg: "すね", Foot: "足首", Toes: "つま先",
};
export const boneLabel = (bone: string) => BONE_LABELS[bone] ?? `${bone.startsWith("left") ? "左" : "右"}${BONE_LABELS[bone.replace(/^(left|right)/, "")] ?? bone}`;

const SIDED: [string, RegExp][] = [
  ["Shoulder", /(shoulder|clavicle|collar)$/], ["UpperArm", /(upperarm|uparm)$|(?<!fore|lower|low)arm$/], ["LowerArm", /(lowerarm|forearm|lowarm|elbow)$/],
  ["Hand", /(hand|wrist)$/], ["UpperLeg", /(upperleg|upleg|thigh)$/], ["LowerLeg", /(lowerleg|lowleg|shin|calf|knee)$|(?<!upper|up|lower|low)leg$/],
  ["Foot", /(foot|ankle)$/], ["Toes", /(toebase|toes?|ball)$/],
];
const FINGER_PARTS: Record<string, string[]> = { Thumb: ["Metacarpal", "Proximal", "Distal"], Index: ["Proximal", "Intermediate", "Distal"], Middle: ["Proximal", "Intermediate", "Distal"],
  Ring: ["Proximal", "Intermediate", "Distal"], Little: ["Proximal", "Intermediate", "Distal"] };

/** Humanoid bone -> the model's bone name, from the names alone. Handles the usual conventions (Mixamo, VRoid, Unreal, Rigify, Unity). */
export function guessBones(names: string[]): Record<string, string> {
  const found: Record<string, string> = {};
  const take = (bone: string, name: string) => { if (!(bone in found)) { found[bone] = name; } };
  const spine: string[] = [];
  for (const name of names) {
    const flat = name.toLowerCase().replace(/^.*[:|]/, "").replace(/mixamorig\d*/g, "").replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
    const side = /(^|_)(l|left)(_|$)|^left|left$/.test(flat) ? "left" : /(^|_)(r|right)(_|$)|^right|right$/.test(flat) ? "right" : "";
    const core = flat.replace(/(^|_)(l|left|r|right)(?=_|$)/g, "").replace(/^(left|right)|(left|right)$/g, "").replace(/_/g, "");
    if (side) {
      const finger = /(thumb|index|middle|ring|pinky|little)0?([1-3])$/.exec(core);
      if (finger) {
        const kind = finger[1] === "pinky" ? "Little" : finger[1][0].toUpperCase() + finger[1].slice(1);
        take(`${side}${kind}${FINGER_PARTS[kind][Number(finger[2]) - 1]}`, name);
        continue;
      }
      const part = SIDED.find(([, pattern]) => pattern.test(core));
      if (part) { take(side + part[0], name); }
    } else if (/(hips?|pelvis)$/.test(core)) { take("hips", name); }
    else if (/head$/.test(core)) { take("head", name); }
    else if (/neck(0?1)?$/.test(core)) { take("neck", name); }
    else if (/(spine|chest)\d*$/.test(core)) { spine.push(name); }
  }
  // Spine bones come in hierarchy order, lowest first: the first three are spine, chest and upper chest.
  spine.slice(0, 3).forEach((name, index) => take(["spine", "chest", "upperChest"][index], name));
  return found;
}

const v1 = new THREE.Vector3(), v2 = new THREE.Vector3(), v3 = new THREE.Vector3();
const q1 = new THREE.Quaternion(), q2 = new THREE.Quaternion(), q3 = new THREE.Quaternion();

export class Rig {
  /** In the scene. Moving or turning it carries the whole model, which the stage uses while a phrase is dragged. */
  readonly root = new THREE.Group();
  /** Turns the model to face +Z and scales it to the motion's body. */
  private readonly fit = new THREE.Group();
  private nodes: (THREE.Object3D | null)[] = [];
  private tpose: THREE.Quaternion[] = [];  // each bone's rotation in the T-pose, in the root's space
  private anchor: number[] = [];  // the nearest driven bone above each bone, or -1
  private chain: THREE.Quaternion[] = [];  // from that anchor (or the root) down to the bone's parent. Constant, as bones between are not driven
  private world: THREE.Quaternion[] = [];
  private hips = -1;
  private hipsFrame = new THREE.Matrix4();  // root space -> the space the hips bone moves in
  private legs: { upper: number; lower: number; foot: number; lift: number }[] = [];
  vrm: VRM | null = null;
  builtin = false;
  /** Every bone name in the model, for matching by hand. Empty for VRM, which names its own bones. */
  boneNames: string[] = [];
  /** The mapping in use: humanoid bone -> model bone name. */
  mapping: Record<string, string> = {};
  missing: string[] = [];

  get ready(): boolean { return this.missing.length === 0; }

  /** Works out the T-pose and everything derived from it. `bones` is the order the motion's turns come in. */
  bind(object: THREE.Object3D, nodeOf: (bone: string) => THREE.Object3D | null, bones: string[]) {
    this.fit.add(object);
    this.root.add(this.fit);
    object.traverse((node) => { node.frustumCulled = false; });  // bounds do not follow the bones
    this.missing = REQUIRED.filter((bone) => !nodeOf(bone));
    if (!this.ready) { return; }
    this.root.updateWorldMatrix(true, true);
    const leftArm = nodeOf("leftUpperArm")!.getWorldPosition(v1).x, rightArm = nodeOf("rightUpperArm")!.getWorldPosition(v2).x;
    if (leftArm < rightArm) { this.fit.rotation.y = Math.PI; }  // it was facing -Z
    this.root.updateWorldMatrix(true, true);
    const hipsNode = nodeOf("hips")!;
    const box = new THREE.Box3().setFromObject(object);
    const feet = Math.min(nodeOf("leftFoot")!.getWorldPosition(v1).y, nodeOf("rightFoot")!.getWorldPosition(v2).y);
    const sole = Number.isFinite(box.min.y) && box.min.y < feet ? box.min.y : feet - 0.08 * (hipsNode.getWorldPosition(v3).y - feet);
    this.fit.scale.setScalar(SOMA_HIPS_HEIGHT / Math.max(1e-6, hipsNode.getWorldPosition(v3).y - sole));
    this.root.updateWorldMatrix(true, true);
    // Into the T-pose, from the top of each limb down: each bone swings the least it must to point the right way.
    const aim = (bone: string, tip: string, direction: THREE.Vector3) => {
      const node = nodeOf(bone), end = nodeOf(tip);
      if (!node || !end) { return; }
      const swing = q1.setFromUnitVectors(end.getWorldPosition(v2).sub(node.getWorldPosition(v1)).normalize(), direction);
      const turned = node.getWorldQuaternion(q2).premultiply(swing);
      node.quaternion.copy(node.parent!.getWorldQuaternion(q3).invert().multiply(turned));
      node.updateWorldMatrix(false, true);
    };
    for (const [side, sign] of [["left", 1], ["right", -1]] as const) {
      const level = new THREE.Vector3(sign, 0, 0);
      aim(`${side}UpperArm`, `${side}LowerArm`, level);
      aim(`${side}LowerArm`, `${side}Hand`, level);
      aim(`${side}ThumbMetacarpal`, `${side}ThumbProximal`, new THREE.Vector3(sign * THUMB.x, THUMB.y, THUMB.z));
      aim(`${side}ThumbProximal`, `${side}ThumbDistal`, level);
      for (const finger of ["Index", "Middle", "Ring", "Little"]) {
        aim(`${side}${finger}Proximal`, `${side}${finger}Intermediate`, level);
        aim(`${side}${finger}Intermediate`, `${side}${finger}Distal`, level);
      }
      aim(`${side}UpperLeg`, `${side}LowerLeg`, THIGH);
      aim(`${side}LowerLeg`, `${side}Foot`, SHIN);
    }
    this.nodes = bones.map((bone) => nodeOf(bone));
    const index = new Map<THREE.Object3D, number>();
    this.nodes.forEach((node, i) => { if (node) { index.set(node, i); } });
    this.tpose = this.nodes.map((node) => (node ? node.getWorldQuaternion(new THREE.Quaternion()) : new THREE.Quaternion()));
    this.world = this.nodes.map(() => new THREE.Quaternion());
    this.anchor = this.nodes.map(() => -1);
    this.chain = this.nodes.map((node, i) => {
      if (!node) { return new THREE.Quaternion(); }
      let above = node.parent;
      while (above && !index.has(above)) { above = above.parent; }
      const parent = node.parent!.getWorldQuaternion(new THREE.Quaternion());
      if (!above) { return parent; }
      this.anchor[i] = index.get(above)!;
      return this.tpose[this.anchor[i]].clone().invert().multiply(parent);
    });
    this.hips = bones.indexOf("hips");
    this.hipsFrame.copy(hipsNode.parent!.matrixWorld).invert();
    const hipsHeight = hipsNode.getWorldPosition(v3).y;
    this.legs = ["left", "right"].map((side) => {
      const foot = bones.indexOf(`${side}Foot`);
      // How much higher this model's ankle stands than the motion's when both feet are flat on the floor.
      const lift = SOMA_HIPS_HEIGHT - (hipsHeight - this.nodes[foot]!.getWorldPosition(v1).y) - SOMA_ANKLE_HEIGHT;
      return { upper: bones.indexOf(`${side}UpperLeg`), lower: bones.indexOf(`${side}LowerLeg`), foot, lift };
    });
  }

  private setLocal(i: number) {
    const parent = this.anchor[i] >= 0 ? q3.copy(this.world[this.anchor[i]]).multiply(this.chain[i]) : q3.copy(this.chain[i]);
    this.nodes[i]!.quaternion.copy(parent.invert().multiply(this.world[i]));
  }

  /**
   * Strike the pose at a (fractional) frame. `turns` holds [frame, bone, 4] quaternions, `hips` and `ankles` are world
   * positions in metres ([left, right] for the ankles).
   */
  pose(turns: Float32Array, frame: number, hips: THREE.Vector3, ankles: [THREE.Vector3, THREE.Vector3] | null) {
    if (!this.ready) { return; }
    const count = this.nodes.length;
    const frames = turns.length / (count * 4);
    const low = Math.max(0, Math.min(frames - 1, Math.floor(frame))), high = Math.min(frames - 1, low + 1), blend = Math.max(0, Math.min(1, frame - low));
    for (let i = 0; i < count; i++) {
      if (!this.nodes[i]) { continue; }
      q1.fromArray(turns, (low * count + i) * 4);
      q2.fromArray(turns, (high * count + i) * 4);
      this.world[i].copy(q1.slerp(q2, blend)).multiply(this.tpose[i]);
      this.setLocal(i);
    }
    const hipsNode = this.nodes[this.hips]!;
    hipsNode.position.copy(hips).applyMatrix4(this.hipsFrame);
    const moved = this.root.position.lengthSq() > 0 || this.root.rotation.y !== 0;
    if (!ankles || moved) { return; }  // while the stage carries the model about, the legs keep the pose as it is
    hipsNode.updateWorldMatrix(false, true);
    this.legs.forEach((leg, side) => this.reach(leg, v3.copy(ankles[side]).setY(ankles[side].y + leg.lift)));
  }

  /** Two-bone IK: bend the leg so the ankle sits on `goal`. The knee keeps the side it bends to, and the foot keeps its rotation. */
  private reach(leg: { upper: number; lower: number; foot: number }, goal: THREE.Vector3) {
    const hip = this.nodes[leg.upper]!.getWorldPosition(new THREE.Vector3());
    const knee = this.nodes[leg.lower]!.getWorldPosition(new THREE.Vector3());
    const ankle = this.nodes[leg.foot]!.getWorldPosition(new THREE.Vector3());
    const upper = knee.distanceTo(hip), lower = ankle.distanceTo(knee);
    const toGoal = goal.clone().sub(hip);
    const length = Math.max(Math.abs(upper - lower) + 1e-4, Math.min(upper + lower - 1e-4, toGoal.length()));
    if (toGoal.lengthSq() < 1e-10 || ankle.distanceToSquared(goal) < 1e-8) { return; }
    const direction = toGoal.normalize();
    const bend = knee.clone().sub(hip);
    bend.addScaledVector(direction, -bend.dot(direction));
    if (bend.lengthSq() < 1e-10) { return; }  // a perfectly straight leg gives no side to bend to
    bend.normalize();
    const along = (upper * upper - lower * lower + length * length) / (2 * length);
    const newKnee = hip.clone().addScaledVector(direction, along).addScaledVector(bend, Math.sqrt(Math.max(0, upper * upper - along * along)));
    const newAnkle = hip.clone().addScaledVector(direction, length);
    const turnUpper = new THREE.Quaternion().setFromUnitVectors(knee.clone().sub(hip).normalize(), newKnee.clone().sub(hip).normalize());
    const shin = ankle.clone().sub(knee).applyQuaternion(turnUpper).normalize();
    const turnLower = new THREE.Quaternion().setFromUnitVectors(shin, newAnkle.sub(newKnee).normalize());
    this.world[leg.upper].premultiply(turnUpper);
    this.world[leg.lower].premultiply(turnUpper).premultiply(turnLower);
    this.setLocal(leg.upper);
    this.setLocal(leg.lower);
    this.setLocal(leg.foot);
  }

  /** Colours the built-in default mannequin per dancer without touching user-selected models. */
  setTint(hex: string) {
    if (!this.builtin) { return; }
    const main = new THREE.Color(hex);
    const joint = new THREE.Color(0x282b36).lerp(main, 0.25);
    this.root.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!mesh.isMesh) { return; }
      for (const material of ([] as THREE.Material[]).concat(mesh.material ?? [])) {
        const std = material as THREE.MeshStandardMaterial;
        if (!std.color) { continue; }
        std.color.copy(std.name === "M_Joints" ? joint : main);
        std.roughness = std.name === "M_Joints" ? 0.62 : 0.52;
        std.metalness = 0.06;
      }
    });
  }

  /** The bones the motion moves, with the hips bone first, for baking an animation. */
  driven(): { hips: THREE.Object3D; nodes: THREE.Object3D[] } {
    return { hips: this.nodes[this.hips]!, nodes: this.nodes.filter((node): node is THREE.Object3D => !!node) };
  }

  dispose() {
    this.root.removeFromParent();
    if (this.vrm) { VRMUtils.deepDispose(this.vrm.scene); return; }
    this.root.traverse((node) => {
      const mesh = node as THREE.Mesh;
      mesh.geometry?.dispose();
      for (const material of ([] as THREE.Material[]).concat(mesh.material ?? [])) {
        for (const value of Object.values(material)) { if ((value as THREE.Texture)?.isTexture) { (value as THREE.Texture).dispose(); } }
        material.dispose();
      }
    });
  }
}

let defaultGlb: Promise<ArrayBuffer> | null = null;

/** Loads the bundled CC0 articulated humanoid mannequin (with full 5-finger joints), caching the GLB bytes across dancers. */
export async function loadDefaultRig(bones: string[], color: string): Promise<Rig> {
  if (!defaultGlb) {
    defaultGlb = fetch("/models/default-mannequin.glb").then((res) => {
      if (!res.ok) { throw new Error(`HTTP ${res.status}`); }
      return res.arrayBuffer();
    }).catch((err) => {
      defaultGlb = null;
      throw err;
    });
  }
  const buffer = await defaultGlb;
  const gltf = await new GLTFLoader().parseAsync(buffer.slice(0), "");
  const rig = new Rig();
  rig.builtin = true;
  bindByName(rig, gltf.scene, {}, bones);
  rig.setTint(color);
  return rig;
}

/** Loads a VRM, GLB or FBX and gets it ready to dance. A model whose bones cannot all be matched comes back with `missing` filled in. */
export async function loadRig(url: string, mapping: Record<string, string>, bones: string[]): Promise<Rig> {
  const rig = new Rig();
  const kind = url.split("?")[0].split(".").pop()!.toLowerCase();
  if (kind === "fbx") {
    const object = await new FBXLoader().loadAsync(url);
    bindByName(rig, object, mapping, bones);
    return rig;
  }
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm as VRM | undefined;
  if (!vrm) {
    bindByName(rig, gltf.scene, mapping, bones);
    return rig;
  }
  VRMUtils.rotateVRM0(vrm);
  vrm.humanoid.autoUpdateHumanBones = false;  // the raw bones are driven directly
  rig.vrm = vrm;
  rig.bind(vrm.scene, (bone) => vrm.humanoid.getRawBoneNode(bone as any), bones);
  return rig;
}

function bindByName(rig: Rig, object: THREE.Object3D, mapping: Record<string, string>, bones: string[]) {
  const byName = new Map<string, THREE.Object3D>();
  object.traverse((node) => { if ((node as THREE.Bone).isBone && node.name && !byName.has(node.name)) { byName.set(node.name, node); } });
  rig.boneNames = [...byName.keys()];
  rig.mapping = { ...guessBones(rig.boneNames), ...Object.fromEntries(Object.entries(mapping).filter(([, name]) => name === "" || byName.has(name))) };
  rig.bind(object, (bone) => byName.get(rig.mapping[bone] ?? "") ?? null, bones);
}
