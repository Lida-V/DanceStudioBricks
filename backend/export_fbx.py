"""Convert a SOMA BVH to FBX with Blender's Python module. Usage: export_fbx.py INPUT.bvh OUTPUT.fbx [--rest]

With --rest the FBX rests in the BVH's own zero pose (for files written on a T-pose or A-pose rest). Without it the rest
is the pose at the start of the motion, as before.
"""
import sys

import bpy
from bpy_extras import anim_utils


def main():
    keep_rest = "--rest" in sys.argv
    source, target = [arg for arg in sys.argv if arg != "--rest"][-2:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_anim.bvh(filepath=source, global_scale=0.01, update_scene_fps=True, update_scene_duration=True, rotate_mode="NATIVE")
    arm = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    # The importer puts the first BVH frame at scene frame 1. Move it to frame 0 so FBX time 0 is the top of the song.
    animation = arm.animation_data
    channels = anim_utils.action_get_channelbag_for_slot(animation.action, animation.action_slot)
    for curve in channels.fcurves:
        for point in curve.keyframe_points:
            point.co.x -= 1.0
            point.handle_left.x -= 1.0
            point.handle_right.x -= 1.0
        curve.update()
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end -= 1
    if keep_rest:
        # With no mesh there is no bind pose in the file, and a reader takes the pose the bones had when the file was written
        # as their rest. Key the rest pose one frame before the motion and stand there, so the rest pose arrives as it is.
        for bone in arm.pose.bones:
            bone.location, bone.rotation_euler, bone.rotation_quaternion = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
            bone.keyframe_insert("location", frame=-1)
            bone.keyframe_insert("rotation_quaternion" if bone.rotation_mode == "QUATERNION" else "rotation_euler", frame=-1)
        bpy.context.scene.frame_set(-1)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.export_scene.fbx(filepath=target, use_selection=True, object_types={"ARMATURE"}, add_leaf_bones=False, bake_anim=True,
                             bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False,
                             bake_anim_simplify_factor=0.0, axis_forward="-Z", axis_up="Y", apply_unit_scale=True)
    print(f"frames {bpy.context.scene.frame_start}-{bpy.context.scene.frame_end} fps {bpy.context.scene.render.fps}")


if __name__ == "__main__":
    main()
