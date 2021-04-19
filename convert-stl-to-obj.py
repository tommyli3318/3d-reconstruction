'''

To use this script on Windows, you must set up a Path to Blender files.
(I.E C:\Program Files\Blender Foundation\Blender2.92)

Run the script using the following command

blender --background --python convert-stl-to-obj.py -- {stl-file-name.stl} {blender-workspace.blend}

'''

import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after "--"

stl_in = argv[0]
obj_out = argv[1]

bpy.ops.import_mesh.stl(filepath=stl_in, axis_forward='-Z', axis_up='Y')
bpy.ops.export_scene.obj(filepath=obj_out, axis_forward='-Z', axis_up='Y')