"""Bouw het Astra-rig parametrisch in FreeCAD en schrijf STEP.

Draai met FreeCAD's eigen interpreter (naast de repo staan):
    freecadcmd demo_cad_twin_step.py
SolidWorks opent het resulterende astra_rig.step native.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cad_twin  # noqa: E402

spec = cad_twin.EquipmentSpec()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astra_rig.step")
cad_twin.export_step(out, spec)
print(f"STEP geschreven: {out} ({os.path.getsize(out)} bytes)")
