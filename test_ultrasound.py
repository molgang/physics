"""Proof suite for the silicon-wash level (ultrasound 28/40 kHz).

Guards the level physics: the viscosity gate on cavitation, the need for
BOTH frequency channels, sono-flotation vs. remixing, overflow capture,
mass balance, generator energy metering and the star logic. This file is
also the parity contract for the JS port.
Run: python3 test_ultrasound.py  (exit 0 = pass)
"""

import math
import sys

from ultrasound import (SiliconWashLevel, UltrasoundBath, ETA_CAV_HI,
                        ETA_CAV_LO, LIGHT_FRACTION_OF_SLAG, P_28KHZ_W,
                        P_40KHZ_W, RIM_FILL_FRACTION, STAR_2_PCT)
from viscosity_core import MixingTank

failures = 0


def check(cond, msg):
    global failures
    print(f"  {'✓' if cond else '✗ FAIL:'} {msg}")
    if not cond:
        failures += 1


def approx(a, b, tol):
    return abs(a - b) <= tol


def new_level(n=32):
    tank = MixingTank(n=n, water_l=14.0, w_pct=20.0)
    tank.placed = True
    tank.stirrer.plugged_in = True
    lvl = SiliconWashLevel(tank)
    lvl.start()
    return tank, lvl


def play(lvl, seconds, dt=0.1, hose_to_rim=False):
    """Advance the level; optionally keep topping up to the overflow rim."""
    t = lvl.tank
    steps = int(seconds / dt)
    for i in range(steps):
        if hose_to_rim and i % 10 == 0 and \
                t.volume_l < t.capacity_l * 0.97:
            t.add_water(0.5)
        t.step(dt)
        lvl.step(dt)


print("\n=== silicon-wash level proof suite ===\n")

print("1. Cavitation efficiency (golden values, parity contract)")
eff_mid = UltrasoundBath.cavitation_efficiency(
    math.sqrt(ETA_CAV_LO * ETA_CAV_HI), 0.0)
check(approx(eff_mid, 0.5, 1e-9), f"log-midpoint viscosity -> 0.5 ({eff_mid})")
check(UltrasoundBath.cavitation_efficiency(ETA_CAV_LO, 0.0) == 1.0,
      "full cavitation at 5 mPa.s, phi=0")
check(UltrasoundBath.cavitation_efficiency(ETA_CAV_HI, 0.0) == 0.0,
      "dead at 50 mPa.s")
check(UltrasoundBath.cavitation_efficiency(0.001, 0.30) == 0.0,
      "dead at phi=0.30 regardless of viscosity")
check(approx(UltrasoundBath.cavitation_efficiency(0.001, 0.15), 0.5, 1e-9),
      "phi=0.15 halves the efficiency")

print("\n2. Level start: locked species, thick sludge above the gate")
tank, lvl = new_level()
check(approx(lvl.light_total, tank.slag_kg * LIGHT_FRACTION_OF_SLAG, 1e-9),
      f"light fraction = 15% of solids ({lvl.light_total:.2f} kg)")
check(approx(lvl.mass_balance_kg(), lvl.light_total, 1e-9),
      "species mass balance closed at start")
check(tank.phi_bulk() > 0.30,
      f"start sludge above cavitation kill (phi={tank.phi_bulk():.2f})")
check(tank.volume_l < tank.capacity_l * 0.75,
      f"headroom to dilute ({tank.volume_l:.1f} L of {tank.capacity_l} L)")

print("\n3. Viscosity gate: thick sludge -> nothing happens")
lvl.bath.on_28 = True
lvl.bath.on_40 = True
play(lvl, 120)
check(lvl.captured_pct() < 1.0,
      f"62% sludge blocks the wash ({lvl.captured_pct():.2f}% captured)")
check(lvl.cav_eff < 0.05, f"cavitation dead (eff={lvl.cav_eff:.3f})")

print("\n4. Full correct play: dilute + 28&40 kHz + gentle stir + rim")
tank2, lvl2 = new_level()
lvl2.bath.on_28 = True
lvl2.bath.on_40 = True
tank2.stirrer.on = True
tank2.stirrer.rpm_set = 60.0
mb0 = lvl2.mass_balance_kg() + lvl2.tank.slag_kg * 0  # species-only balance
play(lvl2, 360, hose_to_rim=True)
pct = lvl2.captured_pct()
check(pct >= STAR_2_PCT,
      f"correct play captures >= {STAR_2_PCT:.0f}% in 6 min ({pct:.0f}%)")
check(approx(lvl2.mass_balance_kg(), lvl2.light_total, 1e-6),
      "species mass conserved through the wash")
check(lvl2.stars() >= 2, f"earns >= 2 stars ({lvl2.stars()})")
check(lvl2.energy_kwh() > 0.02,
      f"generator energy metered ({lvl2.energy_kwh()*1000:.1f} Wh)")
check(tank2.slag_kg < 13.06 - lvl2.captured + 0.02,
      "captured froth left the tank's slag bookkeeping")
check(lvl2.cav_eff > 0.3, f"diluted bath cavitates (eff={lvl2.cav_eff:.2f})")

print("\n5. Single-frequency plateaus: both channels are needed")


def freed_after(on28, on40, seconds=180):
    tk, lv = new_level()
    lv.bath.on_28 = on28
    lv.bath.on_40 = on40
    tk.stirrer.on = True
    tk.stirrer.rpm_set = 60.0
    play(lv, seconds, hose_to_rim=True)
    return lv.light_total - (lv.bound_coarse + lv.bound_fine), lv


freed_28, lv28 = freed_after(True, False)
freed_40, lv40 = freed_after(False, True)
freed_dual, _ = freed_after(True, True)
check(freed_dual > 1.25 * max(freed_28, freed_40),
      f"dual frees >1.25x best single ({freed_dual:.2f} vs "
      f"{freed_28:.2f}/{freed_40:.2f} kg)")
check(lv28.bound_fine > lv28.bound_coarse,
      "28 kHz alone leaves the FINE class locked")
check(lv40.bound_coarse > lv40.bound_fine,
      "40 kHz alone leaves the COARSE class locked")

print("\n6. Hard stirring remixes the froth")


def froth_plus_captured(rpm):
    tk, lv = new_level()
    lv.bath.on_28 = lv.bath.on_40 = True
    tk.stirrer.on = True
    tk.stirrer.rpm_set = rpm
    play(lv, 180, hose_to_rim=True)
    return lv.froth + lv.captured


gentle = froth_plus_captured(60.0)
hard = froth_plus_captured(300.0)
check(gentle > hard * 1.15,
      f"gentle stir beats hard stir ({gentle:.2f} vs {hard:.2f} kg risen)")

print("\n7. Generator power & re-aggregation")
tank3, lvl3 = new_level()
lvl3.bath.on_28 = True
lvl3.bath.on_40 = True
e0 = tank3.meter.energy_kwh
tank3.step(1.0)
lvl3.step(1.0)
gen_wh = (tank3.meter.energy_kwh - e0) * 3.6e6 / 3600 * 1000
check(approx(tank3.meter.p_watt,
             tank3.stirrer.p_electric + P_28KHZ_W + P_40KHZ_W, 1e-6),
      f"meter shows stirrer+generator ({tank3.meter.p_watt:.0f} W)")
tank4, lvl4 = new_level()
tank4.add_water(9.0)            # dilute so things CAN move
lvl4.bath.on_28 = lvl4.bath.on_40 = True
play(lvl4, 60)
lvl4.bath.on_28 = lvl4.bath.on_40 = False
free_before = lvl4.free
play(lvl4, 120)
check(lvl4.free < free_before,
      f"free particles re-aggregate when silent "
      f"({free_before:.2f} -> {lvl4.free:.2f} kg)")

print(f"\n=== {'ALL CHECKS PASSED' if failures == 0 else f'{failures} FAILURES'} ===\n")
sys.exit(1 if failures else 0)
