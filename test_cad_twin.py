"""Proof suite voor de cad_twin digital twin (Astra-apparatuur).

Guards the physical invariants: energy bookkeeping closes (q_in - q_out = dU),
substepping is consistent, stirring and ultrasound heat the slurry where they
physically act, the jacket pulls the steady temperature down, the wall chain
reduces to the analytic cylinder-wall formula, and STEP export demands
FreeCAD instead of silently returning nothing.
Run: python3 test_cad_twin.py  (exit 0 = pass)
"""
import math
import sys

import numpy as np

from cad_twin import EquipmentSpec, H_EXT, H_STIRRED, K_STEEL, ThermalTwin, export_step
from viscosity_core import SlurryRheology

failures = 0


def check(cond, msg):
    global failures
    print(f"  {'✓' if cond else '✗ FAIL:'} {msg}")
    if not cond:
        failures += 1


def approx(a, b, tol):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12)


print("== EquipmentSpec sanity")
spec = EquipmentSpec()
check(spec.radius_o > spec.tank_radius_i, "outer radius encloses the inner radius")
check(approx(spec.h_liq, spec.fill_frac * spec.tank_height, 1e-12), "fill height follows fill_frac")
for bad in ({'fill_frac': 0.0}, {'wall_t': -0.001}, {'w_solids': 1.0}, {'us_heat_frac': 1.5}):
    try:
        EquipmentSpec(**bad)
        check(False, f"rejects {bad}")
    except ValueError:
        check(True, f"rejects {bad}")

print("== ThermalTwin construction")
twin = ThermalTwin(EquipmentSpec())
rheo = SlurryRheology()
check(twin.rho == rheo.rho_mix(spec.w_solids), "slurry density from the shared SlurryRheology")
check(abs(twin.q_stir.sum() - 1) < 1e-9, "stirrer source normalised to total watts")
check(abs(twin.q_us.sum() - 1) < 1e-9, "transducer source normalised to total watts")

print("== heating and energy bookkeeping")
twin.run(600, dt=2.0)
check(twin.bulk_temp > spec.t_init + 1.0, "stirrer + ultrasound heat the slurry")
check(twin.bulk_temp < 100.0, "sane temperatures (no solver blow-up)")
resid = twin.energy_balance_residual()
check(resid < 0.02, f"energy balance closes after 10 min (residual {resid:.4f})")

print("== substep consistency")
a = ThermalTwin(EquipmentSpec()); a.run(120, dt=10.0)
b = ThermalTwin(EquipmentSpec()); b.run(120, dt=1.0)
check(abs(a.bulk_temp - b.bulk_temp) < 0.05, "big-dt auto-substepping matches fine-dt runs")

print("== jacket pulls the steady state down")
hot = ThermalTwin(EquipmentSpec(jacket=False)); hot.run(3600, dt=5.0)
cold = ThermalTwin(EquipmentSpec()); cold.run(3600, dt=5.0)
check(cold.bulk_temp < hot.bulk_temp, "jacket-cooled twin runs cooler after 1 h")
check(hot.bulk_temp > spec.t_init, "un-jacketed twin still heats up")

print("== ultrasound ring heats locally, near the wall at ring height")
pulsed = ThermalTwin(EquipmentSpec(stirrer_power=0.0, us_power=1500.0))
pulsed.step(120.0, stirrer_w=0.0)
i, j = np.unravel_index(np.argmax(pulsed.T), pulsed.T.shape)
check(pulsed.T.max() > pulsed.bulk_temp + 0.5, "a hot spot forms during the pulse")
check(j > pulsed.n_z * 0.4, f"hot spot sits above mid-height (ring at {spec.us_z_frac:.0%})")
check(i > pulsed.n_r * 0.6, "hot spot hugs the wall where the ring is mounted")

print("== analytic steady state: net flux vanishes at T*")
P = 400.0
sp0 = EquipmentSpec(jacket=False, stirrer_power=P, us_power=0.0, us_heat_frac=0.0)
A_i = 2 * math.pi * sp0.tank_radius_i * sp0.h_liq
A_o = 2 * math.pi * sp0.radius_o * sp0.h_liq
r_wall_cond = math.log(sp0.radius_o / sp0.tank_radius_i) / (2 * math.pi * K_STEEL * sp0.h_liq)
r_series = 1 / (H_STIRRED * A_i) + r_wall_cond + 1 / (H_EXT * A_o)
t_star = spec.t_ambient + P * r_series
eq = ThermalTwin(sp0)
eq.T[:, :] = t_star
eq.T_wall[:] = t_star - P * r_wall_cond      # wand zit P*R_wall onder de slurry
eq.run(600, dt=5.0)
check(abs(eq.bulk_temp - t_star) < 2.0,
      f"hand-computed equilibrium {t_star:.1f} °C holds within grid error (now {eq.bulk_temp:.2f} °C)")
check(abs(eq.q_out_j / max(eq.q_in_j, 1.0) - 1.0) < 0.10,
      "at equilibrium the wall loses what the stirrer injects")

print("== wall chain reduces to the analytic cylinder formula")
r_top = eq._wall_resistance()[-1]          # boven de mantel: vrije luchtconvectie
r_analytic = (math.log(spec.radius_o / spec.tank_radius_i) / (2 * math.pi * K_STEEL * eq.dz)
              + 1.0 / (H_EXT * 2 * math.pi * spec.radius_o * eq.dz))
check(approx(r_top, r_analytic, 1e-9), "air-cooled row: R_wall + R_film matches the hand calculation")
r_mid = cold._wall_resistance()[cold.n_z // 2]  # binnen de mantel: koelvloeistof
r_jacket = (math.log(spec.radius_o / spec.tank_radius_i) / (2 * math.pi * K_STEEL * cold.dz)
            + 1.0 / (spec.h_jacket * 2 * math.pi * spec.radius_o * cold.dz))
check(approx(r_mid, r_jacket, 1e-9), "jacket row: R_wall + R_coolant matches the hand calculation")

print("== STEP export")
try:
    export_step("/tmp/should-not-exist.step")
    check(False, "raises without FreeCAD")
except RuntimeError as exc:
    check("FreeCAD" in str(exc), "clear RuntimeError naming FreeCAD (not a silent no-op)")

print()
if failures:
    print(f"{failures} CHECK(S) FAILED")
    sys.exit(1)
print("all cad_twin checks passed")
