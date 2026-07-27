"""Proof suite for the river-flow lab (kreken + rivier-stroomdynamica).

Guards the physical invariants: kinematic-wave flow reaches downstream,
mass/water balance holds, viscosity rises with sediment concentration
(reusing SlurryRheology, not a duplicate), higher sediment slows the flow
(the viscous-drag coupling), ore settles out fastest where flow slows
(the placer-deposit mechanism), and the raw-material supply pays through
the same MolCoin-style economy without inventing a new one.
Run: python3 test_river_flow.py  (exit 0 = pass)
"""

import math
import sys

import numpy as np

from river_flow import RawMaterialSupply, RiverChannel, RHO_GOLD, RHO_ORE
from viscosity_core import SlurryRheology

failures = 0


def check(cond, msg):
    global failures
    print(f"  {'✓' if cond else '✗ FAIL:'} {msg}")
    if not cond:
        failures += 1


def approx(a, b, tol):
    return abs(a - b) <= tol


def straight_river(length_m=2000.0, n_cells=100):
    pts = [(0.0, 0.0), (length_m, 0.0)]
    return RiverChannel(pts, width_m=12.0, slope=0.002, n_cells=n_cells)


print("\n=== river-flow proof suite ===\n")

print("1. Geometry + basic sanity")
r = straight_river()
check(approx(r.length_m, 2000.0, 1e-6), f"channel length = {r.length_m:.0f} m")
check(r.n == 100, f"discretised into {r.n} cells")
try:
    RiverChannel([(0, 0)], width_m=5)
    check(False, "single-point river should raise")
except ValueError:
    check(True, "single-point river raises ValueError")

print("\n2. Kinematic wave: a creek pulse reaches a near-field cell")
# A real river genuinely takes a long time to equilibrate km-scale (that IS
# the correct physics -- kilometre-scale rivers take hours), so this checks
# a cell a few hundred metres downstream, on a timescale a test can afford,
# rather than the far end of a 2 km reach.
r = straight_river(length_m=600.0, n_cells=30)
cid = r.add_creek(Q_m3s=0.0, phi_sediment=0.02, at_fraction=0.0)
for _ in range(120):
    r.step(1.0)
h0 = float(r.h.mean())
r.set_creek_flow(cid, Q_m3s=8.0)
u_before = float(r._velocities()[0][5])
for _ in range(900):
    r.step(1.0)
h1 = float(r.h.mean())
u_after = float(r._velocities()[0][5])
check(h1 > h0 * 1.1, f"a creek inflow raises mean depth ({h0:.3f} -> {h1:.3f} m)")
check(u_after > u_before, f"a near-field cell speeds up once the pulse arrives "
      f"({u_before:.3f} -> {u_after:.3f} m/s)")

print("\n3. Sediment mass conservation (no creek feed, closed system)")
r2 = straight_river(length_m=500.0, n_cells=25)
r2.phi[:] = 0.01
sed0 = float((r2.phi * r2.width * r2.h * r2.ds).sum())
for _ in range(50):
    r2.step(1.0)
sed_susp = float((r2.phi * r2.width * r2.h * r2.ds).sum())
sed_dep = float((r2.deposit_kg_per_m * r2.ds).sum() / r2.rheo.rho_solid)
check(approx(sed_susp + sed_dep, sed0, sed0 * 0.05),
      f"sediment volume conserved (suspended {sed_susp:.4f} + deposited "
      f"{sed_dep:.4f} ~ initial {sed0:.4f} m3)")

print("\n4. Viscosity reuses SlurryRheology (no duplicate physics)")
r3 = straight_river()
rh = SlurryRheology(rho_solid=RHO_ORE)
check(r3.rheo.rho_solid == RHO_ORE, "RiverChannel defaults to the ore rheology")
eta_dilute = rh.apparent_viscosity(0.01, 50.0)
eta_thick = rh.apparent_viscosity(0.25, 50.0)
check(eta_thick > eta_dilute * 3,
      f"apparent viscosity rises sharply with sediment load "
      f"({eta_dilute*1000:.2f} -> {eta_thick*1000:.2f} mPa.s)")

print("\n5. Thicker sediment measurably slows the flow (viscous drag coupling)")


def steady_velocity(phi_fixed, steps=400):
    rc = straight_river(length_m=1000.0, n_cells=50)
    rc.phi[:] = phi_fixed
    cid = rc.add_creek(Q_m3s=6.0, phi_sediment=phi_fixed, at_fraction=0.0)
    for _ in range(steps):
        rc.step(1.0)
    return float(rc._velocities()[0][-5:].mean())


u_clear = steady_velocity(0.001)
u_muddy = steady_velocity(0.30)
check(u_muddy < u_clear,
      f"muddy flow is slower than clear flow at the same discharge "
      f"({u_clear:.3f} vs {u_muddy:.3f} m/s)")
check(u_muddy > 0.02, f"muddy flow still moves, not numerically stuck ({u_muddy:.4f} m/s)")

print("\n6. Ore settles fastest where the flow slows (placer-deposit mechanism)")
r4 = straight_river(length_m=300.0, n_cells=30)
# Widen sharply partway down -> same Q spreads thinner -> velocity drops
# there (see the Manning/continuity derivation in river_flow.py docstring).
r4.width[15:] = 45.0
cid4 = r4.add_creek(Q_m3s=5.0, phi_sediment=0.08, at_fraction=0.0)
for _ in range(1500):
    r4.step(1.0)
dep_narrow = float(r4.deposit_kg_per_m[5])
dep_wide = float(r4.deposit_kg_per_m[22])
check(dep_wide > dep_narrow,
      f"ore deposits more where the channel widens & slows "
      f"({dep_narrow:.2f} vs {dep_wide:.2f} kg/m)")
check(r4.deposit_at(0.75) > 0, f"deposit_at() reports real accumulated mass "
      f"({r4.deposit_at(0.75):.1f} kg)")

print("\n7. Raw-material supply pays through the shared MolCoin-style economy")
r5 = straight_river(length_m=800.0, n_cells=40)
r5.add_creek(Q_m3s=4.0, phi_sediment=0.15, at_fraction=0.0)
for _ in range(1200):
    r5.step(1.0)
supply = RawMaterialSupply(r5, at_fraction=0.6, eur_per_kg=8.0)
avail0 = supply.available_kg()
check(avail0 > 0, f"ore has accumulated at the collection point ({avail0:.2f} kg)")
res = supply.collect()
check(approx(res["payout"], round(avail0 * 8.0), 2),
      f"collect() pays kg*eur_per_kg, same shape as V2O5/harvest payouts "
      f"({res})")
check(supply.available_kg() < 0.01,
      f"collecting drains the accumulator ({supply.available_kg():.3f} kg left)")
for _ in range(60):
    r5.step(1.0)                    # a further minute of flow -> more ore
check(supply.available_kg() > 0,
      f"further flow keeps depositing new ore after a collection "
      f"({supply.available_kg():.3f} kg)")

print("\n8. Gold preset (rivierlab goudpannen precedent) settles even faster than ore")
rh_gold = SlurryRheology(rho_solid=RHO_GOLD)
rh_ore = SlurryRheology(rho_solid=RHO_ORE)
vs_gold = float(rh_gold.settling_velocity(0.02))
vs_ore = float(rh_ore.settling_velocity(0.02))
check(vs_gold > vs_ore,
      f"denser gold settles faster than generic ore at equal concentration "
      f"({vs_gold*1000:.3f} vs {vs_ore*1000:.3f} mm/s)")

print(f"\n=== {'ALL CHECKS PASSED' if failures == 0 else f'{failures} FAILURES'} ===\n")
sys.exit(1 if failures else 0)
