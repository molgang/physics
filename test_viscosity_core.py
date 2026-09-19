"""Proof suite for the MOLGANG viscosity lab core (headless).

Guards the physical invariants of rheology, solver and power model so the
future JS/AR port has a reference to stay in parity with (same idea as
lab3d/chemistry.test.mjs). Run: python3 test_viscosity_core.py  (exit 0 = pass)
"""

import math
import sys

import numpy as np

from viscosity_core import (FluidGrid2D, MixingTank, SlurryRheology, Stirrer,
                            C_THIX, CP_SLAG, CP_WATER, PHI_MAX,
                            PHI_PACK_BED, RHO_SLAG, RHO_WATER, T_REF_C)

failures = 0


def check(cond, msg):
    global failures
    print(f"  {'✓' if cond else '✗ FAIL:'} {msg}")
    if not cond:
        failures += 1


def approx(a, b, tol):
    return abs(a - b) <= tol


print("\n=== viscosity_core proof suite ===\n")

# ---------------------------------------------------------------- rheology
print("1. Rheology")
rh = SlurryRheology()
check(approx(rh.phi_from_w(0.0), 0.0, 1e-12), "phi(0 wt%) = 0")
phi50 = rh.phi_from_w(0.5)
check(approx(phi50, (0.5 / RHO_SLAG) / (0.5 / RHO_SLAG + 0.5 / RHO_WATER),
             1e-9), f"phi(50 wt%) = {phi50:.4f} (exact mixture rule)")
ws = np.linspace(0.01, 1.0, 200)
phis = np.array([rh.phi_from_w(w) for w in ws])
check(bool(np.all(np.diff(phis) > 0)), "phi(w) strictly increasing")
check(approx(rh.eta_infinite(0.0), 1e-3, 1e-9), "eta(phi=0) = water")
etas = np.array([rh.apparent_viscosity(p, 50.0) for p in phis])
check(bool(np.all(np.diff(etas) >= -1e-15)),
      "apparent viscosity non-decreasing in solids fraction")
check(rh.eta_infinite(PHI_MAX * 0.999) > 1.0,
      "viscosity diverges near max packing (>1000x water)")
check(rh.tau_yield(0.15) == 0.0, "no yield stress below onset")
check(rh.tau_yield(0.45) > rh.tau_yield(0.30) > 0.0,
      "yield stress grows with phi")
rho60 = rh.rho_mix(0.6)
check(1000 < rho60 < RHO_SLAG, f"rho_mix(60 wt%) = {rho60:.0f} in bounds")
vs_dilute = float(rh.settling_velocity(0.01))
# Stokes for 5 um slag in water: ~3.3e-5 m/s (~12 cm/h)
check(approx(vs_dilute, 3.27e-5, 6e-6),
      f"Stokes settling 5um = {vs_dilute*3.6e6:.1f} mm/h (~118)")
check(float(rh.settling_velocity(0.45)) < vs_dilute * 0.01,
      "yield-stressed slurry stops settling")

# ------------------------------------------------------------------ solver
print("\n2. Fluid solver: conservation, incompressibility, stability")
g = FluidGrid2D(n=64, tank_diameter=0.40)
rng = np.random.default_rng(7)
g.c[g.liquid] = rng.uniform(0.1, 0.3, int(g.liquid.sum()))
g.u[g.liquid] = rng.uniform(-0.5, 0.5, int(g.liquid.sum()))
g.v[g.liquid] = rng.uniform(-0.5, 0.5, int(g.liquid.sum()))
# Smooth the random field first: raw noise is half checkerboard modes,
# which the collocated 2dx div/grad stencil cannot see (standard Stam
# limitation); real solver fields are smooth after diffusion/advection.
for _ in range(2):
    g.u = g.diffuse(g.u, 5e-3, 1 / 30, iters=10)
    g.v = g.diffuse(g.v, 5e-3, 1 / 30, iters=10)
tot0 = g.total_phi()


def rms_div(gr):
    dv = np.zeros((gr.n, gr.n))
    dv[1:-1, 1:-1] = ((gr.u[1:-1, 2:] - gr.u[1:-1, :-2])
                      + (gr.v[2:, 1:-1] - gr.v[:-2, 1:-1])) / (2 * gr.dx)
    inner = gr.rgrid < gr.radius_cells - 2
    return float(np.sqrt((dv[inner] ** 2).mean()))


d_before = rms_div(g)
g.project(iters=80)
d_after = rms_div(g)
check(d_after < 0.25 * d_before,
      f"projection cuts rms divergence {d_before:.1f} -> {d_after:.2f} (>4x)")
for _ in range(60):
    g.advect_solids(1 / 30)
    g.u = g.diffuse(g.u, 1e-4, 1 / 30)
    g.v = g.diffuse(g.v, 1e-4, 1 / 30)
    g.project(iters=20)
    g.advect_velocity(1 / 30)
tot1 = g.total_phi()
check(approx(tot1, tot0, tot0 * 0.01),
      f"solids conserved under advection: {tot0:.1f} -> {tot1:.1f} (<1% drift)")
check(np.isfinite(g.u).all() and np.isfinite(g.c).all(), "no NaN/Inf")

print("\n3. Stability at extreme viscosity (100% slib paste)")
g2 = FluidGrid2D(n=64)
g2.u[g2.liquid] = 1.0
for _ in range(120):
    g2.u = g2.diffuse(g2.u, 1.0, 1 / 30, iters=10)   # nu = 1 m2/s (paste)
    g2.project(iters=10)
    g2.advect_velocity(1 / 30)
check(np.isfinite(g2.u).all(), "implicit diffusion stable at nu=1 m2/s")
check(g2.max_speed() < 1.5, f"paste flow damped, |u|max={g2.max_speed():.3f}")

# ----------------------------------------------------------------- stirrer
print("\n4. Stirrer power model")
st = Stirrer()
st.plugged_in = True
st.on = True
rho_w, phi_w = 1000.0, 0.0
st.rpm_set = 300.0
st.update(rho_w, rh, phi_w, 1 / 30)
p300 = st.p_shaft
re300 = st.re
check(re300 > 1e4, f"water @300rpm turbulent: Re={re300:.2e}")
check(0.5 < p300 < 100.0, f"P_shaft water 300rpm = {p300:.1f} W (plausible)")
st.rpm_set = 600.0
st.update(rho_w, rh, phi_w, 1 / 30)
ratio = st.p_shaft / p300
check(6.0 < ratio < 10.0,
      f"turbulent P ~ N^3: P(600)/P(300) = {ratio:.2f} (expect ~8)")
check(st.p_electric > st.p_shaft, "electrical power > shaft power")
# Laminar check: paste-like viscosity
st2 = Stirrer(torque_max=1e9)  # no droop, isolate the correlation
st2.plugged_in = True
st2.on = True
rh_hi = SlurryRheology()
phi_hi = 0.55
st2.rpm_set = 30.0
st2.update(1800.0, rh_hi, phi_hi, 1 / 30)
check(st2.re < 10, f"paste laminar: Re={st2.re:.2e}")
p1 = st2.p_shaft
st2.rpm_set = 60.0
st2.update(1800.0, rh_hi, phi_hi, 1 / 30)
# Bingham slurry: tau_y/(ks*N) term makes eta ~ 1/N, so P ~ eta*N^2 ~ N
check(1.5 < st2.p_shaft / p1 < 4.5,
      f"laminar Bingham P scaling P(60)/P(30) = {st2.p_shaft/p1:.2f}")

print("\n5. Overload, droop and breaker trip")
st3 = Stirrer()
st3.plugged_in = True
st3.on = True
st3.rpm_set = 400.0
w85 = 0.85
st3.update(rh.rho_mix(w85), rh, rh.phi_from_w(w85), 1 / 30)
check(st3.overload, "85 wt% slib @400rpm overloads the motor")
check(st3.rpm_actual < 0.9 * st3.rpm_set,
      f"RPM droops/locks: {st3.rpm_actual:.1f} < set {st3.rpm_set:.0f} "
      "(phi > phi_max: paste)")
check(approx(st3.torque, st3.torque_max, 0.05),
      f"droop rides the torque limit ({st3.torque:.2f} ~ {st3.torque_max})")
check(st3.p_electric >= st3.p_stall - 1e-6,
      f"stalled rotor pulls stall current ({st3.p_electric:.0f} W)")
for _ in range(200):
    st3.update(rh.rho_mix(w85), rh, rh.phi_from_w(w85), 1 / 30)
check(st3.tripped and not st3.on, "breaker trips after 5 s overload")
st3.reset_trip()
check(not st3.tripped, "trip resettable")

# -------------------------------------------------------------------- tank
print("\n6. Tank: composition, hose, slag pour, energy meter")
t = MixingTank(n=64, water_l=10.0, w_pct=20.0)
check(approx(t.w(), 0.20, 1e-9), "slider 20% -> w = 0.20")
m_slag0 = t.slag_kg
check(approx(m_slag0, 10.0 * 0.20 / 0.80, 1e-6),
      f"slag mass = {m_slag0:.2f} kg for 10 L water @20wt%")
field_kg = t.grid.total_phi() / t._phi_per_kg()
check(approx(field_kg, t.slag_kg, 0.05 * max(t.slag_kg, 1)),
      f"field mass matches bookkeeping: {field_kg:.2f} ~ {t.slag_kg:.2f} kg")
added = t.add_water(4.0)
check(approx(added, 4.0, 1e-9) and t.w() < 0.20, "hose adds water, dilutes")
over = t.add_water(1e6)
check(t.volume_l <= t.capacity_l + 1e-6,
      f"capacity respected ({t.volume_l:.1f} <= {t.capacity_l} L)")
t.add_slag(2.0)
for _ in range(80):
    t.step(1 / 30)
check(t.pour_queue_kg < 1e-6, "poured slag fully entered the field")
check(np.isfinite(t.grid.c).all(), "field finite after pour")
t.set_composition(60.0)
for _ in range(30):
    t.step(1 / 30)
check(t.volume_l <= t.capacity_l + 1e-6,
      f"set_composition drains water to respect capacity "
      f"({t.volume_l:.1f} <= {t.capacity_l} L)")
check(approx(t.w(), 0.60, 5e-3) or t.pour_queue_kg > 0,
      f"slider up targets 60% (w={t.w()*100:.1f}%, queue "
      f"{t.pour_queue_kg:.1f} kg)")
t.set_composition(88.0, instant=True)
check(t.volume_l <= t.capacity_l + 1e-6 and t.phi_bulk() > PHI_MAX,
      f"88% fits in tank and is paste (phi={t.phi_bulk():.2f})")
t.set_composition(10.0)
check(approx(t.w(), 0.10, 5e-3), f"slider down -> w = {t.w()*100:.1f}%")
check(t.meter.energy_kwh >= 0.0, "energy meter monotone")

print("\n7. Placement / plug gating (AR contract)")
t2 = MixingTank(n=64, water_l=12.0, w_pct=15.0)
t2.stirrer.rpm_set = 300.0
t2.stirrer.on = True            # not plugged in yet
t2.step(1 / 30)
check(t2.stirrer.rpm_actual == 0.0, "stirrer dead without wall socket")
check(t2.meter.p_watt == 0.0, "no standby draw when unplugged")
t2.stirrer.plugged_in = True
t2.placed = False
t2.step(1 / 30)
check(t2.stirrer.rpm_actual == 0.0, "stirrer idle when tank not placed")
check(approx(t2.meter.p_watt, t2.stirrer.p_standby, 1e-9),
      "standby draw once plugged in")
t2.placed = True
t2.step(1 / 30)
check(t2.stirrer.rpm_actual > 0.0, "stirrer runs when placed + plugged")
check(t2.meter.p_watt > t2.stirrer.p_standby, "meter sees motor load")

print("\n8. Mixing raises mixedness; stopping settles the slib")
t3 = MixingTank(n=64, water_l=12.0, w_pct=10.0)
t3.settle_bottom()
m0 = t3.grid.mixedness()
t3.stirrer.plugged_in = True
t3.stirrer.on = True
t3.stirrer.rpm_set = 400.0
for _ in range(240):            # 8 s of stirring
    t3.step(1 / 30)
m1 = t3.grid.mixedness()
check(m1 > m0 + 10.0, f"stirring mixes: {m0:.0f}% -> {m1:.0f}%")
snap = t3.snapshot()
check(snap["p_electric_w"] > 10.0 and snap["energy_kwh"] > 0,
      f"power drawn: {snap['p_electric_w']:.0f} W, {snap['energy_kwh']*1000:.2f} Wh")
t3.stirrer.on = False
com0 = float((t3.grid.c * t3.grid.Y)[t3.grid.liquid].sum()
             / max(t3.grid.total_phi(), 1e-9))
for _ in range(700):            # spin-down (bottom drag) -> boosted settling
    t3.step(1 / 30)
com1 = float((t3.grid.c * t3.grid.Y)[t3.grid.liquid].sum()
             / max(t3.grid.total_phi(), 1e-9))
check(com1 > com0 + 0.5,
      f"solids settle when idle (centroid row {com0:.1f} -> {com1:.1f})")
check(float(t3.grid.c.max()) <= PHI_PACK_BED + 1e-9,
      "sediment respects packed-bed cap")

print("\n9. Thermal coupling: heat thins, stirring warms, the wall sheds")
rh40 = rh.at_conditions(40.0)
rh5 = rh.at_conditions(5.0)
rh20 = rh.at_conditions(20.0)
f40 = rh40.eta_liquid / rh.eta_liquid
f5 = rh5.eta_liquid / rh.eta_liquid
check(rh20.eta_liquid == rh.eta_liquid, "at 20 C the rheology tables are exact")
check(0.4 < f40 < 0.8,
      f"Arrhenius: 40 C thins the liquid phase (x{f40:.2f}, water ~0.6)")
check(1.3 < f5 < 2.0, f"5 C thickens the liquid phase (x{f5:.2f})")
check(rh40.settling_velocity(0.10) > rh.settling_velocity(0.10),
      "warm slurry settles faster (Stokes over eta_liquid)")
t4 = MixingTank(n=48, water_l=14.0, w_pct=20.0)
t4.stirrer.plugged_in = True
t4.stirrer.on = True
t4.stirrer.rpm_set = 300.0
T0 = t4.temperature_c
q_in = q_out = 0.0
for _ in range(1200):            # 120 s of stirring
    q_in += t4.stirrer.p_shaft * 0.1
    q_out += t4.wall_loss_w_per_k * (t4.temperature_c - t4.t_ambient_c) * 0.1
    t4.step(0.1)
m_kg = t4.water_kg + t4.slag_kg
cp = (t4.water_kg * CP_WATER + t4.slag_kg * CP_SLAG) / m_kg
dT = t4.temperature_c - T0
check(dT > 0.005, f"stirring heats the slurry (dT = {dT * 1000:.0f} mK/120 s)")
resid = abs(q_in - q_out - m_kg * cp * dT) / max(q_in, 1.0)
check(resid < 0.01, f"0-D heat balance closes (residual {resid:.4f})")
t5 = MixingTank(n=48, water_l=10.0, w_pct=0.0)
q = 0.0
for _ in range(120):             # calibrated pulse: 60 s at 1000 W
    t5._thermal_step(0.5, 1000.0)
    q += 1000.0 * 0.5
dT_pulse = t5.temperature_c - T_REF_C
check(abs(dT_pulse - q / (t5.water_kg * CP_WATER)) < 0.01 * dT_pulse,
      f"heat pulse matches hand calculation (dT = {dT_pulse:.3f} K)")
t5.temperature_c = 60.0
for _ in range(600):             # 600 s idle at the wall
    t5.step(1.0)
drop = 60.0 - t5.temperature_c
check(0.3 < drop < 1.5 and t5.temperature_c > t5.t_ambient_c,
      f"idle bath cools toward ambient (−{drop:.2f} K in 600 s)")

print("\n10. Thixotropy: shear breaks the gel, rest rebuilds it (opt-in)")
t6 = MixingTank(n=48, water_l=12.0, w_pct=35.0, thixotropy=True)
check(t6.struct_lambda == 1.0, "fresh slurry fully built (lambda = 1)")
t6.stirrer.plugged_in = True
t6.stirrer.on = True
t6.stirrer.rpm_set = 400.0
for _ in range(1800):            # 60 s at 400 rpm
    t6.step(1 / 30)
lam_broken = t6.struct_lambda
check(0.05 < lam_broken < 0.45,
      f"60 s @400 rpm breaks the gel (lambda = {lam_broken:.2f})")
t6.stirrer.on = False
for _ in range(3600):            # 120 s at rest
    t6.step(1 / 30)
check(t6.struct_lambda > lam_broken + 0.5,
      f"rest rebuilds the structure (lambda = {t6.struct_lambda:.2f})")
check(0.0 <= t6.struct_lambda <= 1.0, "lambda stays bounded in [0, 1]")
sur = rh.at_conditions(20.0, structural=1.0 + C_THIX)
check(abs(sur.apparent_viscosity(0.30, 50.0)
          - (1.0 + C_THIX) * rh.apparent_viscosity(0.30, 50.0)) < 1e-12,
      "built gel scales the whole flow curve by (1 + C_thix)")
t7 = MixingTank(n=48, water_l=12.0, w_pct=35.0)   # thixotropy OFF (default)
t7.stirrer.plugged_in = True
t7.stirrer.on = True
t7.stirrer.rpm_set = 300.0
t7.step(1 / 30)
classic = rh.apparent_viscosity(t7.phi_bulk(), 11.0 * t7.stirrer.rpm_set / 60.0)
check(t7.stirrer.eta_app == classic,
      "thixotropy off: eta_app matches the classic tables exactly")

print("\n11. Vortex dip (free surface) and air entrainment flag")


def dip_after(rpm, frames=300):
    tv = MixingTank(n=64, water_l=14.0, w_pct=20.0)
    tv.stirrer.plugged_in = True
    tv.stirrer.on = True
    tv.stirrer.rpm_set = rpm
    for _ in range(frames):      # 10 s spin-up
        tv.step(1 / 30)
    return tv


tq = dip_after(0.0)
check(tq.grid.vortex_dip_m() < 1.0e-4, "quiescent bath: no vortex (<0.1 mm)")
d150 = dip_after(150.0).grid.vortex_dip_m()
d300 = dip_after(300.0).grid.vortex_dip_m()
d600 = dip_after(600.0).grid.vortex_dip_m()
check(0.0 < d150 < d300 < d600,
      f"dip grows with swirl ({d150 * 1e3:.1f} < {d300 * 1e3:.1f} "
      f"< {d600 * 1e3:.1f} mm)")
depth_mm = tq.liquid_depth_m() * 1000.0
snap600 = dip_after(600.0).snapshot()
check(snap600["vortex_dip_mm"] <= depth_mm + 1e-9,
      f"dip readout capped at liquid depth ({depth_mm:.0f} mm)")
check(snap600["air_entrainment"], "600 rpm: Fr > 1 -> air entrainment flagged")
snap150 = dip_after(150.0).snapshot()
check(not snap150["air_entrainment"], "150 rpm: gentle stirring pulls no air")

print(f"\n=== {'ALL CHECKS PASSED' if failures == 0 else f'{failures} FAILURES'} ===\n")
sys.exit(1 if failures else 0)
