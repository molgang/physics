"""Proof suite for the MOLGANG downstream chain (headless).

Guards the physical invariants of vacuum filtration, centrifuge drying,
Nafion 117 membrane options (dubbel / teflon), electrolysis, electrodialysis
and the ton-per-day scale-up -- so the JS/AR port has a reference to stay in
parity with. Run: python3 test_downstream.py  (exit 0 = pass)
"""

import math
import sys

from downstream import (BasketCentrifuge, ElectrodialysisStack,
                        ElectrolysisCell, Nafion117, ProductionScale,
                        VacuumFiltration, steel_slag_cases)

failures = 0


def check(cond, msg):
    global failures
    print(f"  {'✓' if cond else '✗ FAIL:'} {msg}")
    if not cond:
        failures += 1


def approx(a, b, tol):
    return abs(a - b) <= tol


print("\n=== downstream proof suite ===\n")

# ---------------------------------------------------------------- imports
from downstream import F_CONST  # noqa: F401  (contract: Faraday geexporteerd)

# ------------------------------------------------------- vacuumfiltratie
print("1. Vacuumfiltratie: Darcy t/V-vorm, koek, restvocht")
vf = VacuumFiltration(area_m2=1.0)
v = 0.05
t_v = vf.time_for_filtrate(v)
check(approx(vf.filtrate_after(t_v), v, 1e-9),
      "t/V-vorm is omkeerbaar (filtrate_after(time_for_filtrate))")
check(vf.time_for_filtrate(2 * v) > 2.0 * t_v,
      "koekterm maakt de tijd superlineair in V")
vf2 = VacuumFiltration(area_m2=1.0, delta_p_pa=120.0e3)
check(vf2.time_for_filtrate(v) < 0.55 * t_v,
      "twee keer de drukval ~ halveer de filtratietijd (t ~ 1/dp)")
check(approx(vf.cake_mass(v), 200.0 * v, 1e-9),
      "koekmassa = c_solids * V")
check(vf.residual_moisture(30.0e3) > vf.residual_moisture(60.0e3)
      > vf.residual_moisture(120.0e3),
      "restvocht daalt met de drukval (~sqrt-machtsregel)")
vf3 = VacuumFiltration(area_m2=25.0, c_solids_kg_m3=200.0)
check(vf3.time_for_filtrate(5.0) < 2.0 * 3600.0,
      f"5 m3 slib over 25 m2 kamerfilterplaat: "
      f"{vf3.time_for_filtrate(5.0) / 60:.0f} min (< 2 h)")

# -------------------------------------------------------------- centrifuge
print("\n2. Centrifuge: G-kracht, bed-tijd, Ambler-Sigma, spin-dry (5 L)")
cf = BasketCentrifuge(volume_l=5.0, rpm=3000.0)
check(approx(cf.g_force(), 1207.0, 120.0),
      f"3000 rpm op 0,12 m = {cf.g_force():.0f} g (~1200)")
cf15 = BasketCentrifuge(volume_l=5.0, rpm=1500.0)
check(approx(cf15.settling_time_s() / cf.settling_time_s(), 4.0, 0.2),
      "bed-tijd ~ 1/omega^2 (1500 rpm duurt ~4x zo lang)")
check(cf.settling_time_s() < 10.0,
      f"5 um slak zakt in {cf.settling_time_s():.1f} s naar de wand "
      "(daarom werkt een centrifuge)")
check(cf.sigma_m2() > 0.0
      and approx(BasketCentrifuge(volume_l=5.0, rpm=1500.0).sigma_m2(),
                 0.25 * cf.sigma_m2(), 1e-9),
      "Ambler-Sigma ~ omega^2 (continue rating Q = v_g * Sigma)")
m600 = cf.residual_moisture(m0=0.55, spin_s=600.0)
m1200 = cf.residual_moisture(m0=0.55, spin_s=1200.0)
cf_hi = BasketCentrifuge(volume_l=5.0, rpm=6000.0)
m_hi = cf_hi.residual_moisture(m0=0.55, spin_s=600.0)
check(0.0 < m600 < 0.55, f"spin-dry werkt: {0.55:.2f} -> {m600:.3f} in 10 min")
check(m1200 < m600, "langer droogdraaien droogt verder")
check(m_hi < m600, f"hogere G droogt droger ({m_hi:.3f} < {m600:.3f})")
cyc = cf.batch_cycle_s()
check(cyc > 600.0 and cf.batches_per_day() > 50.0,
      f"5 L-cyclus {cyc / 60:.0f} min -> {cf.batches_per_day():.0f} "
      "batches/dag")

# ----------------------------------------------------------------- nafion
print("\n3. Nafion 117: weerstand, doorslag, levensduur; dubbel en teflon")
nm = Nafion117()
check(approx(nm.area_resistance_ohm_m2, 178e-6 / 2.0, 1e-9),
      "ASR = delta/sigma per laag")
nd = Nafion117(layers=2)
nt = Nafion117(teflon_backing=True)
check(approx(nd.area_resistance_ohm_m2, 2.0 * nm.area_resistance_ohm_m2, 1e-9),
      "dubbel nafion = 2x oppervlakteweerstand (serielagen)")
check(approx(nt.area_resistance_ohm_m2, 1.15 * nm.area_resistance_ohm_m2, 1e-9),
      "teflon-rug: kleine ASR-penalty (1.15x)")
check(approx(nd.v_permeability, 0.5 * nm.v_permeability, 1e-15),
      "dubbel nafion halveert de V-doorslag")
check(Nafion117().lifetime_h(60.0) < Nafion117().lifetime_h(40.0)
      < Nafion117().lifetime_h(25.0),
      "levensduur daalt exponentieel met temperatuur")
check(nd.lifetime_h(40.0) > nm.lifetime_h(40.0)
      and nt.lifetime_h(40.0) > nd.lifetime_h(40.0),
      "levensduur: teflon > dubbel > enkel")
check(nt.replacement_downtime_h < nm.replacement_downtime_h,
      "teflon-rug maakt tussentijdse vervanging mogelijk (2 h vs 12 h)")
try:
    Nafion117(layers=3)
    check(False, "layers=3 moet een ValueError geven")
except ValueError:
    check(True, "layers=3 wordt afgewezen (alleen 1 of 2)")

# ------------------------------------------------------------- electrolyse
print("\n4. Electrolyse: Faraday-opbrengst, celspanning, kWh/kg")
ec = ElectrolysisCell(area_m2=1.0, membrane=Nafion117())
p1 = ec.production_kg_s
ec2 = ElectrolysisCell(area_m2=1.0, i_a_m2=800.0, membrane=Nafion117())
check(approx(ec2.production_kg_s / p1, 2.0, 1e-9),
      "Faraday: productie evenredig met de stroom")
check(1.0e-4 < p1 < 5.0e-4,
      f"productie-orde {p1 * 3600:.2f} kg V per uur per m2 (realistisch)")
check(approx(ec.cell_voltage_v,
             1.00 + 0.25 + 400.0 * Nafion117().area_resistance_ohm_m2, 1e-12),
      "celspanning = E_eq + overpotentiaal + i*ASR (exact)")
ec_d = ElectrolysisCell(area_m2=1.0, membrane=Nafion117(layers=2))
check(ec_d.energy_kwh_per_kg > ec.energy_kwh_per_kg,
      f"dubbel nafion kost energie: {ec_d.energy_kwh_per_kg:.3f} vs "
      f"{ec.energy_kwh_per_kg:.3f} kWh/kg")
ec_t = ElectrolysisCell(area_m2=1.0, membrane=Nafion117(teflon_backing=True))
check(ec.energy_kwh_per_kg < ec_t.energy_kwh_per_kg < ec_d.energy_kwh_per_kg,
      "energie-orde: enkel < teflon < dubbel")
kg, kwh = ec.run(hours=10.0)
check(approx(kg, p1 * 36000.0, 1e-9) and kwh > 0.0,
      "run(h) boekt kg en kWh consequent")

# ----------------------------------------------------------- electrodialyse
print("\n5. Electrodialyse: verwijdering per lading, i_lim, kWh/m3")
ed = ElectrodialysisStack()
r1 = ed.removal_mol_s()
ed2 = ElectrodialysisStack(i_a_m2=200.0)
check(approx(ed2.removal_mol_s() / r1, 2.0, 1e-9),
      "verwijdering evenredig met de stroom (Faraday per celpaar)")
check(approx(ed.stack_voltage_v,
             40 * (100.0 * 0.03 + 0.10), 1e-9),
      "stapelspanning = n * (i*ASR + E_diff) (exact)")
check(not ed.limiting_flag, "100 A/m2 onder de grensstroom (150)")
check(ElectrodialysisStack(i_a_m2=200.0).limiting_flag,
      "200 A/m2 boven de grensstroom -> gewaarschuwd")
t_s, kwh, kwh_m3 = ed.treat(1.0, 100.0, 10.0)     # brak 5.8 -> 0.6 g/L NaCl
t_s2, kwh2, kwh_m3_2 = ed.treat(1.0, 100.0, 50.0)
check(t_s2 < t_s and kwh_m3_2 < kwh_m3,
      "minder verwijdering = korter en goedkoper per m3")
check(1.0 < kwh_m3 < 30.0,
      f"energie {kwh_m3:.1f} kWh/m3 in de brakwater-orde")

# --------------------------------------------------------------- opschaling
print("\n6. Opschaling naar tonnen/dag en membraanvervangplan")
v_day = 5000.0 * 8.0 / 100.0 * 0.440          # 5 t slak/dag -> kg V/dag
cell3 = ElectrolysisCell(area_m2=2.5, membrane=Nafion117())
n_lines = max(1, math.ceil(v_day / (cell3.production_kg_s * 24 * 3600.0)))
sc = ProductionScale(kg_per_batch=v_day / n_lines, batch_hours=24.0,
                     lines=n_lines, membrane=Nafion117(),
                     membrane_area_m2=2.5)
check(approx(sc.tonnes_per_day, v_day / 1000.0, v_day / 1000.0 * 0.05),
      f"lijnen opgeteld: {sc.tonnes_per_day:.3f} t V/dag uit 5 t slak/dag")
sc2 = ProductionScale(kg_per_batch=v_day / n_lines, batch_hours=24.0,
                      lines=2 * n_lines, membrane=Nafion117(),
                      membrane_area_m2=2.5)
check(approx(sc2.tonnes_per_day, 2 * sc.tonnes_per_day, 1e-9),
      "tonnage lineair in het aantal lijnen")
plans = {}
for naam, mem in (("enkel", Nafion117()),
                  ("dubbel", Nafion117(layers=2)),
                  ("teflon", Nafion117(teflon_backing=True))):
    s = ProductionScale(kg_per_batch=v_day / n_lines, batch_hours=24.0,
                        lines=n_lines, membrane=mem, membrane_area_m2=2.5)
    plans[naam] = s.membrane_plan()
check(plans["teflon"][1] < plans["enkel"][1],
      f"downtime per jaar: teflon {plans['teflon'][1]:.1f} h < enkel "
      f"{plans['enkel'][1]:.1f} h")
check(all(p[3] > 0.99 for p in plans.values()),
      "beschikbaarheid > 99% voor alle strategieen")

# -------------------------------------------------------------------- cases
print("\n7. Voorbeeldcases met staalslak")
cases = steel_slag_cases()
check(len(cases) == 3, "drie cases gedefinieerd")
c1 = cases[0]
check(approx(c1["v2o5_kg"], 80.0, 1e-9),
      "case 1: 1 t BOF-slak @ 8 wt% -> 80 kg V2O5")
check(c1["filtratie_s"] < 2.0 * 3600.0 and c1["koek_kg"] > 990.0,
      f"case 1: filtratie {c1['filtratie_s'] / 60:.0f} min, koek "
      f"{c1['koek_kg']:.0f} kg")
check(c1["vrfb_electrolyt_l"] > 400.0 and c1["opslag_kwh"] > 1000.0,
      f"case 1: {c1['vrfb_electrolyt_l']:.0f} L VRFB-electrolyt = "
      f"{c1['opslag_kwh'] / 1000:.1f} MWh opslag")
c2 = cases[1]
check(c2["vocht_na_10min"] < 0.10 and c2["water_eruit_kg"] > 0.5,
      f"case 2: 5 L-batch droogt naar {c2['vocht_na_10min'] * 100:.1f}% "
      f"({c2['water_eruit_kg']:.2f} kg water eruit)")
strat = cases[2]["strategieen"]
check(set(strat) == {"enkel", "dubbel", "teflon"},
      "case 3: drie membraanstrategieen vergeleken")
check(strat["dubbel"]["doorslag_relatief"] == 0.5,
      "case 3: dubbel nafion halveert de doorslag")
check(strat["teflon"]["downtime_h_jaar"] < strat["enkel"]["downtime_h_jaar"],
      "case 3: teflon-rug minimaliseert de stilstand")

print(f"\n=== {'ALL CHECKS PASSED' if failures == 0 else f'{failures} FAILURES'} ===\n")
sys.exit(1 if failures else 0)
