"""downstream -- neerstroom-modellen: van slurry naar product (pure numpy/stdlib).

De keten na het roeren/bezinken uit viscosity_core, als eerste-orde
modellen met echte constanten (SI tenzij vermeld, geen GUI-imports):

  VacuumFiltration  -- Darcy constante-druk filtratie (t/V-vorm), koek en
                       restvocht; slib-fijn (alpha ~ 5e10 m/kg).
  BasketCentrifuge  -- batch-centrifuge (vanaf 5 L): G-kracht, radiale
                       Stokes-bedxttijd, Ambler-Sigma voor doorzet-rating en
                       spin-dry droogmodel (restvocht tegen tijd en G).
  Nafion117         -- membraanmodel: oppervlakteweerstand, V-doorslag,
                       degradatielevensduur; opties: DUBBELE nafion
                       (lagen in serie) of teflon (PTFE)-rugversterking
                       voor tussentijdse vervanging zonder lange stilstand.
  ElectrolysisCell  -- Faraday-opbrengst + celspanning (Nernst-basis +
                       overpotentiaal + i*R over het membraan) -> kWh/kg.
  ElectrodialysisStack -- zoutverwijdering per lading; i_lim-wachter,
                       kWh/m3 voor zuur-/waswater-terugwinning.
  ProductionScale   -- opschaling van labbatch naar tonnen/dag incl.
                       membraanvervangplan (enkele vs dubbel vs teflon).
  steel_slag_cases()-- drie rekenbare voorbeeldcases met staalslak.

De membraan-afweging is bewust expliciet: dubbel nafion verlaagt doorslag
en verlengt de levensduur maar verdubbelt de i*R-energie; teflon-rug
kost weinig weerstand en maakt vervanging goedkoop. De suite bewaakt
beide extremen.
"""

from __future__ import annotations

import math

from viscosity_core import SlurryRheology

F_CONST = 96485.0          # C/mol, Faraday
ETA_WATER = 1.0e-3         # Pa.s
GRAVITY = 9.81

# staalslak (BOF/LD): V2O5-gehalte en marktprijs-orde voor de cases
V2O5_WT_PCT_BOF = 8.0      # massa-% V2O5 in gemalen BOF-slak (vooronijs)
V2O5_EUR_PER_KG = 9.0      # EUR/kg, feV/V2O5-markt-orde
NAFION_EUR_PER_M2 = 750.0  # EUR/m2, Nafion 117 orde (2026)
EUR_PER_KWH = 0.30         # zelfde tarief als viscosity_core-meter


# ---------------------------------------------------------------------------
# Vacuumfiltratie (Darcy, constante drukval)
# ---------------------------------------------------------------------------
class VacuumFiltration:
    """Constante-druk nutsch/filter: t = K1*V^2 + K2*V.

    K1 bevat de koeksweerstand (alpha * c, groeit met de koek), K2 het
    medium. Fijn slib is compressibel: alpha hoog. Restvocht daalt
    empirisch met de wortel van de drukval (machtsregel, orde 0.5).
    """

    def __init__(self, area_m2=0.5, delta_p_pa=60.0e3, alpha_m_kg=5.0e10,
                 r_medium_m=1.0e11, c_solids_kg_m3=200.0, eta_pa_s=ETA_WATER,
                 rho_cake_bulk=1800.0, moisture_ref=0.55):
        self.area = area_m2
        self.dp = delta_p_pa
        self.alpha = alpha_m_kg
        self.r_m = r_medium_m
        self.c = c_solids_kg_m3
        self.eta = eta_pa_s
        self.rho_cake = rho_cake_bulk
        self.moisture_ref = moisture_ref      # m_vocht/m_droog bij dp_ref
        self.dp_ref = 60.0e3
        # t/V-vorm: t = K1 V^2 + K2 V
        self.k1 = eta_pa_s * alpha_m_kg * c_solids_kg_m3 \
            / (2.0 * area_m2 ** 2 * delta_p_pa)          # s/m6
        self.k2 = eta_pa_s * r_medium_m / (area_m2 * delta_p_pa)  # s/m3

    def time_for_filtrate(self, v_m3):
        """Filtratietijd (s) voor V m3 filtraat."""
        return self.k1 * v_m3 ** 2 + self.k2 * v_m3

    def filtrate_after(self, t_s):
        """Filtraatvolume (m3) na t seconden (positieve wortel)."""
        if self.k1 <= 0:
            return t_s / max(self.k2, 1e-9)
        return (-self.k2 + math.sqrt(self.k2 ** 2 + 4.0 * self.k1 * t_s)) \
            / (2.0 * self.k1)

    def cake_mass(self, v_m3):
        """Droge koekmassa (kg) bij filtraatvolume V."""
        return self.c * v_m3

    def cake_thickness(self, v_m3):
        """Koekdikte (m) aan het einde van de cyclus."""
        return self.cake_mass(v_m3) / self.rho_cake / self.area

    def throughput_m3_h(self, t_s=3600.0):
        """Gemiddeld filtraat-volume debiet over een cyclus van t_s."""
        return self.filtrate_after(t_s) / t_s * 3600.0

    def residual_moisture(self, delta_p_pa=None):
        """Restvocht m_vocht/m_droog (-), daalt ~ sqrt met de drukval."""
        dp = self.dp if delta_p_pa is None else delta_p_pa
        return self.moisture_ref * math.sqrt(self.dp_ref / max(dp, 1.0e3))


# ---------------------------------------------------------------------------
# Batch-centrifuge en centrifugedroging (vanaf 5 L)
# ---------------------------------------------------------------------------
class BasketCentrifuge:
    """Batch-basket centrifuge: zetten, G-kracht, bed-tijd, spin-dry.

    Radiale Stokes in een centrifugale veld: v_r = v_g * omega^2 r / g,
    dus de bed-tijd over de vloeilaag is (g / (v_g omega^2)) ln(r2/r1).
    De continue rating gebruikt Ambler's Sigma-theorie (tubular-vorm) als
    vergelijkingsmaatstaf Q = v_g * Sigma.
    Spin-dry: restvocht M(t) = M_inf + (M0 - M_inf) exp(-t/tau) met
    M_inf ~ G^-b (empirisch, b = 0.35) en tau ~ G^-0.5.
    """

    def __init__(self, volume_l=5.0, r_inner=0.08, r_wall=0.12, rpm=3000.0,
                 rheo=None):
        self.volume_l = volume_l
        self.r1 = r_inner
        self.r2 = r_wall
        self.rpm = rpm
        self.omega = rpm / 60.0 * 2.0 * math.pi
        self.rheo = rheo or SlurryRheology()

    # -- veld en ratings ----------------------------------------------------
    def g_force(self):
        """Centrifugale versnelling aan de wand, in g's."""
        return self.omega ** 2 * self.r2 / GRAVITY

    def sigma_m2(self, bowl_length_m=0.20):
        """Ambler-Sigma (m2): continue rating Q = v_g * Sigma (tubular-vorm,
        zelfde G-bereik als de basket, voor opschalingsvergelijkingen)."""
        return (2.0 * math.pi * bowl_length_m * self.omega ** 2
                * (self.r2 ** 3 - self.r1 ** 3)
                / (3.0 * GRAVITY * math.log(self.r2 / self.r1)))

    def settling_time_s(self, d_m=5.0e-6, phi=0.10):
        """Tijd om een deeltje d van r1 naar de wand te sedimenteren (s)."""
        v_g = self.rheo.settling_velocity(phi)
        if v_g <= 0:
            return math.inf
        # rho-vervalsing: v_g is voor slak; schaal met (dr) niet nodig,
        # de rheologie-klasse houdt al de 5 um-slak aan.
        return GRAVITY * math.log(self.r2 / self.r1) / (v_g * self.omega ** 2)

    # -- spin-dry ------------------------------------------------------------
    def residual_moisture(self, m0=0.55, spin_s=600.0, m_inf_ref=0.05):
        """Restvocht m_vocht/m_droog na spin_s seconden droogdraaien.

        M_inf ~ G^-0.35 genormaliseerd op 2000 g; tau = 120 s bij 2000 g,
        ~ G^-0.5 (dunnere films en hogere druk drogen sneller).
        """
        g = max(self.g_force(), 1.0)
        m_inf = m_inf_ref * (2000.0 / g) ** 0.35
        tau = 120.0 * (2000.0 / g) ** 0.5
        return m_inf + (m0 - m_inf) * math.exp(-spin_s / tau)

    def batch_cycle_s(self, fill_s=60.0, spin_s=600.0, unload_s=90.0,
                      accel_s=30.0):
        """Kosten van een batchcyclus op 5 L (of schaal): vullen, op toeren,
        bed-tijd, droogdraaien, lossen."""
        t_bed = self.settling_time_s()
        return fill_s + accel_s + max(t_bed, 1.0) + spin_s + unload_s

    def batches_per_day(self, **kw):
        return 24.0 * 3600.0 / self.batch_cycle_s(**kw)


# ---------------------------------------------------------------------------
# Nafion 117 membraan (en vervangstrategieen)
# ---------------------------------------------------------------------------
class Nafion117:
    """Nafion 117 (178 um) in zuur/V-milieu, eerste orde.

    - Oppervlakteweerstand ASR = delta/sigma per laag; lagen in serie
      tellen op (dubbel nafion = 2x ASR).
    - Vanadium-doorslag ~ P/laag; dubbel halveert de doorslag.
    - Degradatie: V(V) oxideert de zijketen; levensduur daalt exponentieel
      met temperatuur en wordt geschaald door de strategie:
        dubbel nafion: factor 1.6 (achterlaag beschermt/overneemt)
        teflon-rug (PTFE-mesh): factor 2.0 mechanisch + vervanging kan
        tussentijds met korte stilstand (downtime 2 h vs 12 h).
    """

    def __init__(self, thickness_m=178.0e-6, sigma_s_m=2.0,
                 v_permeability_m2_s=5.6e-11, life_ref_h=12000.0,
                 t_ref_c=25.0, layers=1, teflon_backing=False):
        if layers not in (1, 2):
            raise ValueError("layers must be 1 or 2 (dubbel nafion)")
        self.delta = thickness_m
        self.sigma = sigma_s_m
        self.p_v = v_permeability_m2_s
        self.life_ref = life_ref_h
        self.t_ref = t_ref_c
        self.layers = layers
        self.teflon = teflon_backing

    @property
    def area_resistance_ohm_m2(self):
        """i*R-term per m2 membraan (serieweerstand van de lagen)."""
        asr = self.layers * self.delta / self.sigma
        return asr * (1.15 if self.teflon else 1.0)

    @property
    def v_permeability(self):
        """Effectieve V-doorslag (m2/s): lagen in serie halveren."""
        return self.p_v / self.layers

    def lifetime_h(self, t_c=40.0):
        """Verwachte levensduur (bedrijfsuren) bij bedrijfstemperatuur."""
        f_t = math.exp(-(t_c - self.t_ref) / 25.0)
        f_strat = 1.0
        if self.layers == 2:
            f_strat *= 1.6
        if self.teflon:
            f_strat *= 2.0
        return self.life_ref * f_t * f_strat

    @property
    def replacement_downtime_h(self):
        """Stilstand per vervanging: teflon-rug = tussentijds wisselbaar."""
        return 2.0 if self.teflon else 12.0


# ---------------------------------------------------------------------------
# Electrolyse (Faraday + celspanning)
# ---------------------------------------------------------------------------
class ElectrolysisCell:
    """Electrolytisch terugschakelen van V-species (of algemeen M/z).

    product = i * A * M * eta_F / (z F)      (kg/s)
    U       = E_eq + eta_over + i * ASR      (V; ASR van het membraan)
    energie = U / (M eta_F / (z F))          -> J/kg, hier kWh/kg
    """

    def __init__(self, area_m2=1.0, i_a_m2=400.0, molar_mass_kg=0.05094,
                 z=1, e_eq_v=1.00, overpot_v=0.25, eta_faraday=0.95,
                 membrane=None):
        self.area = area_m2
        self.i = i_a_m2
        self.M = molar_mass_kg
        self.z = z
        self.e_eq = e_eq_v
        self.over = overpot_v
        self.eta_f = eta_faraday
        self.membrane = membrane or Nafion117()

    @property
    def production_kg_s(self):
        return self.i * self.area * self.M * self.eta_f / (self.z * F_CONST)

    @property
    def cell_voltage_v(self):
        return self.e_eq + self.over + self.i * self.membrane.area_resistance_ohm_m2

    @property
    def power_w(self):
        return self.cell_voltage_v * self.i * self.area

    @property
    def energy_kwh_per_kg(self):
        j_per_kg = self.power_w / self.production_kg_s
        return j_per_kg / 3.6e6

    def run(self, hours):
        """kg product en kWh over een bedrijfsperiode."""
        kg = self.production_kg_s * hours * 3600.0
        kwh = self.power_w * hours
        return kg, kwh


# ---------------------------------------------------------------------------
# Electrodialyse (zuur-/zout-terugwinning)
# ---------------------------------------------------------------------------
class ElectrodialysisStack:
    """ED-stapel met n celparen: verwijdert ionen evenredig met lading.

    mol verwijderd = eta * I * n_paren * t / (z F)
    Stapelspanning = n * (i*ASR_paar + E_diff) ; i Limiet-wachter: de
    aanname i <= i_lim wordt bewaakt (waarschuwing via .limiting_flag).
    """

    def __init__(self, n_pairs=40, area_m2=0.1, i_a_m2=100.0,
                 asr_pair_ohm_m2=0.03, e_diff_v=0.10, eta_current=0.90,
                 i_lim_a_m2=150.0, z=1):
        self.n = n_pairs
        self.area = area_m2
        self.i = i_a_m2
        self.asr = asr_pair_ohm_m2
        self.e_diff = e_diff_v
        self.eta = eta_current
        self.i_lim = i_lim_a_m2
        self.z = z
        self.limiting_flag = self.i > self.i_lim

    @property
    def stack_voltage_v(self):
        return self.n * (self.i * self.asr + self.e_diff)

    def removal_mol_s(self):
        """Ion-verwijdering (mol/s) bij de ingestelde stroom."""
        return self.eta * self.i * self.area * self.n / (self.z * F_CONST)

    def treat(self, volume_m3, c_in_mol_m3, c_out_mol_m3):
        """Behandel tijd (s) en kWh om V m3 van c_in naar c_out te brengen."""
        d_mol = max(c_in_mol_m3 - c_out_mol_m3, 0.0) * volume_m3
        i_total = self.i * self.area
        t_s = d_mol * self.z * F_CONST / (self.eta * i_total * self.n)
        kwh = self.stack_voltage_v * i_total * t_s / 3.6e6
        return t_s, kwh, kwh / max(volume_m3, 1e-9)


# ---------------------------------------------------------------------------
# Opschaling naar tonnen/dag en membraanvervangplan
# ---------------------------------------------------------------------------
class ProductionScale:
    """Van labbatch naar tonnen/dag, met membraanvervangplan.

    De membraan-afweging is de kern: enkele vs DUBBELE nafion vs
    teflon-rug. Dubbel: hogere kWh/kg (i*R x2), langere levensduur,
    halve doorslag. Teflon: kleine ASR-penalty, langste levensduur,
    en tussentijds vervangbaar in 2 h in plaats van 12 h.
    """

    def __init__(self, kg_per_batch, batch_hours, lines=1, hours_per_day=24.0,
                 operating_days=300, membrane=None, membrane_area_m2=1.0):
        self.kg = kg_per_batch
        self.h = batch_hours
        self.lines = lines
        self.hpd = hours_per_day
        self.days = operating_days
        self.membrane = membrane or Nafion117()
        self.area = membrane_area_m2

    @property
    def tonnes_per_day(self):
        return self.kg / self.h * self.hpd * self.lines / 1000.0

    @property
    def operating_hours_year(self):
        return self.hpd * self.days

    def membrane_plan(self):
        """(vervangingen/jaar, downtime h/jaar, membraankosten EUR/jaar,
        beschikbaarheid 0-1) voor de huidige membraanstrategie."""
        life = max(self.membrane.lifetime_h(), 1.0)
        replacements = self.operating_hours_year / life
        downtime = replacements * self.membrane.replacement_downtime_h
        # dubbel nafion: beide lagen vervangen bij plan-stop
        cost_per_swap = self.area * NAFION_EUR_PER_M2 * self.membrane.layers
        avail = 1.0 - downtime / max(self.operating_hours_year, 1.0)
        return replacements, downtime, cost_per_swap * replacements, avail


# ---------------------------------------------------------------------------
# Voorbeeldcases met staalslak
# ---------------------------------------------------------------------------
def steel_slag_cases():
    """Drie rekenbare voorbeeldcases; geeft lijst van dicts met sleutels.

    1) V2O5-terugwinning uit 1 t BOF-slak: malen -> vacuumfilter ->
       electrolyse naar V-electrolyt (VRFB-kwaliteit), incl. energie en
       omzet.
    2) 5 L labbatch: centrifugedroging van het ultrasound-slibresidu.
    3) Opschaling naar 5 t slak/dag: membraanvervangplan voor enkel,
       dubbel en teflon.
    """
    cases = []

    # -- case 1: 1 t BOF-slak -> V2O5 -> VRFB-electrolyt -------------------
    slag_kg = 1000.0
    v2o5_kg = slag_kg * V2O5_WT_PCT_BOF / 100.0
    slurp_m3 = 5.0                                  # 200 kg/m3 vaste stof
    vf = VacuumFiltration(area_m2=25.0, c_solids_kg_m3=200.0)  # kamerfilterplaat
    t_filt = vf.time_for_filtrate(slurp_m3)
    v_moist = vf.residual_moisture() * v2o5_kg      # vastgehouden vocht
    cell = ElectrolysisCell(area_m2=1.0,
                            molar_mass_kg=0.05094, z=1,
                            membrane=Nafion117())
    v_kg = v2o5_kg * 0.440                          # V-gehalte van V2O5
    kg, kwh = cell.run(hours=v_kg / cell.production_kg_s / 3600.0)
    # 1.6 M VRFB-electrolyt: liters die je van dit vanadium maakt
    el_l = (v_kg / 0.05094) / 1600.0 * 1000.0
    cases.append({
        "naam": "V2O5-terugwinning uit 1 t BOF-slak",
        "v2o5_kg": v2o5_kg,
        "filtratie_s": t_filt,
        "koek_kg": vf.cake_mass(slurp_m3),
        "vocht_in_koek_kg": v_moist,
        "v_kg": v_kg,
        "electrolyse_kwh": kwh,
        "kwh_per_kg_v": cell.energy_kwh_per_kg,
        "vrfb_electrolyt_l": el_l,
        "opslag_kwh": el_l * 5.0 * 0.7,             # 25 kWh/L tot 70% DoD
        "omzet_eur": v2o5_kg * V2O5_EUR_PER_KG,
    })

    # -- case 2: 5 L labbatch centrifugedroging ----------------------------
    cf = BasketCentrifuge(volume_l=5.0, rpm=3000.0)
    solids_kg = 5.0e-3 * 300.0                      # 5 L @ 300 kg/m3
    m_start = 0.55                                  # vacuumfilter-koek
    m_droog = cf.residual_moisture(m0=m_start, spin_s=600.0)
    cases.append({
        "naam": "5 L batch centrifugedroging (3000 rpm)",
        "g_force": cf.g_force(),
        "bed_tijd_s": cf.settling_time_s(),
        "cyclus_s": cf.batch_cycle_s(),
        "solids_kg": solids_kg,
        "vocht_start": m_start,
        "vocht_na_10min": m_droog,
        "water_eruit_kg": solids_kg * (m_start - m_droog),
    })

    # -- case 3: opschaling naar 5 t slak/dag, membraanplan ----------------
    #   5 t/dag slak -> 400 kg V2O5/dag -> V-electrolyse over 12 h-lijnen
    v2o5_day = 5000.0 * V2O5_WT_PCT_BOF / 100.0
    v_day = v2o5_day * 0.440
    cell3 = ElectrolysisCell(area_m2=2.5, membrane=Nafion117())
    lines = v_day / (cell3.production_kg_s * 24 * 3600.0)
    n_lines = max(1, math.ceil(lines))
    strategies = {}
    for naam, mem in (("enkel", Nafion117()),
                      ("dubbel", Nafion117(layers=2)),
                      ("teflon", Nafion117(teflon_backing=True))):
        scale = ProductionScale(kg_per_batch=v_day / n_lines,
                                batch_hours=24.0,
                                lines=n_lines,
                                membrane=mem, membrane_area_m2=2.5)
        rep, dt_h, cost, avail = scale.membrane_plan()
        cell_m = ElectrolysisCell(area_m2=2.5, membrane=mem)
        strategies[naam] = {
            "levensduur_h": mem.lifetime_h(40.0),
            "vervangingen_jaar": rep,
            "downtime_h_jaar": dt_h,
            "membraankosten_eur_jaar": cost,
            "beschikbaarheid": avail,
            "kwh_per_kg_v": cell_m.energy_kwh_per_kg,
            "doorslag_relatief": mem.v_permeability / Nafion117().v_permeability,
        }
    cases.append({
        "naam": "Opschaling 5 t slak/dag (membraanstrategieen)",
        "v2o5_kg_dag": v2o5_day,
        "electrolyse_lijnen": max(1, math.ceil(lines)),
        "strategieen": strategies,
    })
    return cases


def print_cases():
    """Leesbare print van steel_slag_cases() (demo Zonder GUI)."""
    for c in steel_slag_cases():
        print(f"== {c['naam']}")
        for k, v in c.items():
            if k == "naam":
                continue
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv:.3g}" if isinstance(vv, float)
                          else f"    {kk}: {vv}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.3g}")
            else:
                print(f"  {k}: {v}")
    return 0
