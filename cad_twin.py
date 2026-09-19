"""cad_twin -- digital-twin bridge for Astra equipment (thermal twin + CAD).

Twee helften, bewust dependency-arm (README-doctrine: pure numpy/stdlib):

1. ThermalTwin  -- gereduceerd-orde transiënt warmtemodel van de cilindrische
   roerbak uit viscosity_core: axisymmetrische 2D-FDM geleiding in de slurry
   (anisotroop rooster), wand als per-z gepaarde weerstandsketen
   (wandgeleiding + buitenconvectie, of jacket-koeling over de mantelhoogte),
   warmtebronnen op de fysisch juiste plek: de impeller-sweep en de
   28/40 kHz-transducentring. Dit is de thermodynamica-basis voor de
   digital twin: inzichtklasse van een SolidWorks-simulatie, maar als
   bewijsbare pure-Python autoriteit.

2. export_step  -- OPTIONELE FreeCAD-brug: bouwt vanuit een EquipmentSpec een
   parametrisch 3D-model (bakwand + jacket + roeras + 2-blads impeller +
   transducentring) en schrijft STEP -- het uitwisselingsformaat dat
   SolidWorks native opent. FreeCAD is níet nodig voor het thermische deel;
   zonder FreeCAD volgt een duidelijke RuntimeError met installatiehint.

Run: python3 test_cad_twin.py  (proof-suite, exit 0 = pass)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from viscosity_core import SlurryRheology

# --- materiaalconstanten (SS304-wand, slurry-mengsel) -----------------------
K_STEEL = 16.2          # W/m/K, AISI 304 bij 20-100 °C
RHO_STEEL = 7900.0      # kg/m3
CP_STEEL = 500.0        # J/kg/K
CP_WATER = 4180.0       # J/kg/K
CP_SLAG = 800.0         # J/kg/K
K_SLURRY_BASE = 0.62    # W/m/K, stilstaand water + fijne deeltjes
STIRRED_K_FACTOR = 2.6  # effectieve geleiding onder roeren (dispersie)
H_STIRRED = 350.0       # W/m2/K, binnenwand-convectie tijdens roeren
H_STATIC = 60.0         # W/m2/K, binnenwand-convectie bij stilstaan
H_EXT = 12.0            # W/m2/K, vrije convectie naar werkplaatslucht


@dataclass
class EquipmentSpec:
    """Parametrische definitie van de Astra-roerinstallatie (SI-eenheden)."""

    tank_radius_i: float = 0.20      # binnenstraal bak (m)
    wall_t: float = 0.004            # wanddikte (m)
    tank_height: float = 0.45        # totale bakhoogte (m)
    fill_frac: float = 0.72          # vulgraad slurry (fractie van hoogte)
    w_solids: float = 0.35           # vaste-stof massafractie (0-1)

    jacket: bool = True              # koelmantel aan/uit
    jacket_frac: float = 0.80        # mantelhoogte (fractie van vulling)
    t_coolant: float = 15.0          # coolant-aanvoertemperatuur (°C)
    h_jacket: float = 450.0          # mantelzijde-convectie (W/m2/K)

    stirrer_power: float = 220.0     # schachtvermogen in steady regime (W)
    impeller_z_frac: float = 0.25    # impellerhoogte (fractie van vulling)
    us_power: float = 900.0          # elektrisch vermogen transducentring (W)
    us_heat_frac: float = 0.80       # fractie van het vermogen -> slurrywarmte
    us_z_frac: float = 0.60          # ringhoogte (fractie van vulling)

    t_ambient: float = 21.0          # werkplaatslucht (°C)
    t_init: float = 21.0             # starttemperatuur slurry (°C)

    def __post_init__(self):
        if not 0.0 < self.fill_frac <= 1.0:
            raise ValueError("fill_frac must be in (0, 1]")
        if self.wall_t <= 0 or self.tank_radius_i <= 0:
            raise ValueError("tank_radius_i and wall_t must be positive")
        if not 0.0 <= self.w_solids < 1.0:
            raise ValueError("w_solids must be in [0, 1)")
        if not 0.0 <= self.us_heat_frac <= 1.0:
            raise ValueError("us_heat_frac must be in [0, 1]")

    @property
    def radius_o(self) -> float:
        return self.tank_radius_i + self.wall_t

    @property
    def h_liq(self) -> float:
        return self.fill_frac * self.tank_height


class ThermalTwin:
    """Transient axisymmetric thermal twin of the stirred slurry tank.

    The slurry is a 2D (r, z) conduction field on an anisotropic grid with a
    stirred effective conductivity; the wall is one lumped node per z-row
    coupled through the cylinder-wall resistance to lab air -- or to jacket
    coolant over the jacket span. Heat enters where it physically enters (the
    impeller sweep and the transducer ring), never as a global multiplier.
    """

    def __init__(self, spec: EquipmentSpec, n_r: int = 40, n_z: int = 60,
                 stir_on: bool = True):
        self.spec = spec
        self.n_r, self.n_z = n_r, n_z
        self.stir_on = stir_on
        sp = self.spec
        self.r = np.linspace(0.0, sp.tank_radius_i, n_r)
        self.z = np.linspace(0.0, sp.h_liq, n_z)
        self.dr = self.r[1] - self.r[0]
        self.dz = self.z[1] - self.z[0]

        rheo = SlurryRheology()
        self.rho = rheo.rho_mix(sp.w_solids)
        self.cp = (1 - sp.w_solids) * CP_WATER + sp.w_solids * CP_SLAG
        self.k_eff = K_SLURRY_BASE * (STIRRED_K_FACTOR if stir_on else 1.0)
        self.alpha = self.k_eff / (self.rho * self.cp)
        self.h_inner = H_STIRRED if stir_on else H_STATIC

        self.T = np.full((n_r, n_z), float(sp.t_init))
        self.T_wall = np.full(n_z, float(sp.t_init))
        self.time = 0.0
        self.q_in_j = 0.0     # cumulatieve energie de slurry in (J)
        self.q_out_j = 0.0    # cumulatieve energie via de wand eruit (J)

        self.cell_v = self._cell_volumes()                         # (n_r, n_z) m3
        self.q_stir = self._source_mask(sp.impeller_z_frac, 0.35, 0.30)
        self.q_us = self._source_mask(sp.us_z_frac, 0.94, 0.12)

    # -- geometrie/hulpgrootheden ------------------------------------------------
    @property
    def volume(self) -> float:
        return math.pi * self.spec.tank_radius_i ** 2 * self.h_liq

    @property
    def bulk_temp(self) -> float:
        return float(np.mean(self.T))

    def _cell_volumes(self) -> np.ndarray:
        """Axisymmetric cell volume incl. the half-cell at the axis (m3)."""
        r_in = np.maximum(self.r - self.dr / 2, 0.0)
        r_out = np.minimum(self.r + self.dr / 2, self.spec.tank_radius_i)
        return (math.pi * (r_out ** 2 - r_in ** 2) * self.dz)[:, None] \
            * np.ones((1, self.n_z))

    def _source_mask(self, z_frac: float, r_frac: float, spread_cells: float) -> np.ndarray:
        """Cell weights normalised so mask.sum() == 1: multiplying by total
        watts injects exactly that power. The per-cell volume weighting happens
        in step() via p_cell = P*mask/(rho*cp*cell_v)."""
        z0 = z_frac * self.spec.h_liq
        rz = r_frac * self.spec.tank_radius_i
        gr = np.exp(-((self.r[:, None] - rz) ** 2) / (2 * (spread_cells * self.dr) ** 2))
        gz = np.exp(-((self.z[None, :] - z0) ** 2) / (2 * (spread_cells * self.dz) ** 2))
        m = gr * gz
        tot = m.sum()
        return m / tot if tot > 0 else m

    def _wall_resistance(self) -> np.ndarray:
        """Per-z outward resistance (K/W): cylinder-wall conduction in series
        with the outer film -- jacket coolant over the jacket span, lab air
        elsewhere."""
        sp = self.spec
        r_wall = math.log(sp.radius_o / sp.tank_radius_i) \
            / (2 * math.pi * K_STEEL * self.dz)
        h_out = np.array([sp.h_jacket if (sp.jacket and zc <= sp.jacket_frac * sp.h_liq)
                          else H_EXT for zc in self.z])
        return r_wall + 1.0 / (h_out * 2 * math.pi * sp.radius_o * self.dz)

    def _t_outside(self) -> np.ndarray:
        sp = self.spec
        return np.array([sp.t_coolant if (sp.jacket and zc <= sp.jacket_frac * sp.h_liq)
                         else sp.t_ambient for zc in self.z])

    # -- integratie ---------------------------------------------------------------
    def step(self, dt: float, stirrer_w: float | None = None,
             us_w: float | None = None):
        """Advance dt wall-clock seconds (auto-substepped for explicit stability).

        The explicit scheme is bounded at Fo <= 0.22; large dt requests are
        split, so callers can drive the twin at GUI frame rate safely.
        """
        dt_sub = 0.22 * min(self.dr, self.dz) ** 2 / self.alpha
        n_sub = max(1, int(math.ceil(dt / dt_sub)))
        h = dt / n_sub
        sp = self.spec
        q_stir_w = sp.stirrer_power if (self.stir_on and stirrer_w is None) else (stirrer_w or 0.0)
        q_us_w = sp.us_power * sp.us_heat_frac if us_w is None else us_w * sp.us_heat_frac
        p_cell = (q_stir_w * self.q_stir + q_us_w * self.q_us) \
            / (self.rho * self.cp * self.cell_v)                   # K/s per cel

        r_chain = self._wall_resistance()
        t_out = self._t_outside()
        c_wall = (RHO_STEEL * CP_STEEL * math.pi
                  * (sp.radius_o ** 2 - sp.tank_radius_i ** 2) * self.dz)
        h_in_area = self.h_inner * 2 * math.pi * sp.tank_radius_i * self.dz

        for _ in range(n_sub):
            T = self.T
            lap = np.zeros_like(T)
            lap[1:-1, :] = ((T[2:, :] - 2 * T[1:-1, :] + T[:-2, :]) / self.dr ** 2
                            + (T[2:, :] - T[:-2, :]) / (2 * self.r[1:-1, None] * self.dr))
            lap[0, :] = 2.0 * (T[1, :] - T[0, :]) / self.dr ** 2    # symmetrie as
            lap[-1, :] = 2.0 * (T[-2, :] - T[-1, :]) / self.dr ** 2  # halfcel wand
            lap[:, 1:-1] += (T[:, 2:] - 2 * T[:, 1:-1] + T[:, :-2]) / self.dz ** 2
            lap[:, 0] += 2.0 * (T[:, 1] - T[:, 0]) / self.dz ** 2    # bodem adiabat
            lap[:, -1] += 2.0 * (T[:, -2] - T[:, -1]) / self.dz ** 2  # vloeispiegel adiabat

            q_wall = h_in_area * (T[-1, :] - self.T_wall)          # W per z-rij
            dT = self.alpha * lap + p_cell
            dT[-1, :] -= q_wall / (self.rho * self.cp
                                   * math.pi * (sp.tank_radius_i ** 2 - (sp.tank_radius_i - self.dr / 2) ** 2) * self.dz)
            q_out = (self.T_wall - t_out) / r_chain                # W per z-rij
            dT_wall = (q_wall - q_out) / c_wall

            self.T = T + h * dT
            self.T_wall = self.T_wall + h * dT_wall
            self.q_in_j += (q_stir_w + q_us_w) * h
            self.q_out_j += float(np.sum(q_out)) * h
        self.time += dt

    def run(self, t_end: float, dt: float = 1.0, sample_every: int = 10):
        """Run to t_end seconds; returns (times, bulk_temps, wall_temps)."""
        times, bulks, walls = [], [], []
        n = int(math.ceil(t_end / dt))
        for k in range(n):
            self.step(dt)
            if k % sample_every == 0 or k == n - 1:
                times.append(self.time)
                bulks.append(self.bulk_temp)
                walls.append(float(np.mean(self.T_wall)))
        return times, bulks, walls

    def energy_balance_residual(self) -> float:
        """|q_in - q_out - dU| / max(q_in, 1 J): 0 = perfect bookkeeping."""
        sp = self.spec
        du_slurry = self.rho * self.cp * float(np.sum((self.T - sp.t_init) * self.cell_v))
        du_wall = (RHO_STEEL * CP_STEEL * math.pi
                   * (sp.radius_o ** 2 - sp.tank_radius_i ** 2) * self.dz
                   * float(np.sum(self.T_wall - sp.t_init)))
        total_in = max(self.q_in_j, 1.0)
        return abs(self.q_in_j - self.q_out_j - du_slurry - du_wall) / total_in


def power_from_tank(tank) -> float:
    """Wall-socket power (W) of a viscosity_core.MixingTank right now: shaft
    power over the drive efficiency plus standby -- i.e. what actually heats
    the shed. Falls back to the meter reading when the stirrer is not exposed."""
    stirrer = getattr(tank, "stirrer", None)
    shaft = getattr(stirrer, "p_shaft", None)
    if shaft is not None:
        return shaft / 0.65 + 5.0
    meter = getattr(tank, "meter", None)
    kw = getattr(meter, "kw_now", None)
    return float(kw) * 1000.0 if kw else 0.0


def export_step(path: str, spec: EquipmentSpec | None = None):
    """Build the parametric Astra rig in FreeCAD and write STEP.

    Requires FreeCAD 0.21+/1.x -- run inside freecadcmd (see
    demo_cad_twin_step.py). Raises RuntimeError with an install hint when
    FreeCAD is absent; the thermal twin never needs it.
    """
    try:
        import FreeCAD  # noqa: F401
        import Part
    except ImportError as exc:
        raise RuntimeError(
            "FreeCAD is required for STEP export (the thermal twin runs "
            "without it). Install FreeCAD 1.x and run inside freecadcmd, e.g. "
            "freecadcmd demo_cad_twin_step.py") from exc

    spec = spec or EquipmentSpec()
    sp = spec
    Ri, Ro, H = sp.tank_radius_i, sp.radius_o, sp.tank_height

    def cyl(r, h, z0=0.0):
        return Part.makeCylinder(r, h, FreeCAD.Vector(0, 0, z0))

    parts = []
    tank_wall = cyl(Ro, H).cut(cyl(Ri, H + 0.01))
    parts.append(tank_wall)
    parts.append(cyl(Ri - 0.002, sp.h_liq))                # slurry-referentielichaam
    parts.append(cyl(0.012, sp.h_liq + 0.05, sp.h_liq * 0.10))  # roeras
    impeller_z = sp.impeller_z_frac * sp.h_liq
    for sign in (-1, 1):
        blade = Part.makeBox(0.09, 0.012, 0.004,
                             FreeCAD.Vector(-0.045 if sign > 0 else -0.045,
                                            -0.006, impeller_z))
        parts.append(blade)
    for i in range(8):                                     # 28/40 kHz-ring
        a = math.radians(45 * i)
        tr = Part.makeBox(0.030, 0.030, 0.055,
                          FreeCAD.Vector((Ro + 0.004) * math.cos(a) - 0.015,
                                         (Ro + 0.004) * math.sin(a) - 0.015,
                                         sp.us_z_frac * sp.h_liq))
        tr.rotate(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 0, 1), 45.0 * i)
        parts.append(tr)
    if sp.jacket:
        jh = sp.jacket_frac * sp.h_liq
        parts.append(cyl(Ro + 0.025, jh).cut(cyl(Ro + 0.004, jh)))

    compound = Part.makeCompound(parts)
    compound.exportStep(path)
    return path
