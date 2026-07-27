"""MOLGANG river-flow lab - physics core (headless, numpy only).

Kreken (creeks) that flow down from the mountains into the river at each
steel-plant site, with flow dynamics and viscosity visible IN the river
itself, carrying raw material (ore-bearing sediment) downstream for later
mining. Extends simulation/viscosity_lab rather than duplicating it:
SlurryRheology (Krieger-Dougherty viscosity, Richardson-Zaki hindered
settling) is imported and reused unchanged for the suspended-sediment
physics — a river is a much more dilute, open, one-directional flow than
the mixing tank, so it gets its own channel-routing solver, but the same
rheology authority.

Numerics: 1D depth-averaged kinematic-wave channel routing (explicit,
CFL-substepped upwind finite volume) -- the standard, textbook-correct
simplification for reach-scale river flow (Saint-Venant with the local
acceleration terms dropped; normal-flow closure via Manning's equation).
A full 2D Stam solver (as used for the mixing tank) is the right tool for
a stirred tank; it is the wrong tool for a many-kilometre river reach
where the flow is overwhelmingly one-directional -- this is a deliberate,
documented choice, not a shortcut.

Components map onto the planned gameplay:
  RiverChannel        <-> the river geometry each steelworks site already
                           carries (rivers[].pts, OSM), now flowing
  RiverChannel.add_creek <-> a kreek entering from the mountains
  RiverChannel.step    <-> advances flow depth, velocity and sediment
  RawMaterialSupply    <-> the mining hand-off point (raw ore accumulates
                           where the flow slows -- a real placer-deposit
                           mechanism, not a scripted spawn)

All comments/units SI unless noted. No pandas, no GUI imports here.
"""

from __future__ import annotations

import math
import numpy as np

from viscosity_core import SlurryRheology, GAMMA_MIN, GRAVITY, RHO_WATER

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO_ORE = 4500.0          # kg/m3, generic ore-bearing sediment (denser than
                           # slag 3400, lighter than native gold 19300 -- a
                           # placer deposit is usually a MIX of the two)
RHO_GOLD = 19300.0        # kg/m3, native gold (rivierlab goudpannen precedent)
MANNING_N = 0.035         # s/m^(1/3), natural winding stream (textbook value)
MIN_DEPTH = 0.02          # m, numerical floor (never fully dry -> no div/0)
KINEMATIC_CELERITY_FACTOR = 5.0 / 3.0   # dQ/dh / u for a wide Manning channel
SHEAR_DEPTH_FACTOR = 1.0  # gamma_dot ~= u / (SHEAR_DEPTH_FACTOR * h)
VISCOUS_DRAG_EXPONENT = 0.15   # thicker (higher eta_app) flow -> more drag
SETTLING_KM = 0.35        # m/s, velocity scale below which ore settles fast


def _manning_velocity(h, slope, n_eff):
    """Manning's equation for a wide rectangular channel (R_h ~= h)."""
    h = np.maximum(h, MIN_DEPTH)
    return (1.0 / np.maximum(n_eff, 1e-6)) * h ** (2.0 / 3.0) * math.sqrt(max(slope, 1e-6))


class RiverChannel:
    """A river reach discretised along its (already-mapped) OSM polyline.

    pts_m: list of (x, y) in local metres, e.g. a site's rivers[i]['pts']
    from the steelworks OSM dataset (tools/build_steel_sites.py). width_m
    matches the render width already used for the ribbon in
    molgang-roblox world.js (river=11 m, canal=6 m).
    """

    def __init__(self, pts_m, width_m=11.0, slope=0.0015, n_cells=None,
                 rheo=None):
        pts = np.asarray(pts_m, dtype=np.float64)
        if len(pts) < 2:
            raise ValueError("river needs at least 2 points")
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length_m = float(s[-1])
        if self.length_m <= 0:
            raise ValueError("river has zero length")
        n = n_cells or max(8, min(400, int(self.length_m / 20.0)))
        self.n = n
        self.ds = self.length_m / n
        cell_s = (np.arange(n) + 0.5) * self.ds
        self.x = np.interp(cell_s, s, pts[:, 0])
        self.y = np.interp(cell_s, s, pts[:, 1])
        self.width = np.full(n, width_m)
        self.slope = slope
        self.rheo = rheo or SlurryRheology(rho_solid=RHO_ORE)
        self.h = np.full(n, MIN_DEPTH * 3)     # depth, m
        self.phi = np.zeros(n)                 # suspended sediment vol frac
        self.deposit_kg_per_m = np.zeros(n)    # settled raw material
        self._creeks = []                      # [(cell_index, Q, phi)]
        self.time_s = 0.0

    # -- creeks --------------------------------------------------------
    def add_creek(self, Q_m3s, phi_sediment, at_fraction=0.0):
        """A kreek joining the river; at_fraction=0 is the upstream end
        (closest to the mountains), 1 is the mouth/downstream end."""
        idx = int(np.clip(at_fraction, 0.0, 0.999) * self.n)
        self._creeks.append([idx, Q_m3s, phi_sediment])
        return idx

    def set_creek_flow(self, creek_id, Q_m3s, phi_sediment=None):
        c = self._creeks[creek_id]
        c[1] = Q_m3s
        if phi_sediment is not None:
            c[2] = phi_sediment

    # -- physics ---------------------------------------------------------
    def _velocities(self):
        eta_app = np.array([self.rheo.apparent_viscosity(p, max(
            self._shear(i), GAMMA_MIN)) for i, p in enumerate(self.phi)])
        # Thicker suspension -> effectively rougher/more resistive flow.
        n_eff = MANNING_N * (eta_app / self.rheo.eta_liquid) ** VISCOUS_DRAG_EXPONENT
        u = _manning_velocity(self.h, self.slope, n_eff)
        return u, eta_app

    def _shear(self, i):
        h = max(self.h[i], MIN_DEPTH)
        u = _manning_velocity(np.array([h]), self.slope, MANNING_N)[0]
        return u / (SHEAR_DEPTH_FACTOR * h)

    def _cfl_dt(self, u, dt_wanted):
        celerity = np.maximum(KINEMATIC_CELERITY_FACTOR * np.abs(u), 1e-6)
        dt_max = 0.9 * self.ds / celerity.max()
        return min(dt_wanted, dt_max)

    def step(self, dt):
        remaining = dt
        while remaining > 1e-9:
            u, eta_app = self._velocities()
            sub_dt = self._cfl_dt(u, remaining)
            Q = self.width * self.h * u                        # m3/s per cell
            phi_Q = Q * self.phi                                # sediment flux

            for idx, Qc, phic in self._creeks:
                Q[idx] += Qc
                phi_Q[idx] += Qc * phic

            Q_up = np.concatenate([[Q[0]], Q[:-1]])             # upwind (from upstream)
            phiQ_up = np.concatenate([[phi_Q[0]], phi_Q[:-1]])

            vol_old = np.maximum(self.width * self.h * self.ds, 1e-9)
            sed_vol_old = self.phi * vol_old                    # m3 of sediment in the cell

            dh = (Q_up - Q) / (self.width * self.ds) * sub_dt
            self.h = np.maximum(self.h + dh, MIN_DEPTH)
            vol_new = np.maximum(self.width * self.h * self.ds, 1e-9)

            sed_vol_new = sed_vol_old + (phiQ_up - phi_Q) * sub_dt   # m3/s * s = m3
            self.phi = np.maximum(0.0, sed_vol_new / vol_new)

            # Hindered settling (Richardson-Zaki via SlurryRheology) removes
            # sediment fastest where the flow has slowed -- a real placer
            # mechanism: ore drops out of suspension where velocity drops.
            v_settle = np.array([self.rheo.settling_velocity(p) for p in self.phi])
            # Smooth (never exactly zero) roll-off with velocity: fast flow
            # resuspends fines and suppresses settling, but a real river
            # always deposits SOME material in the viscous sublayer near the
            # bed, even at speed -- a hard cutoff would let a fed channel
            # permanently "freeze" once it reaches a fast steady state,
            # which would break the ongoing creek->supply gameplay loop.
            slow = np.exp(-u / SETTLING_KM)
            drop_frac = np.clip(v_settle / max(MIN_DEPTH, 1e-3) * slow * sub_dt, 0.0, 0.9)
            dropped = self.phi * drop_frac
            self.phi -= dropped
            self.deposit_kg_per_m += dropped * self.rheo.rho_solid * self.width * self.h

            remaining -= sub_dt
            self.time_s += sub_dt
        self._last_u = u
        self._last_eta = eta_app

    # -- reporting ---------------------------------------------------------
    def snapshot(self):
        u, eta_app = self._velocities()
        return {
            "time_s": self.time_s,
            "length_m": self.length_m,
            "n_cells": self.n,
            "mean_depth_m": float(self.h.mean()),
            "mean_velocity_ms": float(u.mean()),
            "max_velocity_ms": float(u.max()),
            "mean_phi_pct": float(self.phi.mean() * 100),
            "max_phi_pct": float(self.phi.max() * 100),
            "mean_eta_app_pa_s": float(eta_app.mean()),
            "total_flow_m3s": float((self.width * self.h * u).mean()),
            "total_deposit_kg": float((self.deposit_kg_per_m * self.ds).sum()),
        }

    def deposit_at(self, at_fraction):
        idx = int(np.clip(at_fraction, 0.0, 0.999) * self.n)
        return float(self.deposit_kg_per_m[idx] * self.ds)


class RawMaterialSupply:
    """The mining hand-off: raw ore that settled out of the river at a
    named collection point, ready to be converted into the SAME MolCoin
    economy the rest of the game uses (no separate currency -- mirrors
    how V2O5 sales and crop harvests already pay through earn()).
    """

    def __init__(self, channel: RiverChannel, at_fraction=0.5,
                 eur_per_kg=8.0):
        self.channel = channel
        self.at_fraction = at_fraction
        self.eur_per_kg = eur_per_kg
        self.collected_kg = 0.0

    def available_kg(self):
        return max(0.0, self.channel.deposit_at(self.at_fraction) - self.collected_kg)

    def collect(self, max_kg=None):
        avail = self.available_kg()
        take = avail if max_kg is None else min(avail, max_kg)
        self.collected_kg += take
        return {"kg": take, "payout": round(take * self.eur_per_kg)}
