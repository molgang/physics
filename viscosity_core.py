"""MOLGANG viscosity lab - physics core (headless, numpy only).

Simulates stirring of a steel-slag slurry (5 micron particles, 1-100 wt%
solids) in a cylindrical tank, viewed top-down on a 2D grid.

Components map 1:1 onto the planned AR "viscosity room" (Meta Quest 3S):
  MixingTank      <-> grabbable/placeable tank ("bak")
  MixingTank.add_water   <-> hose ("slang")
  MixingTank.add_slag    <-> pouring from the slag container
  Stirrer         <-> stirring device, only runs when tank placed + plugged in
  PowerMeter      <-> wall-socket kW readout + kWh accumulator

Numerics: Jos Stam, "Real-Time Fluid Dynamics for Games" (GDC 2003).
Implicit (Gauss-Seidel) viscous diffusion is unconditionally stable, which
is what lets a single solver span water (1 mPa.s) to slag paste (>1e3 Pa.s).

Rheology: Krieger-Dougherty relative viscosity + a yield stress that grows
near maximum packing (Bingham-like), evaluated at the Metzner-Otto average
shear rate of the impeller. Settling follows Stokes with Richardson-Zaki
hindering; a yield stress suppresses settling (particles held in the gel).

All comments/units SI unless noted. No pandas, no GUI imports here.
"""

from __future__ import annotations

import copy
import math
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants and slurry defaults
# ---------------------------------------------------------------------------
RHO_WATER = 1000.0        # kg/m3
RHO_SLAG = 3400.0         # kg/m3, BOF/LD steel slag particle density
ETA_WATER = 1.0e-3        # Pa.s
PHI_MAX = 0.58            # max packing fraction, fine irregular particles
INTRINSIC_VISC = 2.5      # Einstein coefficient (spheres; slag ~2.5-4)
PHI_YIELD_ONSET = 0.20    # vol fraction where a yield stress appears
TAU0 = 3.0                # Pa, yield stress scale for 5 um slag fines
KS_METZNER_OTTO = 11.0    # average shear rate = ks * N (paddle/turbine)
D_PARTICLE = 5.0e-6       # m, particle diameter (5 micron)
GRAVITY = 9.81            # m/s2
GAMMA_MIN = 0.05          # 1/s, floor for apparent-viscosity evaluation
ETA_CAP = 1.0e6           # Pa.s numerical cap ("locked" paste)
PHI_PACK_BED = 0.62       # local packing cap for settled sediment
ELECTRICITY_EUR_PER_KWH = 0.30

# --- thermal coupling (0-D bulk energy balance) -----------------------------
T_REF_C = 20.0            # °C, reference of the rheology tables
EA_OVER_R = 2400.0        # K, Arrhenius slope of water viscosity (20-60 °C)
CP_WATER = 4180.0         # J/kg/K
CP_SLAG = 800.0           # J/kg/K
UA_TANK_W_PER_K = 2.3     # W/K, steel wall + free convection of the 0.4 m tank

# --- thixotropy (structural breakdown/rebuild, opt-in) ----------------------
C_THIX = 0.8              # max relative viscosity surplus at fully built gel
T_BUILD_S = 45.0          # s, structural rebuild time constant at rest
K_BREAK = 1.15e-3         # (1/s) per 1/s shear; lambda* ~ 0.15 at 300 rpm

# --- evaporation (open-bath water loss, opt-in, part of the 0-D balance) ----
RH_AIR = 0.5              # workshop air, relative humidity (-)
H_MASS = 0.007            # m/s, natural-convection mass transfer coefficient
R_VAPOR = 461.5           # J/kg/K, specific gas constant of water vapour
LATENT_VAP = 2.45e6       # J/kg, latent heat of vaporisation ~40 C

# --- bed erosion (re-suspension of the settled layer under shear) -----------
U_ERODE_CRIT = 0.08       # m/s, critical speed above which the bed erodes
K_ERODE = 2.0             # 1/(s m/s), erosion rate above the threshold


def rho_vapor_sat(t_c):
    """Saturated water vapour density (kg/m3) via the Magnus formula."""
    p = 610.94 * math.exp(17.625 * t_c / (t_c + 243.04))   # Pa
    return p / (R_VAPOR * (t_c + 273.15))


# ---------------------------------------------------------------------------
# Rheology of the slag/water mixture
# ---------------------------------------------------------------------------
class SlurryRheology:
    """Steel-slag slurry rheology as a function of solids mass fraction."""

    def __init__(self, rho_solid=RHO_SLAG, rho_liquid=RHO_WATER,
                 eta_liquid=ETA_WATER, phi_max=PHI_MAX,
                 intrinsic=INTRINSIC_VISC, phi_yield=PHI_YIELD_ONSET,
                 tau0=TAU0, structural_factor=1.0):
        self.rho_solid = rho_solid
        self.rho_liquid = rho_liquid
        self.eta_liquid = eta_liquid
        self.phi_max = phi_max
        self.intrinsic = intrinsic
        self.phi_yield = phi_yield
        self.tau0 = tau0
        # Multiplier on the full flow curve, set by at_conditions(): carries
        # the thixotropy surplus (and only that -- temperature scaling lives
        # in eta_liquid). 1.0 keeps every table exact.
        self.structural_factor = structural_factor

    def at_conditions(self, t_c, structural=1.0):
        """Copy of the rheology evaluated at slurry temperature t_c (°C).

        Liquid viscosity follows an Arrhenius law anchored at T_REF_C (factor
        1.0 exactly at 20 °C, ~0.6x at 40 °C); `structural` scales the whole
        flow curve (thixotropy surplus). Always condition from the base
        instance -- factors are not meant to compound.
        """
        out = copy.copy(self)
        # eta ~ exp(E_a/R * 1/T): warm (T > T_ref) => factor < 1, dunner.
        f = math.exp(EA_OVER_R * (1.0 / (t_c + 273.15)
                                  - 1.0 / (T_REF_C + 273.15)))
        out.eta_liquid = self.eta_liquid * f
        out.structural_factor = float(structural)
        return out

    def phi_from_w(self, w):
        """Solids volume fraction from solids mass fraction w in [0, 1]."""
        w = min(max(w, 0.0), 1.0)
        vs = w / self.rho_solid
        vl = (1.0 - w) / self.rho_liquid
        return vs / (vs + vl) if (vs + vl) > 0 else 0.0

    def rho_mix(self, w):
        """Mixture density (kg/m3) from solids mass fraction."""
        w = min(max(w, 0.0), 1.0)
        return 1.0 / (w / self.rho_solid + (1.0 - w) / self.rho_liquid)

    def eta_infinite(self, phi):
        """High-shear (Krieger-Dougherty) viscosity, Pa.s."""
        ratio = min(phi / self.phi_max, 0.9998)
        eta = self.eta_liquid * (1.0 - ratio) ** (-self.intrinsic * self.phi_max)
        return min(eta, ETA_CAP)

    def tau_yield(self, phi):
        """Yield stress (Pa); zero below onset, diverges near max packing."""
        if np.isscalar(phi):
            if phi <= self.phi_yield:
                return 0.0
            gap = max(self.phi_max - phi, 1.0e-4)
            return self.tau0 * ((phi - self.phi_yield) / gap) ** 2
        phi = np.asarray(phi, dtype=np.float64)
        gap = np.maximum(self.phi_max - phi, 1.0e-4)
        tau = self.tau0 * ((phi - self.phi_yield) / gap) ** 2
        return np.where(phi > self.phi_yield, tau, 0.0)

    def apparent_viscosity(self, phi, gamma_dot):
        """Bingham apparent viscosity at shear rate gamma_dot (Pa.s).

        The structural (thixotropy) factor scales the whole flow curve; the
        ETA_CAP still bounds the result for the solver.
        """
        g = max(gamma_dot, GAMMA_MIN)
        eta = (self.eta_infinite(phi) + self.tau_yield(phi) / g) \
            * self.structural_factor
        return min(eta, ETA_CAP)

    def settling_velocity(self, phi):
        """Hindered Stokes settling speed of 5 um slag in water (m/s).

        Scalar or array phi. Yield stress traps particles: above ~tau_hold
        the suspension is a weak gel and settling effectively stops.
        """
        v0 = ((self.rho_solid - self.rho_liquid) * GRAVITY * D_PARTICLE ** 2
              / (18.0 * self.eta_liquid))
        tau_hold = 0.5  # Pa, gel strength that suspends a 5 um particle bed
        hind = np.clip(1.0 - phi, 0.0, 1.0) ** 4.65  # Richardson-Zaki
        return v0 * hind * np.exp(-self.tau_yield(phi) / tau_hold)


# ---------------------------------------------------------------------------
# 2D masked stable-fluids solver (top-down view of the tank)
# ---------------------------------------------------------------------------
class FluidGrid2D:
    """Stam stable-fluids solver on a circular liquid mask.

    Fields are (n, n) float64; index [row, col] = [y, x], row 0 at the top.
    Velocities are physical m/s; dx converts to grid units where needed.
    Solid cells (outside the tank circle) enforce Neumann/no-through walls.
    """

    def __init__(self, n=96, tank_diameter=0.40):
        self.n = n
        self.diameter = tank_diameter
        self.dx = tank_diameter / n
        self.u = np.zeros((n, n))   # x-velocity (m/s), +right
        self.v = np.zeros((n, n))   # y-velocity (m/s), +down
        self.c = np.zeros((n, n))   # local solids volume fraction phi
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
        self.X, self.Y = xx, yy
        cx = cy = (n - 1) / 2.0
        self.cx, self.cy = cx, cy
        r = np.hypot(xx - cx, yy - cy)
        self.radius_cells = n / 2.0 - 1.5
        self.liquid = r <= self.radius_cells
        self.rgrid = r
        L = self.liquid.astype(np.float64)
        self._L = L
        # Liquid-neighbour count for interior cells (Neumann at solids).
        self.nliq = (L[1:-1, 2:] + L[1:-1, :-2] + L[2:, 1:-1] + L[:-2, 1:-1])
        self.nliq_safe = np.maximum(self.nliq, 1.0)
        ii, jj = np.mgrid[0:n - 2, 0:n - 2]
        self._red = ((ii + jj) % 2 == 0)
        self._black = ~self._red
        self._int_liquid = self.liquid[1:-1, 1:-1]

    # -- linear solvers ----------------------------------------------------
    def _gs_solve(self, x, x0, a, iters):
        """Red-black Gauss-Seidel for (1 + a*nliq) x = x0 + a * sum(nb)."""
        L = self._L
        for _ in range(iters):
            for color in (self._red, self._black):
                sel = color & self._int_liquid
                s = (x[1:-1, 2:] * L[1:-1, 2:] + x[1:-1, :-2] * L[1:-1, :-2]
                     + x[2:, 1:-1] * L[2:, 1:-1] + x[:-2, 1:-1] * L[:-2, 1:-1])
                xi = x[1:-1, 1:-1]
                xi[sel] = ((x0[1:-1, 1:-1] + a * s)[sel]
                           / (1.0 + a * self.nliq)[sel])
        x[~self.liquid] = 0.0
        return x

    def diffuse(self, field, nu, dt, iters=20):
        """Implicit diffusion; unconditionally stable for any nu."""
        a = nu * dt / (self.dx * self.dx)
        if a < 1.0e-9:
            return field
        return self._gs_solve(field.copy(), field, a, iters)

    def project(self, iters=30):
        """Make the velocity field divergence-free (masked Poisson solve)."""
        dx = self.dx
        div = np.zeros((self.n, self.n))
        div[1:-1, 1:-1] = ((self.u[1:-1, 2:] - self.u[1:-1, :-2])
                           + (self.v[2:, 1:-1] - self.v[:-2, 1:-1])) / (2 * dx)
        div[~self.liquid] = 0.0
        p = np.zeros((self.n, self.n))
        L = self._L
        dx2 = dx * dx
        for _ in range(iters):
            for color in (self._red, self._black):
                sel = color & self._int_liquid
                s = (p[1:-1, 2:] * L[1:-1, 2:] + p[1:-1, :-2] * L[1:-1, :-2]
                     + p[2:, 1:-1] * L[2:, 1:-1] + p[:-2, 1:-1] * L[:-2, 1:-1])
                pi = p[1:-1, 1:-1]
                pi[sel] = ((s - dx2 * div[1:-1, 1:-1])[sel]
                           / self.nliq_safe[sel])
        # Gradient with Neumann mirror at solids: a solid neighbour's
        # pressure is the cell's own (zero normal gradient), otherwise the
        # wall cells get a spurious kick from the untouched p=0 outside.
        pc = p[1:-1, 1:-1]
        pr = np.where(L[1:-1, 2:] > 0, p[1:-1, 2:], pc)
        pl = np.where(L[1:-1, :-2] > 0, p[1:-1, :-2], pc)
        pd = np.where(L[2:, 1:-1] > 0, p[2:, 1:-1], pc)
        pu = np.where(L[:-2, 1:-1] > 0, p[:-2, 1:-1], pc)
        self.u[1:-1, 1:-1] -= (pr - pl) / (2 * dx)
        self.v[1:-1, 1:-1] -= (pd - pu) / (2 * dx)
        self._zero_solids()

    def _zero_solids(self):
        self.u[~self.liquid] = 0.0
        self.v[~self.liquid] = 0.0

    def _bilinear(self, field, gx, gy):
        n = self.n
        gx = np.clip(gx, 0.0, n - 1.001)
        gy = np.clip(gy, 0.0, n - 1.001)
        x0 = gx.astype(np.int64)
        y0 = gy.astype(np.int64)
        fx = gx - x0
        fy = gy - y0
        f00 = field[y0, x0]
        f01 = field[y0, x0 + 1]
        f10 = field[y0 + 1, x0]
        f11 = field[y0 + 1, x0 + 1]
        return ((1 - fy) * ((1 - fx) * f00 + fx * f01)
                + fy * ((1 - fx) * f10 + fx * f11))

    def advect(self, field, dt):
        """Semi-Lagrangian advection of a scalar field by (u, v)."""
        k = dt / self.dx
        gx = self.X - self.u * k
        gy = self.Y - self.v * k
        out = self._bilinear(field, gx, gy)
        out[~self.liquid] = 0.0
        return out

    def advect_solids(self, dt):
        """Advect the solids field mass-conservatively.

        Semi-Lagrangian advection loses mass at the wall; the tank is a
        closed system, so rescale to the pre-advection total and shave any
        local overshoot above the packed-bed cap back into the free cells.
        """
        tot0 = self.total_phi()
        # Above bulk packing the field cannot hold all mass; do not force
        # it in (causes cap/redistribute flicker), the bookkeeping in
        # MixingTank stays authoritative for the excess.
        tot0 = min(tot0, float(self.liquid.sum()) * PHI_PACK_BED * 0.995)
        self.c = self.advect(self.c, dt)
        tot1 = self.total_phi()
        if tot1 > 1.0e-12:
            self.c *= tot0 / tot1
        over = np.maximum(self.c - PHI_PACK_BED, 0.0)
        excess = float(over[self.liquid].sum())
        if excess > 0.0:
            self.c = np.minimum(self.c, PHI_PACK_BED)
            room = self.liquid & (self.c < PHI_PACK_BED * 0.98)
            nroom = int(room.sum())
            if nroom:
                self.c[room] += excess / nroom
                self.c = np.minimum(self.c, PHI_PACK_BED)

    def advect_velocity(self, dt):
        k = dt / self.dx
        gx = self.X - self.u * k
        gy = self.Y - self.v * k
        u2 = self._bilinear(self.u, gx, gy)
        v2 = self._bilinear(self.v, gx, gy)
        self.u, self.v = u2, v2
        self._zero_solids()

    # -- forcing -----------------------------------------------------------
    def apply_impeller(self, theta, omega, r_imp_m, dt,
                       n_blades=2, coupling=6.0, u_cap=2.0):
        """Drag-couple fluid toward the tangential speed of rotating blades.

        theta: current blade angle (rad). omega: impeller angular speed
        (rad/s) used for the momentum target (capped at u_cap for the
        visual solver; true RPM is used in the power model). Blades are
        angular sectors so the forcing is unsteady -> chaotic advection.
        """
        r_cells = self.rgrid
        r_m = r_cells * self.dx
        ang = np.arctan2(self.Y - self.cy, self.X - self.cx)
        in_radius = (r_m < r_imp_m) & self.liquid
        blade = np.zeros_like(in_radius)
        half_w = math.pi / n_blades * 0.45
        for b in range(n_blades):
            tb = theta + b * (2 * math.pi / n_blades)
            d = np.angle(np.exp(1j * (ang - tb)))
            blade |= np.abs(d) < half_w
        zone = in_radius & blade
        if not zone.any():
            return
        # Target tangential velocity (right-handed: +omega = CCW on screen).
        speed = np.clip(omega * r_m, -u_cap, u_cap)
        tx = -np.sin(ang) * speed
        ty = np.cos(ang) * speed
        blend = min(1.0, coupling * dt)
        self.u[zone] += (tx[zone] - self.u[zone]) * blend
        self.v[zone] += (ty[zone] - self.v[zone]) * blend

    def splat_velocity(self, gx, gy, ux, uy, radius_cells=5.0, blend=0.6):
        """Manual stirring: push fluid near (gx, gy) toward (ux, uy) m/s."""
        w = np.exp(-((self.X - gx) ** 2 + (self.Y - gy) ** 2)
                   / (2 * radius_cells ** 2))
        w[~self.liquid] = 0.0
        self.u += (ux - self.u) * w * blend
        self.v += (uy - self.v) * w * blend

    # -- solids (dye) transport -------------------------------------------
    def settle(self, rheo, dt):
        """Explicit downward settling flux with a packed-bed cap."""
        vs = rheo.settling_velocity(self.c)          # m/s, array
        k = np.clip(vs * dt / self.dx, 0.0, 0.45)    # CFL-limited fraction
        flux = self.c * k
        room = np.zeros_like(self.c)
        room[:-1, :] = np.maximum(0.0, PHI_PACK_BED - self.c[1:, :])
        both = np.zeros_like(self.c, dtype=bool)
        both[:-1, :] = self.liquid[:-1, :] & self.liquid[1:, :]
        flux = np.where(both, np.minimum(flux, room), 0.0)
        self.c -= flux
        self.c[1:, :] += flux[:-1, :]

    def resuspend(self, dt, u_crit=U_ERODE_CRIT, k_ero=K_ERODE):
        """Bed erosion: the upward mirror of settle().

        Above the critical speed the shear carries material from the
        concentrated layer back up (CFL-limited upward flux, packed-bed
        cap on the receiving cell). Below u_crit the field is untouched --
        a uniform suspension shifts as a whole, only gradients (beds) erode.
        """
        speed = np.hypot(self.u, self.v)
        f = np.clip(k_ero * np.maximum(speed - u_crit, 0.0) * dt, 0.0, 0.45)
        flux = self.c * f
        room = np.zeros_like(self.c)
        room[1:, :] = np.maximum(0.0, PHI_PACK_BED - self.c[:-1, :])
        both = np.zeros_like(self.c, dtype=bool)
        both[1:, :] = self.liquid[1:, :] & self.liquid[:-1, :]
        flux = np.where(both, np.minimum(flux, room), 0.0)
        self.c -= flux
        self.c[:-1, :] += flux[1:, :]

    def add_solids_blob(self, phi_amount, gx, gy, sigma=3.0):
        """Inject phi_amount (sum over cells) as a gaussian blob; returns
        the part that did not fit (local packing cap)."""
        w = np.exp(-((self.X - gx) ** 2 + (self.Y - gy) ** 2)
                   / (2 * sigma ** 2))
        w[~self.liquid] = 0.0
        tot = w.sum()
        if tot <= 0:
            return phi_amount
        add = w / tot * phi_amount
        new = self.c + add
        over = np.maximum(0.0, new - PHI_PACK_BED)
        self.c = np.minimum(new, PHI_PACK_BED)
        return float(over.sum())

    def mean_phi(self):
        m = self.c[self.liquid]
        return float(m.mean()) if m.size else 0.0

    def total_phi(self):
        return float(self.c[self.liquid].sum())

    def mixedness(self):
        """1 - coefficient of variation of phi over liquid cells, in %."""
        m = self.c[self.liquid]
        if m.size == 0 or m.mean() < 1.0e-9:
            return 100.0
        cov = m.std() / m.mean()
        return float(max(0.0, 1.0 - cov) * 100.0)

    def max_speed(self):
        return float(np.hypot(self.u, self.v).max())

    def vortex_dip_m(self):
        """Free-surface centre dip (m) from radial equilibrium, first order.

        A swirling flow needs dh/dr = u_t^2/(g r); for the equivalent
        solid-body core omega_eff = sum(u_t r)/sum(r^2) the wall-to-axis
        level difference is omega_eff^2 R^2 / (2 g). Sign-free (squared) and
        uncapped -- the caller clamps to the liquid depth. A noisy or
        counter-rotating field averages toward zero dip: an honest readout,
        not a hard threshold.
        """
        ang = np.arctan2(self.Y - self.cy, self.X - self.cx)
        u_t = -np.sin(ang) * self.u + np.cos(ang) * self.v
        r_m = self.rgrid * self.dx
        sel = self.liquid & (r_m > 1.0e-6)
        den = float((r_m[sel] ** 2).sum())
        if den <= 0.0:
            return 0.0
        omega_eff = float((u_t[sel] * r_m[sel]).sum()) / den
        R = self.radius_cells * self.dx
        return omega_eff * omega_eff * R * R / (2.0 * GRAVITY)


# ---------------------------------------------------------------------------
# Stirrer (roerapparaat) with mains power draw
# ---------------------------------------------------------------------------
class Stirrer:
    """Overhead paddle stirrer with a torque-limited mains motor.

    Power correlation: Np(Re) = Kp/Re + Np_t * Re/(Re + 300)
    P_shaft = Np * rho * N^3 * D^5      (N in rev/s)
    Re      = rho * N * D^2 / eta_app   (impeller Reynolds number)
    eta_app is evaluated at the Metzner-Otto shear rate ks*N, so the
    torque limit is solved self-consistently (bisection on N).
    """

    def __init__(self, d_impeller=0.12, kp=70.0, np_turb=1.5,
                 torque_max=1.5, motor_eff=0.65, p_standby=5.0,
                 p_stall=300.0):
        self.d = d_impeller
        self.kp = kp
        self.np_turb = np_turb
        self.torque_max = torque_max      # N.m before droop/overload
        self.motor_eff = motor_eff
        self.p_standby = p_standby        # W when plugged in
        self.p_stall = p_stall            # W drawn by a blocked rotor
        self.rpm_set = 0.0
        self.rpm_actual = 0.0
        self.plugged_in = False
        self.on = False
        self.overload = False
        self.tripped = False
        self._overload_s = 0.0
        self.theta = 0.0                  # visual blade angle (rad)
        # Derived diagnostics from the last update():
        self.re = 0.0
        self.torque = 0.0
        self.p_shaft = 0.0
        self.p_electric = 0.0
        self.eta_app = ETA_WATER

    def power_number(self, re):
        re = max(re, 1.0e-9)
        return self.kp / re + self.np_turb * re / (re + 300.0)

    def _shaft(self, n_rps, rho, rheo, phi):
        """Return (p_shaft, torque, re, eta_app) at rotation speed n_rps."""
        gamma = KS_METZNER_OTTO * max(n_rps, 1.0e-6)
        eta = rheo.apparent_viscosity(phi, gamma)
        re = rho * n_rps * self.d ** 2 / eta
        p = self.power_number(re) * rho * n_rps ** 3 * self.d ** 5
        tq = p / (2 * math.pi * n_rps) if n_rps > 1.0e-9 else 0.0
        return p, tq, re, eta

    def update(self, rho, rheo, phi, dt):
        """Resolve actual RPM under the torque limit; track overload/trip."""
        running = self.plugged_in and self.on and not self.tripped
        n_set = self.rpm_set / 60.0
        if not running or n_set < 1.0e-6:
            self.rpm_actual = 0.0
            self.re = 0.0
            self.torque = 0.0
            self.p_shaft = 0.0
            self.p_electric = self.p_standby if self.plugged_in else 0.0
            self.overload = False
            gamma = GAMMA_MIN
            self.eta_app = rheo.apparent_viscosity(phi, gamma)
            return
        p, tq, re, eta = self._shaft(n_set, rho, rheo, phi)
        if tq <= self.torque_max:
            n_act = n_set
        else:
            lo, hi = 0.0, n_set          # torque is monotone increasing in N
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                _, tq_m, _, _ = self._shaft(max(mid, 1.0e-6), rho, rheo, phi)
                if tq_m > self.torque_max:
                    hi = mid
                else:
                    lo = mid
            n_act = lo
            p, tq, re, eta = self._shaft(max(n_act, 1.0e-6), rho, rheo, phi)
        self.rpm_actual = n_act * 60.0
        self.re = re
        self.torque = tq
        self.p_shaft = p
        self.eta_app = eta
        self.p_electric = p / self.motor_eff + self.p_standby
        self.overload = n_act < 0.999 * n_set
        if self.overload:
            # A torque-limited (near-stalled) rotor pulls stall current;
            # that heat is what finally trips the thermal breaker.
            self.p_electric = max(self.p_electric, self.p_stall)
        if self.overload:
            self._overload_s += dt
            if self._overload_s > 5.0:   # thermal breaker trips
                self.tripped = True
                self.on = False
        else:
            self._overload_s = 0.0
        self.theta += min(abs(n_act) * 2 * math.pi, 4 * math.pi) * dt

    def reset_trip(self):
        self.tripped = False
        self._overload_s = 0.0

    def regime(self):
        if self.re <= 0:
            return "stil"
        if self.re < 10:
            return "laminair"
        if self.re < 1.0e4:
            return "overgang"
        return "turbulent"


class PowerMeter:
    """Wall-socket energy meter (kW now, kWh + EUR cumulative)."""

    def __init__(self, eur_per_kwh=ELECTRICITY_EUR_PER_KWH):
        self.eur_per_kwh = eur_per_kwh
        self.p_watt = 0.0
        self.energy_kwh = 0.0

    def add(self, p_watt, dt):
        self.p_watt = p_watt
        self.energy_kwh += p_watt * dt / 3.6e6

    @property
    def cost_eur(self):
        return self.energy_kwh * self.eur_per_kwh


# ---------------------------------------------------------------------------
# The tank: composition bookkeeping + orchestration of one sim step
# ---------------------------------------------------------------------------
class MixingTank:
    """Cylindrical tank holding the slurry; owns solver, stirrer, meter."""

    def __init__(self, n=96, diameter=0.40, capacity_l=20.0,
                 water_l=15.0, w_pct=20.0, rheo=None,
                 thermal=True, thixotropy=False, evaporation=False):
        self.rheo = rheo or SlurryRheology()
        self.grid = FluidGrid2D(n=n, tank_diameter=diameter)
        self.stirrer = Stirrer()
        self.meter = PowerMeter()
        self.capacity_l = capacity_l
        self.water_kg = water_l * RHO_WATER / 1000.0
        self.slag_kg = 0.0
        self.placed = True
        self.pour_queue_kg = 0.0          # slag waiting to fall in
        self.pour_rate_kg_s = 2.0
        self.time_s = 0.0
        self.settle_boost = 200.0         # settling time-lapse when idle
        self.settle_boost_active = False
        # Bottom boundary-layer friction (1/s). The top-down 2D solver has
        # no floor; in a ~16 cm deep tank the bottom layer spins the vortex
        # down in tens of seconds, which this linear drag stands in for.
        self.bottom_drag = 0.3
        # --- thermal coupling: shaft work heats the slurry, the wall sheds
        # it to the workshop; temperature feeds back via Arrhenius rheology.
        self.thermal = thermal
        self.temperature_c = T_REF_C
        self.t_ambient_c = T_REF_C
        self.wall_loss_w_per_k = UA_TANK_W_PER_K
        # --- thixotropy (opt-in for level design): lambda 1 = fully built
        # gel, 0 = fully broken down; ON shifts eta_app by up to (1 + C_THIX).
        self.thix_on = thixotropy
        self.struct_lambda = 1.0
        # --- evaporation (open-bath water loss; lives inside the 0-D bath
        # balance below, so it follows the `thermal` switch).
        self.evaporation = evaporation
        self.evaporated_kg = 0.0
        self.evap_rate_kg_s = 0.0
        if w_pct > 0:
            self.set_composition(w_pct, instant=True)

    # -- volumes/composition ----------------------------------------------
    @property
    def volume_l(self):
        return (self.water_kg / RHO_WATER + self.slag_kg / RHO_SLAG) * 1000.0

    def liquid_depth_m(self):
        area = math.pi * (self.grid.diameter / 2.0) ** 2
        return max(self.volume_l / 1000.0 / area, 1.0e-3)

    def w(self):
        tot = self.water_kg + self.slag_kg
        return self.slag_kg / tot if tot > 0 else 0.0

    def phi_bulk(self):
        return self.rheo.phi_from_w(self.w())

    def _phi_per_kg(self):
        """Field phi units for 1 kg of slag at current liquid depth."""
        cell_vol = self.grid.dx ** 2 * self.liquid_depth_m()
        return (1.0 / RHO_SLAG) / cell_vol

    # -- AR verbs ----------------------------------------------------------
    def add_water(self, liters):
        """Hose: add water, limited by remaining capacity. Returns added L."""
        room_l = max(0.0, self.capacity_l - self.volume_l)
        add = min(liters, room_l)
        self.water_kg += add * RHO_WATER / 1000.0
        if add > 0:  # dilution: same solids in more liquid
            self._rescale_field()
        return add

    def add_slag(self, kg):
        """Pour slag from the second container; falls in over a few frames."""
        room_l = max(0.0, self.capacity_l - self.volume_l)
        add = min(kg, room_l / 1000.0 * RHO_SLAG)
        self.pour_queue_kg += add
        return add

    def set_composition(self, w_pct, instant=False):
        """Slider: set target solids mass fraction (lab mode).

        Capacity-aware: if the target composition cannot fit in the tank
        with the current water, water is drained (instantly) first so the
        tank never exceeds capacity_l.
        """
        w = min(max(w_pct / 100.0, 0.0), 0.995)
        target_slag = self.water_kg * w / (1.0 - w)
        vol_m3 = self.water_kg / RHO_WATER + target_slag / RHO_SLAG
        cap_m3 = self.capacity_l / 1000.0
        if vol_m3 > cap_m3:
            denom = 1.0 / RHO_WATER + w / max((1.0 - w) * RHO_SLAG, 1e-9)
            self.water_kg = cap_m3 / denom
            target_slag = self.water_kg * w / (1.0 - w)
        delta = target_slag - (self.slag_kg + self.pour_queue_kg)
        if delta >= 0:
            if instant:
                # Collapse any pending pour into the instant recomposition.
                self.slag_kg = target_slag
                self.pour_queue_kg = 0.0
                self._spread_uniform()
            else:
                self.pour_queue_kg += delta
        else:
            # Remove solids proportionally (drain/suction, lab mode).
            take_queue = min(self.pour_queue_kg, -delta)
            self.pour_queue_kg -= take_queue
            rest = -delta - take_queue
            if self.slag_kg > 1.0e-12:
                scale = max(0.0, (self.slag_kg - rest) / self.slag_kg)
                self.grid.c *= scale
                self.slag_kg *= scale
        return self.w()

    def _spread_uniform(self):
        nliq = int(self.grid.liquid.sum())
        if nliq == 0:
            return
        phi_tot = self.slag_kg * self._phi_per_kg()
        self.grid.c[:] = 0.0
        self.grid.c[self.grid.liquid] = min(phi_tot / nliq, PHI_PACK_BED)

    def _rescale_field(self):
        """Re-normalize field so total field phi matches slag_kg after
        depth changes (water added)."""
        want = self.slag_kg * self._phi_per_kg()
        have = self.grid.total_phi()
        if have > 1.0e-12:
            self.grid.c *= want / have

    def settle_bottom(self):
        """Start fully settled: all solids as a bottom bed (top-down view:
        a ring near the wall reads wrong, so we use a soft gradient)."""
        self._spread_uniform()
        # Bias concentration downward (screen bottom) to visualise a bed.
        g = self.grid
        bias = 1.0 + 1.5 * (g.Y - g.cy) / g.n
        g.c *= np.clip(bias, 0.1, None)
        self._rescale_field()

    # -- temperature & structure -------------------------------------------
    def _structural(self):
        """Flow-curve multiplier from the thixotropy state (1.0 when off)."""
        return 1.0 + C_THIX * self.struct_lambda if self.thix_on else 1.0

    def _update_structure(self, st, dt):
        """Thixotropy kinetics: rebuild at rest, shear-driven breakdown.

        dlambda/dt = (1 - lambda)/T_build - K_break * gamma_dot * lambda,
        the classic rate equation with lambda in [0, 1].
        """
        n_rps = st.rpm_actual / 60.0
        gamma = KS_METZNER_OTTO * n_rps if n_rps > 1.0e-4 else 0.0
        dl = ((1.0 - self.struct_lambda) / T_BUILD_S
              - K_BREAK * gamma * self.struct_lambda)
        self.struct_lambda = min(max(self.struct_lambda + dl * dt, 0.0), 1.0)

    def _thermal_step(self, dt, p_heat_w):
        """0-D bath balance: viscous shaft work in; wall loss, evaporation
        (mass + latent heat) out.

        Only shaft power heats the fluid -- motor and standby losses stay in
        the drive housing/air, as in reality. Wall loss is Newton cooling
        with a lumped UA; realistic time constants are hours, so the bath
        drifts, it does not jump. Evaporation runs on the vapour-pressure
        deficit against the workshop air and thickens the slurry as pure
        water leaves (the field is rescaled to the new depth).
        """
        m_kg = self.water_kg + self.slag_kg
        if m_kg <= 1.0e-9:
            return
        q_loss = self.wall_loss_w_per_k * (self.temperature_c
                                           - self.t_ambient_c)
        if self.evaporation and self.water_kg > 0.0:
            area = math.pi * (self.grid.diameter / 2.0) ** 2
            deficit = max(rho_vapor_sat(self.temperature_c)
                          - RH_AIR * rho_vapor_sat(self.t_ambient_c), 0.0)
            take = min(area * H_MASS * deficit * dt, self.water_kg)
            self.water_kg -= take
            self.evaporated_kg += take
            self.evap_rate_kg_s = take / dt
            q_loss += take / dt * LATENT_VAP
            self._rescale_field()
        else:
            self.evap_rate_kg_s = 0.0
        cp = (self.water_kg * CP_WATER + self.slag_kg * CP_SLAG) \
            / max(self.water_kg + self.slag_kg, 1.0e-9)
        self.temperature_c += (p_heat_w - q_loss) \
            / max(self.water_kg + self.slag_kg, 1.0e-9) / cp * dt

    # -- simulation step ---------------------------------------------------
    def step(self, dt=1.0 / 30.0, mouse_splat=None):
        """Advance one frame. mouse_splat = (gx, gy, ux, uy) or None."""
        g = self.grid
        st = self.stirrer
        rheo = self.rheo.at_conditions(self.temperature_c,
                                       structural=self._structural())
        w = self.w()
        phi = self.phi_bulk()
        rho = rheo.rho_mix(w)

        # Pour queue -> field injection at the pour point (top of screen).
        if self.pour_queue_kg > 1.0e-9:
            drop = min(self.pour_queue_kg, self.pour_rate_kg_s * dt)
            leftover = g.add_solids_blob(drop * self._phi_per_kg(),
                                         g.cx, g.cy - g.radius_cells * 0.55)
            self.pour_queue_kg -= drop
            self.slag_kg += drop
            if leftover > 1.0e-9:  # did not fit locally; re-queue as mass
                back = leftover / self._phi_per_kg()
                self.slag_kg -= back
                self.pour_queue_kg += back

        if not self.placed:
            self.meter.add(st.p_standby if st.plugged_in else 0.0, dt)
            if self.thermal:
                self._thermal_step(dt, 0.0)
            self.time_s += dt
            return

        st.update(rho, rheo, phi, dt)
        self.meter.add(st.p_electric, dt)
        if self.thix_on:
            self._update_structure(st, dt)
        if self.thermal:
            self._thermal_step(dt, st.p_shaft)
        nu = st.eta_app / rho  # kinematic viscosity for the solver

        omega = st.rpm_actual / 60.0 * 2 * math.pi
        stirring = omega > 1.0e-3
        if stirring or (mouse_splat is not None) or g.max_speed() > 1.0e-4:
            g.u = g.diffuse(g.u, nu, dt)
            g.v = g.diffuse(g.v, nu, dt)
            umax = g.max_speed()
            sub = int(np.clip(math.ceil(umax * dt / (3.0 * g.dx)), 1, 4))
            dts = dt / sub
            for _ in range(sub):
                if stirring:
                    g.apply_impeller(st.theta, omega, st.d / 2.0, dts)
                if mouse_splat is not None:
                    gx, gy, ux, uy = mouse_splat
                    g.splat_velocity(gx, gy, ux, uy)
                g.project(iters=16)
                g.advect_velocity(dts)
            g.project(iters=30)
            drag = math.exp(-self.bottom_drag * dt)
            g.u *= drag
            g.v *= drag
            g.advect_solids(dt)
            if g.max_speed() > U_ERODE_CRIT:
                g.resuspend(dt)          # roeren wervelt de bodem er weer in

        # Settling: time-lapse only when the fluid is quiescent.
        quiescent = (not stirring) and g.max_speed() < 0.05
        self.settle_boost_active = quiescent
        settle_dt = dt * (self.settle_boost if quiescent else 1.0)
        g.settle(rheo, settle_dt)

        self.time_s += dt

    # -- reporting ---------------------------------------------------------
    def snapshot(self):
        st = self.stirrer
        rheo = self.rheo.at_conditions(self.temperature_c,
                                       structural=self._structural())
        w = self.w()
        phi = self.phi_bulk()
        vs = float(rheo.settling_velocity(phi))
        depth = self.liquid_depth_m()
        dip_raw = self.grid.vortex_dip_m()
        # Gas entrainment: the visual field is speed-capped (u_cap), so the
        # drawn dip alone can never reach the floor of a real bath. The flag
        # therefore also uses the standard Froude vortex criterion Fr =
        # N^2 D / g of the TRUE stirrer state (>1: deep vortex, air intake).
        n_rps = st.rpm_actual / 60.0
        fr = n_rps * n_rps * st.d / GRAVITY
        return {
            "time_s": self.time_s,
            "w_pct": w * 100.0,
            "phi_pct": phi * 100.0,
            "rho_mix": rheo.rho_mix(w),
            "eta_app_pa_s": st.eta_app,
            "tau_yield_pa": float(rheo.tau_yield(phi)),
            "volume_l": self.volume_l,
            "re_impeller": st.re,
            "regime": st.regime(),
            "rpm_set": st.rpm_set,
            "rpm_actual": st.rpm_actual,
            "torque_nm": st.torque,
            "p_shaft_w": st.p_shaft,
            "p_electric_w": st.p_electric,
            "energy_kwh": self.meter.energy_kwh,
            "cost_eur": self.meter.cost_eur,
            "mixedness_pct": self.grid.mixedness(),
            "overload": st.overload,
            "tripped": st.tripped,
            "settling_mm_h": vs * 3.6e6,
            "settle_boost": self.settle_boost_active,
            "temperature_c": self.temperature_c,
            "thix_lambda": self.struct_lambda,
            "vortex_dip_mm": min(dip_raw, depth) * 1000.0,
            "air_entrainment": bool(dip_raw >= depth or fr > 1.0),
            "water_kg": self.water_kg,
            "evap_rate_kg_h": self.evap_rate_kg_s * 3600.0,
        }
