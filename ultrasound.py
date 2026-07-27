"""Silicon-wash level: ultrasound-assisted separation of the Si-rich light
fraction from steel-slag slurry (headless, Python authority).

The tank gets a transducer ring driven by an ultrasound generator with two
channels, 28 kHz and 40 kHz. Physical storyline (all first-order but real
mechanisms):

* BOF slag holds a light, silicon-rich fraction (silicates/SiO2-bearing
  phases, ~15 wt% of the solids) locked inside dense Fe-rich aggregates.
* Cavitation de-aggregates the sludge. Bubble size scales inversely with
  frequency: 28 kHz collapses coarse clusters, 40 kHz peels the fine ones,
  so freeing the whole light fraction needs BOTH channels ("instelling
  28 en 40 kHz").
* Freed light particles are lifted by sono-flotation: cavitation micro-
  bubbles attach and rafts rise to the surface as a froth layer. Gentle
  stirring feeds the froth to the rim; hard stirring beats it back in.
* The froth leaves over the tank's overflow rim (fill with the hose!) and
  is captured in the launder ("afvloeien").
* Everything is gated by the slurry rheology: apparent viscosity and
  solids fraction damp cavitation and bubble rise — the "juiste
  viscositeitsparameters" are a window, reached by diluting the sludge.

The species model is 0-D (kg of bound-coarse / bound-fine / free / froth /
captured) on top of the 2-D MixingTank; mass is reconciled with the tank's
slag bookkeeping when froth leaves the tank. Energy for the generator runs
through the tank's wall-socket meter, so the kWh cost of a wash is real.

JS port parity: web/viscosity-sim.js in molgang-knitweb mirrors this file;
change formulas only together.
"""

from __future__ import annotations

import math

from viscosity_core import MixingTank

# Generator electrical draw per channel (W) — small industrial bath drivers.
P_28KHZ_W = 180.0
P_40KHZ_W = 150.0

LIGHT_FRACTION_OF_SLAG = 0.15   # Si-rich light phases, share of solids mass
COARSE_SHARE = 0.60             # of the light fraction, locked in coarse aggl.

# Rate constants (1/s) at full cavitation efficiency.
K_FREE_28 = 0.028               # 28 kHz frees coarse-bound light particles
K_FREE_40 = 0.026               # 40 kHz frees fine-bound light particles
CROSS_TALK = 0.10               # each channel touches the other size class
K_FLOAT = 0.030                 # free -> froth (sono-flotation)
K_OVERFLOW = 0.060              # froth -> captured, at/above the rim
K_REMIX = 0.020                 # froth beaten back to free by hard stirring
K_REAGG = 0.004                 # free re-aggregates when ultrasound is off

RIM_FILL_FRACTION = 0.92        # overflow rim sits at 92% of tank capacity
ETA_CAV_LO = 0.005              # Pa.s: full cavitation at/below 5 mPa.s
ETA_CAV_HI = 0.050              # Pa.s: cavitation dead at/above 50 mPa.s
PHI_CAV_KILL = 0.30             # bubbles fully damped at this solids frac
RPM_GENTLE = 80.0               # stirring that helps feed the froth
RPM_REMIX_ONSET = 140.0         # stirring that starts destroying the froth

STAR_1_PCT = 40.0
STAR_2_PCT = 60.0
STAR_3_PCT = 75.0
STAR_3_KWH = 0.040


class UltrasoundBath:
    """Two-channel (28/40 kHz) transducer ring on the tank."""

    def __init__(self):
        self.on_28 = False
        self.on_40 = False

    def power_w(self):
        return (P_28KHZ_W if self.on_28 else 0.0) + \
               (P_40KHZ_W if self.on_40 else 0.0)

    @staticmethod
    def cavitation_efficiency(eta_app, phi):
        """0..1: how well cavitation develops in the current slurry.

        Log-ramp in apparent viscosity (full below 5 mPa.s, dead above
        50 mPa.s) times a linear solids damping (dead at phi=0.30).
        """
        e = min(max(eta_app, 1e-6), 10.0)
        visc = (math.log10(ETA_CAV_HI) - math.log10(e)) / \
               (math.log10(ETA_CAV_HI) - math.log10(ETA_CAV_LO))
        visc = min(max(visc, 0.0), 1.0)
        solids = min(max(1.0 - phi / PHI_CAV_KILL, 0.0), 1.0)
        return visc * solids


class SiliconWashLevel:
    """Level 1: wash the Si-rich light fraction out of the sludge.

    Owns the species bookkeeping and win/star logic; drives the tank's
    meter for the generator's electrical draw. Call step(dt) right after
    tank.step(dt).
    """

    def __init__(self, tank: MixingTank):
        self.tank = tank
        self.bath = UltrasoundBath()
        self.active = False
        self.bound_coarse = 0.0
        self.bound_fine = 0.0
        self.free = 0.0
        self.froth = 0.0
        self.captured = 0.0
        self.light_total = 0.0
        self.energy_start_kwh = 0.0
        self.time_s = 0.0
        self.cav_eff = 0.0

    # -- lifecycle ---------------------------------------------------------
    def start(self, w_pct=62.0, water_l=8.0):
        """Arm the level: thick sludge at LOW volume, light fraction locked.

        62 wt% starts ABOVE the cavitation window (phi > 0.30) on purpose —
        the player must first fix the viscosity by diluting with the hose,
        and later keep filling to the overflow rim so the froth can leave.
        Starting at 8 L water leaves that headroom in the 20 L tank.
        """
        t = self.tank
        t.water_kg = float(water_l)             # 1 L water = 1 kg
        t.set_composition(w_pct, instant=True)
        light = t.slag_kg * LIGHT_FRACTION_OF_SLAG
        self.light_total = light
        self.bound_coarse = light * COARSE_SHARE
        self.bound_fine = light * (1.0 - COARSE_SHARE)
        self.free = 0.0
        self.froth = 0.0
        self.captured = 0.0
        self.energy_start_kwh = t.meter.energy_kwh
        self.time_s = 0.0
        self.active = True

    # -- physics -----------------------------------------------------------
    def step(self, dt):
        if not self.active:
            return
        t = self.tank
        st = t.stirrer

        # Generator draw through the wall-socket meter (stirrer power was
        # already metered by tank.step; add the generator on top and show
        # the combined instantaneous draw).
        gen_p = self.bath.power_w() if (t.placed and st.plugged_in) else 0.0
        if gen_p > 0:
            t.meter.add(gen_p, dt)
        t.meter.p_watt = st.p_electric + gen_p

        eta = st.eta_app
        phi = t.phi_bulk()
        eff = UltrasoundBath.cavitation_efficiency(eta, phi) if gen_p > 0 else 0.0
        self.cav_eff = eff

        on28 = self.bath.on_28 and gen_p > 0
        on40 = self.bath.on_40 and gen_p > 0

        # 1. De-aggregation: each channel frees its size class (+crosstalk).
        r_coarse = ((K_FREE_28 if on28 else 0.0)
                    + (K_FREE_40 * CROSS_TALK if on40 else 0.0)) * eff
        r_fine = ((K_FREE_40 if on40 else 0.0)
                  + (K_FREE_28 * CROSS_TALK if on28 else 0.0)) * eff
        d_c = self.bound_coarse * min(r_coarse * dt, 0.5)
        d_f = self.bound_fine * min(r_fine * dt, 0.5)
        self.bound_coarse -= d_c
        self.bound_fine -= d_f
        self.free += d_c + d_f

        # 2. Sono-flotation: cavitation bubbles raft free particles up.
        #    Gentle stirring helps feed the froth; hard stirring remixes it.
        rpm = st.rpm_actual
        stir_boost = 1.0 + 0.6 * min(max(rpm / RPM_GENTLE, 0.0), 1.0)
        if rpm > RPM_REMIX_ONSET:
            # High shear detaches bubbles from particles (flotation
            # detachment): the feed itself collapses, not just the froth.
            stir_boost = max(0.25, 1.6 - 1.35 * (rpm - RPM_REMIX_ONSET) / 160.0)
        k_float = K_FLOAT * eff * (1.0 if (on28 or on40) else 0.0) * stir_boost
        d_up = self.free * min(k_float * dt, 0.5)
        self.free -= d_up
        self.froth += d_up
        if rpm > RPM_REMIX_ONSET:
            frac = ((rpm - RPM_REMIX_ONSET) / 160.0) ** 2
            d_remix = self.froth * min(K_REMIX * frac * dt, 0.5)
            self.froth -= d_remix
            self.free += d_remix

        # 3. Overflow: froth leaves over the rim once the bath is filled
        #    high enough (hose!) — this is the actual "afvloeien".
        rim_l = t.capacity_l * RIM_FILL_FRACTION
        head = (t.volume_l - rim_l) / (t.capacity_l - rim_l)
        head = min(max(head, 0.0), 1.0)
        d_out = self.froth * min(K_OVERFLOW * head * dt, 0.5)
        if d_out > 0:
            self.froth -= d_out
            self.captured += d_out
            # The froth mass physically leaves the tank: reconcile the
            # slag bookkeeping and the concentration field.
            if t.slag_kg > 1e-9:
                t.grid.c *= max(0.0, 1.0 - d_out / t.slag_kg)
            t.slag_kg = max(0.0, t.slag_kg - d_out)

        # 4. Re-aggregation when the field is silent.
        if not (on28 or on40):
            d_re = self.free * min(K_REAGG * dt, 0.5)
            self.free -= d_re
            self.bound_coarse += d_re * COARSE_SHARE
            self.bound_fine += d_re * (1.0 - COARSE_SHARE)

        self.time_s += dt

    # -- reporting ---------------------------------------------------------
    def captured_pct(self):
        return 100.0 * self.captured / self.light_total \
            if self.light_total > 0 else 0.0

    def energy_kwh(self):
        return self.tank.meter.energy_kwh - self.energy_start_kwh

    def stars(self):
        pct = self.captured_pct()
        s = 0
        if pct >= STAR_1_PCT:
            s = 1
        if pct >= STAR_2_PCT:
            s = 2
        if pct >= STAR_3_PCT and self.energy_kwh() <= STAR_3_KWH:
            s = 3
        return s

    def hint(self):
        """One-line Dutch coaching hint for the HUD."""
        t = self.tank
        if not self.active:
            return "Start het level"
        if not (t.placed and t.stirrer.plugged_in):
            return "Plaats de bak en steek de stekker in het stopcontact"
        if not (self.bath.on_28 or self.bath.on_40):
            return "Zet de ultrasound-generator aan (28 én 40 kHz)"
        if self.cav_eff < 0.15:
            return ("Slib te dik voor cavitatie — verdun met de slang "
                    "(viscositeit omlaag)")
        if self.bath.on_28 != self.bath.on_40:
            return ("Eén frequentie mist een deeltjesklasse — zet 28 én "
                    "40 kHz aan")
        if t.volume_l < t.capacity_l * RIM_FILL_FRACTION:
            return "Vul tot de overlooprand met de slang zodat de froth afvloeit"
        if t.stirrer.rpm_actual > RPM_REMIX_ONSET:
            return "Te hard geroerd — de froth mengt terug (max ~120 RPM)"
        if t.stirrer.rpm_actual < 10:
            return "Roer zachtjes (30–80 RPM) om de froth aan te voeren"
        return "Goed bezig — silicium vloeit af"

    def mass_balance_kg(self):
        return (self.bound_coarse + self.bound_fine + self.free
                + self.froth + self.captured)

    def snapshot(self):
        return {
            "active": self.active,
            "time_s": self.time_s,
            "cav_eff": self.cav_eff,
            "on_28": self.bath.on_28,
            "on_40": self.bath.on_40,
            "generator_w": self.bath.power_w(),
            "bound_kg": self.bound_coarse + self.bound_fine,
            "free_kg": self.free,
            "froth_kg": self.froth,
            "captured_kg": self.captured,
            "light_total_kg": self.light_total,
            "captured_pct": self.captured_pct(),
            "energy_kwh": self.energy_kwh(),
            "stars": self.stars(),
            "hint": self.hint(),
        }
