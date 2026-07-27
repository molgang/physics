"""MOLGANG viscosity lab - tkinter GUI (Dutch labels, stdlib only + numpy).

Top-down view of the tank. Sliders drive solids fraction (1-100 wt% slag
slib) and stir intensity (RPM); buttons expose the AR verbs (place tank,
plug in, hose water, pour slag). Drag the mouse in the tank to hand-stir.

Run: python3 viscosity_gui.py [grid_n]
"""

from __future__ import annotations

import sys
import time
import tkinter as tk

import numpy as np

try:  # PIL gives a much faster canvas blit (paste); PPM path is fallback
    from PIL import Image, ImageTk
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

from ultrasound import SiliconWashLevel
from viscosity_core import MixingTank

SCALE = 5                       # canvas pixels per grid cell
WATER_RGB = np.array([22.0, 56.0, 112.0])
SLIB_RGB = np.array([150.0, 133.0, 108.0])
BG_RGB = np.array([11.0, 14.0, 19.0])
PHI_VISUAL_FULL = 0.45          # phi rendered as "fully slib-coloured"


class ViscosityApp:
    def __init__(self, root, n=80):
        self.root = root
        self.n = n
        root.title("MOLGANG Viscositeitslab — staalslak-slib 5 µm")
        root.configure(bg="#0b0e13")
        self.tank = self._new_tank()
        self.level = SiliconWashLevel(self.tank)
        self.mouse_splat = None
        self._last_mouse = None
        self._fps_t = time.perf_counter()
        self._fps = 0.0
        self._slider_guard = False

        px = n * SCALE
        self.canvas = tk.Canvas(root, width=px, height=px, bg="#0b0e13",
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.img_id = self.canvas.create_image(0, 0, anchor="nw")
        self.overlay = []

        panel = tk.Frame(root, bg="#0b0e13")
        panel.grid(row=0, column=1, sticky="ns", padx=(0, 12), pady=10)
        self._build_panel(panel)
        self._tick()

    def _new_tank(self):
        tank = MixingTank(n=self.n, diameter=0.40, capacity_l=20.0,
                          water_l=14.0, w_pct=20.0)
        tank.settle_bottom()          # start with a settled bed
        return tank

    # ------------------------------------------------------------- panel --
    def _lbl(self, parent, text, size=10, color="#dbe6f5", bold=False):
        return tk.Label(parent, text=text, bg="#0b0e13", fg=color,
                        font=("Segoe UI", size, "bold" if bold else "normal"),
                        anchor="w", justify="left")

    def _build_panel(self, p):
        self._lbl(p, "Viscositeitslab", 14, bold=True).pack(anchor="w")
        self._lbl(p, "Roeren in staalslak-slib (5 µm)", 9,
                  "#8fa2bd").pack(anchor="w", pady=(0, 8))

        self._lbl(p, "Staalslak-slib (massa-%)", 10).pack(anchor="w")
        self.s_w = tk.Scale(p, from_=1, to=100, orient="horizontal",
                            resolution=1, length=250, bg="#0b0e13",
                            fg="#dbe6f5", highlightthickness=0,
                            troughcolor="#1a2333",
                            command=self._on_w_slider)
        self.s_w.set(20)
        self.s_w.pack(anchor="w")

        self._lbl(p, "Roerintensiteit (RPM)", 10).pack(anchor="w",
                                                       pady=(6, 0))
        self.s_rpm = tk.Scale(p, from_=0, to=600, orient="horizontal",
                              resolution=10, length=250, bg="#0b0e13",
                              fg="#dbe6f5", highlightthickness=0,
                              troughcolor="#1a2333",
                              command=self._on_rpm_slider)
        self.s_rpm.set(300)
        self.s_rpm.pack(anchor="w")

        btns = tk.Frame(p, bg="#0b0e13")
        btns.pack(anchor="w", pady=8)

        def mkbtn(row, col, text, cmd, width=13):
            b = tk.Button(btns, text=text, command=cmd, width=width,
                          bg="#1a2333", fg="#dbe6f5",
                          activebackground="#263247",
                          activeforeground="#ffffff", relief="flat")
            b.grid(row=row, column=col, padx=2, pady=2, sticky="w")
            return b

        self.b_place = mkbtn(0, 0, "Bak: geplaatst", self._toggle_place)
        self.b_plug = mkbtn(0, 1, "Stekker: uit", self._toggle_plug)
        self.b_on = mkbtn(1, 0, "Roerder: UIT", self._toggle_on)
        self.b_trip = mkbtn(1, 1, "Reset zekering", self._reset_trip)
        mkbtn(2, 0, "Slang: +2 L water", self._hose)
        mkbtn(2, 1, "Stort: +1 kg slak", self._pour)
        mkbtn(3, 0, "Opnieuw (bezonken)", self._reset)

        self.status = self._lbl(p, "", 10, "#ffb454", bold=True)
        self.status.pack(anchor="w", pady=(2, 6))

        self._lbl(p, "Level: Si-afvloei (ultrasound)", 10,
                  bold=True).pack(anchor="w", pady=(4, 2))
        lbtns = tk.Frame(p, bg="#0b0e13")
        lbtns.pack(anchor="w")

        def mklvl(col, text, cmd):
            b = tk.Button(lbtns, text=text, command=cmd, width=9,
                          bg="#1a2333", fg="#dbe6f5",
                          activebackground="#263247", relief="flat")
            b.grid(row=0, column=col, padx=2, pady=2)
            return b

        self.b_lvl = mklvl(0, "Start", self._level_start)
        self.b_28 = mklvl(1, "28 kHz: uit", self._toggle_28)
        self.b_40 = mklvl(2, "40 kHz: uit", self._toggle_40)
        self.lvl_hint = self._lbl(p, "", 9, "#8fa2bd")
        self.lvl_hint.pack(anchor="w")
        self.lvl_stat = self._lbl(p, "", 9)
        self.lvl_stat.pack(anchor="w", pady=(0, 4))

        self.rows = {}
        grid = tk.Frame(p, bg="#0b0e13")
        grid.pack(anchor="w", fill="x")
        fields = [
            ("phi", "Vaste stof φ"), ("eta", "Viscositeit η"),
            ("tau", "Zwichtspanning τy"), ("rho", "Dichtheid mengsel"),
            ("vol", "Volume in bak"), ("re", "Reynolds (regime)"),
            ("rpm", "RPM werkelijk"), ("torque", "Koppel"),
            ("pshaft", "Vermogen as"), ("pel", "Vermogen stopcontact"),
            ("kwh", "kWh-meter"), ("mix", "Menggraad"),
            ("settle", "Bezinksnelheid"), ("fps", "Sim FPS"),
        ]
        for i, (key, label) in enumerate(fields):
            self._lbl(grid, label, 9, "#8fa2bd").grid(row=i, column=0,
                                                      sticky="w")
            v = self._lbl(grid, "—", 9)
            v.grid(row=i, column=1, sticky="e", padx=(14, 0))
            self.rows[key] = v
        grid.columnconfigure(1, weight=1)

    # ----------------------------------------------------------- actions --
    def _on_w_slider(self, val):
        if self._slider_guard:
            return
        # Instant lab recomposition; the physical pour visual belongs to
        # the "Stort" button, not the slider.
        self.tank.set_composition(float(val), instant=True)

    def _on_rpm_slider(self, val):
        self.tank.stirrer.rpm_set = float(val)

    def _toggle_place(self):
        t = self.tank
        t.placed = not t.placed
        self.b_place.config(text=f"Bak: {'geplaatst' if t.placed else 'weg'}")

    def _toggle_plug(self):
        st = self.tank.stirrer
        st.plugged_in = not st.plugged_in
        self.b_plug.config(text=f"Stekker: {'in' if st.plugged_in else 'uit'}")

    def _toggle_on(self):
        st = self.tank.stirrer
        st.on = not st.on
        self.b_on.config(text=f"Roerder: {'AAN' if st.on else 'UIT'}")

    def _reset_trip(self):
        self.tank.stirrer.reset_trip()

    def _hose(self):
        self.tank.add_water(2.0)
        self._sync_w_slider()

    def _pour(self):
        self.tank.add_slag(1.0)

    def _reset(self):
        rpm = self.s_rpm.get()
        self.tank = self._new_tank()
        self.tank.set_composition(float(self.s_w.get()), instant=True)
        self.tank.settle_bottom()
        self.tank.stirrer.rpm_set = float(rpm)
        self.level = SiliconWashLevel(self.tank)
        self.b_28.config(text="28 kHz: uit")
        self.b_40.config(text="40 kHz: uit")

    def _level_start(self):
        self.level = SiliconWashLevel(self.tank)
        self.level.bath.on_28 = False
        self.level.bath.on_40 = False
        self.b_28.config(text="28 kHz: uit")
        self.b_40.config(text="40 kHz: uit")
        self.level.start()
        self._sync_w_slider()

    def _toggle_28(self):
        b = self.level.bath
        b.on_28 = not b.on_28
        self.b_28.config(text=f"28 kHz: {'AAN' if b.on_28 else 'uit'}")

    def _toggle_40(self):
        b = self.level.bath
        b.on_40 = not b.on_40
        self.b_40.config(text=f"40 kHz: {'AAN' if b.on_40 else 'uit'}")

    def _sync_w_slider(self):
        self._slider_guard = True
        self.s_w.set(round(self.tank.w() * 100))
        self._slider_guard = False

    def _on_drag(self, ev):
        gx, gy = ev.x / SCALE, ev.y / SCALE
        if self._last_mouse is not None:
            dt = 1.0 / 30.0
            ux = (gx - self._last_mouse[0]) * SCALE * 0.004 / dt
            uy = (gy - self._last_mouse[1]) * SCALE * 0.004 / dt
            cap = 1.5
            ux = max(-cap, min(cap, ux))
            uy = max(-cap, min(cap, uy))
            self.mouse_splat = (gx, gy, ux, uy)
        self._last_mouse = (gx, gy)

    def _on_release(self, _ev):
        self.mouse_splat = None
        self._last_mouse = None

    # ------------------------------------------------------------ render --
    def _render(self):
        g = self.tank.grid
        t = np.clip(g.c / PHI_VISUAL_FULL, 0.0, 1.0) ** 0.7
        rgb = (WATER_RGB[None, None, :] * (1 - t[..., None])
               + SLIB_RGB[None, None, :] * t[..., None])
        speed = np.hypot(g.u, g.v)
        rgb += np.clip(speed * 45.0, 0, 28)[..., None]
        if self.level.active and self.level.froth > 1e-3:
            a = min(0.5, self.level.froth * 0.30)   # froth layer on top
            rgb = rgb * (1 - a) + np.array([232.0, 238.0, 244.0]) * a
        rgb[~g.liquid] = BG_RGB
        img = np.clip(rgb, 0, 255).astype(np.uint8)
        if HAVE_PIL:
            pil = Image.fromarray(img).resize(
                (self.n * SCALE, self.n * SCALE), Image.BILINEAR)
            if not hasattr(self, "photo"):
                self.photo = ImageTk.PhotoImage(pil)
                self.canvas.itemconfig(self.img_id, image=self.photo)
            else:
                self.photo.paste(pil)          # in-place blit, no churn
        else:
            img = np.repeat(np.repeat(img, SCALE, axis=0), SCALE, axis=1)
            h, w, _ = img.shape
            ppm = b"P6 %d %d 255 " % (w, h) + img.tobytes()
            self.photo = tk.PhotoImage(data=ppm)   # keep a reference
            self.canvas.itemconfig(self.img_id, image=self.photo)
        self._draw_overlay()

    def _draw_overlay(self):
        for oid in self.overlay:
            self.canvas.delete(oid)
        self.overlay.clear()
        g = self.tank.grid
        st = self.tank.stirrer
        cx, cy = (g.cx + 0.5) * SCALE, (g.cy + 0.5) * SCALE
        r_tank = g.radius_cells * SCALE
        self.overlay.append(self.canvas.create_oval(
            cx - r_tank, cy - r_tank, cx + r_tank, cy + r_tank,
            outline="#4c8dff", width=2))
        if self.tank.placed:
            r_imp = (st.d / 2.0) / g.dx * SCALE
            for b in range(2):
                a = st.theta + b * np.pi
                x2 = cx + np.cos(a) * r_imp
                y2 = cy + np.sin(a) * r_imp
                self.overlay.append(self.canvas.create_line(
                    cx, cy, x2, y2, fill="#e8edf5", width=4,
                    capstyle="round"))
            self.overlay.append(self.canvas.create_oval(
                cx - 5, cy - 5, cx + 5, cy + 5, fill="#e8edf5", outline=""))
        if self.tank.pour_queue_kg > 1e-6:
            py = (g.cy - g.radius_cells * 0.55) * SCALE
            self.overlay.append(self.canvas.create_line(
                cx, py - 60, cx, py, fill="#b9a689", width=5, dash=(3, 3)))
        if self.tank.settle_boost_active:
            self.overlay.append(self.canvas.create_text(
                cx, 16, text="⏩ bezinken ×200 (tijdversnelling)",
                fill="#8fa2bd", font=("Segoe UI", 10)))

    @staticmethod
    def _fmt_eta(eta):
        if eta < 0.1:
            return f"{eta*1000:.1f} mPa·s"
        if eta < 1000:
            return f"{eta:.2f} Pa·s"
        return f"{eta:.3g} Pa·s"

    def _update_readouts(self, snap):
        r = self.rows
        r["phi"].config(text=f"{snap['phi_pct']:.1f} vol%  "
                             f"({snap['w_pct']:.0f} massa-%)")
        r["eta"].config(text=self._fmt_eta(snap["eta_app_pa_s"]))
        r["tau"].config(text=f"{snap['tau_yield_pa']:.3g} Pa")
        r["rho"].config(text=f"{snap['rho_mix']:.0f} kg/m³")
        r["vol"].config(text=f"{snap['volume_l']:.1f} L")
        r["re"].config(text=f"{snap['re_impeller']:.3g} ({snap['regime']})")
        r["rpm"].config(text=f"{snap['rpm_actual']:.0f} / "
                             f"{snap['rpm_set']:.0f} ingesteld")
        r["torque"].config(text=f"{snap['torque_nm']:.2f} N·m")
        r["pshaft"].config(text=f"{snap['p_shaft_w']:.1f} W")
        r["pel"].config(text=f"{snap['p_electric_w']/1000:.3f} kW")
        r["kwh"].config(text=f"{snap['energy_kwh']*1000:.2f} Wh  "
                             f"(€ {snap['cost_eur']:.4f})")
        r["mix"].config(text=f"{snap['mixedness_pct']:.0f} %")
        r["settle"].config(text=f"{snap['settling_mm_h']:.1f} mm/h")
        r["fps"].config(text=f"{self._fps:.0f}")
        if snap["tripped"]:
            self.status.config(text="⚡ ZEKERING ERUIT — druk Reset zekering",
                               fg="#ff5c5c")
        elif snap["overload"]:
            self.status.config(text="⚠ OVERBELAST — motor blokkeert "
                                    "(slib te dik)", fg="#ffb454")
        elif not self.tank.stirrer.plugged_in:
            self.status.config(text="Stekker zit niet in het stopcontact",
                               fg="#8fa2bd")
        else:
            self.status.config(text="")

    def _update_level_readouts(self):
        lv = self.level
        if not lv.active:
            self.lvl_hint.config(text="Klik Start: was het silicium uit "
                                      "het slib (28+40 kHz)")
            self.lvl_stat.config(text="")
            return
        s = lv.snapshot()
        stars = "★" * s["stars"] + "☆" * (3 - s["stars"])
        self.lvl_stat.config(
            text=f"{stars}  {s['captured_pct']:.0f}% Si afgevangen · "
                 f"froth {s['froth_kg']:.2f} kg · cav {s['cav_eff']*100:.0f}% · "
                 f"{s['energy_kwh']*1000:.1f} Wh",
            fg="#38d39f" if s["stars"] >= 2 else "#dbe6f5")
        self.lvl_hint.config(text=s["hint"])

    # -------------------------------------------------------------- loop --
    def _tick(self):
        t0 = time.perf_counter()
        self.tank.step(1.0 / 30.0, mouse_splat=self.mouse_splat)
        self.level.step(1.0 / 30.0)
        self.mouse_splat = None
        self._frame = getattr(self, "_frame", 0) + 1
        if self._frame % 2 == 0:      # render at 15 Hz, physics at 30 Hz
            self._render()
        if self._frame % 4 == 0:      # readout labels at ~7 Hz
            self._update_readouts(self.tank.snapshot())
            self._update_level_readouts()
        dt = time.perf_counter() - self._fps_t
        self._fps_t = time.perf_counter()
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(dt, 1e-6))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.root.after(max(5, int(33 - elapsed_ms)), self._tick)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    root = tk.Tk()
    ViscosityApp(root, n=n)
    root.mainloop()


if __name__ == "__main__":
    main()
