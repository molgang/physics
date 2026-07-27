"""Headless proof render for the viscosity lab (no GUI needed).

Runs three stirring scenarios (thin / thick / locked slurry) and writes:
  media/viscosity_lab_report.png   4x3 field snapshots + kW & mixedness curves
  media/viscosity_lab_10pct.gif    animated stir of the 10 wt% case
  media/viscosity_lab_metrics.json scenario end-state metrics

Run: python3 demo_headless.py
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from viscosity_core import MixingTank

WATER_RGB = np.array([22.0, 56.0, 112.0])
SLIB_RGB = np.array([150.0, 133.0, 108.0])
BG_RGB = np.array([11.0, 14.0, 19.0])
PHI_VISUAL_FULL = 0.45

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
os.makedirs(OUT, exist_ok=True)


def field_rgb(tank):
    g = tank.grid
    t = np.clip(g.c / PHI_VISUAL_FULL, 0.0, 1.0) ** 0.7
    rgb = (WATER_RGB[None, None, :] * (1 - t[..., None])
           + SLIB_RGB[None, None, :] * t[..., None])
    rgb += np.clip(np.hypot(g.u, g.v) * 45.0, 0, 28)[..., None]
    rgb[~g.liquid] = BG_RGB
    return np.clip(rgb, 0, 255).astype(np.uint8)


def run_scenario(w_pct, rpm=400.0, seconds=12.0, snap_at=(0.0, 2.0, 5.0, 12.0),
                 gif=False):
    tank = MixingTank(n=80, water_l=14.0, w_pct=w_pct)
    tank.settle_bottom()
    tank.stirrer.plugged_in = True
    tank.stirrer.on = True
    tank.stirrer.rpm_set = rpm
    dt = 1.0 / 30.0
    frames = int(seconds / dt)
    snaps, curves, gif_frames = {}, [], []
    snap_frames = {int(s / dt) for s in snap_at}
    t0 = time.perf_counter()
    for i in range(frames + 1):
        if i in snap_frames:
            snaps[round(i * dt, 1)] = field_rgb(tank)
        if gif and i % 3 == 0:
            gif_frames.append(field_rgb(tank))
        s = tank.snapshot()
        curves.append((i * dt, s["p_electric_w"], s["mixedness_pct"],
                       s["rpm_actual"]))
        tank.step(dt)
    wall = time.perf_counter() - t0
    end = tank.snapshot()
    end["sim_fps"] = frames / wall
    return tank, snaps, np.array(curves), gif_frames, end


def main():
    scen = [(10.0, "10% slib — dun, mengt vlot"),
            (60.0, "60% slib — dik, trager"),
            (88.0, "88% slib — pasta: blokkeert & zekering")]
    results = []
    fig, axes = plt.subplots(3, 5, figsize=(16, 9.5), facecolor="#0b0e13")
    metrics = {}
    for row, (w, title) in enumerate(scen):
        tank, snaps, curves, gif_frames, end = run_scenario(
            w, gif=(row == 0))
        results.append((snaps, curves))
        metrics[f"{w:.0f}pct"] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in end.items()}
        for col, (t_s, img) in enumerate(sorted(snaps.items())[:4]):
            ax = axes[row][col]
            ax.imshow(img)
            ax.set_title(f"t = {t_s:.0f} s", color="#dbe6f5", fontsize=9)
            ax.axis("off")
        ax = axes[row][4]
        ax.set_facecolor("#10151d")
        ax.plot(curves[:, 0], curves[:, 1], color="#ffb454",
                label="P stopcontact (W)")
        ax2 = ax.twinx()
        ax2.plot(curves[:, 0], curves[:, 2], color="#4c8dff",
                 label="menggraad (%)")
        ax2.set_ylim(0, 105)
        ax.set_xlabel("tijd (s)", color="#8fa2bd", fontsize=8)
        ax.tick_params(colors="#8fa2bd", labelsize=7)
        ax2.tick_params(colors="#4c8dff", labelsize=7)
        ax.set_title("vermogen & menggraad", color="#dbe6f5", fontsize=9)
        for spine in list(ax.spines.values()) + list(ax2.spines.values()):
            spine.set_color("#263247")
        axes[row][0].set_ylabel(title, color="#dbe6f5", fontsize=10)
        axes[row][0].axis("on")
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])
        for spine in axes[row][0].spines.values():
            spine.set_color("#0b0e13")
        if row == 0 and gif_frames:
            imgs = [Image.fromarray(np.repeat(np.repeat(f, 4, 0), 4, 1))
                    for f in gif_frames]
            imgs[0].save(os.path.join(OUT, "viscosity_lab_10pct.gif"),
                         save_all=True, append_images=imgs[1:],
                         duration=100, loop=0)
    fig.suptitle("MOLGANG viscositeitslab — roeren in staalslak-slib "
                 "(5 µm), 400 RPM", color="#dbe6f5", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "viscosity_lab_report.png"), dpi=110,
                facecolor="#0b0e13")
    with open(os.path.join(OUT, "viscosity_lab_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    for name, m in metrics.items():
        print(f"{name}: mix={m['mixedness_pct']:.0f}% "
              f"P={m['p_electric_w']:.0f}W rpm={m['rpm_actual']:.0f} "
              f"eta={m['eta_app_pa_s']:.3g} Pa.s tripped={m['tripped']} "
              f"simfps={m['sim_fps']:.0f}")


if __name__ == "__main__":
    main()
