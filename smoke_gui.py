"""4-second self-closing GUI smoke test: starts the app with the stirrer
running, screenshots the window (PIL ImageGrab, X11) and reports FPS.
Run: python3 smoke_gui.py
"""

import os
import tkinter as tk

from viscosity_gui import ViscosityApp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
os.makedirs(OUT, exist_ok=True)


def main():
    root = tk.Tk()
    app = ViscosityApp(root, n=80)
    app.tank.stirrer.plugged_in = True
    app.tank.stirrer.on = True
    app.b_plug.config(text="Stekker: in")
    app.b_on.config(text="Roerder: AAN")

    def finish():
        ok = True
        try:
            from PIL import ImageGrab
            root.update_idletasks()
            x, y = root.winfo_rootx(), root.winfo_rooty()
            w, h = root.winfo_width(), root.winfo_height()
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            img.save(os.path.join(OUT, "viscosity_gui_screenshot.png"))
            print("screenshot saved")
        except Exception as exc:  # screenshot is best-effort evidence
            ok = False
            print(f"screenshot failed: {exc}")
        snap = app.tank.snapshot()
        print(f"GUI smoke: fps={app._fps:.0f} mix={snap['mixedness_pct']:.0f}% "
              f"P={snap['p_electric_w']:.0f}W rpm={snap['rpm_actual']:.0f}")
        root.destroy()
        raise SystemExit(0 if ok else 2)

    root.after(4000, finish)
    root.mainloop()


if __name__ == "__main__":
    main()
