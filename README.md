# MOLGANG Viscositeitslab

Interactieve simulator van **roeren in staalslak-slib** (5 µm deeltjes,
1–100 massa-% vaste stof) in een cilindrische bak, bovenaanzicht. Pure
**Python** (numpy + tkinter/stdlib) — bewust géén JavaScript: dit is de
referentie-implementatie waar de 3D/AR-versie (Meta Quest 3S
"viscositeitsruimte") later in parity op poort.

```
python3 viscosity_gui.py          # GUI (slider slib-%, roerintensiteit)
python3 viscosity_gui.py 64       # kleiner grid = hogere FPS
python3 test_viscosity_core.py    # proof-suite (47 checks, exit 0 = pass)
python3 demo_headless.py          # PNG/GIF/JSON bewijs zonder GUI
python3 smoke_gui.py              # 4 s zelfsluitende GUI-smoketest
```

## Vooronderzoek: bestaande GitHub-packages

Geen enkel pakket combineert interactief roeren + instelbare
slurry-rheologie + GUI in Python; daarom een eigen compacte module.

| Package | Oordeel |
|---|---|
| [GregTJ/stable-fluids](https://github.com/GregTJ/stable-fluids) | Minimale NumPy Stable Fluids — bevestigt de solverkeuze, geen rheologie/GUI/roerder |
| [PySPH (pypr)](https://github.com/pypr/pysph) | Volwaardig SPH-framework (Cython/OpenCL) — zwaar, geen interactieve GUI, hoge viscositeit lastig stabiel |
| [Taichi stable_fluid.py](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py) | GPU-DSL, extra dependency + kernelcompilatie — overkill voor dit doel |
| [PhiFlow (tum-pbs)](https://github.com/tum-pbs/PhiFlow) | Differentieerbaar research-framework — zware dependency-keten |
| [AlexandreSajus/Python-Fluid-Simulation](https://github.com/AlexandreSajus/Python-Fluid-Simulation), [taehoon-yoon/SPH-Fluid-Simulation](https://github.com/taehoon-yoon/SPH-Fluid-Simulation) | Demo-SPH; expliciete integratie klapt om bij pasta-viscositeit |
| [PavelDoGreat/WebGL-Fluid-Simulation](https://github.com/PavelDoGreat/WebGL-Fluid-Simulation) | JS/GPU-referentie voor de látere AR-port — let op: heeft geen fysische viscositeitsparameter |
| [mjwatkins2/WebGL-SPH](https://github.com/mjwatkins2/WebGL-SPH), [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH) | JS/C++ referenties (SPlisHSPlasH voor hoge-viscositeitsmethoden) |

**Keuze:** Jos Stam, *Real-Time Fluid Dynamics for Games* (GDC 2003).
De impliciete (Gauss-Seidel) diffusie is onvoorwaardelijk stabiel, zodat
één solver het hele bereik dekt: water (1 mPa·s) → slibpasta (>10³ Pa·s).

## Fysica

- **Samenstelling:** massa-% → volumefractie φ via mengregel
  (ρ_slak = 3400, ρ_water = 1000 kg/m³). 100 massa-% ⇒ φ > φ_max: pasta.
- **Viscositeit:** Krieger–Dougherty η(φ) = η_w·(1−φ/φ_max)^(−[η]·φ_max),
  φ_max = 0,58, [η] = 2,5; plus **zwichtspanning** τ_y(φ) boven φ = 0,20
  (Bingham). Schijnbare viscositeit op Metzner–Otto-afschuifsnelheid
  γ̇ = 11·N.
- **Roerder:** vermogensgetal Np(Re) = Kp/Re + Np_t·Re/(Re+300)
  (Kp = 70, Np_t = 1,5); P_as = Np·ρ·N³·D⁵; koppelbegrensde motor met
  RPM-droop, **stall-stroom 300 W** en thermische zekering na 5 s
  overbelasting. P_stopcontact = P_as/0,65 + 5 W standby → kWh-meter + €.
- **Bezinken:** Stokes voor 5 µm (~110 mm/h in water), Richardson–Zaki
  gehinderd, gel-onderdrukt boven de zwichtgrens; ×200 timelapse zodra de
  vloeistof stilstaat. Bodemwrijvingsdemping (0,3 s⁻¹) vervangt de
  ontbrekende bodemgrenslaag in het 2D-bovenaanzicht.
- **Solver:** 2D stable fluids op gemaskerd cirkelrooster (Neumann-wanden,
  gespiegelde drukgradiënt), massa-conservatieve slib-advectie
  (renormalisatie, packing-cap 0,62), 2-blads impellerforcering met
  CFL-substeps.

Bekende beperking: collocated 2dx-stencils zien checkerboard-divergentie
niet (standaard Stam-artefact); irrelevant voor gladde velden, zie test 2.

## Level 1 — Si-afvloei (ultrasound 28 + 40 kHz)

`ultrasound.py` + `test_ultrasound.py` (24 checks). De bak krijgt een
transducer-ring met een tweekanaals generator (28/40 kHz). Doel: de
Si-rijke lichte fractie (15% van de vaste stof) uit het slib laten
**afvloeien** over de overlooprand.

- **Cavitatie-poort:** het level start op 62 massa-% (φ > 0,30) — te dik;
  cavitatie is dood tot de speler met de slang verdunt
  (η-venster ≤ ~5 mPa·s vol effect, dood ≥ 50 mPa·s; solids-demping tot
  φ = 0,30). Dit zijn de "juiste viscositeitsparameters".
- **Twee frequenties nodig:** 28 kHz de-aggregeert de grove clusters,
  40 kHz de fijne (10% kruiseffect) — één kanaal plateaut, dual bevrijdt
  >1,25× het beste enkele kanaal (bewezen in de suite).
- **Sono-flotatie:** cavitatiebellen hechten aan vrije lichte deeltjes →
  froth-laag. Zacht roeren (≤ ~120 RPM) voert de froth aan; hard roeren
  onthecht de bellen (detachment) en mengt de froth terug.
- **Afvloeien:** froth verlaat de bak boven de overlooprand (92% van
  20 L) — bijvullen met de slang is deel van de puzzel. Afgevangen massa
  wordt afgeboekt op de slak-boekhouding (massabalans gesloten).
- **Score:** ★ ≥ 40% · ★★ ≥ 60% · ★★★ ≥ 75% én ≤ 0,040 kWh (generator
  180 + 150 W loopt via de stopcontact-meter). Correct spel haalt ~95%
  in 6 min ≈ 33 Wh (3★).

## AR-mapping (Quest 3S viscositeitsruimte)

| Python-API | AR-interactie |
|---|---|
| `MixingTank(placed=...)` | bak pakken en neerzetten |
| `tank.add_water(L)` | slang: vocht toevoegen |
| `tank.add_slag(kg)` + `pour_queue` | storten uit de tweede bak (valstroom-visual) |
| `tank.set_composition(w%)` | lab-slider (GUI-modus) |
| `stirrer.plugged_in / on` | stekker in stopcontact, aan/uit |
| `stirrer.rpm_set` → `rpm_actual` | roerintensiteit, droop bij dik slib |
| `PowerMeter` (`p_watt`, `energy_kwh`) | kW-meting aan het stopcontact |
| `grid.c` (φ-veld) + `u,v` | 3D-textuur/particles van het mengsel |
| `snapshot()` | HUD-readouts (η, Re, regime, menggraad …) |

De proof-suite (`test_viscosity_core.py`) is het parity-contract voor de
port, zoals `lab3d/chemistry.test.mjs` dat is voor de chemie-engine.
