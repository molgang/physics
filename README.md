> **Herkomst:** geëxtraheerd uit `febuz/molgang-web` se `simulation/viscosity_lab/`
> (geschiedenis behouden via `git filter-repo`) als losstaande fysica-autoriteit,
> zodat `molgang/webapp` en `molgang/world` 'm als git-submodule kunnen pinnen
> zonder de handelsapp mee te slepen. Zie de MOLGANG-symbiosethesis voor de
> volledige architectuurmotivatie.

# MOLGANG Viscositeitslab

Interactieve simulator van **roeren in staalslak-slib** (5 µm deeltjes,
1–100 massa-% vaste stof) in een cilindrische bak, bovenaanzicht. Pure
**Python** (numpy + tkinter/stdlib) — bewust géén JavaScript: dit is de
referentie-implementatie waar de 3D/AR-versie (Meta Quest 3S
"viscositeitsruimte") later in parity op poort.

```
python3 viscosity_gui.py          # GUI (slider slib-%, roerintensiteit)
python3 viscosity_gui.py 64       # kleiner grid = hogere FPS
python3 test_viscosity_core.py    # proof-suite (83 checks, exit 0 = pass)
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
- **Temperatuur (0-D warmtebalans):** as-vermogen dissipeert als viskeuze
  warmte in de slurry (motorverliezen blijven in de behuizing), wandverlies
  is Newton-koeling (UA ≈ 2,3 W/K). Temperatuur terugkoppelt via een
  Arrhenius-wet op de vloeistofviscositeit (E/R = 2400 K, verankerd op
  20 °C): warme slurry is dunner en bezinkt sneller; tijdconstanten zijn
  realistisch uren, de bad drift dus, hij springt niet.
- **Thixotropie (opt-in, `MixingTank(thixotropy=True)`):**
  structuurparameter λ ∈ [0,1] met dλ/dt = (1−λ)/τ_b − k_b·γ̇·λ
  (τ_b = 45 s); een opgebouwd gel vermenigvuldigt de héle vloeikurve met
  (1 + C_thix·λ), C_thix = 0,8. Off by default: het cavitatie-η-venster
  van level 1 verschuift niet (parity bewaakt met een bit-exacte check).
- **Vortex-trechter:** centrale dip uit het werkelijke snelheidsveld via
  radiaal evenwicht (ω_eff² R²/2g, ω_eff = Σu_t·r/Σr²), afgekapt op de
  vloeistofdiepte. `air_entrainment` gebruikt óók het Froude-criterium
  Fr = N²D/g van de échte roertoestand (> 1 ⇒ lucht inslag), want de
  visuele veldsnelheid is op u_cap gebonden en kan een echte trechter
  alleen onderschatten.
- **Heropwerveling:** de tegenpool van bezinken. Boven een kritische
  stroomsnelheid (u_crit = 0,08 m/s) erodeert de schuif de geconcentreerde
  bodemlaag terug omhoog (CFL-gelimiteerde opwaartse flux, packing-cap op
  de ontvangende cel); eronder is het veld exact onaangeroerd.
- **Verdamping (opt-in, `MixingTank(evaporation=True)`, volgt de
  `thermal`-schakelaar):** open bad verliest zuiver water op het
  dampdrukdeficit tegen de werkplaatslucht (Magnus, 50% RV, natuurlijke-
  convectie massa-overdracht h_m = 0,007 m/s). De latente warmte
  (2,45 MJ/kg) gaat uit de balans — een heet bad koelt door verdamping
  aantoonbaar harder dan door de wand — en waterverlies dikkt de slurry
  op via de gewone boekhouding (massabalans blijft gesloten).

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
| `tank.temperature_c` | thermometer op de bakwand + dunner-worden bij warmte |
| `snapshot().vortex_dip_mm` / `air_entrainment` | zichtbare trechter in de vloeispiegel; waarschuwing bij lucht inslag |
| `tank.thix_on` / `struct_lambda` | schakelaar "thixotroop slib": rust herstelt de dikte |
| `tank.evaporation` / `snapshot().evap_rate_kg_h` | open deksel: zichtbare damp + langzaam dikker worden |
| `snapshot()` | HUD-readouts (η, Re, regime, menggraad …) |

De proof-suite (`test_viscosity_core.py`) is het parity-contract voor de
port, zoals `lab3d/chemistry.test.mjs` dat is voor de chemie-engine.

## Level 2 — Kreken & rivier-stroomdynamica (mijnbouw-aanvoer)

`river_flow.py` + `test_river_flow.py` (18 checks). MOLGANG-016:
kreken uit de bergen voeden de rivier bij elke staalfabriek-site met
zichtbare stroomdynamica/viscositeit, en voeren erts-houdend sediment aan
voor latere mijnbouw. Hergebruikt bewust `SlurryRheology`
(Krieger-Dougherty + Richardson-Zaki bezinken) uit `viscosity_core.py` —
geen tweede fysica-stack.

- **Kanaalroutering, geen tank-solver:** een rivier is overwegend
  eenrichtingsstroming over kilometers, geen geroerde tank. Daarom een
  1D dieptegemiddelde kinematische-golf-oplosser (Manning-vergelijking,
  expliciet, CFL-substepped) langs de al bestaande OSM-rivierpolylijn
  per site, in plaats van de 2D Stam-solver van het viscositeitslab —
  een bewuste, gedocumenteerde keuze (de juiste vereenvoudiging voor
  een riviertak, geen shortcut).
- **Viskeuze koppeling:** schijnbare viscositeit (via de bestaande
  rheologie-klasse) verhoogt de effectieve Manning-ruwheid — dikkere
  modder stroomt aantoonbaar trager bij gelijke afvoer.
- **Placer-depositie:** erts bezinkt (Richardson-Zaki, hergebruikt)
  vloeiend sneller waar de stroming vertraagt (bv. waar het kanaal
  verbreedt) — geen harde snelheidsdrempel (die zou het aanvoermechanisme
  na verloop van tijd laten "bevriezen" zodra de rivier een stabiele
  snelle stroomsnelheid bereikt, ontdekt tijdens het bouwen van de
  proof-suite en gefixt naar een vloeiende exponentiële demping).
- **Gedeelde economie:** `RawMaterialSupply.collect()` betaalt in
  hetzelfde MolCoin-schema als V2O5-verkoop/oogst (kg × €/kg) — geen
  nieuwe, losstaande valuta (zelfde les als bij het tuintje).

## AR-mapping (kreken/rivier)

| Python-API | AR-interactie |
|---|---|
| `RiverChannel(pts_m, width_m)` | de al bestaande OSM-rivier per site, nu stromend |
| `add_creek(Q, phi, at_fraction)` | een kreek die uit de bergen instroomt |
| `step(dt)` | live stroomdynamica + viscositeit in de rivier |
| `deposit_at(fraction)` | zichtbare erts-ophoping (zandbank) |
| `RawMaterialSupply.collect()` | mijnbouw-actie: erts → MolCoins |

## Digital twin: thermisch model + CAD-brug (`cad_twin`)

Brug tussen deze fysica-autoriteit en de 3D/CAD-wereld (Astra-apparatuur,
SolidWorks-interoperabiliteit) — bewust dependency-arm:

- **`ThermalTwin`** — transiënt axisymmetrisch (r, z) warmtemodel van de roerbak:
  anisotroop FDM-rooster in de slurry, wand als per-hoogte weerstandsketen
  (wandgeleiding + vrije luchtconvectie, of jacket-koeling over de mantelhoogte),
  warmtebronnen op de fysisch juiste plek (impeller-sweep, 28/40 kHz-ring).
  Bewaakte invarianten: energieboekhouding sluit, auto-substepping is
  consistent, jacket trekt het evenwicht omlaag, en de handmatige
  cilinderwand-formule komt uit het rooster terug.
  `python3 test_cad_twin.py` (24 checks, exit 0 = pass).
- **`export_step`** — OPTIONELE FreeCAD-brug: bouwt bak + jacket + roeras +
  impeller + transducentring parametrisch en schrijft **STEP** (SolidWorks
  opent dit native, en andersom importeert FreeCAD STEP uit SolidWorks).
  `freecadcmd demo_cad_twin_step.py` — FreeCAD is níet nodig voor het
  thermische model.
