"""MOLGANG viscosity lab: stirring dynamics of steel-slag slurry (5 um).

Public API (pure Python + numpy; GUI is tkinter, stdlib):
  SlurryRheology  - wt% -> volume fraction, viscosity, yield stress, settling
  FluidGrid2D     - masked Stam stable-fluids solver (circular tank)
  Stirrer         - torque-limited mains stirrer + power correlation
  PowerMeter      - kW / kWh / EUR wall-socket meter
  MixingTank      - the tank; owns solver, stirrer, meter; AR verb methods
"""

from .viscosity_core import (FluidGrid2D, MixingTank, PowerMeter,
                             SlurryRheology, Stirrer)

__all__ = ["FluidGrid2D", "MixingTank", "PowerMeter", "SlurryRheology",
           "Stirrer"]
