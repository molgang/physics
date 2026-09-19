"""MOLGANG viscosity lab: stirring dynamics of steel-slag slurry (5 um).

Public API (pure Python + numpy; GUI is tkinter, stdlib):
  SlurryRheology  - wt% -> volume fraction, viscosity, yield stress, settling
  FluidGrid2D     - masked Stam stable-fluids solver (circular tank)
  Stirrer         - torque-limited mains stirrer + power correlation
  PowerMeter      - kW / kWh / EUR wall-socket meter
  MixingTank      - the tank; owns solver, stirrer, meter; AR verb methods
"""

from .cad_twin import EquipmentSpec, ThermalTwin
from .downstream import (BasketCentrifuge, ElectrodialysisStack,
                         ElectrolysisCell, Nafion117, ProductionScale,
                         VacuumFiltration, steel_slag_cases)
from .ultrasound import SiliconWashLevel, UltrasoundBath
from .viscosity_core import (FluidGrid2D, MixingTank, PowerMeter,
                             SlurryRheology, Stirrer)

__all__ = ["BasketCentrifuge", "ElectrodialysisStack", "ElectrolysisCell",
           "EquipmentSpec", "FluidGrid2D", "MixingTank", "Nafion117",
           "PowerMeter", "ProductionScale", "SiliconWashLevel",
           "SlurryRheology", "Stirrer", "ThermalTwin", "UltrasoundBath",
           "VacuumFiltration", "steel_slag_cases"]
