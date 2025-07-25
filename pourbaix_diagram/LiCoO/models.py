import mlipx
import dataclasses
from ase.calculators.calculator import Calculator
import sys
from ase.units import Bohr
from copy import deepcopy


MODELS = {}

MODELS["mace-matpes-pbe-0"] = mlipx.GenericASECalculator(
    module="mace.calculators",
    class_name="MACECalculator",
    device="auto",
    kwargs={"model_paths": "/mnt/nfs/ml_training/pretrained_mace_models/MACE-matpes-pbe-omat-ft.model", "dtype": "float64"},
)

'''
# Example MLIP
MODELS["mace-agnesi"] = mlipx.GenericASECalculator(
    module="mace.calculators",
    class_name="MACECalculator",
    device="auto",
    kwargs={"model_paths": "/mnt/nfs/ml_training/pretrained_mace_models/mace_agnesi_medium.model", "dtype": "float64"},
)

MODELS["mace-mpa-0"] = mlipx.GenericASECalculator(
    module="mace.calculators",
    class_name="mace_mp",
    device="auto",
    kwargs={"model": "/mnt/nfs/ml_training/pretrained_mace_models/mace-mpa-0-medium.model", "dispersion":False, "dispersion_cutoff":40*Bohr, "default_dtype": "float64"},
)

MODELS["sevennet"] = mlipx.GenericASECalculator(
    module="sevenn.sevennet_calculator",
    class_name="SevenNetCalculator",
    device="auto",
    kwargs={"model": "7net-mf-ompa", "modal": 'mpa'},
)

MODELS["mattersim"] = mlipx.GenericASECalculator(
    module="mattersim.forcefield",
    class_name="MatterSimCalculator",
)

MODELS["ocp"] = mlipx.GenericASECalculator(
    module="fairchem.core",
    class_name="OCPCalculator",
    kwargs={
        "checkpoint_path": "/mnt/nfs/ml_training/pretrained_ocp_models/eqV2_31M_omat_mp_salex.pt",
        "seed" : 0
        },
)

MODELS["grace"] = mlipx.GenericASECalculator(
    module="tensorpotential.calculator",
    class_name="grace_fm",
    kwargs={
        "model":"GRACE-2L-OAM",
        }
)

@dataclasses.dataclass
class OrbCalc:
    name: str
    device: str | None = None

    def get_calculator(self, **kwargs):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        method = getattr(pretrained, self.name)
        if self.device is None:
            orbff = method(**kwargs)
            calc = ORBCalculator(orbff, **kwargs)
        elif self.device == "auto":
            orbff = method(device="auto", **kwargs)
            calc = ORBCalculator(orbff, device="auto", **kwargs)
        else:
            orbff = method(device=self.device, **kwargs)
            calc = ORBCalculator(orbff, device=self.device, **kwargs)
        return calc

MODELS["orb-v2"] = OrbCalc(name="orb_v2")

MODELS["pet-mad"] = mlipx.GenericASECalculator(
    module="pet_mad.calculator",
    class_name="PETMADCalculator",
)
'''

import numpy as np
from ase.calculators.calculator import Calculator, all_properties, all_changes
from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read, write

class CustomCalculator(Calculator):
    """Special calculator for a single configuration.

    Used to remember the energy, force and stress for a given
    configuration.  If the positions, atomic numbers, unit cell, or
    boundary conditions are changed, then asking for
    energy/forces/stress will raise an exception."""

    name = 'unknown'
    implemented_properties = ['energy', 'forces']

    def __init__(self, dataset_path, energy_key, identifier_key, **kwargs):
        """Save energy, forces, stress, ... for the current configuration."""
        Calculator.__init__(self)
        self.energy_key = energy_key
        self.identifier_key = identifier_key
        self.images = read(dataset_path, index=':')

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):

        self.atoms = deepcopy(atoms)
        tmp_atoms_list = [a for a in self.images if a.info[self.identifier_key] 
                          == self.atoms.info[self.identifier_key]]
        assert len(tmp_atoms_list) == 1
        tmp_atoms = tmp_atoms_list[0]
        for property in properties:
            assert property in all_properties
            if property == 'energy':
                self.results[property] = tmp_atoms.info[self.energy_key]
            elif property in ['magmom', 'free_energy']:
                self.results[property] = tmp_atoms.info[property]
            else:
                self.results[property] = tmp_atoms.arrays[property]

'''
@dataclasses.dataclass
class CustomCalc:
    name: str
    device: str | None = None

    def get_calculator(self, **kwargs):
        calc = CustomCalculator(dataset_path="/mnt/nfs/mlipx_paper/final_phase_diagram/LiCoO_schmidt_2021.xyz",
                                energy_key="DFT_energy", identifier_key="index")
        return calc

MODELS["dft"] = CustomCalc(name="dft")
'''

@dataclasses.dataclass
class CustomCalc:
    name: str
    device: str | None = None

    def get_calculator(self, **kwargs):
        calc = CustomCalculator(dataset_path="/mnt/nfs/mlipx_paper/final_pourbaix_diagram/LiCoO_schmidt_2021_mad_recomputed.xyz",
                                energy_key="DFT_energy", identifier_key="index")
        return calc

MODELS["dft-qe"] = CustomCalc(name="dft-qe")


# OPTIONAL
# ========
# If you have custom property names you can use the UpdatedFramesCalc
# to set the energy, force and isolated_energies keys mlipx expects.
REFERENCE = mlipx.UpdateFramesCalc(
        results_mapping={"energy": "DFT_energy"}#, "forces": "DFT_forces"},
#    info_mapping={mlipx.abc.ASEKeys.isolated_energies.value: "isolated_free_energies"},
)
