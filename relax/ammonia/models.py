import mlipx
import dataclasses
from ase.calculators.calculator import Calculator
import sys

MODELS = {}

# Example MLIP
#MODELS["mace_medium"] = mlipx.GenericASECalculator(
#    module="mace.calculators",
#    class_name="MACECalculator",
#    device="auto",
#    kwargs={"model_paths": "/mnt/nfs/ml_training/pretrained_mace_models/y7uhwpje-medium.model", "dtype": "float64"},
#)
#MODELS["mace_large"] = mlipx.GenericASECalculator(
#    module="mace.calculators",
#    class_name="MACECalculator",
#    device="auto",
#    kwargs={"model_paths": "/mnt/nfs/ml_training/pretrained_mace_models/43117273-large.model", "dtype": "float64"},
#)
#MODELS["mace_agnesi"] = mlipx.GenericASECalculator(
#    module="mace.calculators",
#    class_name="MACECalculator",
#    device="auto",
#    kwargs={"model_paths": "/mnt/nfs/ml_training/pretrained_mace_models/mace_agnesi_medium.model", "dtype": "float64"},
#)

MODELS["grace"] = mlipx.GenericASECalculator(
    module="tensorpotential.calculator",
    class_name="grace_fm",
    kwargs={
        "model":"GRACE-2L-OAM",
        }
)

MODELS["mace_mpa0"] = mlipx.GenericASECalculator(
    module="mace.calculators",
    class_name="mace_mp",
    device="auto",
    kwargs={"model": "/mnt/nfs/ml_training/pretrained_mace_models/mace-mpa-0-medium.model", "dtype": "float64"},
)

#MODELS["sevennet"] = mlipx.GenericASECalculator(
#    module="sevenn.sevennet_calculator",
#    class_name="SevenNetCalculator",
#    device="auto",
#    kwargs={"model": "7net-mf-ompa", "modal": 'mpa'},
#)
#
#MODELS["mattersim"] = mlipx.GenericASECalculator(
#    module="mattersim.forcefield",
#    class_name="MatterSimCalculator",
#    device="auto",
#)
#
#MODELS["ocp"] = mlipx.GenericASECalculator(
#    module="fairchem.core",
#    class_name="OCPCalculator",
#    kwargs={
#        "checkpoint_path": "/mnt/nfs/ml_training/pretrained_ocp_models/eqV2_31M_omat_mp_salex.pt",
#        "seed" : 0
#        },
#)

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

#MODELS["orb_v2"] = OrbCalc(name="orb_v2", device="cpu")


# OPTIONAL
# ========
# If you have custom property names you can use the UpdatedFramesCalc
# to set the energy, force and isolated_energies keys mlipx expects.
REFERENCE = mlipx.UpdateFramesCalc(
        results_mapping={"energy": "DFT_energy", "forces": "DFT_forces"},
#    info_mapping={mlipx.abc.ASEKeys.isolated_energies.value: "isolated_free_energies"},
)
