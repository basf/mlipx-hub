import dataclasses

import mlipx
from mlipx.nodes.generic_ase import Device

MODELS = {}



# https://github.com/ACEsuit/mace
MODELS["mace_mp"] = mlipx.GenericASECalculator(
    module="mace.calculators",
    class_name="mace_mp",
    device="auto",
    kwargs={"model": "medium"}
)



# https://github.com/MDIL-SNU/SevenNet
MODELS["sevennet"] = mlipx.GenericASECalculator(
    module="sevenn.sevennet_calculator",
    class_name="SevenNetCalculator",
    device="auto",
    kwargs={"model": "7net-0"}
)



# https://github.com/orbital-materials/orb-models
@dataclasses.dataclass
class OrbCalc:
    name: str
    device: Device | None = None

    def get_calculator(self, **kwargs):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        method = getattr(pretrained, self.name)
        if self.device is None:
            orbff = method(**kwargs)
            calc = ORBCalculator(orbff, **kwargs)
        elif self.device == Device.AUTO:
            orbff = method(device=Device.resolve_auto(), **kwargs)
            calc = ORBCalculator(orbff, device=Device.resolve_auto(), **kwargs)
        else:
            orbff = method(device=self.device, **kwargs)
            calc = ORBCalculator(orbff, device=self.device, **kwargs)
        return calc

MODELS["orb_v2"] = OrbCalc(
    name="orb_v2",
    device="auto"
)



# https://github.com/microsoft/mattersim
MODELS["mattersim"] = mlipx.GenericASECalculator(
    module="mattersim.forcefield",
    class_name="MatterSimCalculator",
    device="auto",
)



# https://www.faccts.de/orca/
MODELS["orca"] = mlipx.OrcaSinglePoint(
    orcasimpleinput= "PBE def2-TZVP TightSCF EnGrad",
    orcablocks ="%pal nprocs 8 end",
    orca_shell="/dlls/tools/orca_5_0_4/orca",
)




# OPTIONAL
# ========
# If you have custom property names you can use the UpdatedFramesCalc
# to set the energy, force and isolated_energies keys mlipx expects.

# REFERENCE = mlipx.UpdateFramesCalc(
#     results_mapping={"energy": "DFT_ENERGY", "forces": "DFT_FORCES"},
#     info_mapping={mlipx.abc.ASEKeys.isolated_energies.value: "isol_ene"},
# )