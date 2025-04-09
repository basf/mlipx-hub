import dataclasses

import mlipx
from mlipx.nodes.generic_ase import Device

MODELS = {}



# https://github.com/ACEsuit/mace
MODELS["MACE-MPA-0"] = mlipx.GenericASECalculator(
    module="mace.calculators",
    class_name="mace_mp",
    device="auto",
    kwargs={"model": "../../models/mace-mpa-0-medium.model"}
)



# https://github.com/MDIL-SNU/SevenNet
MODELS["7net-0"] = mlipx.GenericASECalculator(
    module="sevenn.sevennet_calculator",
    class_name="SevenNetCalculator",
    device="auto",
    kwargs={"model": "7net-0"}
)

MODELS["7net-mf-ompa-mpa"] = mlipx.GenericASECalculator(
    module="sevenn.sevennet_calculator",
    class_name="SevenNetCalculator",
    device="auto",
    kwargs={"model": "7net-mf-ompa", "modal": "mpa"}
)


# https://github.com/orbital-materials/orb-models
@dataclasses.dataclass
class OrbCalc:
    name: str
    device: Device | None = None
    kwargs: dict = dataclasses.field(default_factory=dict)

    def get_calculator(self, **kwargs):
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        method = getattr(pretrained, self.name)
        if self.device is None:
            orbff = method(**self.kwargs)
            calc = ORBCalculator(orbff, **self.kwargs)
        elif self.device == Device.AUTO:
            orbff = method(device=Device.resolve_auto(), **self.kwargs)
            calc = ORBCalculator(orbff, device=Device.resolve_auto(), **self.kwargs)
        else:
            orbff = method(device=self.device, **self.kwargs)
            calc = ORBCalculator(orbff, device=self.device, **self.kwargs)
        return calc

MODELS["orb-v2"] = OrbCalc(
    name="orb_v2",
    device="auto"
)

MODELS["orb-v3"] = OrbCalc(
    name="orb_v3_conservative_inf_omat",
    device="auto"
)


# https://github.com/microsoft/mattersim
MODELS["mattersim"] = mlipx.GenericASECalculator(
    module="mattersim.forcefield",
    class_name="MatterSimCalculator",
    device="auto",
)

# https://github.com/microsoft/mattersim
MODELS["GRACE-2L-OMAT"] = mlipx.GenericASECalculator(
    module="tensorpotential.calculator",
    class_name="TPCalculator",
    device=None,
    kwargs={
        "model": "../../models/GRACE-2L-OMAT",
    },
)


# https://www.faccts.de/orca/
MODELS["orca"] = mlipx.OrcaSinglePoint(
    orcasimpleinput= "PBE def2-TZVP TightSCF EnGrad",
    orcablocks ="%pal nprocs 8 end"
)




# OPTIONAL
# ========
# If you have custom property names you can use the UpdatedFramesCalc
# to set the energy, force and isolated_energies keys mlipx expects.

# REFERENCE = mlipx.UpdateFramesCalc(
#     results_mapping={"energy": "DFT_ENERGY", "forces": "DFT_FORCES"},
#     info_mapping={mlipx.abc.ASEKeys.isolated_energies.value: "isol_ene"},
# )