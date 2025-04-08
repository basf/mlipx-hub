import dataclasses

import mlipx
from mlipx.nodes.generic_ase import Device

MODELS = {}


# https://github.com/ACEsuit/mace
MODELS["mace-mp"] = mlipx.GenericASECalculator(
    module="mace.calculators",
    class_name="mace_mp",
    device="auto",
    kwargs={"model": "../../data/mace-mpa-0-medium.model"},
)

# https://github.com/MDIL-SNU/SevenNet
MODELS["sevennet"] = mlipx.GenericASECalculator(
    module="sevenn.sevennet_calculator",
    class_name="SevenNetCalculator",
    device="auto",
    kwargs={"model": "7net-0"},
)

# https://github.com/CederGroupHub/chgnet
MODELS["chgnet"] = mlipx.GenericASECalculator(
    module="chgnet.model",
    class_name="CHGNetCalculator",
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


MODELS["orb_v2"] = OrbCalc(name="orb_v2", device="auto")

# https://github.com/microsoft/mattersim
MODELS["mattersim"] = mlipx.GenericASECalculator(
    module="mattersim.forcefield",
    class_name="MatterSimCalculator",
    device="auto",
)
