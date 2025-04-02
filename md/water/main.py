import mlipx
from models import MODELS

project = mlipx.Project()

with project.group("initialize"):
    water = mlipx.Smiles2Conformers(
        smiles="O",
        num_confs=100,
    )
    box = mlipx.BuildBox(
        data=[water.frames],
        counts=[64],
        density=997,
    )

thermostat = mlipx.LangevinConfig(timestep=0.5, temperature=400, friction=0.05)

for model_name, model in MODELS.items():
    with project.group(model_name, "relax"):
        struct_opt = mlipx.StructureOptimization(data=box.frames, model=model, fmax=0.1)
    with project.group(model_name, "md"):
        md = mlipx.MolecularDynamics(
                model=model,
                thermostat=thermostat,
                data=struct_opt.frames,
                steps=100_000,
            )
        rdf = mlipx.RadialDistributionFunction(
            data=md.frames,
            species=("O", "O"),
        )

project.build()
