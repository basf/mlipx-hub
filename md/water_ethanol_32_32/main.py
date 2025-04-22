import mlipx
from models import MODELS
import znmdakit

project = mlipx.Project()

with project.group("initialize"):
    water = mlipx.Smiles2Conformers(
        smiles="O",
        num_confs=100,
    )
    etoh = mlipx.Smiles2Conformers(
        smiles="CCO",
        num_confs=100,
    )
    box = mlipx.BuildBox(
        data=[water.frames, etoh.frames],
        counts=[32, 32],
        density=910,
        # https://www.engineeringtoolbox.com/ethanol-water-mixture-density-d_2162.html
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

        universe = znmdakit.Universe(
            data=md.frames,
            residues={"H2O": "O", "Eta": "CCO"},
        )

        znmdakit.InterRDF(
            universe=universe.universe,
            g1="name H",
            g2="name H",
            nbins=1000,
            apply_com_transform=False,
        )

        znmdakit.InterRDF(
            universe=universe.universe,
            g1="name O",
            g2="name H",
            nbins=1000,
            apply_com_transform=False,
        )

        znmdakit.InterRDF(
            universe=universe.universe,
            g1="name O",
            g2="name O",
            nbins=1000,
            apply_com_transform=False,
        )

        znmdakit.InterRDF(
            universe=universe.universe,
            g1="name COM and resname H2O",
            g2="name COM and resname H2O",
            nbins=1000,
            apply_com_transform=True,
        )
        znmdakit.InterRDF(
            universe=universe.universe,
            g1="name COM and resname H2O",
            g2="name COM and resname Eta",
            nbins=1000,
            apply_com_transform=True,
        )

        msd = znmdakit.EinsteinMSD(
            universe=universe.universe,
            select="name COM and resname H2O",
            timestep=0.0005, # ps
            sampling_rate=1, # ps
            apply_com_transform=True,
        )

        znmdakit.SelfDiffusionFromMSD(
            data=msd,
            start_time=5.7,
            end_time=40,
        )
        znmdakit.InterRDF(
            universe=universe.universe,
            g1="name COM and resname Eta",
            g2="name COM and resname Eta",
            nbins=1000,
            apply_com_transform=True,
        )

        msd = znmdakit.EinsteinMSD(
            universe=universe.universe,
            select="name COM and resname Eta",
            timestep=0.0005, # ps
            sampling_rate=1, # ps
            apply_com_transform=True,
        )

        znmdakit.SelfDiffusionFromMSD(
            data=msd,
            start_time=5.7,
            end_time=40,
        )


project.build()
