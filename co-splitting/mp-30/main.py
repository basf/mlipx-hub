import mlipx
import zntrack

from models import MODELS

project = zntrack.Project()

frames = []

with project.group("initialize"):
    for material_id in ['mp-30']:
        frames.append(mlipx.MPRester(search_kwargs={"material_ids": [material_id]}))


for model_name, model in MODELS.items():
    with project.group(model_name):
        cosplit = mlipx.COSplitting(
                model=model,
                data=sum([x.frames for x in frames], []),
                miller=[2,1,1],
                supercell=[2,2,1],
                layers=8,
                vacuum=10.0,
                grid_step=3.0,
                freeze_ratio=0.5
            )

project.build()