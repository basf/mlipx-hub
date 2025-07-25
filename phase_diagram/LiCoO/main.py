import mlipx
import zntrack

from models import MODELS

project = zntrack.Project()

frames = []

with project.group("initialize"):
    for path in ['./LiCoO_schmidt_2021_with_terminals_mad_recomputed.xyz']:
        frames.append(mlipx.LoadDataFile(path=path))


for model_name, model in MODELS.items():
    for idx, data in enumerate(frames):
        with project.group(model_name, str(idx)):
            pd = mlipx.PhaseDiagram(data=data.frames, model=model)


project.build()
