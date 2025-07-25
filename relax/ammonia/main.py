import mlipx
import zntrack

from models import MODELS
from glob import glob

range_min = 0
range_max = 499

project = zntrack.Project()

fname = glob("./dataset/*.xyz")

frames = []
with project.group("initialize"):
    for path in fname:
        frames.append(mlipx.LoadDataFile(path=path))

#data = mlipx.LoadDataFile(path='./dataset/Ru_Ni_dft.xyz')
for model_name, model in MODELS.items():
    for data in frames:
        idx = data.path.split(".")[-2].split("/")[-1].split("_")[-1]
        with project.group(model_name, str(idx)):
            geom_opt = mlipx.StructureOptimization(
                   data=data.frames,
                   model=model, 
                   fmax=0.01, 
                   steps=200, 
                   optimizer="FIRE" 
                   )

project.build()
