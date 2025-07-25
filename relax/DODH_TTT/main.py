import zntrack
from models import MODELS
from glob import glob
import mlipx


project = zntrack.Project()

fname = glob("../DODH_TTT.traj")
for DATAPATH in fname:
    with project.group("initialize"):
        data = mlipx.LoadDataFile(path=DATAPATH)

    for model_name, model in MODELS.items():
        with project.group(model_name):
            geom_opt = mlipx.StructureOptimization(data=data.frames, model=model, fmax=0.01, steps=500)

project.build()
