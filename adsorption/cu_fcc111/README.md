```bash
(.venv) $ mlipx recipes adsorption --models MACE-MPA-0,7net-0,7net-mf-ompa-mpa,orb-v2,orb-v3,mattersim,GRACE-2L-OMAT --slab-config '{"crystal": "fcc111", "symbol": "Cu", "size": [3,4,4]}' --smiles CCO --repro
(.venv) $ mlipx compare --glob "*RelaxAdsorptionConfigs"
```
