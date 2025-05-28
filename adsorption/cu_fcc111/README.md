```bash
(.venv) $ mlipx recipes adsorption --models mace-mpa-0,7net-0,7net-mf-ompa-mpa,orb-v2,orb-v3,mattersim,grace-2l-omat --slab-config '{"crystal": "fcc111", "symbol": "Cu", "size": [3,4,4]}' --smiles CCO --repro
(.venv) $ mlipx compare --glob "*RelaxAdsorptionConfigs"
```
