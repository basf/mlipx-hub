```bash
(.venv) $ mlipx recipes vibrational-analysis --models mace-mpa-0,7net-0,7net-mf-ompa-mpa,orb-v2,orb-v3,mattersim,grace-2l-omat  --smiles=CO,CCO,CCCO,CCCCO
(.venv) $ vim main.py # set system="molecule"
(.venv) $ python main.py
(.venv) $ dvc repro
(.venv) $ mlipx compare --glob "*VibrationalAnalysis"
```
