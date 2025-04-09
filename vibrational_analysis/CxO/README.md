```bash
mlipx recipes vibrational-analysis --models MACE-MPA-0,7net-0,7net-mf-ompa-mpa,orb-v2,orb-v3,mattersim,GRACE-2L-OMAT  --smiles=CO,CCO,CCCO,CCCCO
vim main.py # set system="molecule"
python main.py
dvc repro
mlipx compare --glob "*VibrationalAnalysis"
```
