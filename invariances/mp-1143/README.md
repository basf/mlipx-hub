```bash
(.venv) $ mlipx recipes invariances --models mace-mpa-0,7net-0,7net-mf-ompa-mpa,orb-v2,orb-v3,mattersim,grace-2l-omat,chgnet --material-ids=mp-1143 --repro
(.venv) $ mlipx compare --glob "*RotationalInvariance"
(.venv) $ mlipx compare --glob "*TranslationalInvariance"
(.venv) $ mlipx compare --glob "*PermutationInvariance"
```
