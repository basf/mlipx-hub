```bash
mlipx recipes invariances --models MACE-MPA-0,7net-0,7net-mf-ompa-mpa,orb-v2,orb-v3,mattersim,GRACE-2L-OMAT,chgnet --material-ids=mp-1143 --repro
mlipx compare --glob "*RotationalInvariance"
mlipx compare --glob "*TranslationalInvariance"
mlipx compare --glob "*PermutationInvariance"
```
