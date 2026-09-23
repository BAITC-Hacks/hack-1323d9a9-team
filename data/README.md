# Data

Project datasets, sample inputs and sample outputs.

Do not store sensitive information or API keys here.

## SCADA preprocessing

Place the unchanged source exports at `data/raw/turbine_1.csv` and
`data/raw/turbine_2.csv` (the existing `tribune_1.csv` / `tribune_2.csv`
spelling is also accepted), then run:

```powershell
python -m ai.models.scada_preprocessing
```

The command writes `data/processed/turbine_1_hourly.csv` and
`data/processed/turbine_2_hourly.csv`, plus `docs/data-analysis.md`. It only
keeps hours with at least three valid source measurements and does not fill or
interpolate gaps.
