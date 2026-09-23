# Forecasting pipeline

- `models/scada_preprocessing.py`: finite, hourly SCADA targets; no gap filling.
- `models/baseline_forecasting.py`: measured-weather comparison models.
- `models/train.py`: verified archived-weather training, local-time joins,
  chronological validation and independent production artifacts.
- `services/weather_client.py`: explicit historical ECMWF runs and verified cache.
- `services/archive_download.py`: resumable historical cache population.
- `services/forecast_agent.py`: weather → validation → features → inference →
  output validation → atomic save; updated inputs receive new versions and
  existing forecast artifacts are not overwritten.
- `services/rolling_backtest.py`: official Jan 31–Feb 28 local-midnight issue
  simulation; all 48-hour runs plus one forecast per February hour at lead ≥24.
- `services/replay.py`: earlier 23:00 local diagnostic replay.

Commands, source semantics, validation rules and limitations are documented in
the [repository README](../README.md) and [backtesting guide](../docs/backtesting.md).
No LLM API key is required.
