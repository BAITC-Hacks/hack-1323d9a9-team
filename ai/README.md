# Forecasting pipeline

- `models/scada_preprocessing.py`: finite, hourly SCADA targets; no gap filling.
- `models/baseline_forecasting.py`: measured-weather comparison models.
- `models/train.py`: verified archived-weather training, local-time joins,
  chronological validation and independent production artifacts.
- `services/weather_client.py`: explicit historical ECMWF runs and verified cache.
- `services/archive_download.py`: resumable historical cache population.
- `services/forecast_agent.py`: weather → validation → features → inference →
  output validation → atomic save; updated inputs receive new versions.
- `services/replay.py`: daily 24/48-hour forecasts across local February 2026.

Commands, source semantics, validation rules and limitations are documented in
the [repository README](../README.md). No LLM API key is required.
