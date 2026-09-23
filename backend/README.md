# Backend

FastAPI interface for the forecasting workflow.

## Forecast API

Install `requirements.txt`, then run:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The API exposes `/health`, `/api/forecast/run`, `/api/forecast/latest`, and
`/api/metrics`. Forecast orchestration remains in `ai.services.ForecastAgent`.
