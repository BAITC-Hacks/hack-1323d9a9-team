# Backend

Backend part of the HackAlem AI project.

The framework, API structure and business logic will be selected after receiving the hackathon case.

## Forecast API

Install `requirements.txt`, then run:

```powershell
uvicorn backend.app.main:app --reload
```

The API exposes `/health`, `/api/forecast/run`, `/api/forecast/latest`, and
`/api/metrics`. Forecast orchestration remains in `ai.services.ForecastAgent`.
