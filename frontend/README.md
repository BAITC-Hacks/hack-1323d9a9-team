# Wind Power Forecast dashboard

Minimal React/Vite interface for the existing FastAPI forecast service.

## Run locally

Start the API from the repository root:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Then start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` and `/health` to the API on port 8000.

For a separately hosted API, set `VITE_API_BASE_URL` to its origin before starting or building the frontend. That API must allow requests from the frontend origin.

## Production build

```powershell
npm run build
npm run preview
```

The dashboard sends real requests to `POST /api/forecast/run`. It does not contain fallback or demonstration forecast data.
