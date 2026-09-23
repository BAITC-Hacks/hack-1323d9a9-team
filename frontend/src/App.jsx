import { useMemo, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

const WORKFLOW = [
  { key: "FETCH_WEATHER", label: "Fetch weather" },
  { key: "VALIDATE_INPUT", label: "Validate input" },
  { key: "BUILD_FEATURES", label: "Build features" },
  { key: "RUN_MODEL", label: "Run model" },
  { key: "VALIDATE_OUTPUT", label: "Validate output" },
  { key: "SAVE_RESULT", label: "Save" },
];

const TURBINES = [
  { id: "turbine_1", label: "Turbine 1" },
  { id: "turbine_2", label: "Turbine 2" },
];

function utcIsoFromInput(value) {
  if (!value) throw new Error("Choose an issue date and time.");
  const parsed = new Date(`${value}:00Z`);
  if (Number.isNaN(parsed.getTime())) throw new Error("Enter a valid issue date and time.");
  return parsed.toISOString().replace(".000Z", "Z");
}

function formatUtc(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
    timeZoneName: "short",
  }).format(date);
}

function getApiError(response, payload) {
  const detail = payload?.detail;
  if (typeof detail === "string") return detail;
  if (detail?.message) return detail.message;
  if (Array.isArray(detail)) return detail.map((item) => item.msg).filter(Boolean).join(" ");
  return payload?.message || `Forecast request failed with HTTP ${response.status}.`;
}

function errorStage(code) {
  if (code === "weather_unavailable") return "FETCH_WEATHER";
  if (code === "model_unavailable") return "RUN_MODEL";
  return null;
}

function PowerChart({ points, turbineLabel }) {
  const [hovered, setHovered] = useState(null);
  const width = 960;
  const height = 360;
  const padding = { top: 24, right: 24, bottom: 48, left: 54 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const x = (index) => padding.left + (points.length <= 1 ? 0 : (index / (points.length - 1)) * plotWidth);
  const y = (value) => padding.top + (1 - Math.min(1, Math.max(0, value))) * plotHeight;
  const path = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${x(index)} ${y(point.predicted_normalized_power)}`)
    .join(" ");
  const areaPath = path
    ? `${path} L ${x(points.length - 1)} ${padding.top + plotHeight} L ${x(0)} ${padding.top + plotHeight} Z`
    : "";
  const yTicks = [0, 0.25, 0.5, 0.75, 1];
  const xTickIndexes = [...new Set([0, Math.floor((points.length - 1) / 2), points.length - 1])];

  if (!points.length) {
    return <div className="chart-empty">Run a forecast to see hourly predicted power.</div>;
  }

  return (
    <div className="chart-wrap">
      <svg
        className="chart"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${turbineLabel} predicted normalized power by forecast hour`}
      >
        <defs>
          <linearGradient id="power-area" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#46d6a1" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#46d6a1" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {yTicks.map((tick) => (
          <g key={tick}>
            <line className="grid-line" x1={padding.left} x2={width - padding.right} y1={y(tick)} y2={y(tick)} />
            <text className="axis-label" x={padding.left - 14} y={y(tick) + 5} textAnchor="end">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}
        {xTickIndexes.map((index) => (
          <text
            key={index}
            className="axis-label"
            x={x(index)}
            y={height - 16}
            textAnchor={index === 0 ? "start" : index === points.length - 1 ? "end" : "middle"}
          >
            {index === 0 ? "+1h" : `+${points[index].lead_hour}h`}
          </text>
        ))}
        <path className="chart-area" d={areaPath} />
        <path className="chart-line" d={path} />
        {points.map((point, index) => (
          <circle
            className="chart-hit-area"
            cx={x(index)}
            cy={y(point.predicted_normalized_power)}
            key={`${point.turbine}-${point.lead_hour}`}
            onMouseEnter={() => setHovered({ point, index })}
            onMouseLeave={() => setHovered(null)}
            r="10"
          />
        ))}
        {hovered && (
          <g className="chart-focus" pointerEvents="none">
            <line x1={x(hovered.index)} x2={x(hovered.index)} y1={padding.top} y2={padding.top + plotHeight} />
            <circle cx={x(hovered.index)} cy={y(hovered.point.predicted_normalized_power)} r="6" />
          </g>
        )}
      </svg>
      {hovered && (
        <div
          className="chart-tooltip"
          style={{
            left: `${(x(hovered.index) / width) * 100}%`,
            top: `${(y(hovered.point.predicted_normalized_power) / height) * 100}%`,
          }}
        >
          <strong>{hovered.point.predicted_normalized_power.toFixed(3)}</strong>
          <span>Lead +{hovered.point.lead_hour}h</span>
          <span>{formatUtc(hovered.point.valid_time)}</span>
        </div>
      )}
    </div>
  );
}

function Workflow({ phase, states, failedStage }) {
  return (
    <ol className="workflow" aria-label="Agent workflow">
      {WORKFLOW.map((step, index) => {
        const complete = states.includes(step.key);
        const failed = phase === "error" && failedStage === step.key;
        const active = phase === "running" && index === 0;
        const className = complete ? "complete" : failed ? "failed" : active ? "active" : "pending";
        return (
          <li className={className} key={step.key}>
            <span className="step-icon" aria-hidden="true">{complete ? "✓" : failed ? "!" : index + 1}</span>
            <span>{step.label}</span>
          </li>
        );
      })}
    </ol>
  );
}

function MetadataItem({ label, value, wide = false }) {
  return (
    <div className={wide ? "metadata-item wide" : "metadata-item"}>
      <dt>{label}</dt>
      <dd title={value}>{value || "—"}</dd>
    </div>
  );
}

export default function App() {
  const [issueTime, setIssueTime] = useState("2026-01-31T18:00");
  const [horizon, setHorizon] = useState(48);
  const [activeTurbine, setActiveTurbine] = useState("turbine_1");
  const [result, setResult] = useState(null);
  const [phase, setPhase] = useState("idle");
  const [error, setError] = useState(null);

  const points = useMemo(
    () => (result?.forecasts || [])
      .filter((item) => item.turbine === activeTurbine)
      .sort((a, b) => a.lead_hour - b.lead_hour),
    [result, activeTurbine],
  );

  const weather = result?.metadata?.weather?.[activeTurbine] || {};
  const modelVersion = points[0]?.model_version || result?.metadata?.models?.[activeTurbine] || "—";
  const states = result?.metadata?.state_history || [];
  const turbineLabel = TURBINES.find((item) => item.id === activeTurbine)?.label || activeTurbine;
  const stats = points.length
    ? {
        average: points.reduce((sum, point) => sum + point.predicted_normalized_power, 0) / points.length,
        peak: Math.max(...points.map((point) => point.predicted_normalized_power)),
      }
    : null;

  async function runForecast(event) {
    event.preventDefault();
    setPhase("running");
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/forecast/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          issue_time: utcIsoFromInput(issueTime),
          horizon_hours: horizon,
        }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const apiError = new Error(getApiError(response, payload));
        apiError.code = payload?.detail?.code;
        throw apiError;
      }
      setResult(payload);
      setPhase("complete");
    } catch (requestError) {
      setError({
        message: requestError.message || "Unable to run the forecast.",
        code: requestError.code,
      });
      setPhase("error");
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <div>
            <p>Agentic forecasting system</p>
            <h1>Wind Power Forecast</h1>
          </div>
        </div>
        <div className={`api-state ${phase}`}>
          <span className="state-dot" />
          {phase === "running" ? "Agent running" : phase === "complete" ? "Forecast complete" : phase === "error" ? "Run failed" : "Ready"}
        </div>
      </header>

      <section className="control-panel" aria-label="Forecast controls">
        <form onSubmit={runForecast}>
          <label>
            <span>Issue date & time <em>UTC</em></span>
            <input
              aria-label="Issue date and time in UTC"
              onChange={(event) => setIssueTime(event.target.value)}
              required
              step="3600"
              type="datetime-local"
              value={issueTime}
            />
          </label>
          <fieldset>
            <legend>Forecast horizon</legend>
            <div className="segment-control">
              {[24, 48].map((hours) => (
                <button
                  aria-pressed={horizon === hours}
                  className={horizon === hours ? "selected" : ""}
                  key={hours}
                  onClick={() => setHorizon(hours)}
                  type="button"
                >
                  {hours}h
                </button>
              ))}
            </div>
          </fieldset>
          <button className="run-button" disabled={phase === "running"} type="submit">
            {phase === "running" ? <span className="spinner" aria-hidden="true" /> : <span aria-hidden="true">▶</span>}
            {phase === "running" ? "Running agent…" : "Run forecast"}
          </button>
        </form>
      </section>

      {error && (
        <section className="error-banner" role="alert">
          <span aria-hidden="true">!</span>
          <div>
            <strong>Forecast could not be completed</strong>
            <p>{error.message}</p>
          </div>
        </section>
      )}

      <section className="workflow-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Autonomous workflow</p>
            <h2>Agent status</h2>
          </div>
          {result && <span className="run-version">Run v{result.metadata?.forecast_version ?? "—"}</span>}
        </div>
        <Workflow phase={phase} states={states} failedStage={errorStage(error?.code)} />
      </section>

      <section className="forecast-panel">
        <div className="forecast-header">
          <div>
            <p className="eyebrow">Hourly output</p>
            <h2>Predicted normalized power</h2>
          </div>
          <div className="tabs" role="tablist" aria-label="Turbine selection">
            {TURBINES.map((turbine) => (
              <button
                aria-selected={activeTurbine === turbine.id}
                className={activeTurbine === turbine.id ? "active" : ""}
                key={turbine.id}
                onClick={() => setActiveTurbine(turbine.id)}
                role="tab"
                type="button"
              >
                {turbine.label}
              </button>
            ))}
          </div>
        </div>
        <PowerChart points={points} turbineLabel={turbineLabel} />
        <div className="chart-footer">
          <span>Normalized output <strong>0–1</strong></span>
          {stats && <span>Average <strong>{stats.average.toFixed(3)}</strong></span>}
          {stats && <span>Peak <strong>{stats.peak.toFixed(3)}</strong></span>}
          {points.length > 0 && <span>Points <strong>{points.length}</strong></span>}
        </div>
      </section>

      <section className="audit-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Traceability</p>
            <h2>Audit metadata</h2>
          </div>
          {result && <span className="safe-badge">✓ Leakage-safe run</span>}
        </div>
        <dl className="metadata-grid">
          <MetadataItem label="Issue time" value={formatUtc(result?.issue_time)} />
          <MetadataItem label="Weather run time" value={formatUtc(weather.weather_run_time)} />
          <MetadataItem label="Weather provider" value={weather.provider} />
          <MetadataItem label="Weather model" value={weather.model} />
          <MetadataItem label="Forecast source" value={weather.forecast_source} />
          <MetadataItem label="ML model version" value={modelVersion} wide />
        </dl>
        {result?.warnings?.length > 0 && (
          <div className="warnings" role="status">
            <strong>Warnings</strong>
            <ul>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
          </div>
        )}
      </section>

      <footer>
        <span>Open-Meteo Single Runs</span>
        <span>ECMWF IFS HRES</span>
        <span>All times shown in UTC</span>
      </footer>
    </main>
  );
}
