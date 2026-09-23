# AGENTS.md

## Project goal

Build an Agentic AI system that forecasts hourly normalized wind-turbine
active power for 24-48 hour horizons for two turbines.

## Hard constraints

- Historical SCADA data ends on 2026-01-31.
- February 2026 is the test period.
- Never use actual future weather for a historical forecast.
- Weather input must come from a forecast run available at issue time.
- Keep explicit issue_time, weather_run_time and valid_time.
- Output power must always be clipped to [0, 1].
- Do not forward-fill long SCADA gaps.
- Use chronological validation, never random train/test split.
- Do not commit .env or API keys.
- Do not modify unrelated files.

## Engineering priorities

1. Correctness and no data leakage.
2. Reproducibility.
3. Working end-to-end forecast.
4. Tests.
5. UI and extra features.

## Development

Before changing code:
- inspect relevant files;
- state assumptions;
- prefer minimal changes.

After changing code:
- run relevant tests;
- report commands executed;
- report remaining limitations.
