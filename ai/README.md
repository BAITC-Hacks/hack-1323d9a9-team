# AI

AI-related components of the project.

This directory may contain:

- AI API integrations
- prompts
- AI services
- model-related logic

The final implementation will depend on the selected hackathon case.

## Historical SCADA baseline

Run `python -m ai.models.baseline_forecasting` to train deterministic
ExtraTrees and histogram-gradient-boosting candidates for each turbine. It
uses measured historical SCADA wind speed and ambient temperature, so it is a
validation baseline only—not a production weather-forecast model. Artifacts
and January 2026 validation metrics are written to `ai/models/artifacts/`.
