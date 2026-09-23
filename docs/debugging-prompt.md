# Шаблон диагностики ошибок

Перед использованием заполните блок `DEBUGGING TASK`: описание проблемы, ожидаемое и фактическое поведение, команду запуска, полный текст ошибки и шаги воспроизведения. Затем передайте текст ниже вместе с этими данными.

Read @AGENTS.md first and treat it as the source of project-wide constraints.

You are debugging the HackAlem AI wind-power forecasting project.

PROJECT GOAL
============

The project is an Agentic AI system for hourly wind-power forecasting
for two wind turbines.

The system must:

1. Use historical SCADA data to train forecasting models.
2. Obtain archived weather FORECAST data that was actually available
   at the forecast issue time.
3. Produce hourly normalized active-power forecasts for 24–48 hours.
4. Support two turbines.
5. Reproduce forecasting historically through February 2026.
6. Never use future information that was unavailable at issue time.
7. Execute the forecasting process as an autonomous agent workflow:
   weather retrieval
   -> validation
   -> feature preparation
   -> model execution
   -> forecast validation
   -> result storage
   -> rerun/versioning when input data changes.

CURRENT PROJECT STRUCTURE
=========================

Relevant areas may include:

@ai/
@ai/models/
@ai/services/
@ai/prompts/

@backend/
@backend/app/
@backend/app/api/
@backend/app/models/
@backend/app/services/
@backend/app/utils/

@data/
@data/raw/
@data/processed/
@data/weather_cache/
@data/forecasts/
@data/sample/

@frontend/
@frontend/src/

@tests/

@docs/

@README.md
@AGENTS.md
@.env.example
@.gitignore


CRITICAL DOMAIN CONSTRAINTS
===========================

These constraints must NEVER be violated while fixing the bug.

1. Historical SCADA training data ends no later than:

   2026-01-31 23:59:59

2. February 2026 is the official test / rolling forecast period.

3. NEVER use actual February 2026 power values for model training or
   feature construction.

4. NEVER use actual future weather when reproducing a historical forecast.

5. Weather data for a forecast must come from a weather-model run that
   could already have been available at forecast issue_time.

6. Preserve and validate:

   issue_time
   weather_run_time
   valid_time
   lead_hour

7. The no-leakage rule must remain valid:

   weather_run_time + publication_safety_delay <= issue_time

8. If the chosen archived weather run is unavailable, fall back only to
   an EARLIER eligible run.

   Never fall forward to a newer run.

9. The prediction target is normalized active power.

10. Every prediction must remain within:

    0 <= predicted_normalized_power <= 1

11. Do not randomly split time-series data.

12. Model validation must remain chronological.

13. Do not forward-fill large missing SCADA periods.

14. Do not silently fabricate missing weather or SCADA observations.

15. Do not expose API keys or secrets.

16. Never commit .env.

17. Do not replace archived weather forecasts with actual weather just
    to make tests pass.

18. Do not weaken validation or tests merely to remove an error.


DEBUGGING TASK
==============

There is a problem in the project.

PROBLEM DESCRIPTION:

<DESCRIBE THE PROBLEM HERE>


EXPECTED BEHAVIOR:

<DESCRIBE WHAT SHOULD HAPPEN>


ACTUAL BEHAVIOR:

<DESCRIBE WHAT ACTUALLY HAPPENS>


COMMAND THAT TRIGGERS THE PROBLEM:

<PASTE COMMAND HERE>


FULL ERROR / STACK TRACE:

<PASTE THE COMPLETE ERROR OR STACK TRACE HERE>


REPRODUCTION STEPS:

1. <step>
2. <step>
3. <step>


IMPORTANT DEBUGGING PROCEDURE
=============================

Do NOT immediately modify code.

First investigate.

PHASE 1 — UNDERSTAND
--------------------

1. Read @AGENTS.md.

2. Inspect only files relevant to the failing execution path.

3. Trace the execution flow from the entry point to the failure.

For example:

frontend
-> FastAPI route
-> ForecastAgent
-> WeatherClient
-> FeatureBuilder
-> model
-> validator
-> storage

or:

raw CSV
-> preprocessing
-> hourly aggregation
-> feature generation
-> training
-> artifact serialization

4. Identify:

- exact failing component;
- exact failing line or operation;
- expected data shape/type;
- actual data shape/type;
- relevant timestamps;
- relevant environment/configuration values;
- whether the problem is deterministic;
- whether the failure is caused by code, input data, API behavior,
  configuration, model artifact or integration.

5. Before modifying anything, provide a short ROOT CAUSE explanation.

Do not guess.

If evidence is insufficient, say exactly what additional evidence is
required.


PHASE 2 — CHECK HIGH-RISK FAILURE MODES
=======================================

Explicitly check whether the bug is caused by any of the following.

TIME / DATE PROBLEMS
--------------------

Check:

- naive vs timezone-aware datetimes;
- UTC conversion;
- incorrect timezone assumptions;
- wrong date boundaries;
- February data accidentally included in training;
- issue_time after valid_time;
- weather run newer than allowed;
- incorrect 24h/48h lead calculation;
- off-by-one-hour errors;
- wrong resampling boundary;
- duplicate timestamps.


SCADA DATA PROBLEMS
-------------------

Check:

- Russian CSV column names;
- CSV encoding;
- delimiter detection;
- datetime parsing;
- decimal parsing;
- duplicate records;
- missing 10-minute intervals;
- large gaps;
- hourly aggregation;
- sample-count threshold;
- target outside [0,1];
- accidental interpolation over long gaps.


WEATHER DATA PROBLEMS
---------------------

Check:

- latitude / longitude mapping between turbine 1 and turbine 2;
- wrong Open-Meteo endpoint;
- unsupported historical run;
- unavailable weather-model run;
- publication delay logic;
- fallback selecting a FUTURE run;
- units, especially wind speed must be m/s;
- missing hourly weather records;
- wrong weather variable names;
- wrong valid-time alignment;
- malformed cache;
- stale cache;
- mixed forecasts from different model runs.


MODEL PROBLEMS
--------------

Check:

- feature order mismatch;
- missing features;
- unexpected extra features;
- train/inference feature mismatch;
- NaN or infinite values;
- incorrect model artifact path;
- incompatible serialized artifact;
- different scikit-learn versions;
- wrong turbine model loaded;
- random data split;
- model trained on forbidden dates;
- prediction shape;
- predictions outside [0,1].


AGENT PROBLEMS
--------------

Check:

- invalid workflow state transition;
- one stage being skipped;
- failure status not propagated;
- duplicate runs;
- incorrect input hash;
- run version not incremented;
- rerun not triggered after changed input;
- result being saved before validation;
- partial failed runs being treated as COMPLETE.


FASTAPI PROBLEMS
----------------

Check:

- request schema mismatch;
- response schema mismatch;
- datetime serialization;
- incorrect imports;
- incorrect working directory;
- missing environment variables;
- wrong relative paths;
- blocking calls;
- uncaught service exceptions;
- CORS if relevant;
- HTTP status codes.


FRONTEND PROBLEMS
-----------------

Check:

- wrong backend URL;
- wrong endpoint;
- request JSON mismatch;
- field-name mismatch;
- null/undefined response values;
- timestamp formatting;
- frontend assuming fake/mock data;
- chart expecting the wrong response shape;
- stale frontend state.


CACHE / FILE PROBLEMS
---------------------

Check:

- relative path dependent on current working directory;
- missing directories;
- wrong filenames;
- stale cache;
- corrupted JSON;
- malformed CSV;
- partial writes;
- duplicate rows;
- inconsistent schemas.


PHASE 3 — PROPOSE THE FIX
=========================

After identifying the root cause:

1. Propose the SMALLEST safe change.

2. Explain why it fixes the root cause.

3. Explain whether it could affect:

- data leakage;
- model results;
- historical reproducibility;
- API contract;
- stored forecasts;
- tests;
- frontend.

4. Do not refactor unrelated modules.

5. Do not rewrite working architecture unless absolutely necessary.

6. Do not introduce new dependencies unless clearly justified.

7. Prefer fixing the source of the problem rather than adding broad
   try/except blocks.


PHASE 4 — IMPLEMENT
===================

Implement only the required fix.

Requirements:

- preserve existing interfaces unless the interface itself is the bug;
- keep code readable;
- use existing project conventions;
- add useful error messages;
- preserve deterministic behavior where expected;
- do not expose secrets;
- do not remove safety checks.


PHASE 5 — ADD A REGRESSION TEST
===============================

Create or update a test that would FAIL before the fix and PASS after
the fix.

Place it in the most appropriate existing test file.

Possible test locations include:

@tests/test_data_pipeline.py
@tests/test_weather_client.py
@tests/test_no_leakage.py
@tests/test_forecast.py
@tests/test_agent.py

The regression test must target the actual root cause.

Do not merely test that an exception is suppressed.


PHASE 6 — RUN VALIDATION
========================

Run the smallest relevant test first.

For example:

pytest tests/test_weather_client.py -v

or:

pytest tests/test_no_leakage.py -v

Then run related tests.

Finally, if practical, run:

pytest -v

Also run relevant application-level verification.

Examples:

backend:
python -m ...

FastAPI:
uvicorn backend.app.main:app --reload

frontend:
npm test
npm run build

Do not claim a command passed unless you actually executed it.


MANDATORY SAFETY CHECKS AFTER THE FIX
=====================================

After implementing the fix, explicitly verify:

A. DATA LEAKAGE

Confirm that no February actual observations are used in model training.

Confirm:

weather_run_time + publication_safety_delay <= issue_time


B. TARGET RANGE

Confirm:

0 <= prediction <= 1


C. CHRONOLOGY

Confirm timestamps are ordered and there is no random train/test split.


D. TWO TURBINES

Verify the fix works for both turbine 1 and turbine 2 where applicable.


E. REPRODUCIBILITY

Verify deterministic inputs still produce deterministic ML output where
expected.


F. SECRETS

Run or inspect:

git status

Confirm .env is not staged or tracked.


G. UNRELATED CHANGES

Review the diff and remove unrelated modifications.


DO NOT DO THE FOLLOWING
=======================

Do NOT:

- rewrite the entire project;
- change architecture without evidence;
- fabricate data;
- use February actual values;
- use actual future weather for historical predictions;
- choose a newer weather run because an older one fails;
- disable leakage checks;
- remove failing tests without justification;
- replace assertions with broad exception handling;
- expose API keys;
- modify .env;
- add fake frontend forecast data;
- silently change API contracts;
- perform unrelated formatting across many files;
- upgrade all dependencies unless dependency incompatibility is proven;
- commit or push automatically.


FINAL RESPONSE FORMAT
=====================

When finished, respond using this exact structure:

ROOT CAUSE
<short technical explanation>

AFFECTED EXECUTION PATH
<entry point -> components -> failure>

FILES CHANGED
<file list>

FIX
<what was changed and why>

REGRESSION TEST
<what test was added>

COMMANDS EXECUTED
<exact commands>

RESULTS
<tests passed / failed>

SAFETY CHECKS
- No February training leakage: PASS / FAIL
- Weather-run leakage rule: PASS / FAIL
- Predictions constrained to [0,1]: PASS / FAIL
- Chronological split preserved: PASS / FAIL
- .env remains untracked: PASS / FAIL

REMAINING RISKS / LIMITATIONS
<only real remaining issues>

SUGGESTED COMMIT MESSAGE
<one Conventional Commit message>

Do not commit or push anything.
Wait for human review.
