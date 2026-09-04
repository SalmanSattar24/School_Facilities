# School Facilities from Imagery

A reproducible computer-vision pipeline for measuring physical facilities at 25 U.S. public schools from overhead and street-level imagery. It produces structured estimates for rooftop solar, portable classrooms, perimeter fencing, running tracks, full-size athletic fields, hard courts, and outdoor pools, with a separate confidence and review decision for every field.

The pipeline is designed to be run by a human from PowerShell without a coding assistant. Gemini 3.5 Flash Lite performs the declared visual inference; deterministic checks flag contradictions, insufficient evidence, ambiguous campus ownership, and unsupported certainty instead of silently forcing an answer.

## Current result

V1.11 completed for all 25 schools. Twenty-four schools have aerial-plus-Street-View inference; one school without eligible Street View coverage uses an explicitly flagged aerial-only fallback.

Against the frozen reference data, 223 evaluable field-school cells contained 168 correct answers, 41 wrong answers, and 14 explicit unknowns. Answered coverage was 93.7%, answered accuracy was 80.4%, and the uncertainty layer flagged 90.9% of wrong-or-unknown results. Unflagged known values were correct in 113 of 118 cases (95.8%). These are descriptive same-sample results, not independent performance guarantees.

`measurements.csv` is the generated review candidate. It is intentionally **not** presented as the final submission because every row still requires human adjudication.

## Quick start on Windows

Requirements: Windows PowerShell and Python 3.11 or newer.

```powershell
git clone https://github.com/SalmanSattar24/School_Facilities.git
Set-Location .\School_Facilities
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The live workflow requires operator-owned Gemini and Google Street View Static API keys. Follow [the user manual](docs/USER_MANUAL.md) from account creation through key restrictions, quota controls, local secret storage, preflight, execution, and review.

No user key is needed for Microsoft Planetary Computer/USDA NAIP, Esri World Imagery, or OpenStreetMap/Overpass. Real credentials belong only in the ignored `secrets.local.env`; never commit or submit that file.

## Run the pipeline

After completing the API and budget preflight in the manual:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_v1_11.ps1 -ConfirmLive -ConfirmProviderQuota
```

The workflow validates the frozen configuration, plans campus-centered Street View probes, performs metadata-first discovery, deduplicates views, retrieves up to eight eligible views per covered school, runs one Gemini inference per school, freezes predictions before reference comparison, writes `measurements.csv`, and writes the evaluation report.

The run is resumable. Valid existing responses are reused, and schema deviations may be reconciled offline only through documented meaning-preserving vocabulary mappings. Raw provider responses and request ledgers are local generated artifacts and are not included in the repository.

## Verify the checkout

```powershell
.\.venv\Scripts\school-facilities.exe doctor
.\.venv\Scripts\school-facilities.exe validate-config
.\.venv\Scripts\python.exe -m pytest -q
```

To inspect the current candidate:

```powershell
.\.venv\Scripts\school-facilities.exe validate
.\.venv\Scripts\school-facilities.exe workflow-status
```

`validate --final` must remain failing until a human has reviewed every row and changed its status to `reviewed`. Unknowns require a field-specific reason.

## Method

1. Resolve campus scope from public school coordinates and OpenStreetMap geometry. Reliable polygons define authoritative scope; ambiguous cases use soft-boundary or center-only search with explicit review flags.
2. Acquire dated USDA NAIP overhead context/detail imagery through Microsoft Planetary Computer, with Esri as fallback.
3. Discover nearby Google-owned Street View panoramas from public roads and select diverse views under strict request and budget limits.
4. Ask Gemini item-specific visual questions before deriving the nine output measurements. The model records observable evidence such as sports markings, roof and mount type, fencing by boundary side, shadows, visibility, and ownership ambiguity.
5. Apply deterministic consistency guards. Examples include independently playable court counts, fencing-coverage arithmetic, negative-answer visibility, and solar mount/area agreement. Parking-canopy and ground-mounted solar do not count as rooftop solar.
6. Freeze predictions, compare them with the reference data, and report accuracy, abstention, flag capture, silent errors, and empirical confidence by review stratum.
7. Route flagged and unknown fields to human adjudication without rewriting raw model records.

## Repository layout

- `src/school_facilities/`: pipeline, acquisition, validation, uncertainty, and evaluation code.
- `config/`: frozen model, imagery, prompt, schema, rate-limit, and budget configuration.
- `tests/`: offline tests for the operator workflow and safety contracts.
- `docs/USER_MANUAL.md`: complete setup and operating guide.
- `schools_sample.csv`: the supplied 25-school input.
- `measurements.csv`: V1.11 generated candidate requiring human review.
- `measurements_old.csv`: earlier reviewed reference used only after prediction freeze.
- `data/validation/ground_truth.csv`: frozen six-school blind reference labels.
- `data/campus_resolutions/`: reproducible campus-resolution records.
- `outputs/all25_v1_11_predictions_frozen.json`: prediction snapshot created before evaluation.
- `outputs/full_pipeline_v1_11_evaluation.json`: detailed V1.11 evaluation.
- `memo/draft_memo.md` and `output/pdf/memo.pdf`: editable and rendered one-page memo.
- `secrets.local.env.example`: credential template without real values.

Downloaded imagery, raw provider responses, API request ledgers, review packets, local source PDFs, caches, and real secrets are deliberately excluded from Git. A fresh operator can regenerate them using the documented workflow.

## Important limitations

- The six-school blind reference is too small for precise calibration estimates.
- The broader 25-school comparison reuses prior reviewed labels and is descriptive rather than independent validation.
- Courts, fields, and fencing remain difficult under overlapping markings, partial visibility, tree cover, shared facilities, and imagery-vintage mismatch.
- The observed review-routing rate is too high for a national deployment without better field-specific models and selective acquisition.
- Public Overpass access and current provider pricing/quotas must be replaced or rechecked before scaling to roughly 130,000 schools.

See the one-page memo for validation results, named failures, scale estimates, next steps, and complete tooling disclosure.
