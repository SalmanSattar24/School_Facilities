# One-page memo outline

Keep the final memo to one page. Replace every bracketed item with measured evidence; do not claim accuracy before running the calibration command.

## Method and validation

- Imagery source(s), capture/retrieval vintage, campus-resolution procedure, and any VLM/assistant used.
- Blind same-reviewer reference schools labeled before raw-output exposure: `[n schools; selection rule]`; do not call these independent expert labels.
- Raw VLM outcomes: `[correct / wrong / abstained among evaluable fields]`; answered coverage and selective accuracy: `[values with denominators]`.
- Error awareness: `[problem cases flagged / all wrong-or-abstained]`; silent unflagged errors: `[wrong and unflagged / evaluable]`.
- Count MAE and solar-area error where the validation data support them.
- Raw model-confidence evidence only when its sample size is worth showing; keep final human confidence separate, label sparse bins descriptive, and state that six schools cannot support strong calibration claims.
- Repeatability: `[schema-valid runs, per-field agreement, unknown/flag consistency across three repeated pilot responses]`; state clearly that repeatability does not prove correctness.

## Failures

- Name specific schools/fields affected by coordinate error, boundary ambiguity, tree cover, shadows, low resolution, stale imagery, or missing metadata.
- State how unknown values and low confidence were encoded.

## Cost at scale

- Observed wall time and any API/model cost per school.
- Multiply the measured per-school figures for the four take-home attribute groups by approximately 130,000 schools; distinguish serial from parallel time.
- Note any licensing, rate-limit, storage, and human-review assumptions.

## With 100 hours

- Resolve campus polygons first; obtain dated imagery; expand independent labels; stratify validation; fit confidence calibration; automate only fields that validate well.

## Tooling disclosure

- List each imagery service, exact model and version, code assistant, and manual step with its purpose.
- State that the VLM uses a user-supplied key, and disclose the request-rate and run-level caps.

## Required personal answers to include with submission

1. Current GRA/RA appointment: `[answer]`.
2. Fall-start work authorization: `[answer]`.
