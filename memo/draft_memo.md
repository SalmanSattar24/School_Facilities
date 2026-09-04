# School Facilities Measurement Pipeline - Memo

## Approach

I built a reproducible pipeline that takes the supplied school coordinates, resolves campus scope, acquires imagery, requests structured visual evidence from a VLM, applies deterministic consistency checks, and writes one row per school with field-level confidence and audit trails. Dated USDA NAIP imagery, accessed through Microsoft Planetary Computer, is the primary overhead source; Esri is a fallback. Reliable OpenStreetMap/Overpass school polygons define campus scope. Weak or missing polygons produce a soft or center-only search area plus ownership/boundary review flags, not a fabricated boundary.

Gemini 3.5 Flash Lite receives a frozen context/detail overhead pair and, when available, up to eight Google Street View Static images from distinct nearby panoramas. Before returning values, it answers item-specific questions about geometry, markings, shadows, roof and mount type, boundary sides, ownership, visibility, and exclusions. Code then checks fencing-coverage arithmetic, independently playable court/field counts, support for negative answers, and solar mount/area consistency. Parking-canopy and ground-mounted solar are excluded. Contradictions become `unknown` or review flags; raw responses are preserved. The PowerShell workflow requires no AI coding assistant, uses operator-supplied keys, and records imagery vintages, hashes, provenance, and request ledgers.

## Validation and results

Before unblinding predictions, I hand-labeled a stratified six-school subset from imagery alone and froze its 54 values by SHA-256. The pipeline returned 38 correct answers, 11 wrong answers, and 5 unknowns: 90.7% coverage and 77.6% accuracy among answered fields (22.4% answered error). All 16 wrong-or-unknown cases were flagged, yielding zero silent errors. Six schools are too few for precise calibration, so I report counts and rates.

For a broader descriptive check, I froze all 25 predictions before comparing them with the six blind labels and prior reviewed labels for the other 19 schools. Of 223 evaluable cells, 168 were correct, 41 wrong, and 14 unknown: 93.7% coverage and 80.4% answered accuracy. Flags captured 90.9% of wrong-or-unknown cases. Unflagged known values were correct 113/118 (95.8%); flagged known values were correct 55/91 (60.4%). I therefore assign empirical confidences of 0.96 and 0.60 to those frozen strata, with 0.20 for unknown. These are descriptive same-sample estimates, not independent guarantees. Street View raised coverage 17.5 percentage points over aerial-only V1.10 but reduced answered accuracy 4.9 points; I treat it as a coverage and uncertainty supplement, not automatic improvement.

## Failures

Full-size fields (13 errors), hard courts (10), fencing extent (8), and fence type (7) were weakest because of overlapping markings, partial/tree-obscured boundaries, shared facilities, and imagery-vintage mismatch. The 14 abstentions were Kofa fencing/type/fields/courts; perimeter fencing at Wells and Pleasant Valley; and solar presence/area at Beverly Hills, Cerra Vista, Bel Air, and Spring Lake Heights. Pleasant Valley had no eligible Street View panorama; the other abstentions reflect insufficient views or failed mount/area guards. Lincoln County and Ridgeview court references also remained unmeasurable. Five pooled errors escaped flags: Thompson and Rundlett fields, Spring Lake courts, India Hook track, and Northside fields.

## Cost and wall-clock time at scale

The 23-school supplement ran in 47 minutes with retrieval and inference overlapped, or about 2.1 minutes per new school, excluding prior NAIP acquisition and review. The VLM averaged 13,119 input and 2,688 output tokens per covered school. At current paid Gemini 3.5 Flash-Lite rates ($0.30/$2.50 per million input/output tokens), 130,000 schools would cost about $1,385 for inference. Eight Street View requests per school add about $5,054, giving a **$6,439 direct-API estimate ($0.050/school)** before compute, storage, and labor. Serial execution would take roughly 4,550 hours; parallel quota-controlled workers are essential. At three minutes per flagged school, the observed 24/25 routing rate would also exceed 6,000 reviewer-hours, so national use should reserve Street View and review for ambiguous/high-value cases.

## Next with 100 hours

I would build a larger multi-reviewer blind set; measure agreement; recalibrate by field and imagery quality; add targeted high-resolution court/field crops; match imagery vintages; test repeatability; replace public Overpass calls with regional extracts; and optimize selective routing while protecting silent-error capture.

## Tooling

Production uses Python, Rasterio/Shapely, Planetary Computer STAC/NAIP, Esri, OpenStreetMap/Overpass, Google Street View Static, and Gemini 3.5 Flash Lite. Gemini 3.1 Flash Lite supported a development auditor diagnostic. OpenAI Codex assisted requirements review, implementation/debugging, tests, and documentation; it is not required to execute or normalize the pipeline.
