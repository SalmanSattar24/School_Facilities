# Configuration files

The active frozen configuration is imagery Version 1.6, boundary resolver Version 1.0, VLM Version 1.10, and evidence-auditor Version 1.2. Runtime code loads these frozen files:

- `imagery.json`
- `vlm.json`
- `vlm_prompt.txt`
- `vlm_field_protocol.json`
- `vlm_response_schema.json`
- `vlm_auditor.json`
- `vlm_auditor_prompt.txt`
- `vlm_auditor_response_schema.json`
- `boundary_vlm.json`
- `boundary_vlm_prompt.txt`
- `boundary_vlm_response_schema.json`
- `pilot_schools.json`
- `frozen_config.sha256`, which records their approved hashes

`imagery.json` keeps a fixed context product and defines authoritative, soft-boundary, and center-only detail modes. Authoritative polygons use the standard 60 m buffer; soft boundaries use at least 600 m and 150 m of safety buffer on each side. The boundary resolver uses one context image and `gemini-3.5-flash-lite` only when the public polygon stage fails. Its polygons can guide the crop but can never silently become hard measurement masks. `vlm.json` names the same model as the sole primary facility model. Version 1.10 declares boundary authority and full-image search where required, prevents model-side `0.95`, requires seven feature assessments plus nine measurement packets, and derives fencing from boundary-weighted minimum/maximum coverage. `vlm_field_protocol.json` is the shared definition and question registry used by both the visual model and auditor. The solar polygon-area tolerance remains `max(25 m², 25%)`. The text-only auditor uses stable `gemini-3.1-flash-lite`, receives no images or labels, and cannot overwrite primary values. All Gemini clients use the environment override or ignored `secrets.local.env`; there is no Gemini 3.7 unlock path.

The files whose names contain `development` are retained only as historical evidence of the earlier transport-and-schema test. They are not loaded by configuration validation, the CLI, or the VLM client, and their outputs are not final measurements.
