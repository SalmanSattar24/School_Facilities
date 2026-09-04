# V1.11 terminal user manual

This project runs without a coding assistant. The only AI service in the production path is the declared Gemini VLM. The operator supplies their own API keys and reviews fields routed to adjudication.

## APIs used by the pipeline

| Service | Purpose | Does the operator need a key? |
|---|---|---|
| Google Gemini Developer API | Runs the frozen Gemini VLM inference | Yes: `GEMINI_API_KEY` |
| Google Street View Static API | Adds ground-level views to V1.11 | Yes: `GOOGLE_STREET_VIEW_API_KEY` |
| Microsoft Planetary Computer / USDA NAIP | Supplies dated overhead imagery | No |
| Esri World Imagery | Supplies the no-key aerial fallback | No |
| OpenStreetMap / Overpass | Finds roads and campus geometry for centering | No |

Only the first two services require setup. Map Tiles API, Maps JavaScript API, Geocoding API, Maps Grounding, and a Street View URL-signing secret are **not** used by this repository. Use separate Gemini and Street View keys; do not reuse one key for both services.

## 1. Install the project

Open PowerShell in the repository:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\school-facilities.exe doctor
```

Do not continue until `doctor` reports that the project files, input data, and frozen configuration are valid. A missing Gemini key is expected at this point.

## 2. Create and save the Gemini API key

1. Sign in to the [Google AI Studio API Keys page](https://aistudio.google.com/api-keys).
2. Create or select a Google Cloud project. If an existing Cloud project is absent, open **Dashboard > Projects > Import projects**, import it, and return to **API Keys**.
3. Click **Create API key**. New AI Studio keys may be authorization keys; this is acceptable. If AI Studio shows an older unrestricted standard key, create a current key or apply the restriction offered by AI Studio.
4. Confirm that the project has the intended Gemini tier and quota. The frozen project configuration requires no active Gemini billing and does not allow paid overage.
5. Copy the key. Treat it like a password: do not paste it into chat, source code, screenshots, documentation, or a tracked file.
6. Save it through the repository's hidden prompt:

```powershell
.\.venv\Scripts\school-facilities.exe save-gemini-key
```

Paste the key only when `Paste Gemini API key (hidden):` appears, then press Enter. The command stores it as `GEMINI_API_KEY` in the ignored local file `secrets.local.env`; it does not place the key in PowerShell command history.

The project uses Gemini's unpaid service. Send only the public, non-sensitive school imagery and public school metadata intended for this assessment. Every new operator must review and accept Google's terms for their own account.

Verify the saved Gemini credential without displaying it:

```powershell
.\.venv\Scripts\school-facilities.exe doctor --require-key
```

The `Gemini credential` check must say `PASS`.

## 3. Create and secure the Street View key

Google Maps Platform setup is separate from Gemini. Google may require a billing-enabled Cloud project even when usage remains inside a no-charge allowance. Creating or changing billing is an operator-controlled cloud action; inspect the current pricing, quota, and billing settings before proceeding.

1. Open the [Google Cloud Console](https://console.cloud.google.com/) and create or select the project that will own Street View usage.
2. Open **APIs & Services > Library**, search for **Street View Static API**, open it, and click **Enable**.
3. Open **APIs & Services > Credentials**.
4. Select **Create credentials > API key**, then open **Edit API key**.
5. Under **API restrictions**, choose **Restrict key**, select only **Street View Static API**, and save.
6. Under **Application restrictions**, use **IP addresses** when the computer has a stable public IP address. Add the public IP address seen by Google, not a private address such as `192.168.x.x`. If the computer's public IP changes regularly, an IP restriction can break terminal requests; at minimum keep the API restriction above, keep the key local, monitor usage, and disable it after the assessment.
7. Open **APIs & Services > Street View Static API > Quotas & System Limits**. Set the request-per-minute provider quota to **10 requests per minute or lower**. Also configure Google Cloud budget alerts and review the current Street View pricing. A budget alert warns—it is not a hard spending cap.
8. Copy the API key. The URL-signing secret shown by some Maps pages is not needed by this terminal pipeline.
9. Save the key through the second hidden prompt:

```powershell
.\.venv\Scripts\school-facilities.exe save-streetview-key
```

This adds `GOOGLE_STREET_VIEW_API_KEY` to the same ignored secrets file without replacing the Gemini key.

Confirm that both variable names are present without printing either value:

```powershell
$secretNames = Get-Content .\secrets.local.env | ForEach-Object { ($_ -split '=', 2)[0].Trim() }
@('GEMINI_API_KEY', 'GOOGLE_STREET_VIEW_API_KEY') | ForEach-Object {
    if ($secretNames -contains $_) { "PASS: $_ is present" } else { "FAIL: $_ is missing" }
}
Remove-Variable secretNames
```

## 4. Record the Street View usage guard

Before a live run, use the Google Cloud console to obtain the current calendar month's number of **Street View Static image requests**. Record that month-to-date count locally; do not substitute an all-time or 28-day number.

```powershell
$billingMonth = Get-Date -Format 'yyyy-MM'
.\.venv\Scripts\school-facilities.exe streetview-record-usage `
    --month $billingMonth `
    --used-requests 0 `
    --source "Google Cloud console"
.\.venv\Scripts\school-facilities.exe streetview-budget-status
Remove-Variable billingMonth
```

Replace `0` with the actual month-to-date image-request count shown in the provider console. The snapshot must be less than 24 hours old when the live pipeline starts. V1.11 also enforces 10 requests per minute, no billable image retry, a per-run cap, a reserve below the configured monthly no-charge allowance, and a local paid allowance of `$0`. These code checks reduce risk but do not replace provider-side restrictions, quota monitoring, or billing alerts.

## 5. How secrets are stored and shared

The resulting local file contains plain-text credentials in this form:

```text
GEMINI_API_KEY=your-own-gemini-key
GOOGLE_STREET_VIEW_API_KEY=your-own-street-view-key
```

The file is ignored by Git. Never commit, submit, attach, print, or copy `secrets.local.env` into a project archive. Submit only `secrets.local.env.example`. Every reviewer creates their own keys and runs the two `save-...-key` commands on their checkout.

A temporary environment variable, when present, overrides the saved value. This is optional and normally unnecessary. To replace a key, rerun its save command. If a key is exposed, create a replacement, verify it, then disable or delete the compromised key and inspect provider usage.

## 6. Final preflight

```powershell
.\.venv\Scripts\school-facilities.exe doctor --require-key
.\.venv\Scripts\school-facilities.exe validate-config
.\.venv\Scripts\school-facilities.exe streetview-budget-status
```

All checks must pass before the live command. Common failures are a key copied with extra characters, the Street View Static API not enabled, billing prerequisites not satisfied, an incompatible IP restriction, exhausted provider quota, or a Street View usage snapshot older than 24 hours. Never solve a `403`, `REQUEST_DENIED`, or quota error by removing all key restrictions; correct the specific project, API, billing, IP, or quota setting.

## Run V1.11 end to end

```powershell
powershell -ExecutionPolicy Bypass -File .\run_v1_11.ps1 -ConfirmLive -ConfirmProviderQuota
```

The script validates configuration; creates campus-centered probes; performs metadata-only discovery; rotates previously reserved duplicate headings; retrieves at most eight views per covered school; invokes Gemini once per covered school; retains the exact source JPEGs locally for human review; freezes the fused predictions; and writes the evaluation.

Inference is resumable: valid V1.11 outputs are reused. Rejected responses remain as `<school_id>-rejected.json` and may be reconciled offline only through documented, meaning-preserving vocabulary mappings. Do not spend a retry to repair formatting.

## Outputs

- `measurements.csv`: new fused V1.11 candidate deliverable.
- `measurements_old.csv`: prior reviewed/reference file, opened only after prediction freeze and never used to alter values.
- `outputs/all25_v1_11_predictions_frozen.json`: pre-reference snapshot, hashes, sources, and flags.
- `outputs/full_pipeline_v1_11_evaluation.json`: accuracy, abstention, flag capture, silent errors, and confidence reliability.
- `data/model_outputs/streetview/v1.11/` plus the validation quarantine: preserved V1.11 records.
- `data/streetview/control/request_ledger_v1.11.jsonl`: append-only provider accounting.

Street View JPEGs are retained locally for human review but remain ignored by Git and are not submission artifacts. NAIP/Esri imagery and model/evaluation artifacts are also retained locally.

## Required human review

Before submission, inspect every `needs-review` row, every `unknown`, all fencing and solar-area results, shared-facility decisions, and the five silent-error cells in the evaluation report. Record corrections as human adjudications without rewriting raw VLM records. Then run:

```powershell
.\.venv\Scripts\school-facilities.exe validate --final
.\.venv\Scripts\python.exe -m pytest -q
```

The current `measurements.csv` is a pipeline candidate, not a final human-approved submission.
