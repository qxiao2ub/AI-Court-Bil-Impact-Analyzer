# AI Court-Bill Impact Analyzer

**Author:** Claire Yuan  
**Advisor:** Dr. Qingyang Xiao  
**License:** MIT

A Streamlit research prototype derived from Claire Yuan's Colab notebook. The app accepts a redacted court-issued notice, citation, fine, or bill; extracts model features; runs a classical machine-learning + neural-network ensemble; and produces a plain-language impact report with calibrated prediction ranges.

> **Important:** The bundled model is trained entirely on synthetic `Demo-*` records. It is an educational prototype, not legal advice and not a tool for guilt, sentencing, eligibility, creditworthiness, or other high-impact decisions.

## Streamlit Cloud deployment fix

This revision is specifically optimized for the current Streamlit Community Cloud Python 3.14 runtime.

The earlier repository could spend a very long time in the dependency-install stage because it pinned an older pandas release and installed PyTorch during every cloud build. The fixed version changes the deployment architecture:

- **No PyTorch package is installed by Streamlit Cloud.**
- The multi-task neural network is still **trained with PyTorch offline**, but its learned weights are exported as NumPy arrays.
- The Streamlit app performs the neural-network forward pass with **NumPy-only inference**.
- The model's classical scikit-learn pipelines are preserved.
- pandas is updated to a Python-3.14-compatible wheel release.
- Large/unneeded cloud dependencies such as Matplotlib and an explicit SciPy pin were removed.
- `requirements-training.txt` is separated from `requirements.txt`, so optional model retraining does not burden normal app deployment.


## Migrated UI design

This revision translates the attached TanStack/React editorial UI into native Streamlit styling while preserving the Court-Bill AI pipeline. The migrated interface uses the source design's warm paper background, fine borders, JetBrains Mono interface typography, Instrument Serif display typography, orange accent, ticker treatment, oversized editorial hero, compact uppercase labels, rule-based result cards, principles block, and responsive behavior.

The unrelated AP News, legislation lookup, Supabase, Cloudflare, watchlist, and server-side React features from the source UI are intentionally excluded. They are not needed for Claire Yuan's court-notice prototype and would make Streamlit Community Cloud deployment heavier. See [`UI_MIGRATION.md`](UI_MIGRATION.md) for details.

## Features

- Streamlit Community Cloud-ready `app.py` entrypoint
- Attached editorial UI design translated into native Streamlit + CSS
- Python 3.14-compatible deployment dependencies
- PDF text extraction with OCR fallback
- Image OCR through Tesseract
- TXT, CSV, and Markdown ingestion
- Transparent bill-feature parser
- scikit-learn Ridge and logistic-regression pipelines
- Multi-task deep neural network trained with PyTorch
- Lightweight NumPy neural-network inference on Streamlit Cloud
- Classical + DNN ensemble predictions
- Approximate 90% empirical calibration ranges
- Constrained reinforcement-learning demo for report presentation only
- Markdown and JSON report downloads
- Privacy and responsible-use warnings
- Original Colab notebook included under `notebooks/`

## Repository structure

```text
.
├── app.py
├── LICENSE
├── README.md
├── DEPLOYMENT_FIX.md
├── UI_MIGRATION.md
├── requirements.txt
├── requirements-training.txt
├── packages.txt
├── .streamlit/
│   └── config.toml
├── assets/
│   └── ui-reference-favicon.ico
├── examples/
│   └── demo_speeding_notice.txt
├── models/
│   └── claire_yuan_court_bill_impact_models.joblib
├── notebooks/
│   └── Claire_Yuan_AI_Court_Bill_Impact_Analyzer_Colab.ipynb
├── scripts/
│   └── train_models.py
├── tests/
│   ├── smoke_test.py
│   └── ui_migration_test.py
└── src/
    ├── __init__.py
    └── modeling.py
```

## Deploy on Streamlit Community Cloud

1. Replace the contents of the existing GitHub repository with the contents of this fixed package.
2. Confirm these files are in the **repository root**:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `LICENSE`
3. Keep the entrypoint as:

```text
app.py
```

4. Python **3.14** can be used. This fixed build no longer requires switching back to Python 3.13.
5. Commit and push the changes to GitHub.
6. Streamlit Community Cloud should detect the changed dependencies and perform a clean rebuild.
7. If the old build is still displayed, open the app management menu and reboot/redeploy the app so the new dependency file is installed.

## Runtime dependencies

Normal cloud deployment uses only `requirements.txt`:

```text
streamlit==1.63.0
pandas==3.0.5
numpy==2.3.5
scikit-learn==1.8.0
joblib==1.5.3
pypdf==5.9.0
Pillow==12.3.0
pytesseract>=0.3.13,<0.4
pdf2image>=1.17.0,<2
```

Linux OCR utilities are declared in `packages.txt`:

```text
tesseract-ocr
poppler-utils
```

## Run locally

Python 3.13 or Python 3.14 can be used for inference.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

For local OCR, also install Tesseract and Poppler using your operating system's package manager.

## Smoke test without launching Streamlit

```bash
python tests/smoke_test.py
```

The test checks that the model artifact loads, the demo court notice parses, the classical models run, the NumPy DNN inference runs, and a complete impact report is produced.

## Retrain the model (optional)

PyTorch is needed only when training a new model. Install the separate training dependency file:

```bash
pip install -r requirements-training.txt
python scripts/train_models.py
```

The training script automatically exports the trained PyTorch network to NumPy arrays before saving the Streamlit model artifact. The resulting cloud app therefore remains PyTorch-free at inference time.

## Model architecture

The deployed ensemble combines:

1. **TF-IDF + structured features** for document text and parsed case attributes.
2. **Ridge regression** for continuous prototype estimates.
3. **Logistic regression** for escalation and risk categories.
4. **Multi-task neural network**, trained offline in PyTorch.
5. **NumPy forward-pass inference**, reproducing the trained DNN's linear, ReLU, BatchNorm, and output-head operations without importing PyTorch.
6. **Calibration ranges** from held-out synthetic records.
7. **Constrained reinforcement learning**, used only to select report presentation style.

## Responsible development notes

For future real-data research, use only appropriately obtained, de-identified, documented data. Separate official court outcomes from downstream third-party outcomes such as insurance changes. Evaluate by jurisdiction and time period, document missingness and selection effects, test calibration and subgroup error patterns, and require human review. Do not train on unnecessary protected or identifying attributes.

The prototype must not be used to make sentencing, guilt, eligibility, creditworthiness, or other high-impact determinations about a person. Official consequences must be verified from the appropriate court, motor-vehicle agency, insurer, or qualified professional.

## MIT License

Copyright (c) 2026 Claire Yuan. See [`LICENSE`](LICENSE).

## Credits

- **Author:** Claire Yuan
- **Advisor:** Dr. Qingyang Xiao
