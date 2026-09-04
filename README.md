# AI Court-Bill Impact Analyzer

**Author:** Claire Yuan  
**Advisor:** Dr. Qingyang Xiao  
**License:** MIT

A Streamlit research prototype derived from Claire Yuan's Colab notebook. The app accepts a redacted court-issued notice, citation, fine, or bill; extracts model features; runs classical machine-learning and PyTorch multi-task models; and produces a plain-language impact report with calibrated prediction ranges.

> **Important:** The bundled model is trained entirely on synthetic `Demo-*` records. It is an educational prototype, not legal advice and not a tool for guilt, sentencing, eligibility, creditworthiness, or other high-impact decisions.

## Features

- Streamlit Community Cloud-ready `app.py` entrypoint
- PDF text extraction with OCR fallback
- Image OCR through Tesseract
- TXT, CSV, and Markdown ingestion
- Transparent bill-feature parser
- scikit-learn Ridge and logistic-regression pipelines
- PyTorch multi-task deep neural network
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
├── requirements.txt
├── packages.txt
├── .streamlit/
│   └── config.toml
├── examples/
│   └── demo_speeding_notice.txt
├── models/
│   └── claire_yuan_court_bill_impact_models.joblib
├── notebooks/
│   └── Claire_Yuan_AI_Court_Bill_Impact_Analyzer_Colab.ipynb
├── scripts/
│   └── train_models.py
└── src/
    └── modeling.py
```

## Run locally

Use Python 3.13 to match the pre-trained model environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For local OCR, install Tesseract and Poppler with your operating system's package manager. `packages.txt` handles these Linux dependencies automatically on Streamlit Community Cloud.

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository and upload all files/folders from this package.
2. In Streamlit Community Cloud, create a new app from that GitHub repository.
3. Set the entrypoint to `app.py`.
4. In **Advanced settings**, select Python 3.13 to match the bundled model artifact.
5. Deploy.

The app keeps `requirements.txt` in the repository root, requests the CPU-only PyTorch wheel for a lighter cloud deployment, and uses `packages.txt` for Tesseract/Poppler.

## Retrain the bundled model

The repository includes the training script used to generate the Streamlit model artifact from the notebook's synthetic-data logic:

```bash
python scripts/train_models.py
```

This overwrites:

```text
models/claire_yuan_court_bill_impact_models.joblib
```

## Responsible development notes

For any future real-data research, use only appropriately obtained, de-identified, documented data. Separate official court outcomes from downstream third-party outcomes such as insurance changes. Evaluate by jurisdiction and time period, document missingness and selection effects, test calibration and subgroup error patterns, and require human review. Do not train on unnecessary protected or identifying attributes.

## MIT License

Copyright (c) 2026 Claire Yuan. See [`LICENSE`](LICENSE).

## Credits

- **Author:** Claire Yuan
- **Advisor:** Dr. Qingyang Xiao
