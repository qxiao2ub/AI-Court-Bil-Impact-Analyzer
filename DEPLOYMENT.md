# Streamlit Community Cloud Deployment

**Author:** Claire Yuan  
**Advisor:** Dr. Qingyang Xiao

## Required repository settings

- Branch: `main` (or the branch containing the app)
- Main file path: `app.py`
- Python: `3.14`

## Secret

Add this under **App settings → Secrets**:

```toml
CONGRESS_API_KEY = "your-free-congress-gov-api-key"
```

Do not upload `.streamlit/secrets.toml` to GitHub. The included `.gitignore` excludes it.

## Why this build starts quickly

- No PyTorch or TensorFlow runtime
- No OCR system package installation
- No Node/React build step
- Local TF-IDF and rules for explainable analysis
- Official text downloaded only when a user requests a live bill or version comparison
- API responses cached for 15 minutes during the running app instance
- Empty root `packages.txt`, so no Tesseract/Poppler apt build is triggered

## Clean replacement of the earlier repository

Upload the contents of this package to the repository root and remove the old court-notice files, model artifacts, and dependency files. In particular, replace both `requirements.txt` and `packages.txt`; leaving the old OCR/PyTorch dependencies in GitHub can cause a slow or failed Community Cloud rebuild.

## Banner-safe layout

The CSS reserves `5.25rem` above the first custom navigation element and applies the rule to current and legacy Streamlit block-container selectors. The Streamlit header remains visible with an opaque background and a high stacking order. This prevents the Share/Edit toolbar from covering the app banner.

## Troubleshooting

### Live lookup says an API key is missing

Confirm the secret name is exactly `CONGRESS_API_KEY`, then reboot the app.

### A bill has metadata but no text

Newly introduced bills may not yet have a published downloadable text version. Use the official bill link and try again later.

### A PDF produces too little text

The file is probably scanned. Convert it to a searchable PDF or paste OCR text. This cloud build intentionally omits OCR packages.

### Accounts or follows disappeared

Local SQLite is a classroom prototype and Streamlit Cloud storage may be reset. Use a managed external database for persistence.
