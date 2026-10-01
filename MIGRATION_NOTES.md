# Migration from the Earlier Court-Notice Repository

This revision changes the product from a court-notice consequence prototype to a U.S. federal legislative-bill research app.

Before deploying, remove the earlier repository contents or replace the repository with this package. In particular:

- Replace the old `app.py` and `src/` modules.
- Replace `requirements.txt`.
- Replace the old `packages.txt` with the empty `packages.txt` included here. This removes Tesseract/Poppler apt installation and prevents the earlier long “app is in the oven” build.
- Remove old trained model artifacts and court-notice demo files if they remain in the GitHub repository.
- Add `CONGRESS_API_KEY` in Streamlit secrets.

The new app keeps the editorial UI and the toolbar-safe top spacing while implementing live Congress.gov retrieval, legislative analysis, personalization, following, comparison, and visitor counting.
