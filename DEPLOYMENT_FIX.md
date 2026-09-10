# Streamlit Community Cloud Deployment Fix

## Symptom

The app remains on **"Your app is in the oven"** while the build log reaches the Python dependency stage and appears to stop after package resolution.

## Root cause addressed by this revision

The previous cloud dependency set included both an old pandas pin and a full PyTorch installation. On newer Streamlit Community Cloud Python runtimes this can make dependency installation extremely slow or force unnecessary heavy package work.

## What changed

1. Removed PyTorch from `requirements.txt`.
2. Converted the already-trained neural-network weights from PyTorch tensors to NumPy arrays.
3. Reimplemented only the DNN **inference** forward pass in `src/modeling.py` using NumPy.
4. Updated pandas to a Python-3.14-compatible release.
5. Removed Matplotlib and explicit SciPy from normal cloud dependencies because the app does not import Matplotlib and scikit-learn resolves its own compatible SciPy dependency.
6. Added a separate `requirements-training.txt`; PyTorch is now needed only for retraining.
7. Updated `scripts/train_models.py` so future training also exports a cloud-ready NumPy DNN state.
8. Added `tests/smoke_test.py` for deployment-artifact validation.

## Redeploy procedure

Replace the old GitHub repository contents with this revision, commit the changes, and allow Streamlit Community Cloud to rebuild. Keep `app.py` as the entrypoint. Python 3.14 is supported by this repository.

If Streamlit continues to display a cached old build, reboot/redeploy the application from its management menu after the GitHub commit is visible.
