# Streamlit top-banner visibility fix

The Streamlit Community Cloud toolbar is fixed at the top of the browser. The
previous UI started the custom `court_impact` banner almost at `0px`, so the
Cloud toolbar overlapped and hid the upper portion of that banner.

## Changes in this build

- Reserves a `4.75rem` desktop safe area above the application content.
- Applies the fix to current and legacy Streamlit block-container selectors.
- Uses a `4.5rem` mobile safe area.
- Gives the Streamlit header an opaque paper-colored background and separator.
- Adds scroll padding so in-page anchor navigation is not hidden by the toolbar.
- Keeps the custom navigation banner above ordinary page content with a stable
  stacking context.

No model, report, OCR, licensing, author, or advisor functionality was removed.
