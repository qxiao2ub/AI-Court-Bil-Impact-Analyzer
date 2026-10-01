# Validation Record

Validated on 2026-09-24 in the provided build environment.

## Passed checks

- Python bytecode compilation for `app.py`, `src/`, and `tests/`
- 16 unit/static repository tests
- Core analysis smoke test
- Full Streamlit entrypoint execution against a local Streamlit test double
- MIT license, root dependency files, toolbar-safe CSS, author/advisor credits, visitor counter, Congress.gov workflow, personalization, follow/contact, comparison, and version-diff markers

## Important validation boundary

The build environment did not contain Streamlit and did not allow package downloads or live network/API calls. Therefore, an actual browser-based Streamlit server and a live Congress.gov request were not executed here. The pinned package releases and deployment layout were checked against current official package and Streamlit documentation, and the repository includes a fictional demo mode that works without an API key.
