# Legislative-Bill Revision Release Notes

**Author:** Claire Yuan  
**Advisor:** Dr. Qingyang Xiao  
**License:** MIT

This release replaces the earlier court-notice consequence workflow with a neutral U.S. federal legislative-bill research application.

## Implemented in this release

- Live Congress.gov lookup by citation or recent-bill selection
- Official bill metadata, actions, amendments, committees, sponsors/cosponsors, related measures, summaries, text versions, titles, and subjects
- Automatic full-text selection and download from allowlisted official hosts
- Searchable-PDF and pasted-text ingestion
- Evidence-based legislative-stage detection
- Plain-English extractive summaries and section-by-section analysis
- Stated-purpose versus operative-language view
- Conservative rider/scope-mismatch review flags
- Legal/procedural glossary, fiscal-language detection, CBO metadata, and effective-date signals
- Profile-based relevance checklist with prototype login and saved profiles
- Official source links, confidence explanations, and official-version diffing
- Sponsor/cosponsor/vote display, current-member lookup, official contact directories, and editable constituent-message drafts
- Saved/followed bills and manually refreshed in-app status-change notifications
- Side-by-side bill comparison
- Unique visitor counter based on one random identifier per Streamlit browser session

## Cloud reliability improvements

- Toolbar-safe top spacing retained from the prior UI repair
- No PyTorch, TensorFlow, Tesseract, Poppler, or Node build step
- Python 3.14-compatible pinned packages
- Transient-request retries and paginated Congress.gov collection retrieval
- Standard-library XML parsing to avoid an optional `lxml` dependency
- Searchable-PDF extraction only

## Prototype boundaries

The local SQLite database may reset on Streamlit Community Cloud. Notifications require a signed-in user to press the status-check button; there is no background email or SMS worker. The analysis is educational, does not provide legal advice, does not recommend a political position, and does not predict passage.
