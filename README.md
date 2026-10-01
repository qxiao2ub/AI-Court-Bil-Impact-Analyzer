# AI Legislative Bill Impact Analyzer

**Author:** Claire Yuan  
**Advisor:** Dr. Qingyang Xiao  
**License:** MIT

A Streamlit Community Cloud-ready educational application that helps users navigate U.S. federal legislative bills without replacing the official source. It can retrieve a live bill from the official Congress.gov API, analyze a searchable PDF, or process pasted bill text.

> This is a neutral educational research prototype. It does not provide legal advice, endorse or oppose legislation, determine constitutionality, or predict whether a bill will pass.

## Major capabilities

### Input and ingestion

- Search a live federal measure by citation, including `H.R.`, `S.`, `H.J.Res.`, `S.J.Res.`, `H.Con.Res.`, `S.Con.Res.`, `H.Res.`, and `S.Res.`
- Browse recent measures by Congress and type
- Pull official bill detail, actions, amendments, committees, cosponsors, related bills, subjects, summaries, text versions, and titles from Congress.gov
- Download a selected official full-text version from Congress.gov, GovInfo, or GPO
- Upload a searchable PDF or paste bill text manually
- Auto-detect the procedural stage from official action evidence

### Core analysis

- Plain-English extractive summary
- Section-by-section summaries with original section text available for verification
- Operative-language extraction for duties, permissions, amendments, appropriations, reports, and effective dates
- “Stated purpose” versus “operative text” view
- Low-to-medium-confidence scope-mismatch review flags for possible unrelated provisions
- Legal and procedural glossary
- Fiscal-language and dollar-reference detection
- Congress.gov CBO cost-estimate metadata display when supplied by the API
- Confidence and uncertainty reporting
- Markdown and JSON report downloads

### Personalized relevance

- Optional profile fields: age, state, income bracket, occupation, industry, student, veteran, business owner, and caregiver
- A “How this may affect you” checklist that identifies provisions and eligibility questions to verify
- Prototype account login and saved profiles

### Trust, tracking, and civic engagement

- Official text, summary, and bill-page links remain visible
- Official text-version comparison with a unified line diff
- Sponsor, cosponsor, committee, related-bill, and vote-reference displays
- Current-member lookup by state and optional House district
- Official House and Senate contact-directory buttons
- Neutral, editable constituent-message builder; the app never chooses a position or sends a message
- Followed bills and in-app status-change notifications when the user manually checks for updates

### Comparison and usage analytics

- Side-by-side comparison of a current bill with another live or pasted bill
- TF-IDF text similarity, shared topics, status, section counts, and fiscal-signal comparison
- Unique visitor counter: one count per Streamlit browser session

## Explainable AI approach

The cloud app intentionally avoids a large GPU model so it starts reliably on Streamlit Community Cloud. The analysis pipeline uses:

1. Section-header parsing and fallback document chunking
2. Sentence tokenization
3. TF-IDF sentence representations
4. Centrality and position scoring for extractive summaries
5. Transparent legalese substitutions
6. Rule-based status, obligation, fiscal, date, and topic extraction
7. Confidence based on source completeness and text coverage

The app does not invent missing provisions. Its paraphrases should always be checked against the official text.

## Repository structure

```text
.
├── app.py
├── requirements.txt
├── packages.txt
├── LICENSE
├── README.md
├── DEPLOYMENT.md
├── FEATURE_IMPLEMENTATION.md
├── MIGRATION_NOTES.md
├── RELEASE_NOTES.md
├── VALIDATION.md
├── examples/
│   └── demo_bill.txt
├── src/
│   ├── bill_analysis.py
│   ├── congress_client.py
│   ├── persistence.py
│   └── ui.py
├── tests/
│   ├── smoke_test.py
│   ├── streamlit_stub_smoke.py
│   ├── test_core.py
│   └── test_repository_static.py
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

## Local run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Manual-paste, PDF, and fictional-demo modes work without an API key.

## Congress.gov API setup

1. Request a free key from the official Congress.gov API sign-up page.
2. For local development, create `.streamlit/secrets.toml`:

```toml
CONGRESS_API_KEY = "your-key-here"
```

3. For Streamlit Community Cloud, add the same entry under **App settings → Secrets**.
4. Do not commit the real secret.

## Streamlit Community Cloud deployment

1. Create a GitHub repository.
2. Upload the **contents of this folder** to the repository root.
3. In Streamlit Community Cloud, select the repository and branch.
4. Set **Main file path** to `app.py`.
5. Select **Python 3.14** in Advanced settings.
6. Add `CONGRESS_API_KEY` under Secrets.
7. Deploy or reboot the app.

The repository pins Streamlit 1.64.0 and Python-3.14-compatible scientific-package releases. The top layout reserves a safe area below Streamlit Community Cloud’s fixed Share/Edit toolbar, preventing the custom navigation banner from being hidden.

The Congress.gov client retries transient API failures and paginates collection endpoints so actions, amendments, cosponsors, titles, summaries, and text-version lists are not silently limited to the API's first page.

## Prototype account and notification limitations

The classroom prototype stores accounts, profiles, follows, visitor sessions, and notifications in local SQLite. Passwords use salted PBKDF2-HMAC-SHA256 hashes. Streamlit Community Cloud’s local filesystem is not a production database and can reset during redeploys, restarts, or platform maintenance. Notifications are created only when a signed-in user manually checks followed bills; this repository does not run a background email or SMS service.

A production version should use managed authentication, a persistent managed database, encryption, role-based access, audit logs, secret management, and a scheduled notification worker.

## PDF limitation

The lightweight cloud build extracts embedded text from searchable PDFs. Scanned image-only PDFs require OCR, which is deliberately not bundled because system OCR packages previously caused slow Streamlit builds. Users can paste OCR text manually.

## Privacy

Do not enter sensitive personal information. The visitor counter stores a random session hash and timestamps in the local prototype database; it does not intentionally store IP addresses. A production privacy notice should reflect the hosting platform, analytics, authentication provider, and data-retention policy actually used.

## Validation

Run the repository checks from the project root:

```bash
python -m compileall -q app.py src tests
python -m unittest discover -s tests -p "test_*.py" -v
python tests/smoke_test.py
python tests/streamlit_stub_smoke.py
```

The stub smoke test executes the full Streamlit entrypoint against a minimal local test double. It catches import-time errors and widget/tab integration mistakes without requiring a browser. A final deployment should still be verified in Streamlit Community Cloud with a valid Congress.gov API key.

## License

Released under the [MIT License](LICENSE). Copyright © 2026 Claire Yuan.
