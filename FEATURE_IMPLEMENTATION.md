# Feature Implementation Map

| Requested capability | Implementation |
|---|---|
| Paste, PDF, or live bill number | Four ingestion modes in `app.py` |
| Auto-detect bill status | Evidence-based rules in `detect_status()` |
| Congress.gov full text and metadata | `CongressClient.get_bill_bundle()` plus official text downloader |
| Plain-English summary | TF-IDF extractive summary plus transparent wording substitutions |
| Section-by-section breakdown | `split_sections()` and `analyze_sections()` |
| Claims versus operative effect | Purpose extraction and obligation-focused sentence selection |
| Possible riders / unrelated additions | Conservative topic-scope mismatch signal with explicit uncertainty |
| Key-term glossary | Built-in legal/procedural glossary matched against each section |
| Fiscal/CBO callout | CBO metadata, spending terms, dollar references, and section mapping |
| Personalized impact | Optional profile and relevance checklist; saved for signed-in users |
| Timeline and effective date | Official action history plus text date extraction |
| Official verification links | Bill, summary, and text links shown with results |
| Confidence/uncertainty | Source-completeness score and per-section confidence |
| Version tracking | Select two official text versions and generate a unified diff |
| Sponsor/cosponsor/vote info | Official bundle tables and recorded-vote references |
| Find/contact representatives | Congress.gov member lookup and official House/Senate directories |
| Save/follow/status changes | SQLite follows and manual in-app update checks |
| Compare two bills | Live or pasted comparison with text/topic/fiscal/status dimensions |
| User counter | One unique count per random Streamlit session hash |

## Deliberate boundaries

- No prediction of whether legislation will pass
- No recommendation to support or oppose a bill
- No automatic lobbying message transmission
- No assertion that a scope mismatch is legally or procedurally an improper rider
- No claimed individualized legal, tax, benefit, or budget outcome
- No production-grade identity, persistence, or background notification service in the free cloud demo
