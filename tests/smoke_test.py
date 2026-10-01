from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bill_analysis import analyze_bill, analysis_to_json, analysis_to_markdown



def main() -> None:
    required = [
        ROOT / "app.py",
        ROOT / "requirements.txt",
        ROOT / "LICENSE",
        ROOT / "README.md",
        ROOT / ".streamlit" / "config.toml",
        ROOT / "src" / "congress_client.py",
        ROOT / "src" / "bill_analysis.py",
        ROOT / "src" / "persistence.py",
        ROOT / "src" / "ui.py",
        ROOT / "examples" / "demo_bill.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"Missing required repository files: {missing}")

    for path in [ROOT / "app.py", *sorted((ROOT / "src").glob("*.py"))]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    ui_text = (ROOT / "src" / "ui.py").read_text(encoding="utf-8")
    required_strings = [
        "Claire Yuan",
        "Dr. Qingyang Xiao",
        "CONGRESS_API_KEY",
        "Live Congress.gov bill",
        "HOW THIS AFFECTS YOU",
        "FOLLOW + CONTACT",
        "COMPARE BILLS",
        "record_visit",
    ]
    for value in required_strings:
        if value not in app_text and value not in ui_text:
            raise AssertionError(f"Expected app feature marker not found: {value}")
    for selector in ("stAppViewBlockContainer", "stMainBlockContainer", "streamlit-toolbar-safe-area"):
        if selector not in ui_text:
            raise AssertionError(f"Top-banner safe-area selector missing: {selector}")

    text = (ROOT / "examples" / "demo_bill.txt").read_text(encoding="utf-8")
    analysis = analyze_bill(
        text,
        source_kind="fictional demonstration",
        citation="H.R. DEMO",
        title="Digital Skills and Rural Clinic Support Act of 2026",
    )
    if len(analysis.sections) < 8:
        raise AssertionError("Demo section parsing did not produce the expected section map.")
    if "$120,000,000" not in analysis.fiscal.get("detected_amounts", []):
        raise AssertionError("Fiscal amount detection failed.")
    if not analysis_to_markdown(analysis).startswith("# H.R. DEMO"):
        raise AssertionError("Markdown report generation failed.")
    if '"analysis"' not in analysis_to_json(analysis):
        raise AssertionError("JSON report generation failed.")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
