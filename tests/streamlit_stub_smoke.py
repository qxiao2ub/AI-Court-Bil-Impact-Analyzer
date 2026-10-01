"""Execute app.py against a tiny Streamlit stub.

This is not a browser/UI test. It catches import-time errors, missing names, tab
unpacking mistakes, and widget-call problems in environments where the full
Streamlit dependency is unavailable.
"""
from __future__ import annotations

import os
import runpy
import sys
import tempfile
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _Context:
    def __enter__(self) -> "_Context":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def __getattr__(self, name: str):
        return getattr(ST, name)


class _Progress:
    def progress(self, _value: float) -> None:
        return None


class _Secrets(dict):
    pass


ST = types.ModuleType("streamlit")
ST.session_state = {}
ST.secrets = _Secrets()


def _cache_data(*args: Any, **kwargs: Any):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(func):
        return func

    return decorator


def _columns(spec: Any, **_kwargs: Any) -> list[_Context]:
    count = int(spec) if isinstance(spec, int) else len(spec)
    return [_Context() for _ in range(count)]


def _tabs(labels: Any) -> list[_Context]:
    return [_Context() for _ in labels]


def _selectbox(_label: str, options: Any, index: int = 0, **_kwargs: Any) -> Any:
    values = list(options)
    return values[index] if values else None


def _radio(_label: str, options: Any, **_kwargs: Any) -> Any:
    values = list(options)
    return values[0] if values else None


def _text_input(_label: str, value: str = "", **_kwargs: Any) -> str:
    return value


def _text_area(_label: str, value: str = "", *args: Any, **kwargs: Any) -> str:
    if "value" in kwargs:
        return str(kwargs["value"])
    if args:
        return str(args[0])
    return value


def _number_input(_label: str, **kwargs: Any) -> Any:
    return kwargs.get("value", 0)


def _checkbox(_label: str, value: bool = False, **_kwargs: Any) -> bool:
    return value


def _no(*_args: Any, **_kwargs: Any) -> bool:
    return False


def _none(*_args: Any, **_kwargs: Any) -> None:
    return None


ST.cache_data = _cache_data
ST.columns = _columns
ST.tabs = _tabs
ST.selectbox = _selectbox
ST.radio = _radio
ST.text_input = _text_input
ST.text_area = _text_area
ST.number_input = _number_input
ST.checkbox = _checkbox
ST.button = _no
ST.form_submit_button = _no
ST.download_button = _no
ST.file_uploader = lambda *_args, **_kwargs: None
ST.expander = lambda *_args, **_kwargs: _Context()
ST.spinner = lambda *_args, **_kwargs: _Context()
ST.form = lambda *_args, **_kwargs: _Context()
ST.progress = lambda *_args, **_kwargs: _Progress()
ST.rerun = _none
for function_name in (
    "set_page_config",
    "markdown",
    "caption",
    "code",
    "dataframe",
    "error",
    "info",
    "json",
    "link_button",
    "success",
    "warning",
    "write",
    "metric",
):
    setattr(ST, function_name, _none)

sys.modules["streamlit"] = ST

from src.bill_analysis import analyze_bill, personalize_impact  # noqa: E402


def main() -> None:
    demo_text = (ROOT / "examples" / "demo_bill.txt").read_text(encoding="utf-8")
    bundle = {
        "ref": {
            "congress": 119,
            "bill_type": "hr",
            "number": 99999,
            "bill_id": "demo-hr-99999",
            "citation": "H.R. DEMO",
            "official_url": "",
        },
        "detail": {
            "title": "Digital Skills and Rural Clinic Support Act of 2026 (fictional demonstration)",
            "introducedDate": "2026-02-12",
            "originChamber": "House",
            "latestAction": {"actionDate": "2026-02-12", "text": "Referred to committee."},
            "cboCostEstimates": [],
        },
        "actions": [{"actionDate": "2026-02-12", "text": "Introduced and referred to committee."}],
        "summaries": [{"actionDate": "2026-02-12", "text": "Creates demonstration grant programs."}],
        "amendments": [],
        "committees": [],
        "cosponsors": [],
        "related_bills": [],
        "subjects": {},
        "text_versions": [],
        "titles": [],
    }
    analysis = analyze_bill(
        demo_text,
        bundle=bundle,
        source_kind="fictional demonstration",
        citation="H.R. DEMO",
        title=bundle["detail"]["title"],
    )
    ST.session_state.update(
        {
            "analysis": analysis,
            "working_document": {
                "text": demo_text,
                "title": analysis.title,
                "citation": analysis.citation,
                "source_kind": "fictional demonstration",
                "bundle": bundle,
                "source_url": "",
                "source_name": "demo_bill.txt",
                "selected_version": {},
            },
            "personalization": personalize_impact(
                analysis,
                {
                    "age": 35,
                    "state": "MD",
                    "income_bracket": "$50,000–$99,999",
                    "occupation": "teacher",
                    "industry": "education",
                },
            ),
        }
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["APP_DB_PATH"] = str(Path(temp_dir) / "stub.sqlite3")
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")

    print("STREAMLIT STUB SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
