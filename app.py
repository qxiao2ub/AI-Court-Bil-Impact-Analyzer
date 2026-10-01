from __future__ import annotations

import html
import io
import json
import os
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from src.bill_analysis import (
    BillAnalysis,
    analysis_to_json,
    analysis_to_markdown,
    analyze_bill,
    build_contact_message,
    compare_analyses,
    extract_votes,
    personalize_impact,
    sponsor_rows,
    version_diff,
)
from src.congress_client import (
    BILL_TYPE_DISPLAY,
    BillRef,
    CongressAPIError,
    CongressClient,
    current_congress_fallback,
    latest_text_version,
    parse_bill_citation,
    preferred_text_format,
)
from src.persistence import (
    authenticate,
    create_user,
    follow_bill,
    get_profile,
    init_db,
    list_follows,
    list_notifications,
    mark_notifications_read,
    record_visit,
    save_profile,
    unfollow_bill,
    update_follow_and_notify,
    visitor_stats,
)
from src.ui import callout, footer, hero, inject_styles, metric_cards, section_intro, source_box, top_chrome


ROOT = Path(__file__).resolve().parent
DEMO_PATH = ROOT / "examples" / "demo_bill.txt"
HOUSE_CONTACT_URL = "https://www.house.gov/representatives/find-your-representative"
SENATE_CONTACT_URL = "https://www.senate.gov/senators/senators-contact.htm"
API_SIGNUP_URL = "https://api.congress.gov/sign-up/"

st.set_page_config(
    page_title="AI Legislative Bill Impact Analyzer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_styles()


# -----------------------------------------------------------------------------
# Persistence and visitor counter
# -----------------------------------------------------------------------------
try:
    init_db()
    if "visit_session_id" not in st.session_state:
        st.session_state["visit_session_id"] = str(uuid.uuid4())
        st.session_state["visitor_count"] = record_visit(st.session_state["visit_session_id"])
    stats = visitor_stats()
    visitor_count = int(stats.get("unique_sessions", st.session_state.get("visitor_count", 0)))
except Exception as exc:  # The app remains usable if Streamlit's ephemeral disk is unavailable.
    visitor_count = 0
    st.session_state["persistence_error"] = str(exc)


def secret_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        candidate = st.secrets.get(name, "")
        return str(candidate).strip() if candidate else ""
    except Exception:
        return ""


def active_api_key() -> str:
    return str(st.session_state.get("session_api_key") or secret_value("CONGRESS_API_KEY") or "").strip()


def demo_bundle() -> dict[str, Any]:
    return {
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
            "policyArea": {"name": "Education"},
            "latestAction": {
                "actionDate": "2026-02-12",
                "text": "Fictional demonstration: referred to the Committees on Education and the Workforce and Energy and Commerce.",
            },
            "sponsors": [
                {
                    "fullName": "Representative Demo Sponsor",
                    "party": "—",
                    "state": "—",
                    "district": "—",
                }
            ],
            "cboCostEstimates": [],
        },
        "actions": [
            {
                "actionDate": "2026-02-12",
                "text": "Fictional demonstration: introduced in House and referred to committee.",
            }
        ],
        "amendments": [],
        "committees": [
            {"name": "Committee on Education and the Workforce (fictional demo)"},
            {"name": "Committee on Energy and Commerce (fictional demo)"},
        ],
        "cosponsors": [],
        "related_bills": [],
        "subjects": {"legislativeSubjects": [{"name": "Digital skills"}, {"name": "Rural health"}]},
        "summaries": [
            {
                "actionDate": "2026-02-12",
                "text": "Fictional demonstration summary: establishes rural digital-skills and telehealth grant programs and authorizes appropriations.",
                "versionCode": "00",
            }
        ],
        "text_versions": [],
        "titles": [],
        "errors": {},
    }


def extract_pdf_text(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 80:
        raise ValueError(
            "The PDF contains too little extractable text. Upload a searchable/text PDF or paste the bill text. "
            "This lightweight Streamlit build does not install OCR system packages."
        )
    return text


@st.cache_data(ttl=900, show_spinner=False)
def cached_bill_bundle(api_key: str, congress: int, bill_type: str, number: int) -> dict[str, Any]:
    client = CongressClient(api_key)
    return client.get_bill_bundle(BillRef(int(congress), bill_type, int(number)))


@st.cache_data(ttl=86_400, show_spinner=False)
def cached_current_congress(api_key: str) -> int:
    return CongressClient(api_key).get_current_congress()


@st.cache_data(ttl=900, show_spinner=False)
def cached_official_text(api_key: str, url: str, source_format: str) -> dict[str, str]:
    client = CongressClient(api_key)
    downloaded = client.download_text_url(url, source_format)
    return asdict(downloaded)


@st.cache_data(ttl=900, show_spinner=False)
def cached_recent_bills(api_key: str, congress: int, bill_type: str, limit: int = 40) -> list[dict[str, Any]]:
    return CongressClient(api_key).list_bills(congress, bill_type, limit=limit)


@st.cache_data(ttl=900, show_spinner=False)
def cached_members(api_key: str, state: str, district: int | None) -> list[dict[str, Any]]:
    return CongressClient(api_key).get_current_members(state, district)


def version_label(version: Mapping[str, Any], index: int = 0) -> str:
    date = str(version.get("date") or "date unavailable")
    version_type = str(version.get("type") or version.get("versionName") or "official text")
    return f"{date} · {version_type} · version {index + 1}"


def official_text_for_version(api_key: str, version: Mapping[str, Any]) -> dict[str, str]:
    fmt = preferred_text_format(dict(version))
    if not fmt:
        raise CongressAPIError("The selected official version does not include a downloadable text format.")
    return cached_official_text(api_key, str(fmt.get("url") or ""), str(fmt.get("type") or "Official text"))


def store_document(
    *,
    text: str,
    title: str,
    citation: str,
    source_kind: str,
    bundle: Mapping[str, Any] | None = None,
    source_url: str = "",
    source_name: str = "",
    selected_version: Mapping[str, Any] | None = None,
) -> None:
    st.session_state["working_document"] = {
        "text": text,
        "title": title,
        "citation": citation,
        "source_kind": source_kind,
        "bundle": dict(bundle or {}),
        "source_url": source_url,
        "source_name": source_name,
        "selected_version": dict(selected_version or {}),
    }
    # A new source/version invalidates analysis artifacts created for the prior
    # document. Clearing them prevents a stale version diff, comparison, or
    # profile result from being displayed beside a newly loaded bill.
    for stale_key in ("analysis", "personalization", "version_diff", "compare_analysis"):
        st.session_state.pop(stale_key, None)


def load_live_bill(api_key: str, ref: BillRef) -> None:
    with st.spinner(f"Retrieving {ref.display} metadata, actions, summaries, and text versions from Congress.gov…"):
        bundle = cached_bill_bundle(api_key, ref.congress, ref.bill_type, ref.number)
    versions = [item for item in bundle.get("text_versions", []) if isinstance(item, dict)]
    version = latest_text_version(versions)
    if not version:
        raise CongressAPIError(
            "Congress.gov returned the bill metadata, but no downloadable full-text version is currently available. "
            "Open the official source or try again after a text version is published."
        )
    with st.spinner("Downloading the latest official bill text…"):
        downloaded = official_text_for_version(api_key, version)
    detail = bundle.get("detail") if isinstance(bundle.get("detail"), Mapping) else {}
    store_document(
        text=downloaded["text"],
        title=str(detail.get("title") or ref.display),
        citation=ref.citation,
        source_kind="official Congress.gov text",
        bundle=bundle,
        source_url=downloaded.get("source_url", ref.official_url),
        source_name=ref.bill_id,
        selected_version=version,
    )


def render_api_key_box(prefix: str) -> str:
    existing = active_api_key()
    if existing:
        st.success("Congress.gov API access is configured for this session.")
        return existing
    st.info(
        "Live lookup requires a free Congress.gov API key. Add `CONGRESS_API_KEY` to Streamlit secrets, "
        "or enter a key below for this browser session. The key is not written to the app database."
    )
    entered = st.text_input("Congress.gov API key", type="password", key=f"{prefix}_api_key")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("USE API KEY FOR THIS SESSION", key=f"{prefix}_save_api_key", use_container_width=True):
            if entered.strip():
                st.session_state["session_api_key"] = entered.strip()
                st.rerun()
            st.error("Enter an API key first.")
    with c2:
        st.link_button("GET A FREE OFFICIAL API KEY ↗", API_SIGNUP_URL, use_container_width=True)
    return active_api_key()


def source_link_override(analysis: BillAnalysis, url: str) -> None:
    if url:
        analysis.source_links["official_bill_page"] = url
        analysis.source_links["official_text"] = url
        if not analysis.source_links.get("official_summary"):
            analysis.source_links["official_summary"] = url


def current_analysis() -> BillAnalysis | None:
    value = st.session_state.get("analysis")
    return value if isinstance(value, BillAnalysis) else None


def current_document() -> dict[str, Any] | None:
    value = st.session_state.get("working_document")
    return value if isinstance(value, dict) else None


def action_snapshot(bundle: Mapping[str, Any]) -> tuple[str, str]:
    detail = bundle.get("detail") if isinstance(bundle.get("detail"), Mapping) else {}
    latest = detail.get("latestAction") if isinstance(detail.get("latestAction"), Mapping) else {}
    if latest:
        return str(latest.get("actionDate") or ""), str(latest.get("text") or "")
    actions = [item for item in bundle.get("actions", []) if isinstance(item, Mapping)]
    if not actions:
        return "", ""
    latest_action = max(actions, key=lambda item: str(item.get("actionDate") or ""))
    return str(latest_action.get("actionDate") or ""), str(latest_action.get("text") or "")


def member_display(member: Mapping[str, Any]) -> dict[str, Any]:
    terms = member.get("terms")
    term_list: list[Any] = []
    if isinstance(terms, dict):
        term_list = terms.get("item") if isinstance(terms.get("item"), list) else []
    elif isinstance(terms, list):
        term_list = terms
    latest_term = term_list[-1] if term_list else {}
    if not isinstance(latest_term, Mapping):
        latest_term = {}
    return {
        "Name": member.get("name") or member.get("directOrderName") or "Unknown",
        "Party": member.get("partyName") or member.get("party") or latest_term.get("partyName") or "",
        "State": member.get("state") or latest_term.get("stateCode") or "",
        "District": member.get("district") or latest_term.get("district") or "",
        "Chamber": latest_term.get("chamber") or "",
        "Bioguide ID": member.get("bioguideId") or "",
    }


def render_analysis_results(analysis: BillAnalysis, document: Mapping[str, Any]) -> None:
    st.markdown('<div class="bb-rule"></div>', unsafe_allow_html=True)
    section_intro(
        "analysis results / official text remains primary",
        f"{analysis.citation}: {analysis.title}",
        "The AI output is a navigational layer over the source text. Use the official links and section text to verify every paraphrase.",
    )
    metric_cards(
        [
            ("Legislative stage", analysis.status.label.title(), f"{analysis.status.confidence} source confidence"),
            ("Analysis confidence", str(analysis.confidence.get("label", "unknown")).title(), f"{analysis.confidence.get('score', 0)}/{analysis.confidence.get('maximum', 9)} source score"),
            ("Sections mapped", str(len(analysis.sections)), f"{analysis.metadata.get('word_count', 0):,} words analyzed"),
        ]
    )
    source_box(analysis.source_links)
    for warning in analysis.warnings:
        st.caption(f"⚠ {warning}")

    overview, sections_tab, purpose_tab, fiscal_tab, process_tab, glossary_tab, versions_tab, metadata_tab = st.tabs(
        [
            "OVERVIEW",
            "SECTION BREAKDOWN",
            "CLAIMS VS OPERATIVE TEXT",
            "FISCAL / CBO",
            "STATUS + TIMELINE",
            "GLOSSARY",
            "VERSIONS",
            "OFFICIAL METADATA",
        ]
    )

    with overview:
        section_intro("plain-English AI summary", "What the bill says, condensed.", "Extractive NLP selects and simplifies the most central provisions while preserving a path back to the official text.")
        callout(analysis.plain_summary, "Read the source sections below before relying on this paraphrase.", "plain-English summary")
        if analysis.topics:
            st.markdown("**Detected subject areas:** " + " · ".join(analysis.topics))
        st.markdown("### Confidence and uncertainty")
        for reason in analysis.confidence.get("reasons", []):
            st.write(f"- {reason}")
        st.info(str(analysis.confidence.get("uncertainty_note", "")))
        st.markdown("### Official summary, when available")
        summaries = [item for item in document.get("bundle", {}).get("summaries", []) if isinstance(item, Mapping)]
        if summaries:
            newest = max(summaries, key=lambda item: str(item.get("updateDate") or item.get("actionDate") or ""))
            official_summary = re.sub(r"<[^>]+>", " ", str(newest.get("text") or ""))
            official_summary = re.sub(r"\s+", " ", official_summary).strip()
            st.write(official_summary or "The official summary record did not include readable text.")
        else:
            st.caption("No official summary was returned for this source. This is common for newly introduced measures or manual uploads.")

    with sections_tab:
        section_intro("long-form bill navigation", "Every section gets its own map.", "Sections are summarized separately so a long measure is not reduced to one global paragraph.")
        section_rows = [
            {
                "Section": item.number,
                "Title": item.title,
                "Words": item.word_count,
                "Topics": ", ".join(item.topics),
                "Confidence": item.confidence,
            }
            for item in analysis.sections
        ]
        st.dataframe(pd.DataFrame(section_rows), use_container_width=True, hide_index=True)
        for item in analysis.sections:
            with st.expander(f"SECTION {item.number} — {item.title}", expanded=False):
                st.markdown(f"**Plain-English section summary:** {item.summary}")
                if item.operative_points:
                    st.markdown("**Operative points detected:**")
                    for point in item.operative_points:
                        st.write(f"- {point}")
                cols = st.columns(3)
                cols[0].write("**Topics**")
                cols[0].write(", ".join(item.topics) or "No clear topic")
                cols[1].write("**Dates / deadlines**")
                cols[1].write(", ".join(item.dates) or "None detected")
                cols[2].write("**Dollar references**")
                cols[2].write(", ".join(item.dollar_amounts) or "None detected")
                if item.key_terms:
                    st.markdown("**Inline term definitions:**")
                    for term in item.key_terms:
                        definition = analysis.glossary.get(term, "See the bill's definitions section and official procedural guidance.")
                        st.write(f"- **{term.title()}** — {definition}")
                with st.expander("VIEW SOURCE TEXT FOR THIS SECTION", expanded=False):
                    st.text_area(
                        f"Official/source text — section {item.number}",
                        value=item.raw_text,
                        height=260,
                        disabled=True,
                        key=f"section_raw_{item.number}_{hash(item.title)}",
                    )

    with purpose_tab:
        section_intro("purpose versus operation", "What it says it is for — and what the clauses do.", "The left side is derived from the title, findings, and official summary. The right side emphasizes mandatory, permissive, amending, spending, and reporting language.")
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("### Stated purpose / official framing")
            st.write(analysis.stated_purpose)
        with right:
            st.markdown("### Operative effect detected in the text")
            st.write(analysis.operative_effect)
        st.markdown("### Possible scope-mismatch review signals")
        if analysis.scope_mismatch_flags:
            st.warning(
                "These flags identify vocabulary that appears outside the bill's main detected topics. They do not prove that a section is a rider or unrelated amendment."
            )
            st.dataframe(pd.DataFrame(analysis.scope_mismatch_flags), use_container_width=True, hide_index=True)
        else:
            st.success("No strong section-level scope mismatch was detected by the vocabulary heuristic.")
        amendments = [item for item in document.get("bundle", {}).get("amendments", []) if isinstance(item, Mapping)]
        st.markdown("### Linked amendments")
        if amendments:
            amendment_rows = []
            for item in amendments:
                amendment_rows.append(
                    {
                        "Amendment": item.get("number") or item.get("type") or item.get("url") or "Record",
                        "Purpose": item.get("purpose") or item.get("description") or "",
                        "Latest action": (item.get("latestAction") or {}).get("text") if isinstance(item.get("latestAction"), Mapping) else "",
                        "URL": item.get("url") or "",
                    }
                )
            st.dataframe(pd.DataFrame(amendment_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No amendment records were returned for this source.")

    with fiscal_tab:
        section_intro("budget lens", "Spending, revenue, appropriations, and CBO records.", "The app highlights text signals and Congress.gov CBO metadata. It does not calculate a federal budget score.")
        fiscal = analysis.fiscal
        c1, c2, c3 = st.columns(3)
        c1.metric("CBO estimate record", "Available" if fiscal.get("has_cbo_estimate") else "Not returned")
        c2.metric("Dollar references", len(fiscal.get("detected_amounts", [])))
        c3.metric("Fiscal terms", len(fiscal.get("detected_terms", [])))
        st.info(fiscal.get("caution", ""))
        if fiscal.get("cbo_estimates"):
            st.markdown("### Congress.gov CBO cost-estimate metadata")
            st.dataframe(pd.DataFrame(fiscal["cbo_estimates"]), use_container_width=True, hide_index=True)
            cbo_links = []
            for index, estimate in enumerate(fiscal["cbo_estimates"], start=1):
                if not isinstance(estimate, Mapping):
                    continue
                url = str(estimate.get("url") or estimate.get("cboUrl") or "").strip()
                if url:
                    label = str(estimate.get("title") or estimate.get("description") or f"CBO estimate {index}")
                    cbo_links.append((label[:90], url))
            for label, url in cbo_links[:8]:
                st.link_button(f"OPEN CBO RECORD — {label} ↗", url, use_container_width=True)
        if fiscal.get("detected_terms"):
            st.markdown("**Detected fiscal language:** " + ", ".join(fiscal["detected_terms"]))
        if fiscal.get("section_amounts"):
            st.markdown("### Dollar references by section")
            st.dataframe(pd.DataFrame(fiscal["section_amounts"]), use_container_width=True, hide_index=True)
        elif not fiscal.get("cbo_estimates"):
            st.caption("No CBO estimate record or explicit dollar amount was detected in the retrieved material.")

    with process_tab:
        section_intro("legislative process", "Current stage, evidence, and effective-date signals.", "Status is inferred from official action history when available. Manual uploads receive a conservative low-confidence label.")
        callout(analysis.status.label.title(), analysis.status.explanation, f"{analysis.status.confidence} confidence")
        if analysis.status.evidence:
            st.markdown("### Status evidence")
            for item in analysis.status.evidence:
                st.write(f"- {item}")
        if analysis.timeline:
            st.markdown("### Timeline")
            st.dataframe(pd.DataFrame(analysis.timeline), use_container_width=True, hide_index=True)
        else:
            st.caption("No official action timeline was available.")

    with glossary_tab:
        section_intro("inline terminology", "Legal and procedural terms, decoded.", "Definitions are general educational explanations. The bill's own definitions control within the legislation.")
        if analysis.glossary:
            for term, definition in analysis.glossary.items():
                st.markdown(f"**{term.title()}** — {definition}")
        else:
            st.caption("No glossary terms from the built-in dictionary were detected.")

    with versions_tab:
        section_intro("amendment awareness", "Compare official text versions line by line.", "Bills change during the process. Select two Congress.gov text versions to see additions and removals.")
        versions = [item for item in document.get("bundle", {}).get("text_versions", []) if isinstance(item, Mapping)]
        api_key = active_api_key()
        if len(versions) >= 2 and api_key:
            options = list(range(len(versions)))
            left_index = st.selectbox(
                "Older/base version",
                options,
                index=max(0, len(versions) - 1),
                format_func=lambda i: version_label(versions[i], i),
                key="version_left",
            )
            right_index = st.selectbox(
                "Newer/comparison version",
                options,
                index=0,
                format_func=lambda i: version_label(versions[i], i),
                key="version_right",
            )
            if st.button("COMPARE SELECTED OFFICIAL VERSIONS", key="run_version_diff", use_container_width=True):
                try:
                    with st.spinner("Downloading and comparing official versions…"):
                        old = official_text_for_version(api_key, versions[left_index])
                        new = official_text_for_version(api_key, versions[right_index])
                        diff = version_diff(old["text"], new["text"])
                    st.session_state["version_diff"] = diff
                except Exception as exc:
                    st.error(f"Version comparison failed: {exc}")
            diff = st.session_state.get("version_diff")
            if isinstance(diff, Mapping):
                c1, c2 = st.columns(2)
                c1.metric("Added lines", diff.get("added_lines", 0))
                c2.metric("Removed lines", diff.get("removed_lines", 0))
                st.write(diff.get("summary", ""))
                diff_text = str(diff.get("unified_diff") or "No line-level changes.")
                st.code(diff_text, language="diff")
                st.download_button(
                    "DOWNLOAD VERSION DIFF",
                    data=diff_text,
                    file_name=f"{re.sub(r'[^A-Za-z0-9_-]+', '_', analysis.citation)}_version_diff.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="download_version_diff",
                )
                if diff.get("truncated"):
                    st.warning("The displayed diff was truncated for browser performance.")
        elif versions:
            st.info("One official text version is available. Version comparison becomes available after another version is published.")
        else:
            st.info("Version tracking requires a live Congress.gov bill with at least two published text versions.")
        if versions:
            st.markdown("### Published versions")
            rows = []
            for index, version in enumerate(versions):
                fmt = preferred_text_format(dict(version))
                rows.append(
                    {
                        "Version": version_label(version, index),
                        "Type": version.get("type") or "",
                        "Date": version.get("date") or "",
                        "Official format": (fmt or {}).get("type") or "",
                        "Official URL": (fmt or {}).get("url") or "",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with metadata_tab:
        section_intro("source record", "Sponsors, cosponsors, committees, votes, and related measures.", "These records come from the current source bundle. Open official links for the complete and most current record.")
        bundle = document.get("bundle", {})
        sponsors = sponsor_rows(bundle)
        if sponsors:
            st.markdown("### Sponsor and cosponsors")
            st.dataframe(pd.DataFrame(sponsors), use_container_width=True, hide_index=True)
        committees = [item for item in bundle.get("committees", []) if isinstance(item, Mapping)]
        if committees:
            st.markdown("### Committees")
            st.dataframe(pd.DataFrame(committees), use_container_width=True, hide_index=True)
        votes = extract_votes(bundle)
        st.markdown("### Recorded vote links / roll-call references")
        if votes:
            st.dataframe(pd.DataFrame(votes), use_container_width=True, hide_index=True)
        else:
            action_votes = []
            for action in bundle.get("actions", []):
                if not isinstance(action, Mapping):
                    continue
                text = str(action.get("text") or "")
                if re.search(r"\bvote|roll(?: call| no\.)|yeas|nays\b", text, flags=re.I):
                    action_votes.append({"date": action.get("actionDate") or "", "action": text})
            if action_votes:
                st.dataframe(pd.DataFrame(action_votes), use_container_width=True, hide_index=True)
            else:
                st.caption("No recorded-vote metadata was returned in the bill action bundle.")
        related = [item for item in bundle.get("related_bills", []) if isinstance(item, Mapping)]
        if related:
            st.markdown("### Related bills")
            st.dataframe(pd.DataFrame(related), use_container_width=True, hide_index=True)
        with st.expander("VIEW RAW RETRIEVED METADATA", expanded=False):
            st.json(bundle)

    st.markdown("### Download this analysis")
    personalization = st.session_state.get("personalization")
    c1, c2 = st.columns(2)
    c1.download_button(
        "DOWNLOAD MARKDOWN REPORT",
        data=analysis_to_markdown(analysis, personalization if isinstance(personalization, Mapping) else None),
        file_name=f"{re.sub(r'[^A-Za-z0-9_-]+', '_', analysis.citation)}_analysis.md",
        mime="text/markdown",
        use_container_width=True,
    )
    c2.download_button(
        "DOWNLOAD JSON REPORT",
        data=analysis_to_json(analysis, personalization if isinstance(personalization, Mapping) else None),
        file_name=f"{re.sub(r'[^A-Za-z0-9_-]+', '_', analysis.citation)}_analysis.json",
        mime="application/json",
        use_container_width=True,
    )


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
top_chrome(visitor_count)
hero()

if st.session_state.get("persistence_error"):
    st.warning(
        "The prototype database is unavailable, so visitor counting, accounts, saved profiles, follows, and notifications are disabled for this session. "
        f"Technical detail: {st.session_state['persistence_error']}"
    )

main_tab, personal_tab, civic_tab, compare_tab, account_tab, methodology_tab = st.tabs(
    [
        "ANALYZE A BILL",
        "HOW THIS AFFECTS YOU",
        "FOLLOW + CONTACT",
        "COMPARE BILLS",
        "ACCOUNT",
        "METHODOLOGY",
    ]
)


# -----------------------------------------------------------------------------
# Analyze tab
# -----------------------------------------------------------------------------
with main_tab:
    section_intro(
        "input + ingestion / step 01",
        "Start with a live bill, PDF, or pasted text.",
        "Live mode pulls full text and metadata from the official Congress.gov API. Manual modes preserve an optional source link but cannot verify that the text is the latest version.",
    )
    input_mode = st.radio(
        "Input method",
        ["Live Congress.gov bill", "Paste bill text", "Upload bill PDF", "Use fictional demo bill"],
        horizontal=True,
        key="input_mode",
    )

    candidate_document: dict[str, Any] | None = None

    if input_mode == "Live Congress.gov bill":
        api_key = render_api_key_box("live")
        if api_key:
            try:
                current_congress = cached_current_congress(api_key)
            except Exception:
                current_congress = current_congress_fallback()
            lookup_mode = st.radio(
                "Live lookup method",
                ["Enter a bill number", "Browse recent bills"],
                horizontal=True,
                key="live_lookup_mode",
            )
            if lookup_mode == "Enter a bill number":
                c1, c2 = st.columns([0.7, 0.3], gap="large")
                with c1:
                    citation_value = st.text_input(
                        "Federal bill or resolution number",
                        value=st.session_state.get("live_citation", "H.R. 1"),
                        placeholder="Examples: H.R. 3076, S. 1, H.J.Res. 7",
                        key="live_citation_input",
                    )
                with c2:
                    congress_value = st.number_input(
                        "Congress",
                        min_value=1,
                        max_value=200,
                        value=current_congress,
                        step=1,
                        key="live_congress_input",
                    )
                if st.button("FETCH FULL TEXT + OFFICIAL METADATA", type="primary", use_container_width=True, key="fetch_live_exact"):
                    try:
                        ref = parse_bill_citation(citation_value, int(congress_value))
                        st.session_state["live_citation"] = ref.citation
                        load_live_bill(api_key, ref)
                        st.success(f"Loaded {ref.display} from official sources.")
                    except Exception as exc:
                        st.error(f"Live bill retrieval failed: {exc}")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    browse_congress = st.number_input(
                        "Congress",
                        min_value=1,
                        max_value=200,
                        value=current_congress,
                        step=1,
                        key="browse_congress",
                    )
                with c2:
                    browse_type = st.selectbox(
                        "Measure type",
                        list(BILL_TYPE_DISPLAY),
                        format_func=lambda key: BILL_TYPE_DISPLAY[key],
                        key="browse_type",
                    )
                if st.button("LOAD RECENT OFFICIAL RECORDS", use_container_width=True, key="browse_load"):
                    try:
                        with st.spinner("Loading recent bill records…"):
                            st.session_state["recent_bills"] = cached_recent_bills(
                                api_key, int(browse_congress), browse_type, 60
                            )
                    except Exception as exc:
                        st.error(f"Recent-bill lookup failed: {exc}")
                recent = st.session_state.get("recent_bills")
                if isinstance(recent, list) and recent:
                    labels: list[str] = []
                    refs: list[BillRef] = []
                    for item in recent:
                        try:
                            item_type = str(item.get("type") or browse_type).lower().replace(".", "")
                            item_type = {"house bill": "hr", "senate bill": "s"}.get(item_type, item_type)
                            ref = BillRef(int(item.get("congress") or browse_congress), item_type, int(item.get("number")))
                        except Exception:
                            continue
                        refs.append(ref)
                        labels.append(f"{ref.citation} — {item.get('title') or 'Title unavailable'}")
                    if refs:
                        selected = st.selectbox("Select a live bill", range(len(refs)), format_func=lambda i: labels[i])
                        if st.button("FETCH SELECTED BILL", type="primary", use_container_width=True, key="fetch_live_selected"):
                            try:
                                load_live_bill(api_key, refs[selected])
                                st.success(f"Loaded {refs[selected].display} from official sources.")
                            except Exception as exc:
                                st.error(f"Live bill retrieval failed: {exc}")

        live_doc = current_document()
        if live_doc and live_doc.get("source_kind") == "official Congress.gov text":
            candidate_document = live_doc
            bundle = live_doc.get("bundle", {})
            versions = [item for item in bundle.get("text_versions", []) if isinstance(item, Mapping)]
            st.markdown("### Loaded official source")
            st.write(f"**{live_doc.get('citation')} — {live_doc.get('title')}**")
            st.caption(f"Text source: {live_doc.get('source_url')}")
            if versions and api_key:
                selected_version_index = st.selectbox(
                    "Choose another official text version",
                    range(len(versions)),
                    format_func=lambda i: version_label(versions[i], i),
                    key="load_version_index",
                )
                if st.button("LOAD SELECTED VERSION FOR ANALYSIS", use_container_width=True, key="load_alt_version"):
                    try:
                        downloaded = official_text_for_version(api_key, versions[selected_version_index])
                        store_document(
                            text=downloaded["text"],
                            title=str(live_doc.get("title") or ""),
                            citation=str(live_doc.get("citation") or ""),
                            source_kind="official Congress.gov text",
                            bundle=bundle,
                            source_url=downloaded.get("source_url", ""),
                            source_name=str(live_doc.get("source_name") or ""),
                            selected_version=versions[selected_version_index],
                        )
                        st.success("The selected official version is now the analysis source.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not load the selected text version: {exc}")

    elif input_mode == "Paste bill text":
        c1, c2 = st.columns([0.35, 0.65], gap="large")
        with c1:
            manual_citation = st.text_input("Bill citation / label", value="Manual bill text", key="paste_citation")
            manual_title = st.text_input("Bill title", value="", key="paste_title")
            manual_url = st.text_input(
                "Official source URL (recommended)",
                value="",
                placeholder="Paste the Congress.gov bill page URL",
                key="paste_url",
            )
        with c2:
            manual_text = st.text_area(
                "Paste legislative text",
                height=330,
                placeholder="Paste the full bill text or a substantial section here…",
                key="paste_text",
            )
        if manual_text.strip():
            candidate_document = {
                "text": manual_text.strip(),
                "title": manual_title.strip() or "Untitled pasted legislative text",
                "citation": manual_citation.strip() or "Manual text",
                "source_kind": "pasted text",
                "bundle": {},
                "source_url": manual_url.strip(),
                "source_name": "pasted_text",
                "selected_version": {},
            }

    elif input_mode == "Upload bill PDF":
        upload = st.file_uploader("Upload a searchable/text PDF", type=["pdf"], key="bill_pdf")
        c1, c2, c3 = st.columns(3)
        upload_citation = c1.text_input("Bill citation / label", value="Uploaded bill", key="upload_citation")
        upload_title = c2.text_input("Bill title", value="", key="upload_title")
        upload_url = c3.text_input("Official source URL (recommended)", value="", key="upload_url")
        if upload is not None:
            try:
                with st.spinner("Extracting text from the PDF…"):
                    pdf_text = extract_pdf_text(upload.getvalue())
                st.success(f"Extracted {len(pdf_text):,} characters from {upload.name}.")
                candidate_document = {
                    "text": pdf_text,
                    "title": upload_title.strip() or Path(upload.name).stem,
                    "citation": upload_citation.strip() or "Uploaded bill",
                    "source_kind": "uploaded PDF",
                    "bundle": {},
                    "source_url": upload_url.strip(),
                    "source_name": upload.name,
                    "selected_version": {},
                }
                with st.expander("PREVIEW EXTRACTED PDF TEXT", expanded=False):
                    st.text_area("Extracted text", pdf_text, height=280, disabled=True)
            except Exception as exc:
                st.error(f"PDF extraction failed: {exc}")

    else:
        demo_text = DEMO_PATH.read_text(encoding="utf-8")
        candidate_document = {
            "text": demo_text,
            "title": "Digital Skills and Rural Clinic Support Act of 2026 (fictional demonstration)",
            "citation": "H.R. DEMO",
            "source_kind": "fictional demonstration",
            "bundle": demo_bundle(),
            "source_url": "",
            "source_name": DEMO_PATH.name,
            "selected_version": {},
        }
        st.info("This fictional measure is included only to demonstrate every analysis panel without an API key.")
        with st.expander("VIEW FICTIONAL DEMO BILL", expanded=False):
            st.code(demo_text, language="text")

    if candidate_document:
        st.markdown('<div class="bb-rule"></div>', unsafe_allow_html=True)
        section_intro(
            "input + ingestion / step 02",
            "Run the neutral analysis pipeline.",
            "The pipeline separates source retrieval from AI paraphrase, reports uncertainty, and keeps official verification links visible whenever available.",
        )
        preview_cols = st.columns(3)
        preview_cols[0].metric("Source", str(candidate_document.get("source_kind", "")))
        preview_cols[1].metric("Words", f"{len(str(candidate_document.get('text', '')).split()):,}")
        preview_cols[2].metric("Official URL", "Provided" if candidate_document.get("source_url") or candidate_document.get("bundle", {}).get("ref", {}).get("official_url") else "Missing")
        if st.button("RUN AI BILL ANALYSIS →", type="primary", use_container_width=True, key="run_analysis"):
            try:
                with st.spinner("Mapping sections, status, fiscal signals, terminology, and uncertainty…"):
                    analysis = analyze_bill(
                        str(candidate_document["text"]),
                        bundle=candidate_document.get("bundle", {}),
                        source_kind=str(candidate_document.get("source_kind") or "pasted text"),
                        text_source_url=str(candidate_document.get("source_url") or ""),
                        citation=str(candidate_document.get("citation") or "Manual text"),
                        title=str(candidate_document.get("title") or ""),
                    )
                    source_link_override(analysis, str(candidate_document.get("source_url") or ""))
                    st.session_state["working_document"] = candidate_document
                    st.session_state["analysis"] = analysis
                    st.session_state.pop("personalization", None)
                    st.session_state.pop("version_diff", None)
                    st.session_state.pop("compare_analysis", None)
                st.success("Analysis complete. Review the official source links and section text alongside the paraphrase.")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    analysis = current_analysis()
    document = current_document()
    if analysis and document:
        render_analysis_results(analysis, document)


# -----------------------------------------------------------------------------
# Personalization tab
# -----------------------------------------------------------------------------
with personal_tab:
    section_intro(
        "profile-based relevance",
        "How this may affect someone with your profile.",
        "The tool checks age, state, income bracket, occupation, and optional status fields against bill language. It identifies what to verify; it does not determine legal eligibility or a guaranteed financial result.",
    )
    analysis = current_analysis()
    if not analysis:
        st.info("Analyze a bill first. The profile checklist will then connect your information to specific provisions and topics.")
    else:
        user = st.session_state.get("user")
        saved = get_profile(int(user["id"])) if isinstance(user, Mapping) and "id" in user else {}
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=0, max_value=120, value=int(saved.get("age") or 0), step=1, key="profile_age")
            state = st.text_input("State (two-letter code)", value=str(saved.get("state") or ""), max_chars=2, key="profile_state").upper()
        with c2:
            income_options = ["Not provided", "Under $25,000", "$25,000–$49,999", "$50,000–$99,999", "$100,000–$199,999", "$200,000+"]
            saved_income = str(saved.get("income_bracket") or "Not provided")
            income = st.selectbox("Household income bracket", income_options, index=income_options.index(saved_income) if saved_income in income_options else 0, key="profile_income")
            occupation = st.text_input("Occupation", value=str(saved.get("occupation") or ""), key="profile_occupation")
        with c3:
            industry = st.text_input("Industry", value=str(saved.get("industry") or ""), key="profile_industry")
            household = st.selectbox("Household context", ["Not provided", "One adult", "Multiple adults", "Household with children", "Multigenerational household"], index=0, key="profile_household")
        flags = st.columns(4)
        student = flags[0].checkbox("Student", value=bool(saved.get("student")), key="profile_student")
        veteran = flags[1].checkbox("Veteran", value=bool(saved.get("veteran")), key="profile_veteran")
        business_owner = flags[2].checkbox("Business owner", value=bool(saved.get("business_owner")), key="profile_business")
        caregiver = flags[3].checkbox("Caregiver", value=bool(saved.get("caregiver")), key="profile_caregiver")
        profile = {
            "age": int(age),
            "state": state,
            "income_bracket": income,
            "occupation": occupation,
            "industry": industry,
            "household": household,
            "student": student,
            "veteran": veteran,
            "business_owner": business_owner,
            "caregiver": caregiver,
        }
        c1, c2 = st.columns(2)
        with c1:
            if st.button("GENERATE PERSONAL IMPACT CHECKLIST", type="primary", use_container_width=True, key="run_personal"):
                st.session_state["personalization"] = personalize_impact(analysis, profile)
        with c2:
            if user and st.button("SAVE PROFILE TO MY ACCOUNT", use_container_width=True, key="save_profile"):
                try:
                    save_profile(int(user["id"]), profile)
                    st.success("Profile saved to the prototype account database.")
                except Exception as exc:
                    st.error(f"Profile save failed: {exc}")
            elif not user:
                st.caption("Create or sign in to an account to save this profile. You can still run the checklist without logging in.")
        result = st.session_state.get("personalization")
        if isinstance(result, Mapping):
            callout(str(result.get("overall", "")), str(result.get("disclaimer", "")), "profile relevance")
            signals = result.get("signals", [])
            if signals:
                st.dataframe(pd.DataFrame(signals), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Follow and civic action tab
# -----------------------------------------------------------------------------
with civic_tab:
    section_intro(
        "engagement + verification",
        "Follow the record, find representatives, and write in your own voice.",
        "The app supplies official directories and a neutral editable draft. It does not choose a position, send messages automatically, or make electoral recommendations.",
    )
    analysis = current_analysis()
    document = current_document()
    if not analysis or not document:
        st.info("Analyze a live or uploaded bill first.")
    else:
        bundle = document.get("bundle", {})
        live_ref = bundle.get("ref") if isinstance(bundle.get("ref"), Mapping) else {}
        user = st.session_state.get("user")
        c1, c2 = st.columns([0.58, 0.42], gap="large")
        with c1:
            st.markdown("### Sponsor, cosponsors, and vote record")
            sponsors = sponsor_rows(bundle)
            if sponsors:
                st.dataframe(pd.DataFrame(sponsors), use_container_width=True, hide_index=True)
            else:
                st.caption("No sponsor/cosponsor metadata was supplied for this source.")
            votes = extract_votes(bundle)
            if votes:
                st.dataframe(pd.DataFrame(votes), use_container_width=True, hide_index=True)
            else:
                st.caption("No recorded-vote links were returned in the action metadata.")

            if live_ref.get("bill_id") and user:
                action_date, action_text = action_snapshot(bundle)
                if st.button("FOLLOW THIS BILL", type="primary", use_container_width=True, key="follow_current"):
                    try:
                        follow_bill(
                            int(user["id"]),
                            bill_id=str(live_ref["bill_id"]),
                            citation=analysis.citation,
                            title=analysis.title,
                            official_url=str(live_ref.get("official_url") or ""),
                            status=analysis.status.label,
                            latest_action_date=action_date,
                            latest_action_text=action_text,
                        )
                        st.success("Bill saved. Use Account → Followed bills to check for live changes.")
                    except Exception as exc:
                        st.error(f"Could not follow the bill: {exc}")
            elif not user:
                st.info("Sign in under Account to save/follow bills and receive in-app change notifications.")
            elif not live_ref.get("bill_id"):
                st.caption("Follow/status-change checks require a live Congress.gov bill identifier.")

        with c2:
            st.markdown("### Find your federal representatives")
            rep_state = st.text_input("State code", value=str((get_profile(int(user["id"])) if user else {}).get("state") or ""), max_chars=2, key="rep_state").upper()
            rep_district = st.number_input("House district (optional; 0 = statewide lookup)", min_value=0, max_value=99, value=0, step=1, key="rep_district")
            api_key = active_api_key()
            if st.button("LOOK UP CURRENT MEMBERS", use_container_width=True, key="lookup_members"):
                if not api_key:
                    st.error("Add a Congress.gov API key in the Analyze tab first.")
                elif len(rep_state) != 2:
                    st.error("Enter a two-letter state code.")
                else:
                    try:
                        with st.spinner("Retrieving current member records…"):
                            members = cached_members(api_key, rep_state, int(rep_district) if rep_district else None)
                        st.session_state["member_results"] = members
                    except Exception as exc:
                        st.error(f"Member lookup failed: {exc}")
            members = st.session_state.get("member_results")
            if isinstance(members, list) and members:
                st.dataframe(pd.DataFrame([member_display(item) for item in members]), use_container_width=True, hide_index=True)
            link_cols = st.columns(2)
            link_cols[0].link_button("HOUSE DIRECTORY ↗", HOUSE_CONTACT_URL, use_container_width=True)
            link_cols[1].link_button("SENATE DIRECTORY ↗", SENATE_CONTACT_URL, use_container_width=True)

        st.markdown('<div class="bb-rule"></div>', unsafe_allow_html=True)
        st.markdown("### Build an editable constituent message")
        position = st.selectbox(
            "Choose your own purpose",
            ["Request information", "Express support", "Express opposition", "Share a concern"],
            key="contact_position",
        )
        c1, c2 = st.columns(2)
        sender_name = c1.text_input("Your name (optional)", value=str(user.get("display_name") if isinstance(user, Mapping) else ""), key="contact_name")
        sender_state = c2.text_input("Your state (optional)", value=rep_state, max_chars=2, key="contact_state")
        custom_note = st.text_area(
            "Your own question, reason, or concern",
            placeholder="Describe the provision you care about in your own words…",
            key="contact_note",
        )
        message = build_contact_message(
            analysis,
            position=position,
            sender_name=sender_name,
            state=sender_state,
            custom_note=custom_note,
        )
        edited_message = st.text_area("Editable draft", value=message, height=330, key="contact_draft")
        st.download_button(
            "DOWNLOAD CONTACT DRAFT",
            data=edited_message,
            file_name=f"contact_representative_{re.sub(r'[^A-Za-z0-9]+', '_', analysis.citation)}.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.caption("The app does not send the message. Review the latest official bill version, edit the draft, and use the official House or Senate directory.")


# -----------------------------------------------------------------------------
# Comparison tab
# -----------------------------------------------------------------------------
with compare_tab:
    section_intro(
        "side-by-side comparison",
        "Compare a House bill, Senate bill, or competing text.",
        "The comparison reports text similarity, shared topics, status, section counts, and budget signals without declaring a winner or predicting passage.",
    )
    left_analysis = current_analysis()
    if not left_analysis:
        st.info("Analyze the first bill in the Analyze tab. It will become the left side of the comparison.")
    else:
        st.markdown(f"**Left bill:** {left_analysis.citation} — {left_analysis.title}")
        compare_mode = st.radio("Second bill source", ["Live Congress.gov bill", "Paste text"], horizontal=True, key="compare_mode")
        if compare_mode == "Live Congress.gov bill":
            api_key = active_api_key()
            if not api_key:
                st.info("Add a Congress.gov API key in the Analyze tab before loading a live comparison bill.")
            compare_citation = st.text_input("Second bill citation", value="S. 1", key="compare_citation")
            compare_congress = st.number_input("Second bill Congress", min_value=1, max_value=200, value=current_congress_fallback(), step=1, key="compare_congress")
            if st.button("LOAD + ANALYZE SECOND LIVE BILL", type="primary", use_container_width=True, key="compare_live_run"):
                if not api_key:
                    st.error("A Congress.gov API key is required.")
                else:
                    try:
                        ref = parse_bill_citation(compare_citation, int(compare_congress))
                        with st.spinner("Retrieving and analyzing the second official bill…"):
                            bundle = cached_bill_bundle(api_key, ref.congress, ref.bill_type, ref.number)
                            version = latest_text_version([v for v in bundle.get("text_versions", []) if isinstance(v, dict)])
                            if not version:
                                raise CongressAPIError("No downloadable official text version was returned for the second bill.")
                            downloaded = official_text_for_version(api_key, version)
                            detail = bundle.get("detail") if isinstance(bundle.get("detail"), Mapping) else {}
                            right_analysis = analyze_bill(
                                downloaded["text"],
                                bundle=bundle,
                                source_kind="official Congress.gov text",
                                text_source_url=downloaded.get("source_url", ""),
                                citation=ref.citation,
                                title=str(detail.get("title") or ref.display),
                            )
                        st.session_state["compare_analysis"] = right_analysis
                    except Exception as exc:
                        st.error(f"Second-bill analysis failed: {exc}")
        else:
            c1, c2 = st.columns(2)
            compare_label = c1.text_input("Second bill citation / label", value="Comparison text", key="compare_label")
            compare_title = c2.text_input("Second bill title", value="", key="compare_title")
            compare_text = st.text_area("Paste second bill text", height=280, key="compare_text")
            if st.button("ANALYZE SECOND PASTED BILL", type="primary", use_container_width=True, key="compare_paste_run"):
                try:
                    st.session_state["compare_analysis"] = analyze_bill(
                        compare_text,
                        source_kind="pasted comparison text",
                        citation=compare_label,
                        title=compare_title or "Pasted comparison text",
                    )
                except Exception as exc:
                    st.error(f"Second-bill analysis failed: {exc}")

        right_analysis = st.session_state.get("compare_analysis")
        if isinstance(right_analysis, BillAnalysis):
            comparison = compare_analyses(left_analysis, right_analysis)
            metric_cards(
                [
                    ("Text similarity", f"{comparison['similarity_pct']:.1f}%", "TF-IDF overlap, not legal equivalence"),
                    ("Left stage", comparison["status"]["left"].title(), f"{comparison['sections']['left']} mapped sections"),
                    ("Right stage", comparison["status"]["right"].title(), f"{comparison['sections']['right']} mapped sections"),
                ]
            )
            st.info(comparison["note"])
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown(f"### {left_analysis.citation}: {left_analysis.title}")
                st.write(left_analysis.plain_summary)
                st.write("**Topics:** " + (", ".join(left_analysis.topics) or "None detected"))
            with c2:
                st.markdown(f"### {right_analysis.citation}: {right_analysis.title}")
                st.write(right_analysis.plain_summary)
                st.write("**Topics:** " + (", ".join(right_analysis.topics) or "None detected"))
            st.markdown("### Topic overlap")
            st.json(
                {
                    "shared_topics": comparison["shared_topics"],
                    "left_only_topics": comparison["left_only_topics"],
                    "right_only_topics": comparison["right_only_topics"],
                    "fiscal_signals": comparison["fiscal"],
                }
            )


# -----------------------------------------------------------------------------
# Account tab
# -----------------------------------------------------------------------------
with account_tab:
    section_intro(
        "prototype account layer",
        "Save a profile, follow bills, and view in-app changes.",
        "This demo uses a local SQLite database and salted password hashes. On Streamlit Community Cloud, local storage may reset during redeploys, restarts, or platform maintenance; production use requires a managed identity and database service.",
    )
    user = st.session_state.get("user")
    if not user:
        login_tab, register_tab = st.tabs(["SIGN IN", "CREATE ACCOUNT"])
        with login_tab:
            with st.form("login_form"):
                login_email = st.text_input("Email")
                login_password = st.text_input("Password", type="password")
                login_submit = st.form_submit_button("SIGN IN", use_container_width=True)
            if login_submit:
                try:
                    authenticated = authenticate(login_email, login_password)
                    if authenticated:
                        st.session_state["user"] = authenticated
                        st.success("Signed in.")
                        st.rerun()
                    else:
                        st.error("Email or password was not recognized.")
                except Exception as exc:
                    st.error(f"Sign-in failed: {exc}")
        with register_tab:
            with st.form("register_form"):
                register_name = st.text_input("Display name")
                register_email = st.text_input("Email", key="register_email")
                register_password = st.text_input("Password (8+ characters)", type="password", key="register_password")
                register_submit = st.form_submit_button("CREATE PROTOTYPE ACCOUNT", use_container_width=True)
            if register_submit:
                try:
                    created = create_user(register_email, register_password, register_name)
                    st.session_state["user"] = created
                    st.success("Account created and signed in.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Account creation failed: {exc}")
    else:
        c1, c2 = st.columns([0.75, 0.25])
        c1.write(f"Signed in as **{user.get('display_name') or user.get('email')}**")
        if c2.button("SIGN OUT", use_container_width=True):
            st.session_state.pop("user", None)
            st.session_state.pop("personalization", None)
            st.rerun()

        follows = list_follows(int(user["id"]))
        notifications = list_notifications(int(user["id"]))
        metric_cards(
            [
                ("Followed bills", str(len(follows)), "saved in this prototype database"),
                ("Unread notifications", str(sum(1 for item in notifications if not item.get("is_read"))), "created during manual status checks"),
                ("Registered profile", "Saved" if get_profile(int(user["id"])) else "Empty", "edit under How This Affects You"),
            ]
        )
        st.markdown("### Followed bills")
        if follows:
            st.dataframe(pd.DataFrame(follows), use_container_width=True, hide_index=True)
            api_key = active_api_key()
            if st.button("CHECK FOLLOWED BILLS FOR STATUS CHANGES", type="primary", use_container_width=True, key="check_follows"):
                if not api_key:
                    st.error("Add a Congress.gov API key in the Analyze tab first.")
                else:
                    changed = 0
                    failures: list[str] = []
                    progress = st.progress(0)
                    for index, item in enumerate(follows):
                        try:
                            match = re.fullmatch(r"(\d+)-([a-z]+)-(\d+)", str(item.get("bill_id") or ""))
                            if not match:
                                continue
                            ref = BillRef(int(match.group(1)), match.group(2), int(match.group(3)))
                            bundle = cached_bill_bundle(api_key, ref.congress, ref.bill_type, ref.number)
                            status = analyze_bill(
                                "This placeholder text supports status refresh only. " * 5,
                                bundle=bundle,
                                source_kind="official metadata refresh",
                                citation=ref.citation,
                                title=str((bundle.get("detail") or {}).get("title") or ref.display),
                            ).status
                            action_date, action_text = action_snapshot(bundle)
                            if update_follow_and_notify(
                                int(user["id"]),
                                str(item["bill_id"]),
                                new_status=status.label,
                                latest_action_date=action_date,
                                latest_action_text=action_text,
                            ):
                                changed += 1
                        except Exception as exc:
                            failures.append(f"{item.get('citation')}: {exc}")
                        progress.progress((index + 1) / max(len(follows), 1))
                    if changed:
                        st.success(f"Detected changes for {changed} followed bill(s).")
                    else:
                        st.info("No changes were detected in the retrieved official records.")
                    if failures:
                        st.warning("Some checks failed: " + " | ".join(failures[:5]))
                    st.rerun()
            selected_unfollow = st.selectbox(
                "Remove a followed bill",
                options=[item["bill_id"] for item in follows],
                format_func=lambda bill_id: next((f"{item['citation']} — {item['title']}" for item in follows if item["bill_id"] == bill_id), bill_id),
                key="unfollow_select",
            )
            if st.button("UNFOLLOW SELECTED BILL", use_container_width=True, key="unfollow_button"):
                unfollow_bill(int(user["id"]), selected_unfollow)
                st.success("Bill removed from followed list.")
                st.rerun()
        else:
            st.caption("No followed bills yet. Analyze a live bill, then use Follow + Contact.")

        st.markdown("### In-app notifications")
        notifications = list_notifications(int(user["id"]))
        if notifications:
            st.dataframe(pd.DataFrame(notifications), use_container_width=True, hide_index=True)
            if st.button("MARK ALL NOTIFICATIONS READ", use_container_width=True, key="mark_notifications"):
                mark_notifications_read(int(user["id"]))
                st.rerun()
        else:
            st.caption("No notifications. This prototype creates notifications only when you manually check followed bills.")


# -----------------------------------------------------------------------------
# Methodology tab
# -----------------------------------------------------------------------------
with methodology_tab:
    section_intro(
        "transparent methodology",
        "What the AI does — and what it does not do.",
        "The default cloud build uses explainable local NLP and official public data. It avoids a heavyweight model dependency so Streamlit Community Cloud can start quickly and users can inspect the logic.",
    )
    st.markdown(
        """
### 1. Official retrieval
Live mode uses the Congress.gov API for bill detail, actions, amendments, committees, cosponsors, related bills, subjects, summaries, text versions, and titles. The selected official text file is downloaded only from allowlisted Congress.gov, GovInfo, or GPO hosts.

### 2. Plain-English AI
The app splits the bill into sections, tokenizes sentences, builds TF-IDF representations, and ranks central sentences using document relevance plus position. A transparent legalese-replacement layer simplifies selected wording. This is an extractive NLP system: it does not invent missing provisions.

### 3. Clause and topic mapping
Rules identify obligation terms such as *shall*, *must*, *may*, *authorize*, *appropriate*, *amend*, and *repeal*. Topic keywords create section-level subject tags. A low-to-medium-confidence scope mismatch flag appears when a substantial section uses a topic vocabulary outside the bill's main detected subjects.

### 4. Status and timeline
When official action history is available, rule-based evidence maps the record to stages such as introduced, in committee, reported, passed one chamber, passed both chambers, presented to the President, enacted, vetoed, or failed/rejected. The app never estimates the probability of passage.

### 5. Fiscal and profile signals
Budget analysis surfaces Congress.gov CBO metadata, appropriations/revenue terms, and textual dollar references; it is not a budget score. Personalization creates a relevance checklist from user-supplied fields; it is not a legal eligibility or financial-impact determination.

### 6. Trust controls
Official source links remain visible, every section retains source text, confidence reflects source completeness, and version comparison uses a line-level unified diff. User decisions and civic positions remain the user's own.
"""
    )
    st.markdown("### Streamlit deployment and privacy")
    st.write(
        "Set `CONGRESS_API_KEY` in Streamlit Community Cloud secrets. Manual and demo modes work without it. "
        "The account/follow layer uses local SQLite for a classroom prototype; Cloud local files may be ephemeral. "
        "Do not store sensitive personal information in this demo. A production deployment should use managed authentication, "
        "a persistent database, encryption, access controls, audit logs, and a notification service."
    )
    st.markdown("### User counter")
    st.write(
        "A visitor is counted once per Streamlit browser session. The counter records only a random session hash and timestamps; "
        "it does not store an IP address in the application database. Browser refreshes within the same live session increase page-view metadata but not the unique-session count."
    )
    try:
        st.json(visitor_stats())
    except Exception:
        st.caption("Visitor statistics are unavailable in this runtime.")
    st.markdown("### Credits")
    st.write("**Author:** Claire Yuan  \n**Advisor:** Dr. Qingyang Xiao  \n**License:** MIT")

footer()
