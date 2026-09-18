from __future__ import annotations

import html
import io
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from src.modeling import (
    CASE_TYPE_MAP,
    DEMO_BILL_TEXT,
    DEMO_JURISDICTIONS,
    VIOLATION_CODES,
    build_dnn_from_bundle,
    load_bundle,
    make_impact_report,
    parse_bill_text,
    predict_case,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "claire_yuan_court_bill_impact_models.joblib"

st.set_page_config(
    page_title="AI Court-Bill Impact Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# UI migration
# -----------------------------------------------------------------------------
# The attached UI source is a TanStack/React design. Streamlit Community Cloud
# launches Python entrypoints, so the visual system is translated here into
# native Streamlit + CSS rather than requiring a second Node server/runtime.
# Design cues preserved: warm paper background, JetBrains Mono body type,
# Instrument Serif display type, fine rules, editorial spacing, orange accent,
# ticker treatment, uppercase micro-labels, minimal square controls, and
# report-page typography.
# -----------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

:root {
  --paper: #faf8f3;
  --ink: #1a1a1a;
  --muted: #7a766c;
  --line: #d6d3ca;
  --soft: #f3efe4;
  --accent: #d07a2d;
  --green: #3a7a4a;
  --red: #a04438;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
  font-feature-settings: "ss01", "cv02";
}

[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"] {
  background: var(--paper) !important;
}


#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }

/*
 * Streamlit Community Cloud keeps its Share/Edit toolbar fixed above the app.
 * Reserve a stable safe area so the custom navigation banner always starts
 * below that toolbar instead of being covered by it.
 */
:root { --streamlit-toolbar-safe-area: 4.75rem; }

.block-container,
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"] {
  max-width: 1160px !important;
  padding-top: var(--streamlit-toolbar-safe-area) !important;
  padding-bottom: 4rem !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
  overflow: visible !important;
}

/* Keep Streamlit's own toolbar readable while preventing it from visually
   merging with the first app banner. */
[data-testid="stHeader"] {
  min-height: 3.75rem !important;
  background: rgba(250,248,243,.97) !important;
  border-bottom: 1px solid rgba(214,211,202,.72) !important;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}

/* Anchor links should also stop below the fixed Streamlit toolbar. */
html { scroll-padding-top: 5.25rem; }

h1, h2, h3, h4, h5, h6 {
  color: var(--ink) !important;
  letter-spacing: -0.025em;
}

p, li, label, div { color: var(--ink); }

.bb-nav {
  position: relative;
  z-index: 3;
  overflow: visible;
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 72px;
  margin: 0 -2rem;
  padding: 0 2rem;
}
.bb-brand {
  display: flex;
  align-items: center;
  gap: .6rem;
  font-size: .82rem;
  font-weight: 500;
  letter-spacing: -.02em;
}
.bb-square { width: 8px; height: 8px; background: var(--ink); display: inline-block; }
.bb-version { color: var(--muted); font-weight: 400; }
.bb-nav-right {
  display: flex;
  align-items: center;
  gap: 1.3rem;
  font-size: .65rem;
  text-transform: uppercase;
  letter-spacing: .18em;
  color: var(--muted);
}
.bb-nav-right a { color: var(--ink); text-decoration: none; border: 1px solid var(--ink); padding: .62rem .82rem; }

.bb-ticker {
  margin: 0 -2rem;
  border-bottom: 1px solid var(--line);
  overflow: hidden;
  white-space: nowrap;
  display: flex;
  align-items: stretch;
  min-height: 40px;
}
.bb-ticker-label {
  background: var(--ink);
  color: var(--paper);
  padding: .78rem 1rem;
  font-size: .59rem;
  text-transform: uppercase;
  letter-spacing: .22em;
  z-index: 2;
}
.bb-dot {
  width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
  display: inline-block; margin-right: .45rem; animation: bbpulse 1.6s ease-in-out infinite;
}
.bb-ticker-track {
  color: var(--muted);
  font-size: .64rem;
  text-transform: uppercase;
  letter-spacing: .18em;
  padding: .78rem 0;
  display: inline-block;
  min-width: max-content;
  animation: bbmarquee 45s linear infinite;
}
.bb-ticker-track span { margin: 0 1.4rem; color: var(--muted); }
.bb-ticker:hover .bb-ticker-track { animation-play-state: paused; }
@keyframes bbmarquee { from { transform: translateX(0); } to { transform: translateX(-35%); } }
@keyframes bbpulse { 0%,100% { opacity:.35; } 50% { opacity:1; } }

.bb-hero {
  border-bottom: 1px solid var(--line);
  margin: 0 -2rem 0;
  padding: 5.3rem 2rem 4.8rem;
}
.bb-hero-inner { max-width: 1000px; margin: 0 auto; }
.bb-eyebrow {
  color: var(--muted) !important;
  font-size: .61rem;
  text-transform: uppercase;
  letter-spacing: .28em;
  margin-bottom: 1.4rem;
}
.bb-eyebrow .bb-dot { vertical-align: middle; }
.bb-display {
  font-family: "JetBrains Mono", monospace;
  font-size: clamp(3rem, 7vw, 5.4rem);
  font-weight: 300;
  line-height: 1.03;
  letter-spacing: -.055em;
  margin: 0;
  color: var(--ink);
}
.bb-display .serif {
  font-family: "Instrument Serif", Georgia, serif;
  font-style: italic;
  font-weight: 400;
  color: var(--muted);
  letter-spacing: -.02em;
}
.bb-display .accent-word {
  text-decoration: underline;
  text-decoration-color: var(--accent);
  text-decoration-thickness: 3px;
  text-underline-offset: 11px;
}
.bb-lede {
  max-width: 720px;
  margin-top: 2.2rem;
  color: var(--muted) !important;
  font-size: .91rem;
  line-height: 1.85;
}
.bb-creditline {
  margin-top: 1.6rem;
  font-size: .66rem;
  text-transform: uppercase;
  letter-spacing: .15em;
  color: var(--muted) !important;
}
.bb-creditline strong { color: var(--ink); font-weight: 500; }

.bb-warning {
  border-left: 2px solid var(--accent);
  background: var(--soft);
  margin: 2rem 0 .7rem;
  padding: 1rem 1.15rem;
  font-family: "Instrument Serif", Georgia, serif;
  font-style: italic;
  font-size: 1.02rem;
  line-height: 1.55;
  color: #4c4941 !important;
}

.bb-section-kicker {
  margin-top: 1.2rem;
  color: var(--muted) !important;
  font-size: .61rem;
  text-transform: uppercase;
  letter-spacing: .27em;
}
.bb-section-title {
  font-family: "Instrument Serif", Georgia, serif;
  font-size: clamp(2.15rem, 4.2vw, 3.5rem);
  line-height: 1.08;
  margin: .55rem 0 1.25rem;
  color: var(--ink) !important;
  font-weight: 400;
}
.bb-section-copy {
  max-width: 760px;
  color: var(--muted) !important;
  font-size: .83rem;
  line-height: 1.7;
  margin-bottom: 1.7rem;
}
.bb-rule { border-top: 1px solid var(--line); margin: 2.7rem 0 2.4rem; }

.bb-aside {
  border-top: 1px solid var(--ink);
  padding-top: 1rem;
  margin-top: .2rem;
}
.bb-aside-title {
  color: var(--muted) !important;
  font-size: .59rem;
  text-transform: uppercase;
  letter-spacing: .24em;
  margin-bottom: .8rem;
}
.bb-aside p {
  color: #5a5650 !important;
  font-family: "Instrument Serif", Georgia, serif;
  font-style: italic;
  font-size: 1.04rem;
  line-height: 1.5;
}

.bb-step-grid, .bb-principle-grid, .bb-metric-grid {
  display: grid;
  gap: 1.2rem;
}
.bb-step-grid { grid-template-columns: repeat(3, minmax(0,1fr)); }
.bb-principle-grid { grid-template-columns: 5fr 7fr; }
.bb-metric-grid { grid-template-columns: repeat(5, minmax(0,1fr)); margin-top: 1.3rem; }
.bb-step, .bb-metric {
  border-top: 1px solid var(--ink);
  padding-top: 1.05rem;
}
.bb-step-no, .bb-metric-label {
  color: var(--muted) !important;
  font-size: .58rem;
  text-transform: uppercase;
  letter-spacing: .22em;
}
.bb-step h3, .bb-metric-value {
  font-family: "Instrument Serif", Georgia, serif;
  font-weight: 400;
  margin: .65rem 0 .35rem;
}
.bb-step h3 { font-size: 1.75rem; }
.bb-step p { color: var(--muted) !important; font-size: .74rem; line-height: 1.65; }
.bb-metric-value { font-size: 1.8rem; line-height: 1.1; }
.bb-metric-sub { color: var(--muted) !important; font-size: .62rem; line-height: 1.45; }

.bb-impact {
  border-left: 2px solid var(--accent);
  padding: .4rem 0 .4rem 1.2rem;
  margin: 1.8rem 0 1.5rem;
}
.bb-impact-label {
  color: var(--muted) !important;
  font-size: .58rem;
  text-transform: uppercase;
  letter-spacing: .24em;
}
.bb-impact-headline {
  font-family: "Instrument Serif", Georgia, serif;
  font-size: 1.65rem;
  line-height: 1.35;
  margin-top: .45rem;
}
.bb-range-strip {
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  padding: .9rem 0;
  color: #5a5650 !important;
  font-family: "Instrument Serif", Georgia, serif;
  font-style: italic;
  line-height: 1.55;
  margin: 1rem 0 1.5rem;
}

.bb-principles {
  border-top: 1px solid var(--line);
  margin: 4.5rem -2rem 0;
  padding: 4rem 2rem 1.2rem;
}
.bb-principles h2 {
  font-family: "JetBrains Mono", monospace;
  font-weight: 300;
  font-size: 2.4rem;
  line-height: 1.15;
  margin: 1rem 0;
}
.bb-principles h2 em { font-family:"Instrument Serif", Georgia, serif; color:var(--muted); font-weight:400; }
.bb-principle-list { border-top: 1px solid var(--line); }
.bb-principle-item {
  display:flex; gap:1rem; border-bottom:1px solid var(--line); padding: 1rem 0;
  font-size:.77rem; line-height:1.6;
}
.bb-arrow { color: var(--accent) !important; }

.bb-footer {
  border-top: 1px solid var(--line);
  margin: 3rem -2rem 0;
  padding: 1.6rem 2rem .5rem;
  display: flex;
  justify-content: space-between;
  gap: 1.2rem;
  flex-wrap: wrap;
  color: var(--muted) !important;
  font-size: .59rem;
  text-transform: uppercase;
  letter-spacing: .2em;
}

/* Streamlit tabs — translated from the attached sticky editorial tab strip. */
.stTabs [data-baseweb="tab-list"] {
  gap: 0 !important;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  background: var(--paper);
  margin-top: 1.6rem;
}
.stTabs [data-baseweb="tab"] {
  height: 54px;
  padding: 0 1.2rem !important;
  color: var(--muted) !important;
  font-family: "JetBrains Mono", monospace !important;
  font-size: .64rem !important;
  text-transform: uppercase;
  letter-spacing: .18em;
}
.stTabs [aria-selected="true"] { color: var(--ink) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--ink) !important; height: 1px !important; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Streamlit inputs */
.stRadio label, .stCheckbox label, .stSelectbox label, .stNumberInput label,
.stTextArea label, .stFileUploader label, .stTextInput label {
  font-family: "JetBrains Mono", monospace !important;
  font-size: .66rem !important;
  letter-spacing: .04em;
}
.stTextArea textarea, .stTextInput input, .stNumberInput input,
[data-baseweb="select"] > div {
  border-radius: 0 !important;
  border-color: var(--line) !important;
  background: #fffdf8 !important;
  color: var(--ink) !important;
  box-shadow: none !important;
}
[data-testid="stFileUploaderDropzone"] {
  border-radius: 0 !important;
  border: 1px dashed #aaa59a !important;
  background: #fffdf8 !important;
  padding: 1.2rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--muted) !important; }

.stButton > button, .stDownloadButton > button {
  border-radius: 0 !important;
  border: 1px solid var(--ink) !important;
  background: transparent !important;
  color: var(--ink) !important;
  font-family: "JetBrains Mono", monospace !important;
  font-size: .64rem !important;
  text-transform: uppercase;
  letter-spacing: .16em;
  min-height: 44px;
  transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--ink) !important;
  color: var(--paper) !important;
  border-color: var(--ink) !important;
}
.stButton > button[kind="primary"] {
  background: var(--ink) !important;
  color: var(--paper) !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}

[data-testid="stExpander"] {
  border: 1px solid var(--line) !important;
  border-radius: 0 !important;
  background: transparent !important;
}
[data-testid="stDataFrame"] { border: 1px solid var(--line); }

/* Alerts & progress */
[data-testid="stAlert"] { border-radius: 0 !important; }
[data-testid="stProgress"] > div > div > div > div { background-color: var(--accent) !important; }

/* Markdown report styling */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
  font-family: "Instrument Serif", Georgia, serif !important;
  font-weight: 400 !important;
}
[data-testid="stMarkdownContainer"] table { font-size: .75rem; }
[data-testid="stMarkdownContainer"] blockquote {
  border-left: 2px solid var(--accent);
  color: #5a5650;
  font-family: "Instrument Serif", Georgia, serif;
  font-style: italic;
}

@media (max-width: 900px) {
  .bb-nav-right span { display:none; }
  .bb-hero { padding-top: 3.6rem; padding-bottom: 3.7rem; }
  .bb-step-grid, .bb-metric-grid, .bb-principle-grid { grid-template-columns: 1fr; }
  .bb-metric { padding-bottom: .4rem; }
  .block-container,
  [data-testid="stAppViewBlockContainer"],
  [data-testid="stMainBlockContainer"] {
    padding-top: 4.5rem !important;
    padding-left: 1.15rem !important;
    padding-right: 1.15rem !important;
  }
  .bb-nav, .bb-ticker, .bb-hero, .bb-principles, .bb-footer { margin-left:-1.15rem; margin-right:-1.15rem; }
  .bb-nav, .bb-hero, .bb-principles, .bb-footer { padding-left:1.15rem; padding-right:1.15rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained ML + neural ensemble…")
def get_models():
    bundle = load_bundle(MODEL_PATH)
    dnn_state = build_dnn_from_bundle(bundle)
    return bundle, dnn_state


def extract_text_from_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.getvalue()

    if suffix in {".txt", ".md", ".csv"}:
        return raw.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        text_parts: list[str] = []
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            text_parts = [(page.extract_text() or "") for page in reader.pages]
        except Exception:
            text_parts = []

        text = "\n".join(text_parts).strip()
        if len(text) >= 40:
            return text

        from pdf2image import convert_from_bytes
        import pytesseract

        images = convert_from_bytes(raw, dpi=220)
        return "\n".join(pytesseract.image_to_string(img) for img in images).strip()

    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        from PIL import Image
        import pytesseract

        return pytesseract.image_to_string(Image.open(io.BytesIO(raw))).strip()

    raise ValueError(f"Unsupported file type: {suffix}")


def section_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
<div class="bb-section-kicker">{html.escape(kicker)}</div>
<div class="bb-section-title">{html.escape(title)}</div>
<div class="bb-section-copy">{html.escape(copy)}</div>
""",
        unsafe_allow_html=True,
    )


def impact_headline(risk_level: str) -> str:
    return {
        "low": "The model sees a lower relative impact profile — verify the official consequences before acting.",
        "medium": "The model sees a meaningful impact profile that deserves deadline and consequence verification.",
        "high": "The model flags a higher synthetic impact profile — prioritize official verification and qualified guidance.",
    }.get(risk_level, "Review the modeled ranges and verify every consequence with an official source.")


def render_metrics(pred) -> None:
    risk = html.escape(pred.risk_level.upper())
    st.markdown(
        f"""
<div class="bb-metric-grid">
  <div class="bb-metric">
    <div class="bb-metric-label">Insurance estimate</div>
    <div class="bb-metric-value">{pred.insurance_mid_pct:.1f}%</div>
    <div class="bb-metric-sub">synthetic ensemble midpoint</div>
  </div>
  <div class="bb-metric">
    <div class="bb-metric-label">Impact duration</div>
    <div class="bb-metric-value">{pred.duration_mid_years:.1f} yr</div>
    <div class="bb-metric-sub">modeled duration midpoint</div>
  </div>
  <div class="bb-metric">
    <div class="bb-metric-label">License points</div>
    <div class="bb-metric-value">{pred.license_points_mid:.1f}</div>
    <div class="bb-metric-sub">prototype estimate only</div>
  </div>
  <div class="bb-metric">
    <div class="bb-metric-label">Escalation risk</div>
    <div class="bb-metric-value">{100 * pred.escalation_probability:.1f}%</div>
    <div class="bb-metric-sub">classification probability</div>
  </div>
  <div class="bb-metric">
    <div class="bb-metric-label">Impact category</div>
    <div class="bb-metric-value">{risk}</div>
    <div class="bb-metric-sub">synthetic training category</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# Top navigation translated from the attached UI.
st.markdown(
    """
<div class="bb-nav">
  <div class="bb-brand">
    <span class="bb-square"></span>
    <span>court_impact</span>
    <span class="bb-version">/v2.1</span>
  </div>
  <div class="bb-nav-right">
    <span>AI research prototype</span>
    <a href="#analysis-workbench">Analyze a notice</a>
  </div>
</div>
<div class="bb-ticker">
  <div class="bb-ticker-label"><span class="bb-dot"></span>model notes</div>
  <div class="bb-ticker-track">
    <span>synthetic demo data · not legal advice</span> ·
    <span>scikit-learn + PyTorch-trained DNN</span> ·
    <span>NumPy cloud inference</span> ·
    <span>OCR + transparent parsing</span> ·
    <span>calibrated ranges</span> ·
    <span>human verification required</span> ·
    <span>synthetic demo data · not legal advice</span> ·
    <span>scikit-learn + PyTorch-trained DNN</span> ·
    <span>NumPy cloud inference</span> ·
    <span>OCR + transparent parsing</span> ·
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<section class="bb-hero">
  <div class="bb-hero-inner">
    <div class="bb-eyebrow"><span class="bb-dot"></span>educational prototype · privacy-first · explainable workflow</div>
    <h1 class="bb-display">court bills,<br><span class="serif">in plain</span> <span class="accent-word">English.</span></h1>
    <div class="bb-lede">
      Upload or paste a redacted court-issued notice, citation, fine, or bill. The prototype extracts structured facts,
      runs a classical ML + neural-network ensemble, and produces a readable impact report with calibrated ranges.
    </div>
    <div class="bb-creditline"><strong>Author: Claire Yuan</strong> &nbsp;·&nbsp; <strong>Advisor: Dr. Qingyang Xiao</strong> &nbsp;·&nbsp; MIT License</div>
  </div>
</section>
<div class="bb-warning">
  Educational research prototype only. The bundled model is trained on synthetic demonstration data. It does not provide legal advice,
  determine guilt, predict an official court disposition, or establish insurer or credit-bureau action.
</div>
<div id="analysis-workbench"></div>
""",
    unsafe_allow_html=True,
)

main_tab, methodology_tab, about_tab = st.tabs(
    ["ANALYZE A NOTICE", "MODEL METHODOLOGY", "ABOUT + DEPLOYMENT"]
)

with main_tab:
    section_intro(
        "analysis workbench / step 01",
        "Bring the notice. We’ll map the possible impact.",
        "Choose a redacted file, paste text, or use the bundled demo. The visual layer follows the attached editorial UI while the original Streamlit ML pipeline remains intact.",
    )

    left, right = st.columns([1.65, 0.75], gap="large")
    with left:
        input_mode = st.radio(
            "Input method",
            ["Upload a document", "Paste text", "Use demo notice"],
            horizontal=True,
        )

        source_text = ""
        source_name = ""

        if input_mode == "Upload a document":
            uploaded = st.file_uploader(
                "Upload a redacted PDF, image, TXT, CSV, or Markdown file",
                type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "txt", "csv", "md"],
            )
            if uploaded is not None:
                try:
                    with st.spinner("Extracting document text…"):
                        source_text = extract_text_from_upload(uploaded)
                    source_name = uploaded.name
                    st.success(f"Extracted {len(source_text):,} characters from {uploaded.name}.")
                except Exception as exc:
                    st.error(f"Text extraction failed: {exc}")

        elif input_mode == "Paste text":
            source_text = st.text_area(
                "Paste redacted notice text",
                height=235,
                placeholder="Paste the court notice or citation text here…",
            ).strip()
            source_name = "pasted_text"

        else:
            source_text = DEMO_BILL_TEXT
            source_name = "demo_speeding_notice.txt"
            st.code(DEMO_BILL_TEXT, language="text")

    with right:
        st.markdown(
            """
<div class="bb-aside">
  <div class="bb-aside-title">Privacy before prediction</div>
  <p>Redact names, addresses, dates of birth, account numbers, barcodes, license numbers, and other identifying information before upload.</p>
</div>
<div class="bb-aside" style="margin-top:2rem;">
  <div class="bb-aside-title">Model scope</div>
  <p>The included model uses fictional <strong>Demo-*</strong> jurisdiction labels and synthetic outcomes. Treat every output as a prototype estimate.</p>
</div>
""",
            unsafe_allow_html=True,
        )

    if source_text:
        parsed = parse_bill_text(
            source_text,
            defaults={"prior_case_count": 0, "jurisdiction": "Demo-MD"},
        )

        with st.expander("REVIEW EXTRACTED TEXT", expanded=False):
            st.text_area("Extracted text", source_text, height=225, disabled=True)

        st.markdown('<div class="bb-rule"></div>', unsafe_allow_html=True)
        section_intro(
            "analysis workbench / step 02",
            "Check the machine-read facts.",
            "OCR and rule-based extraction can be wrong. Confirm each field against the original notice before you run the model.",
        )

        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            parsed_jurisdiction = parsed.get("jurisdiction", "Demo-MD")
            if parsed_jurisdiction not in DEMO_JURISDICTIONS:
                parsed_jurisdiction = "Demo-MD"
            jurisdiction = st.selectbox(
                "Fictional model jurisdiction",
                DEMO_JURISDICTIONS,
                index=DEMO_JURISDICTIONS.index(parsed_jurisdiction),
            )

            parsed_violation = parsed.get("violation_code", VIOLATION_CODES[0])
            if parsed_violation not in VIOLATION_CODES:
                parsed_violation = VIOLATION_CODES[0]
            violation = st.selectbox(
                "Violation code",
                VIOLATION_CODES,
                index=VIOLATION_CODES.index(parsed_violation),
            )
            case_type = CASE_TYPE_MAP[violation]

        with c2:
            fine_amount = st.number_input(
                "Listed fine ($)",
                min_value=0.0,
                max_value=100000.0,
                value=float(parsed["fine_amount"]),
                step=5.0,
            )
            days_to_due = st.number_input(
                "Response window (days)",
                min_value=1,
                max_value=365,
                value=int(parsed["days_to_due"]),
                step=1,
            )
            prior_cases = st.number_input(
                "Prior similar case count",
                min_value=0,
                max_value=20,
                value=int(parsed["prior_case_count"]),
                step=1,
            )

        with c3:
            speed_over = st.number_input(
                "Speed above limit (mph)",
                min_value=0,
                max_value=100,
                value=int(parsed["speed_over_mph"]),
                step=1,
            )
            court_required = st.checkbox(
                "Court appearance marked required",
                value=bool(parsed["court_appearance_required"]),
            )
            st.text_input("Case type", value=case_type, disabled=True)

        case = {
            "bill_text": source_text,
            "jurisdiction": jurisdiction,
            "case_type": case_type,
            "violation_code": violation,
            "fine_amount": float(fine_amount),
            "days_to_due": int(days_to_due),
            "prior_case_count": int(prior_cases),
            "speed_over_mph": int(speed_over),
            "court_appearance_required": int(court_required),
        }

        if st.button("RUN AI IMPACT ANALYSIS →", type="primary", use_container_width=True):
            if len(source_text.strip()) < 10:
                st.error("The extracted text is too short to analyze reliably.")
            else:
                try:
                    bundle, dnn_state = get_models()
                    pred = predict_case(bundle, dnn_state, case)
                    report = make_impact_report(case, pred)
                    result_json = json.dumps(
                        {
                            "source_name": source_name,
                            "parsed_case": case,
                            "prediction": asdict(pred),
                        },
                        indent=2,
                    )
                    st.session_state["analysis_result"] = (
                        case,
                        pred,
                        report,
                        result_json,
                    )
                except Exception as exc:
                    st.exception(exc)

    if "analysis_result" in st.session_state:
        case, pred, report, result_json = st.session_state["analysis_result"]

        st.markdown('<div class="bb-rule"></div>', unsafe_allow_html=True)
        section_intro(
            "AI breakdown / step 03",
            "The plain-English read.",
            "The figures below are ensemble estimates from synthetic training records. The calibration ranges show model uncertainty, not guaranteed legal or financial outcomes.",
        )

        risk_color = {
            "low": "#3a7a4a",
            "medium": "#d07a2d",
            "high": "#a04438",
        }.get(pred.risk_level, "#d07a2d")
        st.markdown(
            f"""
<div class="bb-impact" style="border-color:{risk_color};">
  <div class="bb-impact-label" style="color:{risk_color} !important;">personal impact / synthetic {html.escape(pred.risk_level)} profile</div>
  <div class="bb-impact-headline">{html.escape(impact_headline(pred.risk_level))}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        render_metrics(pred)
        st.progress(
            min(max(float(pred.escalation_probability), 0.0), 1.0),
            text="Modeled nonpayment / escalation probability",
        )
        st.markdown(
            f"""
<div class="bb-range-strip">
  Approximate 90% calibration ranges — insurance: <strong>{pred.insurance_low_pct:.1f}%–{pred.insurance_high_pct:.1f}%</strong>;
  duration: <strong>{pred.duration_low_years:.1f}–{pred.duration_high_years:.1f} years</strong>;
  points: <strong>{pred.license_points_low:.1f}–{pred.license_points_high:.1f}</strong>.
</div>
""",
            unsafe_allow_html=True,
        )

        report_view, inputs_view, download_view = st.tabs(
            ["IMPACT REPORT", "PARSED INPUTS", "DOWNLOADS"]
        )
        with report_view:
            st.markdown(report)
        with inputs_view:
            st.dataframe(
                pd.DataFrame([case]).T.rename(columns={0: "value"}),
                use_container_width=True,
            )
        with download_view:
            d1, d2 = st.columns(2, gap="large")
            d1.download_button(
                "DOWNLOAD MARKDOWN REPORT",
                data=report.encode("utf-8"),
                file_name="claire_yuan_court_bill_impact_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
            d2.download_button(
                "DOWNLOAD STRUCTURED JSON",
                data=result_json.encode("utf-8"),
                file_name="claire_yuan_court_bill_impact_result.json",
                mime="application/json",
                use_container_width=True,
            )

with methodology_tab:
    section_intro(
        "prototype architecture",
        "From document to calibrated report.",
        "The pipeline preserves the notebook's ML logic while the cloud app uses lightweight NumPy inference for the PyTorch-trained neural network.",
    )

    st.markdown(
        """
<div class="bb-step-grid">
  <div class="bb-step"><div class="bb-step-no">step 01</div><h3>Ingest</h3><p>Accept PDF, image, TXT, CSV, or Markdown. OCR is used only when normal text extraction is insufficient.</p></div>
  <div class="bb-step"><div class="bb-step-no">step 02</div><h3>Parse</h3><p>Transparent regular expressions and keyword rules extract amount, deadline, violation, speed, and court-appearance signals.</p></div>
  <div class="bb-step"><div class="bb-step-no">step 03</div><h3>Represent</h3><p>TF-IDF text features are combined with one-hot categorical variables and standardized numeric features.</p></div>
  <div class="bb-step"><div class="bb-step-no">step 04</div><h3>Predict</h3><p>Classical Ridge/logistic models and a multi-task DNN estimate continuous and categorical prototype outcomes.</p></div>
  <div class="bb-step"><div class="bb-step-no">step 05</div><h3>Ensemble</h3><p>Classical and neural predictions are blended rather than treating either model family as a sole decision-maker.</p></div>
  <div class="bb-step"><div class="bb-step-no">step 06</div><h3>Calibrate</h3><p>Held-out synthetic residuals form approximate 90% empirical ranges around the regression estimates.</p></div>
  <div class="bb-step"><div class="bb-step-no">step 07</div><h3>Present</h3><p>A constrained RL demo selects report presentation style only. It never changes an official legal consequence.</p></div>
  <div class="bb-step"><div class="bb-step-no">cloud</div><h3>Stay light</h3><p>The DNN is trained in PyTorch offline, then exported to NumPy arrays so Streamlit Cloud does not download PyTorch at startup.</p></div>
  <div class="bb-step"><div class="bb-step-no">human</div><h3>Verify</h3><p>Every parsed field and predicted consequence requires comparison with the original notice and an authoritative source.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="bb-rule"></div>', unsafe_allow_html=True)
    st.info(
        "The bundled training data are synthetic Demo-* records. Replace them only with lawful, de-identified, documented, jurisdiction-approved records before any serious evaluation."
    )

    if MODEL_PATH.exists():
        bundle, _ = get_models()
        meta = bundle.get("metadata", {})
        st.markdown("### Bundled model metadata")
        st.json(meta)
        q = bundle.get("report_style_q_table")
        if isinstance(q, pd.DataFrame):
            st.markdown("### RL report-style value table")
            st.dataframe(q.round(3), use_container_width=True)

with about_tab:
    section_intro(
        "project + deployment",
        "A Streamlit-native migration of the attached UI system.",
        "The attached TanStack/React design was translated into native Streamlit so Community Cloud needs only one Python app process. The visual language is preserved without carrying over unrelated legislation-news or Supabase services.",
    )

    st.markdown(
        """
<div class="bb-step-grid">
  <div class="bb-step"><div class="bb-step-no">author</div><h3>Claire Yuan</h3><p>Student author and product developer for the AI Court-Bill Impact Analyzer prototype.</p></div>
  <div class="bb-step"><div class="bb-step-no">advisor</div><h3>Dr. Qingyang Xiao</h3><p>Project advisor credited in the application, README, notebook metadata, and model bundle.</p></div>
  <div class="bb-step"><div class="bb-step-no">license</div><h3>MIT</h3><p>The repository includes the standard MIT License with 2026 copyright attribution to Claire Yuan.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="bb-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        """
### Streamlit Community Cloud

Deploy the repository with **`app.py`** as the entrypoint. The inference environment intentionally excludes PyTorch; `requirements.txt` contains cloud runtime dependencies and `packages.txt` contains Tesseract/Poppler system packages for OCR.

### UI migration boundary

The attached source UI also contains legislation-news, bill-search, external service, and Supabase/server features. Those unrelated services were **not** copied into this court-notice prototype. The design system and interaction language were migrated while the existing AI Court-Bill analysis functionality remained the source of truth.

### Responsible-use boundary

This application is an educational model demonstration. It must not be used to make sentencing, guilt, eligibility, creditworthiness, or other high-impact determinations about a person. Official consequences must be verified with the appropriate court, motor-vehicle agency, insurer, or qualified professional.
"""
    )

# Principles section mirrors the editorial principles block in the attached UI.
st.markdown(
    """
<section class="bb-principles">
  <div class="bb-section-kicker">principles</div>
  <div class="bb-principle-grid">
    <div>
      <h2>Explain first.<br><em>Never decide for you.</em></h2>
    </div>
    <div class="bb-principle-list">
      <div class="bb-principle-item"><span class="bb-arrow">→</span><span>Predictions are synthetic research estimates, not official legal outcomes.</span></div>
      <div class="bb-principle-item"><span class="bb-arrow">→</span><span>Users review machine-extracted fields before analysis.</span></div>
      <div class="bb-principle-item"><span class="bb-arrow">→</span><span>Personally identifying information should be removed before upload or model development.</span></div>
      <div class="bb-principle-item"><span class="bb-arrow">→</span><span>High-impact legal, credit, insurance, and eligibility decisions remain outside this prototype.</span></div>
    </div>
  </div>
</section>
<div class="bb-footer">
  <span>© 2026 Claire Yuan</span>
  <span>AI Court-Bill Impact Analyzer</span>
  <span>Advisor · Dr. Qingyang Xiao</span>
  <span>MIT License</span>
</div>
""",
    unsafe_allow_html=True,
)
