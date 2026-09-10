from __future__ import annotations

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
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {max-width: 1250px; padding-top: 1.8rem; padding-bottom: 3rem;}
.hero {padding: 1.3rem 1.5rem; border: 1px solid rgba(128,128,128,.25); border-radius: 18px; margin-bottom: 1rem;}
.hero h1 {margin: 0 0 .35rem 0; font-size: 2.25rem;}
.hero p {margin: .2rem 0; opacity: .86;}
.credit {font-size: .98rem; font-weight: 600;}
.small-note {font-size: .86rem; opacity: .8;}
[data-testid="stMetricValue"] {font-size: 1.7rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>⚖️ AI Court-Bill Impact Analyzer</h1>
  <p>Upload or paste a court-issued notice, citation, fine, or bill and receive a research-prototype impact report.</p>
  <p class="credit">Author: Claire Yuan &nbsp;•&nbsp; Advisor: Dr. Qingyang Xiao</p>
</div>
""",
    unsafe_allow_html=True,
)

st.warning(
    "Educational research prototype only. The bundled model was trained on synthetic demonstration data. "
    "It does not provide legal advice, determine guilt, predict an official court disposition, or establish insurer/credit-bureau action."
)


@st.cache_resource(show_spinner="Loading trained ML and neural-network models...")
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
        text_parts = []
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


with st.sidebar:
    st.header("Project")
    st.markdown("**Author:** Claire Yuan")
    st.markdown("**Advisor:** Dr. Qingyang Xiao")
    st.caption("Python • scikit-learn • PyTorch-trained DNN / NumPy inference • OCR • calibration • constrained RL")
    st.divider()
    st.subheader("Privacy first")
    st.caption(
        "Before uploading any real document, redact names, addresses, dates of birth, account numbers, barcodes, license numbers, and other identifying information."
    )
    st.divider()
    st.subheader("Model scope")
    st.caption(
        "The included model uses fictional Demo-* jurisdiction labels and synthetic outcomes from the attached Colab prototype."
    )

input_tab, methodology_tab, about_tab = st.tabs(["Analyze a bill", "Model methodology", "About & deployment"])

with input_tab:
    st.subheader("1. Provide a court notice")
    input_mode = st.radio("Input method", ["Upload a document", "Paste text", "Use demo notice"], horizontal=True)

    source_text = ""
    source_name = ""
    if input_mode == "Upload a document":
        uploaded = st.file_uploader(
            "Upload a redacted PDF, image, TXT, CSV, or Markdown file",
            type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "txt", "csv", "md"],
        )
        if uploaded is not None:
            try:
                with st.spinner("Extracting text from the uploaded document..."):
                    source_text = extract_text_from_upload(uploaded)
                source_name = uploaded.name
                st.success(f"Extracted {len(source_text):,} characters from {uploaded.name}.")
            except Exception as exc:
                st.error(f"Text extraction failed: {exc}")
    elif input_mode == "Paste text":
        source_text = st.text_area(
            "Paste redacted notice text",
            height=220,
            placeholder="Paste the court notice or citation text here...",
        ).strip()
        source_name = "pasted_text"
    else:
        source_text = DEMO_BILL_TEXT
        source_name = "demo_speeding_notice.txt"
        st.code(DEMO_BILL_TEXT, language="text")

    if source_text:
        parsed = parse_bill_text(source_text, defaults={"prior_case_count": 0, "jurisdiction": "Demo-MD"})
        with st.expander("Review extracted text", expanded=False):
            st.text_area("Extracted text", source_text, height=220, disabled=True)

        st.subheader("2. Review and correct model inputs")
        st.caption("OCR and regular-expression parsing can be wrong. Confirm every field against the original notice before analysis.")

        c1, c2, c3 = st.columns(3)
        with c1:
            jurisdiction = st.selectbox("Fictional model jurisdiction", DEMO_JURISDICTIONS, index=DEMO_JURISDICTIONS.index(parsed["jurisdiction"]))
            violation = st.selectbox("Violation code", VIOLATION_CODES, index=VIOLATION_CODES.index(parsed["violation_code"]))
            case_type = CASE_TYPE_MAP[violation]
        with c2:
            fine_amount = st.number_input("Listed fine ($)", min_value=0.0, max_value=100000.0, value=float(parsed["fine_amount"]), step=5.0)
            days_to_due = st.number_input("Response window (days)", min_value=1, max_value=365, value=int(parsed["days_to_due"]), step=1)
            prior_cases = st.number_input("Prior similar case count", min_value=0, max_value=20, value=int(parsed["prior_case_count"]), step=1)
        with c3:
            speed_over = st.number_input("Speed above limit (mph)", min_value=0, max_value=100, value=int(parsed["speed_over_mph"]), step=1)
            court_required = st.checkbox("Court appearance marked required", value=bool(parsed["court_appearance_required"]))
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

        if st.button("Analyze bill impact", type="primary", use_container_width=True):
            if len(source_text.strip()) < 10:
                st.error("The extracted text is too short to analyze reliably.")
            else:
                try:
                    bundle, dnn_state = get_models()
                    pred = predict_case(bundle, dnn_state, case)
                    report = make_impact_report(case, pred)
                    result_json = json.dumps({"source_name": source_name, "parsed_case": case, "prediction": asdict(pred)}, indent=2)
                    st.session_state["analysis_result"] = (case, pred, report, result_json)
                except Exception as exc:
                    st.exception(exc)

    if "analysis_result" in st.session_state:
        case, pred, report, result_json = st.session_state["analysis_result"]
        st.divider()
        st.subheader("3. Modeled impact summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Insurance estimate", f"{pred.insurance_mid_pct:.1f}%")
        m2.metric("Impact duration", f"{pred.duration_mid_years:.1f} yrs")
        m3.metric("License points", f"{pred.license_points_mid:.1f}")
        m4.metric("Escalation risk", f"{100 * pred.escalation_probability:.1f}%")
        m5.metric("Impact category", pred.risk_level.upper())

        st.progress(min(max(float(pred.escalation_probability), 0.0), 1.0), text="Modeled nonpayment / escalation probability")
        st.caption(
            f"Approximate 90% calibration ranges — insurance: {pred.insurance_low_pct:.1f}%–{pred.insurance_high_pct:.1f}%; "
            f"duration: {pred.duration_low_years:.1f}–{pred.duration_high_years:.1f} years; "
            f"points: {pred.license_points_low:.1f}–{pred.license_points_high:.1f}."
        )

        report_view, inputs_view, download_view = st.tabs(["Impact report", "Parsed inputs", "Downloads"])
        with report_view:
            st.markdown(report)
        with inputs_view:
            st.dataframe(pd.DataFrame([case]).T.rename(columns={0: "value"}), use_container_width=True)
        with download_view:
            c1, c2 = st.columns(2)
            c1.download_button(
                "Download Markdown report",
                data=report.encode("utf-8"),
                file_name="claire_yuan_court_bill_impact_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
            c2.download_button(
                "Download structured JSON",
                data=result_json.encode("utf-8"),
                file_name="claire_yuan_court_bill_impact_result.json",
                mime="application/json",
                use_container_width=True,
            )

with methodology_tab:
    st.subheader("How the prototype works")
    st.markdown(
        """
1. **Document ingestion:** PDF/image/text input is converted to text, using OCR when needed.
2. **Transparent feature parsing:** regular expressions and keyword rules extract bill amount, response window, violation type, speed-over-limit, and court-appearance indicators.
3. **Classical ML:** TF-IDF text features, one-hot categorical features, and standardized numeric features feed Ridge and logistic-regression models.
4. **Deep learning:** a multi-task neural network trained offline with PyTorch jointly estimates insurance change, impact duration, license points, escalation probability, and overall impact class. For cloud deployment, its learned weights are exported to NumPy so the app does not download PyTorch at startup.
5. **Ensemble:** classical and neural predictions are averaged for the three regression outputs and probability outputs.
6. **Calibration:** held-out calibration residuals form approximate 90% empirical prediction ranges.
7. **Constrained reinforcement learning:** RL only chooses a report-presentation style (concise, detailed, or action-first). It does **not** change legal consequences or official decisions.
"""
    )
    st.info("The bundled training data are synthetic Demo-* records. Replace them only with lawful, de-identified, documented, jurisdiction-approved records before any serious evaluation.")

    if MODEL_PATH.exists():
        bundle, _ = get_models()
        meta = bundle.get("metadata", {})
        st.markdown("#### Bundled model metadata")
        st.json(meta)
        q = bundle.get("report_style_q_table")
        if isinstance(q, pd.DataFrame):
            st.markdown("#### RL report-style value table")
            st.dataframe(q.round(3), use_container_width=True)

with about_tab:
    st.subheader("Project credits")
    st.markdown("**Author:** Claire Yuan  ")
    st.markdown("**Advisor:** Dr. Qingyang Xiao")
    st.markdown("**License:** MIT License")
    st.markdown(
        """
### Streamlit Community Cloud
Deploy this repository with **`app.py`** as the entrypoint. The cloud runtime is compatible with Python 3.14 and intentionally avoids a PyTorch download; `requirements.txt` contains only inference dependencies and `packages.txt` contains Linux OCR dependencies.

### Responsible-use boundary
This application is an educational model demonstration. It must not be used to make sentencing, guilt, eligibility, creditworthiness, or other high-impact determinations about a person. Official consequences must be verified from the appropriate court, motor-vehicle agency, insurer, or qualified professional.
"""
    )

st.divider()
st.caption("AI Court-Bill Impact Analyzer • Author: Claire Yuan • Advisor: Dr. Qingyang Xiao • MIT License")
