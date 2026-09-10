from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

SEED = 42
INPUT_COLUMNS = [
    "bill_text", "jurisdiction", "case_type", "violation_code",
    "fine_amount", "days_to_due", "prior_case_count",
    "speed_over_mph", "court_appearance_required",
]
REG_TARGETS = ["insurance_change_pct", "impact_years", "license_points"]
RISK_TO_ID = {"low": 0, "medium": 1, "high": 2}
ID_TO_RISK = {v: k for k, v in RISK_TO_ID.items()}
DEMO_JURISDICTIONS = ["Demo-MD", "Demo-VA", "Demo-DC", "Demo-NY", "Demo-CA"]
VIOLATION_CODES = [
    "PARKING", "TOLL", "SEATBELT", "REGISTRATION", "RED_LIGHT",
    "SPEED_1_9", "SPEED_10_19", "SPEED_20_PLUS", "RECKLESS",
]
CASE_TYPE_MAP = {
    "PARKING": "parking", "TOLL": "civil", "SEATBELT": "traffic",
    "REGISTRATION": "traffic", "RED_LIGHT": "traffic", "SPEED_1_9": "traffic",
    "SPEED_10_19": "traffic", "SPEED_20_PLUS": "traffic", "RECKLESS": "traffic",
}

DEMO_BILL_TEXT = """
OFFICIAL CITATION / COURT NOTICE
Jurisdiction: Demo-MD
Violation: SPEED_10_19
The vehicle was recorded traveling 52 mph in a 35 mph zone, 17 mph above the limit.
Fine amount due: $225.00
Payment or response is due within 30 days.
Court appearance is not required unless the citation is contested.
""".strip()



@dataclass
class ImpactPrediction:
    insurance_mid_pct: float
    insurance_low_pct: float
    insurance_high_pct: float
    duration_mid_years: float
    duration_low_years: float
    duration_high_years: float
    license_points_mid: float
    license_points_low: float
    license_points_high: float
    escalation_probability: float
    risk_level: str
    report_style: str
    model_scope: str = "Synthetic educational prototype"


def infer_speed_over(text: str) -> int:
    patterns = [
        r"(\d{1,2})\s*mph\s*(?:over|above)",
        r"(\d{1,2})\s*miles?\s+per\s+hour\s*(?:over|above)",
        r"(\d{1,3})\s*mph\s+in\s+(?:a\s+)?(\d{1,3})\s*mph",
        r"traveling\s+(\d{1,3})\s*mph\s+in\s+(?:a\s+)?(\d{1,3})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            nums = [int(x) for x in match.groups()]
            return max(nums[0] - nums[1], 0) if len(nums) == 2 else nums[0]
    return 0


def infer_violation_code(text: str) -> str:
    upper = text.upper()
    direct_codes = [
        "RECKLESS", "SPEED_20_PLUS", "SPEED_10_19", "SPEED_1_9",
        "RED_LIGHT", "REGISTRATION", "SEATBELT", "TOLL", "PARKING",
    ]
    for code in direct_codes:
        if code in upper:
            return code
    if "RECKLESS" in upper:
        return "RECKLESS"
    if "RED LIGHT" in upper or "TRAFFIC SIGNAL" in upper:
        return "RED_LIGHT"
    if "SEAT BELT" in upper or "SEATBELT" in upper:
        return "SEATBELT"
    if "REGISTRATION" in upper:
        return "REGISTRATION"
    if "TOLL" in upper:
        return "TOLL"
    if "PARKING" in upper:
        return "PARKING"
    speed_over = infer_speed_over(text)
    if speed_over >= 20:
        return "SPEED_20_PLUS"
    if speed_over >= 10:
        return "SPEED_10_19"
    if speed_over > 0:
        return "SPEED_1_9"
    return "PARKING"


def parse_bill_text(text: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    defaults = defaults or {}
    clean = re.sub(r"\s+", " ", text).strip()
    fine_match = re.search(
        r"(?:fine|amount|penalty)[^$\d]{0,25}\$?\s*([\d,]+(?:\.\d{1,2})?)",
        clean,
        re.IGNORECASE,
    )
    due_match = re.search(r"(?:within|due in|respond in)\s+(\d{1,3})\s+days", clean, re.IGNORECASE)
    jur_match = re.search(r"Jurisdiction\s*[:\-]\s*([A-Za-z-]+)", clean, re.IGNORECASE)

    violation = infer_violation_code(clean)
    court_required = int(bool(re.search(r"court appearance\s+(?:is\s+)?required", clean, re.IGNORECASE)))
    if re.search(r"court appearance\s+(?:is\s+)?not required", clean, re.IGNORECASE):
        court_required = 0

    jurisdiction = jur_match.group(1) if jur_match else defaults.get("jurisdiction", "Demo-MD")
    if jurisdiction not in DEMO_JURISDICTIONS:
        jurisdiction = defaults.get("jurisdiction", "Demo-MD")

    return {
        "bill_text": clean,
        "jurisdiction": jurisdiction,
        "case_type": CASE_TYPE_MAP.get(violation, defaults.get("case_type", "traffic")),
        "violation_code": violation,
        "fine_amount": float(fine_match.group(1).replace(",", "")) if fine_match else float(defaults.get("fine_amount", 150.0)),
        "days_to_due": int(due_match.group(1)) if due_match else int(defaults.get("days_to_due", 30)),
        "prior_case_count": int(defaults.get("prior_case_count", 0)),
        "speed_over_mph": int(infer_speed_over(clean)),
        "court_appearance_required": court_required,
    }


def to_dense_float32(x):
    """Convert scikit-learn sparse/dense output to a NumPy float32 array."""
    return x.toarray().astype(np.float32) if hasattr(x, "toarray") else np.asarray(x, dtype=np.float32)


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load the cloud-ready model bundle (no PyTorch runtime required)."""
    return joblib.load(path)


def build_dnn_from_bundle(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return the exported neural-network weights stored as NumPy arrays.

    The network was trained with PyTorch offline, then its inference weights were
    exported to NumPy so Streamlit Community Cloud does not need to download the
    very large PyTorch runtime package during deployment.
    """
    state = bundle.get("dnn_numpy_state")
    if not isinstance(state, dict):
        raise RuntimeError(
            "This model bundle is not the lightweight cloud artifact. "
            "Re-run scripts/train_models.py or use the FIXED repository bundle."
        )
    return state


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def dnn_predict(bundle: dict[str, Any], state: dict[str, np.ndarray], case_df: pd.DataFrame):
    """Run the trained multi-task DNN with NumPy-only inference.

    This reproduces the original PyTorch evaluation graph:
    Linear -> ReLU -> BatchNorm -> Linear -> ReLU -> Linear -> ReLU -> heads.
    Dropout layers are inactive during evaluation and therefore do not appear
    in this inference implementation.
    """
    x = to_dense_float32(bundle["dnn_preprocessor"].transform(case_df[INPUT_COLUMNS]))

    h = x @ state["shared.0.weight"].T + state["shared.0.bias"]
    h = _relu(h)

    # BatchNorm1d evaluation mode, using the running statistics learned in training.
    h = (h - state["shared.2.running_mean"]) / np.sqrt(state["shared.2.running_var"] + 1e-5)
    h = h * state["shared.2.weight"] + state["shared.2.bias"]

    h = h @ state["shared.4.weight"].T + state["shared.4.bias"]
    h = _relu(h)
    h = h @ state["shared.7.weight"].T + state["shared.7.bias"]
    h = _relu(h)

    reg_scaled = h @ state["reg_head.weight"].T + state["reg_head.bias"]
    esc_logits = h @ state["escalation_head.weight"].T + state["escalation_head.bias"]
    risk_logits = h @ state["risk_head.weight"].T + state["risk_head.bias"]

    reg = reg_scaled * np.asarray(bundle["reg_std"], dtype=np.float32) + np.asarray(
        bundle["reg_mean"], dtype=np.float32
    )
    esc_prob = _sigmoid(esc_logits).ravel()
    risk_prob = _softmax(risk_logits)
    return reg, esc_prob, risk_prob


def predict_case(
    bundle: dict[str, Any],
    state: dict[str, np.ndarray],
    case: dict[str, Any],
) -> ImpactPrediction:
    case_df = pd.DataFrame([case])
    models = bundle["classical_models"]

    ml_ins = float(models["insurance"].predict(case_df[INPUT_COLUMNS])[0])
    ml_dur = float(models["duration"].predict(case_df[INPUT_COLUMNS])[0])
    ml_pts = float(models["points"].predict(case_df[INPUT_COLUMNS])[0])
    ml_esc = float(models["escalation"].predict_proba(case_df[INPUT_COLUMNS])[0, 1])
    ml_risk_prob = models["risk"].predict_proba(case_df[INPUT_COLUMNS])[0]
    ml_risk_classes = list(models["risk"].named_steps["model"].classes_)

    dnn_reg, dnn_esc, dnn_risk_prob = dnn_predict(bundle, state, case_df)

    ins_mid = max(0.0, 0.5 * ml_ins + 0.5 * float(dnn_reg[0, 0]))
    dur_mid = max(0.0, 0.5 * ml_dur + 0.5 * float(dnn_reg[0, 1]))
    pts_mid = max(0.0, 0.5 * ml_pts + 0.5 * float(dnn_reg[0, 2]))
    esc_prob = float(np.clip(0.5 * ml_esc + 0.5 * dnn_esc[0], 0, 1))

    ml_aligned = np.zeros(3, dtype=float)
    for cls, prob in zip(ml_risk_classes, ml_risk_prob):
        ml_aligned[RISK_TO_ID[cls]] = prob
    combined_risk = 0.5 * ml_aligned + 0.5 * dnn_risk_prob[0]
    risk_level = ID_TO_RISK[int(np.argmax(combined_risk))]

    qs = bundle["calibration_quantiles"]
    q_table = bundle["report_style_q_table"]
    style = q_table.loc[risk_level].idxmax()

    return ImpactPrediction(
        insurance_mid_pct=ins_mid,
        insurance_low_pct=max(0.0, ins_mid - float(qs["insurance"])),
        insurance_high_pct=max(0.0, ins_mid + float(qs["insurance"])),
        duration_mid_years=dur_mid,
        duration_low_years=max(0.0, dur_mid - float(qs["duration"])),
        duration_high_years=max(0.0, dur_mid + float(qs["duration"])),
        license_points_mid=pts_mid,
        license_points_low=max(0.0, pts_mid - float(qs["points"])),
        license_points_high=max(0.0, pts_mid + float(qs["points"])),
        escalation_probability=esc_prob,
        risk_level=risk_level,
        report_style=style,
    )


def money(x: float) -> str:
    return f"${x:,.2f}"


def make_impact_report(case: dict[str, Any], pred: ImpactPrediction) -> str:
    risk_pct = 100 * pred.escalation_probability
    action_items = [
        "Verify the citation number, jurisdiction, due date, and listed response options on the original notice.",
        "Use the official court or motor-vehicle-agency source to confirm points, deadlines, and hearing rights.",
        "Ask the current insurer for a quote or written explanation; the model cannot know an insurer-specific surcharge.",
        "Do not ignore the payment or response deadline. Seek qualified legal help when contesting the notice or when a court appearance is required.",
        "Redact names, addresses, account numbers, barcodes, and other identifiers before using documents for model development.",
    ]
    direct_credit_note = (
        "The prototype does not predict a fixed direct credit-score deduction from this notice. "
        "A separate escalation probability is shown because unpaid obligations may be handled differently, "
        "including possible collection activity, depending on the jurisdiction and facts."
    )
    report = f"""# Prototype Court-Bill Impact Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**Model scope:** {pred.model_scope}  
**Suggested report presentation:** {pred.report_style.replace('_', ' ').title()}

## 1. Extracted notice facts

- Fictional model jurisdiction: **{case['jurisdiction']}**
- Case type: **{case['case_type']}**
- Parsed violation: **{case['violation_code']}**
- Listed fine: **{money(case['fine_amount'])}**
- Response window: **{case['days_to_due']} days**
- Speed above limit: **{case['speed_over_mph']} mph**
- Court appearance marked required: **{'Yes' if case['court_appearance_required'] else 'No / not detected'}**

## 2. Modeled consequence ranges

| Output | Ensemble estimate | Approximate calibrated range |
|---|---:|---:|
| Possible insurance premium change | {pred.insurance_mid_pct:.1f}% | {pred.insurance_low_pct:.1f}% to {pred.insurance_high_pct:.1f}% |
| Possible impact duration | {pred.duration_mid_years:.1f} years | {pred.duration_low_years:.1f} to {pred.duration_high_years:.1f} years |
| Possible license points | {pred.license_points_mid:.1f} | {pred.license_points_low:.1f} to {pred.license_points_high:.1f} |
| Nonpayment / escalation probability | {risk_pct:.1f}% | Classification probability, not a legal finding |
| Overall modeled impact category | **{pred.risk_level.upper()}** | Synthetic training categories |

## 3. Credit-report interpretation

{direct_credit_note}

## 4. Recommended verification steps

"""
    report += "\n".join(f"{i + 1}. {item}" for i, item in enumerate(action_items))
    report += """

## 5. Limitations

- The model was trained on generated demonstration data, not official court records.
- Actual license points, insurance effects, deadlines, collections, and collateral consequences vary by jurisdiction, disposition, insurer, policy, and individual circumstances.
- OCR and rule-based parsing can misread documents; compare every extracted field with the original notice.
- The report is not a determination of guilt, legal liability, sentence, eligibility, or creditworthiness.
- A real deployment must include human review, data-use agreements, privacy controls, bias evaluation, audit logs, model/version documentation, and a process for correction and appeal.
"""
    return report


def structured_result(case: dict[str, Any], pred: ImpactPrediction) -> str:
    return json.dumps({"parsed_case": case, "prediction": asdict(pred)}, indent=2)
