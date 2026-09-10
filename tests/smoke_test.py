from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling import (  # noqa: E402
    DEMO_BILL_TEXT,
    build_dnn_from_bundle,
    load_bundle,
    make_impact_report,
    parse_bill_text,
    predict_case,
)


def main():
    model_path = ROOT / "models" / "claire_yuan_court_bill_impact_models.joblib"
    assert model_path.exists(), f"Missing model: {model_path}"

    bundle = load_bundle(model_path)
    assert "dnn_numpy_state" in bundle, "Cloud bundle is missing NumPy DNN weights"
    assert "dnn_state_dict" not in bundle, "Cloud bundle still contains PyTorch tensors"

    state = build_dnn_from_bundle(bundle)
    case = parse_bill_text(
        DEMO_BILL_TEXT,
        defaults={"jurisdiction": "Demo-MD", "prior_case_count": 0},
    )
    pred = predict_case(bundle, state, case)
    report = make_impact_report(case, pred)

    assert case["violation_code"] == "SPEED_10_19"
    assert 0.0 <= pred.escalation_probability <= 1.0
    assert pred.risk_level in {"low", "medium", "high"}
    assert pred.insurance_high_pct >= pred.insurance_low_pct >= 0.0
    assert pred.duration_high_years >= pred.duration_low_years >= 0.0
    assert pred.license_points_high >= pred.license_points_low >= 0.0
    assert "Prototype Court-Bill Impact Report" in report

    meta = bundle.get("metadata", {})
    assert meta.get("author") == "Claire Yuan"
    assert meta.get("advisor") == "Dr. Qingyang Xiao"

    print("SMOKE TEST PASSED")
    print(f"Parsed violation: {case['violation_code']}")
    print(f"Insurance estimate: {pred.insurance_mid_pct:.3f}%")
    print(f"Impact duration: {pred.duration_mid_years:.3f} years")
    print(f"License points: {pred.license_points_mid:.3f}")
    print(f"Escalation probability: {pred.escalation_probability:.4f}")
    print(f"Risk category: {pred.risk_level}")
    print(f"Report style: {pred.report_style}")


if __name__ == "__main__":
    main()
