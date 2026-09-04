from __future__ import annotations

import platform
import random
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling import INPUT_COLUMNS, REG_TARGETS, RISK_TO_ID, MultiTaskImpactNet

SEED = 42
N_SAMPLES = 2400
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
DEVICE = torch.device("cpu")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic_dataset(n=2400, seed=42):
    rng = np.random.default_rng(seed)
    jurisdictions = ["Demo-MD", "Demo-VA", "Demo-DC", "Demo-NY", "Demo-CA"]
    jurisdiction_mult = {"Demo-MD": 1.00, "Demo-VA": 0.95, "Demo-DC": 1.08, "Demo-NY": 1.12, "Demo-CA": 1.05}
    violations = {
        "PARKING": {"case_type": "parking", "fine": 55, "points": 0, "ins": 0.0, "years": 0.5, "severity": 0.5},
        "TOLL": {"case_type": "civil", "fine": 85, "points": 0, "ins": 0.0, "years": 0.5, "severity": 0.8},
        "SEATBELT": {"case_type": "traffic", "fine": 75, "points": 0, "ins": 0.8, "years": 1.0, "severity": 1.0},
        "REGISTRATION": {"case_type": "traffic", "fine": 140, "points": 0, "ins": 1.0, "years": 1.0, "severity": 1.3},
        "RED_LIGHT": {"case_type": "traffic", "fine": 180, "points": 2, "ins": 5.0, "years": 2.5, "severity": 2.5},
        "SPEED_1_9": {"case_type": "traffic", "fine": 125, "points": 1, "ins": 3.0, "years": 2.0, "severity": 1.8},
        "SPEED_10_19": {"case_type": "traffic", "fine": 210, "points": 2, "ins": 7.0, "years": 3.0, "severity": 3.0},
        "SPEED_20_PLUS": {"case_type": "traffic", "fine": 390, "points": 4, "ins": 14.0, "years": 4.0, "severity": 4.5},
        "RECKLESS": {"case_type": "traffic", "fine": 760, "points": 6, "ins": 24.0, "years": 5.0, "severity": 6.0},
    }
    violation_keys = list(violations)
    violation_probs = np.array([0.12, 0.08, 0.08, 0.08, 0.12, 0.18, 0.17, 0.11, 0.06])
    templates = [
        "Official notice from {jur}. Alleged code {code}. Fine due is ${fine:.2f} within {days} days. {detail} {court}",
        "Citation / court bill. Jurisdiction: {jur}. Violation: {code}. Amount: ${fine:.2f}. Payment deadline: {days} days. {detail} {court}",
        "Case notice for {code} in {jur}. The listed monetary penalty is ${fine:.2f}. Respond in {days} days. {detail} {court}",
    ]
    rows = []
    for i in range(n):
        code = rng.choice(violation_keys, p=violation_probs)
        spec = violations[code]
        jur = rng.choice(jurisdictions)
        prior = int(np.clip(rng.poisson(0.65), 0, 5))
        days = int(rng.choice([15, 20, 30, 45, 60], p=[0.08, 0.10, 0.50, 0.22, 0.10]))
        if code == "SPEED_1_9":
            speed_over = int(rng.integers(1, 10))
        elif code == "SPEED_10_19":
            speed_over = int(rng.integers(10, 20))
        elif code == "SPEED_20_PLUS":
            speed_over = int(rng.integers(20, 36))
        elif code == "RECKLESS":
            speed_over = int(rng.integers(20, 46))
        else:
            speed_over = 0
        fine = spec["fine"] * jurisdiction_mult[jur]
        fine += 22 * prior + max(speed_over - 10, 0) * 4.5 + rng.normal(0, 22)
        fine = float(np.clip(fine, 20, 2500))
        court_required = int(spec["severity"] >= 4.5 or fine > 650 or rng.random() < 0.03)
        points = spec["points"] + 0.25 * prior + (0.6 if court_required else 0) + rng.normal(0, 0.55)
        points = float(np.clip(points, 0, 12))
        insurance = spec["ins"] * jurisdiction_mult[jur]
        insurance += 2.2 * prior + 0.18 * speed_over + (3.0 if court_required else 0) + rng.normal(0, 2.4)
        insurance = float(np.clip(insurance, 0, 65))
        impact_years = spec["years"] + 0.25 * prior + (0.45 if court_required else 0) + rng.normal(0, 0.35)
        impact_years = float(np.clip(impact_years, 0, 7))
        escalation_prob = sigmoid(-3.1 + 0.0032 * fine + 0.38 * prior + 0.75 * court_required - 0.018 * days)
        escalation_risk = int(rng.random() < escalation_prob)
        impact_score = insurance + points * 4.2 + impact_years * 2.1 + escalation_risk * 10
        risk_level = "low" if impact_score < 17 else "medium" if impact_score < 40 else "high"
        detail = f"Recorded speed is {speed_over} mph above the limit." if speed_over > 0 else "No speed-over-limit value is listed."
        court = "Court appearance is marked required." if court_required else "Court appearance is not marked as required on this notice."
        bill_text = rng.choice(templates).format(jur=jur, code=code, fine=fine, days=days, detail=detail, court=court)
        rows.append({
            "case_id_hash": f"SYN-{i:06d}", "bill_text": bill_text, "jurisdiction": jur,
            "case_type": spec["case_type"], "violation_code": code, "fine_amount": round(fine, 2),
            "days_to_due": days, "prior_case_count": prior, "speed_over_mph": speed_over,
            "court_appearance_required": court_required, "insurance_change_pct": round(insurance, 2),
            "impact_years": round(impact_years, 2), "license_points": round(points, 2),
            "escalation_risk": escalation_risk, "risk_level": risk_level,
        })
    return pd.DataFrame(rows)


TEXT_COL = "bill_text"
CAT_COLS = ["jurisdiction", "case_type", "violation_code"]
NUM_COLS = ["fine_amount", "days_to_due", "prior_case_count", "speed_over_mph", "court_appearance_required"]


def make_preprocessor(max_text_features=1400):
    return ColumnTransformer([
        ("text", TfidfVectorizer(max_features=max_text_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True), TEXT_COL),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", StandardScaler(with_mean=False), NUM_COLS),
    ], remainder="drop", sparse_threshold=0.3)


def make_regression_pipeline(alpha=5.0):
    return Pipeline([("preprocessor", make_preprocessor()), ("model", Ridge(alpha=alpha, solver="lsqr"))])


def make_classification_pipeline():
    return Pipeline([("preprocessor", make_preprocessor()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs"))])


def to_dense_float32(x):
    return x.toarray().astype(np.float32) if sp.issparse(x) else np.asarray(x, dtype=np.float32)


def main():
    data = make_synthetic_dataset(N_SAMPLES, SEED)
    train_df, temp_df = train_test_split(data, test_size=0.40, random_state=SEED, stratify=data["risk_level"])
    cal_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["risk_level"])

    models = {
        "insurance": make_regression_pipeline(7.0),
        "duration": make_regression_pipeline(5.0),
        "points": make_regression_pipeline(5.0),
        "escalation": make_classification_pipeline(),
        "risk": make_classification_pipeline(),
    }
    targets = {"insurance": "insurance_change_pct", "duration": "impact_years", "points": "license_points", "escalation": "escalation_risk", "risk": "risk_level"}
    for name, model in models.items():
        model.fit(train_df[INPUT_COLUMNS], train_df[targets[name]])
        print("trained", name, flush=True)

    calibration_quantiles = {}
    for name in ["insurance", "duration", "points"]:
        true = cal_df[targets[name]].to_numpy()
        pred = models[name].predict(cal_df[INPUT_COLUMNS])
        calibration_quantiles[name] = float(np.quantile(np.abs(true - pred), 0.90, method="higher"))

    dnn_preprocessor = make_preprocessor(max_text_features=1000)
    X_train = to_dense_float32(dnn_preprocessor.fit_transform(train_df[INPUT_COLUMNS]))
    X_cal = to_dense_float32(dnn_preprocessor.transform(cal_df[INPUT_COLUMNS]))
    reg_mean = train_df[REG_TARGETS].mean().to_numpy(dtype=np.float32)
    reg_std = train_df[REG_TARGETS].std().replace(0, 1).to_numpy(dtype=np.float32)

    def targets_for(df):
        yreg = (df[REG_TARGETS].to_numpy(dtype=np.float32) - reg_mean) / reg_std
        yesc = df["escalation_risk"].to_numpy(dtype=np.float32).reshape(-1, 1)
        yrisk = df["risk_level"].map(RISK_TO_ID).to_numpy(dtype=np.int64)
        return yreg, yesc, yrisk

    def loader(X, targets, shuffle=False):
        yreg, yesc, yrisk = targets
        ds = TensorDataset(torch.tensor(X), torch.tensor(yreg), torch.tensor(yesc), torch.tensor(yrisk))
        return DataLoader(ds, batch_size=96, shuffle=shuffle)

    train_loader = loader(X_train, targets_for(train_df), True)
    cal_loader = loader(X_cal, targets_for(cal_df), False)
    net = MultiTaskImpactNet(X_train.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_reg, loss_bin, loss_multi = nn.MSELoss(), nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()

    def eval_loss():
        net.eval(); total = 0.0; count = 0
        with torch.no_grad():
            for xb, yreg, yesc, yrisk in cal_loader:
                preg, pesc, prisk = net(xb)
                loss = loss_reg(preg, yreg) + 0.35 * loss_bin(pesc, yesc) + 0.35 * loss_multi(prisk, yrisk)
                total += float(loss.item()) * len(xb); count += len(xb)
        return total / max(count, 1)

    best_state, best_cal, no_improve = None, float("inf"), 0
    for epoch in range(1, 29):
        net.train()
        for xb, yreg, yesc, yrisk in train_loader:
            opt.zero_grad(); preg, pesc, prisk = net(xb)
            loss = loss_reg(preg, yreg) + 0.35 * loss_bin(pesc, yesc) + 0.35 * loss_multi(prisk, yrisk)
            loss.backward(); opt.step()
        cal_loss = eval_loss()
        if cal_loss < best_cal - 1e-4:
            best_cal = cal_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch == 1 or epoch % 4 == 0:
            print(f"epoch {epoch:02d} calibration={cal_loss:.4f}", flush=True)
        if no_improve >= 6:
            break
    if best_state is not None:
        net.load_state_dict(best_state)

    report_styles = ["concise", "detailed", "action_first"]
    contexts = ["low", "medium", "high"]
    Q = pd.DataFrame(0.0, index=contexts, columns=report_styles)
    preference = {
        "low": {"concise": 0.78, "detailed": 0.45, "action_first": 0.58},
        "medium": {"concise": 0.52, "detailed": 0.72, "action_first": 0.76},
        "high": {"concise": 0.35, "detailed": 0.70, "action_first": 0.88},
    }
    rng = np.random.default_rng(SEED)
    for _ in range(3500):
        context = rng.choice(contexts, p=[0.45, 0.38, 0.17])
        action = rng.choice(report_styles) if rng.random() < 0.15 else Q.loc[context].idxmax()
        reward = float(rng.random() < preference[context][action])
        Q.loc[context, action] += 0.12 * (reward - Q.loc[context, action])

    artifact = {
        "classical_models": models,
        "dnn_preprocessor": dnn_preprocessor,
        "dnn_state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "dnn_input_dim": X_train.shape[1],
        "reg_mean": reg_mean,
        "reg_std": reg_std,
        "calibration_quantiles": calibration_quantiles,
        "risk_to_id": RISK_TO_ID,
        "report_style_q_table": Q,
        "input_columns": INPUT_COLUMNS,
        "metadata": {
            "author": "Claire Yuan",
            "advisor": "Dr. Qingyang Xiao",
            "seed": SEED,
            "synthetic_training_data": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    out = ROOT / "models" / "claire_yuan_court_bill_impact_models.joblib"
    joblib.dump(artifact, out, compress=3)
    print("saved", out, out.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
