from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    cfg = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")

    assert "Author: Claire Yuan" in app
    assert "Advisor: Dr. Qingyang Xiao" in app
    assert "court bills," in app
    assert "JetBrains Mono" in app
    assert "Instrument Serif" in app
    assert "#faf8f3" in app.lower()
    assert "#d07a2d" in app.lower()
    assert "bb-ticker" in app
    assert "bb-principles" in app
    assert "torch" not in [line.strip().split("==")[0] for line in req.splitlines() if line.strip() and not line.startswith("#")]
    assert 'backgroundColor = "#FAF8F3"' in cfg
    assert 'primaryColor = "#D07A2D"' in cfg

    print("UI MIGRATION TEST PASSED")


if __name__ == "__main__":
    main()
