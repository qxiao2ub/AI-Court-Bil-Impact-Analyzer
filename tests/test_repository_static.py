from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryStaticTests(unittest.TestCase):
    def test_toolbar_safe_layout_and_credits_are_present(self) -> None:
        ui_text = (ROOT / "src" / "ui.py").read_text(encoding="utf-8")
        self.assertIn("--streamlit-toolbar-safe-area: 5.25rem", ui_text)
        self.assertIn('[data-testid="stMainBlockContainer"]', ui_text)
        self.assertIn("Author: Claire Yuan", ui_text)
        self.assertIn("Advisor: Dr. Qingyang Xiao", ui_text)

    def test_streamlit_entrypoint_contains_requested_workflows(self) -> None:
        app_text = (ROOT / "app.py").read_text(encoding="utf-8")
        for marker in (
            "Live Congress.gov bill",
            "Paste bill text",
            "Upload bill PDF",
            "HOW THIS AFFECTS YOU",
            "FOLLOW + CONTACT",
            "COMPARE BILLS",
            "record_visit",
            "version_diff",
            "LOOK UP CURRENT MEMBERS",
        ):
            self.assertIn(marker, app_text)

    def test_cloud_dependency_files_are_at_repository_root(self) -> None:
        self.assertTrue((ROOT / "requirements.txt").is_file())
        self.assertTrue((ROOT / "packages.txt").is_file())
        self.assertTrue((ROOT / ".streamlit" / "config.toml").is_file())
        self.assertTrue((ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License"))


if __name__ == "__main__":
    unittest.main()
