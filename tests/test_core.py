from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.bill_analysis import (
    analyze_bill,
    compare_analyses,
    detect_status,
    extract_votes,
    personalize_impact,
    version_diff,
)
from src.congress_client import (
    BillRef,
    CongressClient,
    current_congress_fallback,
    parse_bill_citation,
    preferred_text_format,
)
import src.persistence as persistence


ROOT = Path(__file__).resolve().parents[1]
DEMO_TEXT = (ROOT / "examples" / "demo_bill.txt").read_text(encoding="utf-8")


def bundle() -> dict:
    return {
        "ref": {
            "congress": 119,
            "bill_type": "hr",
            "number": 99999,
            "bill_id": "119-hr-99999",
            "citation": "H.R. DEMO",
            "official_url": "https://www.congress.gov/",
        },
        "detail": {
            "title": "Digital Skills and Rural Clinic Support Act of 2026",
            "introducedDate": "2026-02-12",
            "originChamber": "House",
            "policyArea": {"name": "Education"},
            "latestAction": {"actionDate": "2026-02-12", "text": "Referred to committee."},
            "cboCostEstimates": [],
        },
        "actions": [
            {"actionDate": "2026-02-12", "text": "Introduced in House and referred to the Committee on Education."}
        ],
        "summaries": [
            {"actionDate": "2026-02-12", "text": "Establishes rural digital-skills and telehealth grants."}
        ],
        "amendments": [],
        "committees": [],
        "cosponsors": [],
        "related_bills": [],
        "subjects": {},
        "text_versions": [],
        "titles": [],
    }


class CongressReferenceTests(unittest.TestCase):
    def test_citation_formats(self) -> None:
        self.assertEqual(parse_bill_citation("H.R. 123", 119), BillRef(119, "hr", 123))
        self.assertEqual(parse_bill_citation("S 7", 119), BillRef(119, "s", 7))
        self.assertEqual(parse_bill_citation("119-HJRES-5"), BillRef(119, "hjres", 5))
        self.assertEqual(parse_bill_citation("S.Con.Res. 2", 119), BillRef(119, "sconres", 2))

    def test_current_congress_boundary(self) -> None:
        self.assertEqual(current_congress_fallback(date(2025, 1, 2)), 118)
        self.assertEqual(current_congress_fallback(date(2025, 1, 3)), 119)
        self.assertEqual(current_congress_fallback(date(2026, 9, 1)), 119)

    def test_current_congress_item_response_shape(self) -> None:
        client = CongressClient("test-key")
        client._get = lambda *_args, **_kwargs: {"congress": {"number": 119}}  # type: ignore[method-assign]
        self.assertEqual(client.get_current_congress(), 119)

    def test_formatted_text_is_preferred_for_cloud_portability(self) -> None:
        selected = preferred_text_format(
            {
                "formats": [
                    {"type": "Formatted XML", "url": "https://www.congress.gov/a.xml"},
                    {"type": "PDF", "url": "https://www.congress.gov/a.pdf"},
                    {"type": "Formatted Text", "url": "https://www.congress.gov/a.htm"},
                ]
            }
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["type"], "Formatted Text")

    def test_collection_pagination_combines_pages(self) -> None:
        client = CongressClient("test-key")

        def fake_get(_path: str, *, params=None, allow_404=False):  # type: ignore[no-untyped-def]
            offset = int((params or {}).get("offset", 0))
            if offset == 0:
                return {
                    "actions": [{"actionDate": "2026-01-01", "text": "Introduced"}],
                    "pagination": {"count": 2},
                }
            return {
                "actions": [{"actionDate": "2026-01-02", "text": "Referred"}],
                "pagination": {"count": 2},
            }

        client._get = fake_get  # type: ignore[method-assign]
        actions = client._get_collection("/bill/example/actions", "actions")
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[-1]["text"], "Referred")


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analysis = analyze_bill(
            DEMO_TEXT,
            bundle=bundle(),
            source_kind="fictional demonstration",
            citation="H.R. DEMO",
            title="Digital Skills and Rural Clinic Support Act of 2026",
        )

    def test_section_and_status_pipeline(self) -> None:
        self.assertGreaterEqual(len(self.analysis.sections), 8)
        self.assertEqual(self.analysis.status.label, "in committee")
        self.assertIn("$120,000,000", self.analysis.fiscal["detected_amounts"])
        self.assertIn("health", self.analysis.topics)
        self.assertTrue(self.analysis.plain_summary)

    def test_personalization(self) -> None:
        result = personalize_impact(
            self.analysis,
            {
                "age": 67,
                "state": "MD",
                "income_bracket": "$50,000–$99,999",
                "occupation": "nurse",
                "industry": "health care",
                "veteran": True,
            },
        )
        self.assertIn("signals", result)
        self.assertGreaterEqual(len(result["signals"]), 3)
        self.assertIn("disclaimer", result)

    def test_compare_and_diff(self) -> None:
        second = analyze_bill(
            DEMO_TEXT.replace("$120,000,000", "$150,000,000"),
            source_kind="pasted comparison text",
            citation="S. DEMO",
            title="Rural Technology Demonstration Act",
        )
        comparison = compare_analyses(self.analysis, second)
        self.assertGreater(comparison["similarity_pct"], 50)
        diff = version_diff("line one\nline two", "line one\nline three")
        self.assertEqual(diff["added_lines"], 1)
        self.assertEqual(diff["removed_lines"], 1)

    def test_nested_recorded_vote_shape(self) -> None:
        votes = extract_votes(
            {
                "actions": [
                    {
                        "actionDate": "2026-06-01",
                        "text": "Passed the House.",
                        "recordedVotes": {
                            "recordedVote": [
                                {
                                    "chamber": "House",
                                    "date": "2026-06-01",
                                    "rollNumber": 321,
                                    "url": "https://clerk.house.gov/Votes/2026321",
                                }
                            ]
                        },
                    }
                ]
            }
        )
        self.assertEqual(len(votes), 1)
        self.assertEqual(votes[0]["roll_number"], 321)
        self.assertEqual(votes[0]["chamber"], "House")

    def test_enactment_takes_precedence_over_historic_veto_text(self) -> None:
        status = detect_status(
            {
                "ref": {"bill_type": "hr"},
                "detail": {"laws": [{"number": "123-1", "type": "Public Law"}]},
                "actions": [
                    {"actionDate": "2026-07-01", "text": "Vetoed by President."},
                    {"actionDate": "2026-07-10", "text": "Became Public Law after veto override."},
                ],
            }
        )
        self.assertEqual(status.label, "enacted")

    def test_rejected_amendment_does_not_mark_bill_failed(self) -> None:
        status = detect_status(
            {
                "ref": {"bill_type": "hr"},
                "detail": {"introducedDate": "2026-01-10"},
                "actions": [
                    {"actionDate": "2026-01-10", "text": "Introduced in House and referred to committee."},
                    {"actionDate": "2026-03-01", "text": "Amendment No. 3 rejected by the committee."},
                    {"actionDate": "2026-03-02", "text": "Committee consideration and mark-up session held."},
                ],
            }
        )
        self.assertEqual(status.label, "in committee")

    def test_official_summary_link_uses_stable_bill_page(self) -> None:
        self.assertEqual(
            self.analysis.source_links["official_summary"],
            self.analysis.source_links["official_bill_page"],
        )


class PersistenceTests(unittest.TestCase):
    def test_accounts_visits_and_follows(self) -> None:
        old_path = persistence.DB_PATH
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                persistence.DB_PATH = Path(temp_dir) / "test.sqlite3"
                persistence.init_db()
                self.assertEqual(persistence.record_visit("session-a"), 1)
                self.assertEqual(persistence.record_visit("session-a"), 1)
                self.assertEqual(persistence.record_visit("session-b"), 2)

                user = persistence.create_user("claire@example.com", "password123", "Claire")
                self.assertIsNotNone(persistence.authenticate("claire@example.com", "password123"))
                persistence.save_profile(user["id"], {"state": "MD", "age": 17})
                self.assertEqual(persistence.get_profile(user["id"])["state"], "MD")

                persistence.follow_bill(
                    user["id"],
                    bill_id="119-hr-1",
                    citation="H.R. 1",
                    title="Test Bill",
                    status="introduced",
                )
                self.assertEqual(len(persistence.list_follows(user["id"])), 1)
                changed = persistence.update_follow_and_notify(
                    user["id"],
                    "119-hr-1",
                    new_status="in committee",
                    latest_action_date="2026-01-01",
                    latest_action_text="Referred to committee.",
                )
                self.assertTrue(changed)
                self.assertEqual(len(persistence.list_notifications(user["id"])), 1)
        finally:
            persistence.DB_PATH = old_path


if __name__ == "__main__":
    unittest.main()
