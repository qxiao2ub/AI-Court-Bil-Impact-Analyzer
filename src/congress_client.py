from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = "https://api.congress.gov/v3"

BILL_TYPE_DISPLAY: dict[str, str] = {
    "hr": "H.R.",
    "s": "S.",
    "hjres": "H.J.Res.",
    "sjres": "S.J.Res.",
    "hconres": "H.Con.Res.",
    "sconres": "S.Con.Res.",
    "hres": "H.Res.",
    "sres": "S.Res.",
}

BILL_TYPE_SLUG: dict[str, str] = {
    "hr": "house-bill",
    "s": "senate-bill",
    "hjres": "house-joint-resolution",
    "sjres": "senate-joint-resolution",
    "hconres": "house-concurrent-resolution",
    "sconres": "senate-concurrent-resolution",
    "hres": "house-resolution",
    "sres": "senate-resolution",
}

BILL_TYPE_OPTIONS = list(BILL_TYPE_DISPLAY.keys())


class CongressAPIError(RuntimeError):
    """A user-facing Congress.gov API error."""

    def __init__(self, message: str, *, status_code: int | None = None, url: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url


@dataclass(frozen=True, slots=True)
class BillRef:
    congress: int
    bill_type: str
    number: int

    def __post_init__(self) -> None:
        normalized = self.bill_type.lower()
        if normalized not in BILL_TYPE_DISPLAY:
            raise ValueError(f"Unsupported bill type: {self.bill_type}")
        if self.congress < 1:
            raise ValueError("Congress number must be positive.")
        if self.number < 1:
            raise ValueError("Bill number must be positive.")
        object.__setattr__(self, "bill_type", normalized)

    @property
    def bill_id(self) -> str:
        return f"{self.congress}-{self.bill_type}-{self.number}"

    @property
    def citation(self) -> str:
        return f"{BILL_TYPE_DISPLAY[self.bill_type]} {self.number}"

    @property
    def display(self) -> str:
        return f"{self.citation} ({self.congress}th Congress)"

    @property
    def api_path(self) -> str:
        return f"/bill/{self.congress}/{self.bill_type}/{self.number}"

    @property
    def official_url(self) -> str:
        slug = BILL_TYPE_SLUG[self.bill_type]
        return f"https://www.congress.gov/bill/{self.congress}th-congress/{slug}/{self.number}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "congress": self.congress,
            "bill_type": self.bill_type,
            "number": self.number,
            "bill_id": self.bill_id,
            "citation": self.citation,
            "official_url": self.official_url,
        }


@dataclass(slots=True)
class DownloadedBillText:
    text: str
    source_url: str
    source_format: str
    content_type: str


def current_congress_fallback(today: date | None = None) -> int:
    """Return the current Congress number without requiring the API.

    A new Congress begins on January 3 of each odd-numbered year. This fallback
    is used only when /congress/current cannot be reached.
    """

    today = today or date.today()
    year = today.year
    if year % 2 == 1 and (today.month, today.day) < (1, 3):
        year -= 1
    return ((year - 1789) // 2) + 1


def parse_bill_citation(value: str, congress: int | None = None) -> BillRef:
    """Parse common federal bill citations such as H.R. 123, S. 9, or 119-HR-123."""

    raw = (value or "").strip().upper()
    if not raw:
        raise ValueError("Enter a bill citation such as H.R. 3076 or S. 1.")

    explicit_congress: int | None = None
    congress_match = re.match(r"^\s*(\d{1,3})\s*[-/:]\s*(.+)$", raw)
    if congress_match:
        explicit_congress = int(congress_match.group(1))
        raw = congress_match.group(2)

    compact = re.sub(r"[^A-Z0-9]", "", raw)
    aliases = ["HCONRES", "SCONRES", "HJRES", "SJRES", "HRES", "SRES", "HR", "S"]
    bill_type = ""
    number = 0
    for alias in aliases:
        match = re.search(rf"{alias}(\d+)$", compact)
        if match:
            bill_type = alias.lower()
            number = int(match.group(1))
            break

    if not bill_type:
        raise ValueError(
            "Could not parse the citation. Use H.R. 123, S. 45, H.J.Res. 7, "
            "S.J.Res. 7, H.Con.Res. 2, S.Con.Res. 2, H.Res. 10, or S.Res. 10."
        )

    selected_congress = explicit_congress or congress or current_congress_fallback()
    return BillRef(selected_congress, bill_type, number)


def _as_list(value: Any, *preferred_keys: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        for key in preferred_keys:
            candidate = value.get(key)
            if isinstance(candidate, list):
                return candidate
        for key in ("item", "items"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return candidate
        return [value]
    return [value]


def extract_api_items(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    items = _as_list(value, key, "item", "items")
    return [item for item in items if isinstance(item, dict)]


def preferred_text_format(version: dict[str, Any]) -> dict[str, Any] | None:
    formats = _as_list(version.get("formats"), "item", "formats")
    formats = [item for item in formats if isinstance(item, dict) and item.get("url")]
    if not formats:
        return None

    priorities = {
        # Congress.gov's Formatted Text is normally the most portable choice
        # for a lightweight Streamlit deployment. XML remains supported as a
        # fallback, but does not require the optional lxml parser.
        "formatted text": 0,
        "text": 1,
        "formatted xml": 2,
        "xml": 3,
        "pdf": 4,
    }
    return min(formats, key=lambda item: priorities.get(str(item.get("type", "")).lower(), 99))


def latest_text_version(versions: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    versions = [item for item in versions if isinstance(item, dict)]
    if not versions:
        return None
    return max(versions, key=lambda item: str(item.get("date") or ""))


def official_summary_url(ref: BillRef, summaries: list[dict[str, Any]]) -> str:
    # The bill landing page exposes the official CRS summary and remains valid
    # even when a summary-version slug is absent or changes. Avoid constructing
    # a route that Congress.gov may not recognize for every measure/version.
    return ref.official_url


class CongressClient:
    """Thin, defensive client for the official Congress.gov v3 API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = API_BASE,
        timeout: int = 30,
        user_agent: str = "Claire-Yuan-AI-Legislative-Bill-Analyzer/3.0",
    ) -> None:
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise CongressAPIError(
                "A Congress.gov API key is required for live bill lookup. Add it in Streamlit secrets or enter it for this session."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        retry = Retry(
            total=4,
            connect=4,
            read=4,
            status=4,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=12, pool_maxsize=12)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        allow_404: bool = False,
    ) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        merged = {"format": "json", "api_key": self.api_key}
        if params:
            merged.update({k: v for k, v in params.items() if v is not None})
        try:
            response = self.session.get(url, params=merged, timeout=self.timeout)
        except requests.RequestException as exc:
            raise CongressAPIError(f"Congress.gov request failed: {exc}", url=url) from exc

        if allow_404 and response.status_code == 404:
            return {}
        if response.status_code == 403:
            raise CongressAPIError(
                "Congress.gov rejected the request. Check that the API key is valid and has not exceeded its quota.",
                status_code=403,
                url=response.url,
            )
        if response.status_code == 404:
            raise CongressAPIError(
                "Congress.gov did not return a record for that bill or endpoint. The bill may be invalid or the requested text/summary may not yet be published.",
                status_code=404,
                url=response.url,
            )
        if not response.ok:
            detail = response.text[:400].strip()
            raise CongressAPIError(
                f"Congress.gov returned HTTP {response.status_code}. {detail}",
                status_code=response.status_code,
                url=response.url,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise CongressAPIError("Congress.gov returned a response that was not valid JSON.", url=response.url) from exc
        if not isinstance(payload, dict):
            raise CongressAPIError("Congress.gov returned an unexpected response structure.", url=response.url)
        return payload

    def get_current_congress(self) -> int:
        payload = self._get("/congress/current")
        current = payload.get("congress")
        if isinstance(current, dict):
            number = current.get("number") or current.get("congress")
            if number:
                return int(number)
        congresses = _as_list(payload.get("congresses"), "item", "congresses")
        for item in congresses:
            if isinstance(item, dict):
                number = item.get("number") or item.get("congress")
                if number:
                    return int(number)
        return current_congress_fallback()

    def list_bills(
        self,
        congress: int,
        bill_type: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        bill_type = bill_type.lower()
        if bill_type not in BILL_TYPE_DISPLAY:
            raise ValueError(f"Unsupported bill type: {bill_type}")
        payload = self._get(
            f"/bill/{int(congress)}/{bill_type}",
            params={"limit": min(max(int(limit), 1), 250), "offset": max(int(offset), 0)},
        )
        return extract_api_items(payload, "bills")

    def _get_collection(
        self,
        path: str,
        payload_key: str,
        *,
        allow_404: bool = True,
        max_records: int = 2_000,
    ) -> list[dict[str, Any]]:
        """Retrieve a Congress.gov collection defensively across API pages.

        Congress.gov caps list responses, so relying on a single default page
        can silently omit actions, cosponsors, amendments, titles, or text
        versions. This helper follows offset-based pages up to a conservative
        per-endpoint ceiling to keep the classroom app responsive and within
        API quotas.
        """

        limit = 250
        offset = 0
        collected: list[dict[str, Any]] = []
        while len(collected) < max_records:
            page_limit = min(limit, max_records - len(collected))
            payload = self._get(
                path,
                params={"limit": page_limit, "offset": offset},
                allow_404=allow_404,
            )
            page = extract_api_items(payload, payload_key)
            if not page:
                break
            collected.extend(page)

            pagination = payload.get("pagination") if isinstance(payload.get("pagination"), dict) else {}
            count_raw = pagination.get("count") if isinstance(pagination, dict) else None
            try:
                total_count = int(count_raw) if count_raw is not None else None
            except (TypeError, ValueError):
                total_count = None

            offset += len(page)
            if total_count is not None:
                if offset >= total_count:
                    break
            elif len(page) < page_limit:
                break

        return collected[:max_records]

    def get_bill(self, ref: BillRef) -> dict[str, Any]:
        payload = self._get(ref.api_path)
        bill = payload.get("bill")
        if isinstance(bill, dict):
            return bill
        raise CongressAPIError("Congress.gov returned no bill detail object for the requested citation.")

    def get_bill_bundle(self, ref: BillRef) -> dict[str, Any]:
        collection_endpoints = {
            "actions": (f"{ref.api_path}/actions", "actions"),
            "amendments": (f"{ref.api_path}/amendments", "amendments"),
            "committees": (f"{ref.api_path}/committees", "committees"),
            "cosponsors": (f"{ref.api_path}/cosponsors", "cosponsors"),
            "related_bills": (f"{ref.api_path}/relatedbills", "relatedBills"),
            "summaries": (f"{ref.api_path}/summaries", "summaries"),
            "text_versions": (f"{ref.api_path}/text", "textVersions"),
            "titles": (f"{ref.api_path}/titles", "titles"),
        }

        raw: dict[str, Any] = {}
        errors: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures: dict[Any, str] = {
                pool.submit(self._get, ref.api_path): "detail",
                pool.submit(self._get, f"{ref.api_path}/subjects", allow_404=True): "subjects",
            }
            for key, (path, payload_key) in collection_endpoints.items():
                futures[pool.submit(self._get_collection, path, payload_key)] = key
            for future in as_completed(futures):
                key = futures[future]
                try:
                    raw[key] = future.result()
                except CongressAPIError as exc:
                    if key == "detail":
                        raise
                    raw[key] = [] if key in collection_endpoints else {}
                    errors[key] = str(exc)

        detail = raw.get("detail", {}).get("bill") or {}
        if not isinstance(detail, dict) or not detail:
            raise CongressAPIError("Congress.gov returned an incomplete bill record.")

        subjects_payload = raw.get("subjects", {}).get("subjects") or {}
        if isinstance(subjects_payload, list):
            subjects_payload = {"legislativeSubjects": subjects_payload}

        return {
            "ref": ref.to_dict(),
            "detail": detail,
            "actions": raw.get("actions", []) if isinstance(raw.get("actions"), list) else [],
            "amendments": raw.get("amendments", []) if isinstance(raw.get("amendments"), list) else [],
            "committees": raw.get("committees", []) if isinstance(raw.get("committees"), list) else [],
            "cosponsors": raw.get("cosponsors", []) if isinstance(raw.get("cosponsors"), list) else [],
            "related_bills": raw.get("related_bills", []) if isinstance(raw.get("related_bills"), list) else [],
            "subjects": subjects_payload if isinstance(subjects_payload, dict) else {},
            "summaries": raw.get("summaries", []) if isinstance(raw.get("summaries"), list) else [],
            "text_versions": raw.get("text_versions", []) if isinstance(raw.get("text_versions"), list) else [],
            "titles": raw.get("titles", []) if isinstance(raw.get("titles"), list) else [],
            "errors": errors,
        }

    def get_current_members(self, state_code: str, district: int | None = None) -> list[dict[str, Any]]:
        state = re.sub(r"[^A-Za-z]", "", state_code or "").upper()
        if len(state) != 2:
            raise ValueError("State must be a two-letter postal abbreviation.")

        members: list[dict[str, Any]] = []
        state_payload = self._get(
            f"/member/{state}",
            params={"currentMember": "true", "limit": 250},
        )
        members.extend(extract_api_items(state_payload, "members"))

        if district is not None:
            district_payload = self._get(
                f"/member/{state}/{int(district)}",
                params={"currentMember": "true"},
            )
            members.extend(extract_api_items(district_payload, "members"))

        deduped: dict[str, dict[str, Any]] = {}
        for member in members:
            key = str(member.get("bioguideId") or member.get("name") or "")
            if key:
                deduped[key] = member
        return list(deduped.values())

    def get_member_detail(self, bioguide_id: str) -> dict[str, Any]:
        payload = self._get(f"/member/{bioguide_id}")
        member = payload.get("member")
        return member if isinstance(member, dict) else {}

    def download_text_version(self, version: dict[str, Any]) -> DownloadedBillText:
        fmt = preferred_text_format(version)
        if not fmt:
            raise CongressAPIError("The selected bill version does not include a downloadable text format.")
        return self.download_text_url(str(fmt["url"]), str(fmt.get("type") or "Unknown"))

    def download_text_url(self, url: str, source_format: str = "") -> DownloadedBillText:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        allowed_hosts = (
            "congress.gov",
            "www.congress.gov",
            "api.congress.gov",
            "govinfo.gov",
            "www.govinfo.gov",
            "gpo.gov",
            "www.gpo.gov",
        )
        if host not in allowed_hosts and not any(host.endswith(f".{item}") for item in allowed_hosts):
            raise CongressAPIError("For safety, the app downloads bill text only from official Congress.gov, GovInfo, or GPO hosts.")

        headers = {
            "User-Agent": self.session.headers.get("User-Agent", "Mozilla/5.0"),
            "Accept": "text/html,application/xml,text/xml,text/plain,application/pdf,*/*",
        }
        try:
            response = self.session.get(url, headers=headers, timeout=max(self.timeout, 45))
        except requests.RequestException as exc:
            raise CongressAPIError(f"The official bill text could not be downloaded: {exc}", url=url) from exc
        if not response.ok:
            raise CongressAPIError(
                f"The official bill text host returned HTTP {response.status_code}. Try the source link directly or select another version.",
                status_code=response.status_code,
                url=response.url,
            )

        content_type = (response.headers.get("content-type") or "").lower()
        lower_url = response.url.lower()
        if "pdf" in content_type or lower_url.endswith(".pdf"):
            try:
                reader = PdfReader(io.BytesIO(response.content))
                text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
            except Exception as exc:
                raise CongressAPIError("The PDF bill text downloaded, but its text could not be extracted.", url=response.url) from exc
        else:
            response.encoding = response.encoding or "utf-8"
            body = response.text
            if "xml" in content_type or lower_url.endswith(".xml"):
                # Use the standard library so Streamlit Cloud does not need the
                # optional lxml wheel merely to read Congress.gov XML.
                try:
                    root = ET.fromstring(body)
                    text = "\n".join(part.strip() for part in root.itertext() if part and part.strip())
                    soup = None
                except ET.ParseError:
                    soup = BeautifulSoup(body, "html.parser")
            elif "html" in content_type or "formatted text" in source_format.lower():
                soup = BeautifulSoup(body, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
            else:
                soup = None
            if soup is not None:
                text = soup.get_text("\n", strip=True)
            elif not ("xml" in content_type or lower_url.endswith(".xml")):
                text = body

        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) < 80:
            raise CongressAPIError("The selected official text version contained too little extractable text.", url=response.url)
        return DownloadedBillText(
            text=text,
            source_url=response.url,
            source_format=source_format or content_type or "Official text",
            content_type=content_type,
        )
