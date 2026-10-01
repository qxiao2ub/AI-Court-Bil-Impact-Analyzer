from __future__ import annotations

import difflib
import html
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STATUS_ORDER = {
    "introduced": 1,
    "in committee": 2,
    "reported by committee": 3,
    "passed one chamber": 4,
    "passed both chambers": 5,
    "presented to president": 6,
    "enacted": 7,
    "vetoed": 7,
    "failed or rejected": 7,
    "agreed to": 7,
    "unknown": 0,
}

STATUS_HELP = {
    "introduced": "Formally introduced and assigned a bill number.",
    "in committee": "Referred to one or more committees for review.",
    "reported by committee": "A committee reported the measure to its chamber.",
    "passed one chamber": "Passed or agreed to in either the House or the Senate, but not both based on the retrieved actions.",
    "passed both chambers": "Both chambers appear to have approved the measure or identical text.",
    "presented to president": "Presented to the President for approval or veto.",
    "enacted": "Signed or otherwise became law according to the official action record.",
    "vetoed": "The official action record includes a presidential veto or pocket veto.",
    "failed or rejected": "The official action record includes a failed, rejected, or tabled disposition.",
    "agreed to": "A resolution was agreed to by the relevant chamber or chambers.",
    "unknown": "The available source does not provide enough procedural history to identify a stage confidently.",
}

LEGAL_GLOSSARY: dict[str, str] = {
    "appropriation": "A law provision that provides authority for federal agencies to spend money for specified purposes.",
    "authorization": "A provision that creates or continues a program or sets policy; it does not always provide the money needed to operate it.",
    "concurrent resolution": "A measure adopted by both chambers that generally does not go to the President and usually does not have the force of law.",
    "continuing resolution": "Temporary appropriations legislation that keeps federal programs funded when regular appropriations are unfinished.",
    "covered entity": "A person, organization, or other party that the bill defines as subject to a requirement.",
    "effective date": "The date on which a provision begins to operate, which may differ from the date of enactment.",
    "fiscal year": "The federal government's accounting year, running from October 1 through September 30.",
    "joint resolution": "A legislative measure that generally follows the same process as a bill and, if approved by both chambers and the President, can become law; constitutional amendments are an exception.",
    "mandatory spending": "Spending controlled by eligibility rules or other statutory criteria rather than annual appropriations.",
    "notwithstanding": "A phrase meaning that the stated rule applies despite another law or provision that might otherwise conflict.",
    "obligation": "A legally binding commitment by the federal government that will result in immediate or future spending.",
    "outlay": "An actual payment made by the federal government.",
    "preemption": "A rule under which federal law displaces or limits state or local law in a specified area.",
    "rescission": "Cancellation of previously enacted budget authority.",
    "rider": "A provision added to a measure that may address a subject different from the measure's main topic; whether a provision is a rider can be contested and requires human review.",
    "rulemaking": "The process by which an agency develops regulations to implement a statute.",
    "severability": "A clause stating that if one provision is invalidated, the remaining provisions can continue to operate.",
    "sunset clause": "A provision that automatically ends a program, authority, or rule on a specified date unless Congress extends it.",
    "cloture": "A Senate procedure for limiting debate and moving toward a vote.",
    "quorum": "The minimum number of members required for a chamber or committee to conduct specified business.",
    "discretionary spending": "Spending provided and controlled through annual appropriations acts.",
    "score": "In a budget context, an estimate of how legislation would affect federal spending, revenues, or deficits over a stated period.",
    "sense of congress": "A nonbinding statement expressing Congress's position or recommendation.",
    "shall": "In statutory drafting, usually imposes a mandatory duty.",
    "may": "In statutory drafting, usually grants permission or discretion rather than imposing a duty.",
}

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "agriculture": ("agriculture", "farm", "crop", "livestock", "usda", "rural development"),
    "budget & taxation": ("tax", "revenue", "appropriation", "budget", "treasury", "irs", "deduction", "credit"),
    "civil rights": ("civil rights", "discrimination", "equal protection", "voting rights", "accessibility"),
    "defense & veterans": ("defense", "armed forces", "military", "veteran", "department of defense", "national guard"),
    "education": ("education", "school", "student", "teacher", "college", "university", "pell grant"),
    "energy & environment": ("environment", "climate", "energy", "emission", "epa", "conservation", "wildlife"),
    "health": ("health", "medicare", "medicaid", "hospital", "drug", "public health", "patient"),
    "housing": ("housing", "mortgage", "tenant", "landlord", "homeless", "hud"),
    "immigration": ("immigration", "immigrant", "visa", "citizenship", "border", "alien"),
    "justice & public safety": ("criminal", "court", "justice", "law enforcement", "firearm", "prison", "sentencing"),
    "labor & employment": ("employee", "employer", "labor", "wage", "workforce", "occupation", "worker"),
    "technology & privacy": ("technology", "cyber", "privacy", "artificial intelligence", "data", "internet", "telecommunications"),
    "transportation": ("transportation", "highway", "transit", "vehicle", "aviation", "rail", "infrastructure"),
    "government operations": ("federal agency", "inspector general", "government accountability", "procurement", "administrative"),
    "foreign affairs": ("foreign", "international", "sanction", "diplomatic", "aid", "treaty"),
}

PLAIN_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\bnotwithstanding any other provision of law\b", "even if another law would otherwise conflict"),
    (r"\bnotwithstanding\b", "despite"),
    (r"\bpursuant to\b", "under"),
    (r"\bprior to\b", "before"),
    (r"\bsubsequent to\b", "after"),
    (r"\bin accordance with\b", "under"),
    (r"\bfor the purposes of\b", "for"),
    (r"\bshall be deemed to\b", "will be treated as"),
    (r"\bis hereby amended\b", "is changed"),
    (r"\bpromulgate regulations\b", "issue regulations"),
    (r"\bterminate\b", "end"),
    (r"\bcommence\b", "begin"),
    (r"\butilize\b", "use"),
    (r"\bthereof\b", "of it"),
    (r"\btherein\b", "in it"),
    (r"\bhereinafter\b", "later in this text"),
)

SECTION_PATTERN = re.compile(
    r"(?im)^(?P<header>\s*(?:SEC(?:TION)?\.?\s+|§\s*)"
    r"(?P<number>\d+[A-Z0-9]*(?:[-–]\d+[A-Z0-9]*)?)\.?\s*"
    r"(?P<title>[^\n]{0,240}))\s*$"
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n{2,}")
DOLLAR_PATTERN = re.compile(
    r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand))?",
    flags=re.IGNORECASE,
)
DATE_PATTERNS = (
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b", re.I),
    re.compile(r"\b\d{1,3}\s+days?\s+after\s+(?:the date of )?enactment\b", re.I),
    re.compile(r"\b(?:on|beginning|effective)\s+(?:on\s+)?the date of enactment\b", re.I),
    re.compile(r"\b(?:fiscal year|FY)\s+20\d{2}\b", re.I),
)


@dataclass(slots=True)
class SectionAnalysis:
    number: str
    title: str
    summary: str
    operative_points: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    key_terms: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    dollar_amounts: list[str] = field(default_factory=list)
    confidence: str = "medium"
    word_count: int = 0
    raw_text: str = ""


@dataclass(slots=True)
class StatusResult:
    label: str
    confidence: str
    explanation: str
    evidence: list[str] = field(default_factory=list)
    last_action_date: str = ""


@dataclass(slots=True)
class BillAnalysis:
    title: str
    citation: str
    plain_summary: str
    sections: list[SectionAnalysis]
    status: StatusResult
    topics: list[str]
    stated_purpose: str
    operative_effect: str
    scope_mismatch_flags: list[dict[str, Any]]
    glossary: dict[str, str]
    fiscal: dict[str, Any]
    timeline: list[dict[str, str]]
    confidence: dict[str, Any]
    metadata: dict[str, Any]
    source_links: dict[str, str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = html.unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html(value: str) -> str:
    return _clean_text(re.sub(r"<[^>]+>", " ", value or ""))


def safe_get(mapping: Mapping[str, Any] | None, *path: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return current if current is not None else default


def as_list(value: Any, *keys: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        for key in keys:
            item = value.get(key)
            if isinstance(item, list):
                return item
        for key in ("item", "items"):
            item = value.get(key)
            if isinstance(item, list):
                return item
        return [value]
    return [value]


def sentence_split(text: str, *, min_chars: int = 25) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    sentences: list[str] = []
    for item in SENTENCE_SPLIT.split(cleaned):
        item = re.sub(r"\s+", " ", item).strip(" \n\t")
        if len(item) >= min_chars:
            sentences.append(item)
    if not sentences and cleaned:
        sentences = [cleaned]
    return sentences


def plainify(text: str) -> str:
    result = re.sub(r"\s+", " ", strip_html(text)).strip()
    for pattern, replacement in PLAIN_REPLACEMENTS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = re.sub(r"\bshall\b", "must", result, flags=re.IGNORECASE)
    result = re.sub(r"\bmay\b", "may", result, flags=re.IGNORECASE)
    result = re.sub(r"\bSecretary\b", "the Secretary", result)
    if result and result[-1] not in ".!?":
        result += "."
    return result


def _tfidf_sentence_scores(sentences: Sequence[str], query: str = "") -> list[float]:
    if not sentences:
        return []
    if len(sentences) == 1:
        return [1.0]
    documents = list(sentences)
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=8000,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(documents + ([query] if query.strip() else []))
        sent_matrix = matrix[: len(documents)]
        centrality = cosine_similarity(sent_matrix, sent_matrix).mean(axis=1)
        position = [1.0 / math.sqrt(index + 1) for index in range(len(documents))]
        scores = [0.82 * float(centrality[i]) + 0.18 * position[i] for i in range(len(documents))]
        if query.strip():
            query_vector = matrix[-1]
            relevance = cosine_similarity(sent_matrix, query_vector).ravel()
            scores = [0.7 * score + 0.3 * float(relevance[i]) for i, score in enumerate(scores)]
        return scores
    except ValueError:
        return [1.0 / math.sqrt(index + 1) for index in range(len(documents))]


def extractive_summary(text: str, *, max_sentences: int = 4, max_chars: int = 1300, query: str = "") -> str:
    sentences = sentence_split(text)
    if not sentences:
        return "No readable bill text was available for summarization."
    scores = _tfidf_sentence_scores(sentences, query=query)
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    selected: list[int] = []
    total = 0
    for index in ranked:
        candidate = sentences[index]
        if total + len(candidate) > max_chars and selected:
            continue
        selected.append(index)
        total += len(candidate)
        if len(selected) >= max_sentences or total >= max_chars:
            break
    ordered = [plainify(sentences[index]) for index in sorted(selected)]
    return " ".join(ordered)


def split_sections(text: str) -> list[dict[str, str]]:
    cleaned = _clean_text(text)
    matches = list(SECTION_PATTERN.finditer(cleaned))
    sections: list[dict[str, str]] = []
    if matches:
        preamble = cleaned[: matches[0].start()].strip()
        if len(preamble) >= 80:
            sections.append({"number": "PREAMBLE", "title": "Preamble and findings", "text": preamble})
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
            body = cleaned[start:end].strip()
            title = re.sub(r"^[.—\-–:\s]+", "", match.group("title") or "").strip()
            title = re.sub(r"[.—\-–:\s]+$", "", title).strip()
            sections.append(
                {
                    "number": match.group("number").strip(),
                    "title": title or f"Section {match.group('number').strip()}",
                    "text": body,
                }
            )
        return sections

    # Fallback for texts that lost section headers during PDF extraction.
    paragraphs = [item.strip() for item in re.split(r"\n{2,}", cleaned) if len(item.strip()) >= 40]
    if not paragraphs:
        return [{"number": "DOCUMENT", "title": "Full text", "text": cleaned}]
    chunks: list[str] = []
    current: list[str] = []
    word_count = 0
    for paragraph in paragraphs:
        count = len(paragraph.split())
        if current and word_count + count > 900:
            chunks.append("\n\n".join(current))
            current = []
            word_count = 0
        current.append(paragraph)
        word_count += count
    if current:
        chunks.append("\n\n".join(current))
    return [
        {"number": f"CHUNK-{index:02d}", "title": f"Document segment {index}", "text": chunk}
        for index, chunk in enumerate(chunks, start=1)
    ]


def detect_topics(text: str, *, max_topics: int = 5) -> list[str]:
    lower = _clean_text(text).lower()
    scores: Counter[str] = Counter()
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            count = lower.count(keyword)
            if count:
                scores[topic] += count * (2 if " " in keyword else 1)
    return [topic for topic, _ in scores.most_common(max_topics)]


def find_key_terms(text: str) -> list[str]:
    lower = _clean_text(text).lower()
    found = [term for term in LEGAL_GLOSSARY if re.search(rf"\b{re.escape(term)}s?\b", lower)]
    return sorted(found)


def _extract_dates(text: str) -> list[str]:
    found: list[str] = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            if value not in found:
                found.append(value)
    return found[:12]


def _extract_dollars(text: str) -> list[str]:
    found: list[str] = []
    for match in DOLLAR_PATTERN.finditer(text):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        if value not in found:
            found.append(value)
    return found[:20]


def _operative_points(text: str, max_points: int = 5) -> list[str]:
    signals = (
        "shall", "must", "may", "prohibit", "require", "establish", "authorize", "appropriate",
        "amend", "repeal", "direct", "submit", "report", "effective", "terminate", "eligible",
    )
    candidates = sentence_split(text)
    scored: list[tuple[int, str]] = []
    for index, sentence in enumerate(candidates):
        lower = sentence.lower()
        score = sum(1 for signal in signals if re.search(rf"\b{re.escape(signal)}\w*\b", lower))
        if score:
            scored.append((score * 1000 - index, sentence))
    scored.sort(reverse=True)
    points: list[str] = []
    for _, sentence in scored:
        simplified = plainify(sentence)
        if simplified not in points:
            points.append(simplified)
        if len(points) >= max_points:
            break
    return points


def analyze_sections(text: str, title: str = "") -> list[SectionAnalysis]:
    results: list[SectionAnalysis] = []
    for raw in split_sections(text):
        body = raw["text"]
        words = len(body.split())
        summary = extractive_summary(body, max_sentences=2, max_chars=700, query=title)
        confidence = "high" if words >= 70 else "medium" if words >= 25 else "low"
        results.append(
            SectionAnalysis(
                number=raw["number"],
                title=plainify(raw["title"]).rstrip("."),
                summary=summary,
                operative_points=_operative_points(body),
                topics=detect_topics(f"{raw['title']} {body}", max_topics=3),
                key_terms=find_key_terms(body),
                dates=_extract_dates(body),
                dollar_amounts=_extract_dollars(body),
                confidence=confidence,
                word_count=words,
                raw_text=body,
            )
        )
    return results


def _action_text(action: Mapping[str, Any]) -> str:
    pieces = [
        action.get("text"),
        action.get("actionCode"),
        action.get("type"),
        safe_get(action, "sourceSystem", "name"),
    ]
    return " ".join(str(piece) for piece in pieces if piece).strip()


def _action_date(action: Mapping[str, Any]) -> str:
    return str(action.get("actionDate") or action.get("date") or action.get("updateDate") or "")


def detect_status(bundle: Mapping[str, Any] | None, text: str = "") -> StatusResult:
    bundle = bundle or {}
    detail = bundle.get("detail") if isinstance(bundle.get("detail"), Mapping) else {}
    actions = [item for item in as_list(bundle.get("actions")) if isinstance(item, Mapping)]
    latest = detail.get("latestAction") if isinstance(detail, Mapping) and isinstance(detail.get("latestAction"), Mapping) else {}
    if latest:
        actions.append(latest)
    actions = sorted(actions, key=_action_date)
    joined = "\n".join(_action_text(item) for item in actions).lower()
    recent_joined = "\n".join(_action_text(item) for item in actions[-5:]).lower()

    laws = as_list(detail.get("laws"), "item", "laws") if isinstance(detail, Mapping) else []
    is_resolution = str(safe_get(bundle, "ref", "bill_type", default="")).lower() in {"hres", "sres", "hconres", "sconres"}

    house_pass = bool(re.search(r"passed/agreed to in house|passed house|on passage passed|house agreed to", joined))
    senate_pass = bool(re.search(r"passed/agreed to in senate|passed senate|senate agreed to", joined))
    presented = bool(re.search(r"presented to president|cleared for white house|sent to the president", joined))
    enacted = bool(laws) or bool(re.search(r"became (?:public|private) law|signed by president|became law", joined))
    vetoed = bool(re.search(r"pocket veto|vetoed by president|president vetoed|veto message", joined))
    # Keep failure detection narrow and recent. Bill histories often contain
    # rejected amendments or failed procedural motions that do not mean the
    # underlying measure itself failed.
    failed = bool(
        re.search(
            r"failed of passage|measure (?:was )?rejected|bill (?:was )?rejected|"
            r"joint resolution (?:was )?rejected|not agreed to in (?:the )?(?:house|senate)|"
            r"motion to (?:suspend the rules and )?pass (?:the measure|the bill|h\.?r\.?\s*\d+|s\.?\s*\d+).*?failed",
            recent_joined,
        )
    )
    reported = bool(re.search(r"reported by|ordered to be reported|committee report", joined))
    committee = bool(re.search(r"referred to|committee consideration|committee on", joined))
    agreed = bool(re.search(r"agreed to", joined))

    # Final/latest legal outcomes take precedence over earlier procedural
    # events that remain in the action history. For example, a bill can be
    # vetoed and later enacted after an override, or fail an interim motion
    # before eventually passing.
    if enacted:
        label = "enacted"
    elif vetoed:
        label = "vetoed"
    elif presented:
        label = "presented to president"
    elif house_pass and senate_pass:
        label = "passed both chambers"
    elif house_pass or senate_pass:
        label = "passed one chamber"
    elif is_resolution and agreed:
        label = "agreed to"
    elif failed:
        label = "failed or rejected"
    elif reported:
        label = "reported by committee"
    elif committee:
        label = "in committee"
    elif actions or safe_get(detail, "introducedDate", default=""):
        label = "introduced"
    else:
        # Manual-text fallback. This is deliberately conservative.
        lower = text.lower()
        if "public law" in lower and re.search(r"public law\s+\d+[-–]\d+", lower):
            label = "enacted"
        elif "introduced in" in lower or re.search(r"\b(?:h\.?r\.?|s\.?)\s*\d+", lower):
            label = "introduced"
        else:
            label = "unknown"

    evidence: list[str] = []
    for action in reversed(actions):
        action_text = _action_text(action)
        if action_text and action_text not in evidence:
            date = _action_date(action)
            evidence.append(f"{date}: {action_text}" if date else action_text)
        if len(evidence) >= 4:
            break
    confidence = "high" if actions and label != "unknown" else "medium" if label != "unknown" else "low"
    last_date = _action_date(actions[-1]) if actions else ""
    return StatusResult(
        label=label,
        confidence=confidence,
        explanation=STATUS_HELP[label],
        evidence=evidence,
        last_action_date=last_date,
    )


def _latest_summary(bundle: Mapping[str, Any]) -> str:
    summaries = [item for item in as_list(bundle.get("summaries")) if isinstance(item, Mapping)]
    if not summaries:
        return ""
    newest = max(summaries, key=lambda item: str(item.get("updateDate") or item.get("actionDate") or ""))
    return strip_html(str(newest.get("text") or newest.get("summary") or ""))


def _purpose_text(bundle: Mapping[str, Any], title: str, text: str) -> str:
    official = _latest_summary(bundle)
    if official:
        return extractive_summary(official, max_sentences=3, max_chars=900, query=title)
    first_sentences = sentence_split(text)[:8]
    findings = " ".join(first_sentences)
    return extractive_summary(findings, max_sentences=2, max_chars=700, query=title)


def _scope_mismatch_flags(sections: Sequence[SectionAnalysis], title: str, official_summary: str) -> list[dict[str, Any]]:
    primary_topics = detect_topics(f"{title} {official_summary} " + " ".join(s.summary for s in sections[:3]), max_topics=3)
    primary_set = set(primary_topics)
    flags: list[dict[str, Any]] = []
    if not primary_set or len(sections) < 2:
        return flags
    for section in sections:
        section_set = set(section.topics)
        if not section_set or section.number == "PREAMBLE":
            continue
        overlap = primary_set & section_set
        # Administrative and definitions sections are intentionally excluded.
        admin_text = f"{section.title} {section.summary}".lower()
        if any(term in admin_text for term in ("definition", "effective date", "severability", "technical amendment", "table of contents")):
            continue
        if not overlap and section.word_count >= 90:
            flags.append(
                {
                    "section": section.number,
                    "title": section.title,
                    "primary_topics": primary_topics,
                    "section_topics": section.topics,
                    "reason": "This section's detected subject terms do not overlap the bill's main detected topics.",
                    "confidence": "low-to-medium",
                    "review_note": "This is a vocabulary-based scope-mismatch signal, not a legal conclusion that the section is a rider.",
                }
            )
    return flags[:12]


def _amendment_scope_flags(
    bundle: Mapping[str, Any],
    *,
    primary_topics: Sequence[str],
) -> list[dict[str, Any]]:
    primary_set = set(primary_topics)
    if not primary_set:
        return []
    flags: list[dict[str, Any]] = []
    amendments = [item for item in as_list(bundle.get("amendments")) if isinstance(item, Mapping)]
    for amendment in amendments:
        purpose = _clean_text(
            amendment.get("purpose")
            or amendment.get("description")
            or amendment.get("title")
            or safe_get(amendment, "latestAction", "text", default="")
        )
        if not purpose:
            continue
        amendment_topics = detect_topics(purpose, max_topics=3)
        if amendment_topics and not (primary_set & set(amendment_topics)):
            number = str(amendment.get("number") or amendment.get("type") or "amendment")
            flags.append(
                {
                    "section": f"Amendment {number}",
                    "title": purpose[:180],
                    "primary_topics": list(primary_topics),
                    "section_topics": amendment_topics,
                    "reason": "The amendment purpose metadata uses subject terms outside the bill's main detected topics.",
                    "confidence": "low-to-medium",
                    "review_note": "This metadata-only signal is not a conclusion that the amendment is unrelated or improper; read the amendment text and official action record.",
                }
            )
    return flags[:12]


def _fiscal_analysis(bundle: Mapping[str, Any], text: str, sections: Sequence[SectionAnalysis]) -> dict[str, Any]:
    detail = bundle.get("detail") if isinstance(bundle.get("detail"), Mapping) else {}
    cbo = [item for item in as_list(detail.get("cboCostEstimates"), "item", "cboCostEstimates") if isinstance(item, Mapping)]
    amounts = _extract_dollars(text)
    spending_terms = []
    lower = text.lower()
    for term in (
        "appropriation", "authorize to be appropriated", "outlay", "budget authority", "revenue",
        "tax credit", "tax deduction", "mandatory spending", "discretionary spending", "rescission",
    ):
        if term in lower:
            spending_terms.append(term)
    section_hits = [
        {"section": section.number, "title": section.title, "amounts": section.dollar_amounts}
        for section in sections
        if section.dollar_amounts
    ]
    return {
        "has_cbo_estimate": bool(cbo),
        "cbo_estimates": cbo,
        "spending_or_revenue_language": bool(spending_terms or amounts),
        "detected_terms": spending_terms,
        "detected_amounts": amounts,
        "section_amounts": section_hits,
        "caution": (
            "Detected dollar figures are textual references, not a calculated budget score. Use the linked CBO estimate or official source when available."
        ),
    }


def _timeline(bundle: Mapping[str, Any], sections: Sequence[SectionAnalysis]) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    detail = bundle.get("detail") if isinstance(bundle.get("detail"), Mapping) else {}
    introduced = str(detail.get("introducedDate") or "")
    if introduced:
        events.append({"date": introduced, "event": "Introduced", "source": "Congress.gov metadata"})
    actions = [item for item in as_list(bundle.get("actions")) if isinstance(item, Mapping)]
    for action in sorted(actions, key=_action_date):
        text = _action_text(action)
        if text:
            events.append({"date": _action_date(action), "event": text, "source": "Congress.gov action"})
    effective_candidates: list[str] = []
    for section in sections:
        for date_value in section.dates:
            if date_value not in effective_candidates:
                effective_candidates.append(date_value)
    for value in effective_candidates[:8]:
        events.append({"date": value, "event": "Potential effective-date or implementation reference in bill text", "source": "AI text extraction"})
    return events


def _source_links(bundle: Mapping[str, Any], text_source_url: str = "") -> dict[str, str]:
    ref = bundle.get("ref") if isinstance(bundle.get("ref"), Mapping) else {}
    official = str(ref.get("official_url") or "")
    return {
        "official_bill_page": official,
        # The bill landing page consistently exposes the official CRS summary.
        # Avoid guessing at an unstable summary-version route.
        "official_summary": official,
        "official_text": text_source_url or official,
    }


def _analysis_confidence(text: str, source_kind: str, sections: Sequence[SectionAnalysis], bundle: Mapping[str, Any]) -> dict[str, Any]:
    word_count = len(text.split())
    has_official_metadata = bool(bundle.get("detail"))
    score = 0
    reasons: list[str] = []
    if word_count >= 1500:
        score += 3
        reasons.append("Substantial bill text was available.")
    elif word_count >= 350:
        score += 2
        reasons.append("A moderate amount of bill text was available.")
    else:
        score += 1
        reasons.append("The available text is short or incomplete.")
    if source_kind == "official Congress.gov text":
        score += 3
        reasons.append("Text was retrieved from an official live source.")
    elif source_kind == "uploaded PDF":
        score += 2
        reasons.append("Text was extracted from an uploaded PDF; extraction errors remain possible.")
    else:
        score += 1
        reasons.append("Text was pasted or supplied manually and was not independently verified.")
    if has_official_metadata:
        score += 2
        reasons.append("Official metadata and action history were available.")
    if len(sections) >= 2:
        score += 1
        reasons.append("The analyzer identified multiple bill sections.")
    label = "high" if score >= 8 else "medium" if score >= 5 else "low"
    return {
        "label": label,
        "score": score,
        "maximum": 9,
        "reasons": reasons,
        "uncertainty_note": "Confidence reflects source completeness and text extraction, not whether the bill is good, bad, constitutional, or likely to pass.",
    }


def analyze_bill(
    text: str,
    *,
    bundle: Mapping[str, Any] | None = None,
    source_kind: str = "pasted text",
    text_source_url: str = "",
    citation: str = "Manual text",
    title: str = "",
) -> BillAnalysis:
    cleaned = _clean_text(text)
    if len(cleaned) < 80:
        raise ValueError("The bill text is too short to analyze. Provide at least 80 readable characters.")
    bundle = bundle or {}
    detail = bundle.get("detail") if isinstance(bundle.get("detail"), Mapping) else {}
    resolved_title = _clean_text(title or detail.get("title") or detail.get("shortTitle") or "Untitled legislative text")
    ref = bundle.get("ref") if isinstance(bundle.get("ref"), Mapping) else {}
    resolved_citation = _clean_text(citation or ref.get("citation") or "Manual text")
    sections = analyze_sections(cleaned, resolved_title)
    official_summary = _latest_summary(bundle)
    plain_summary = extractive_summary(cleaned, max_sentences=5, max_chars=1600, query=resolved_title)
    status = detect_status(bundle, cleaned)
    topics = detect_topics(f"{resolved_title} {official_summary} {cleaned}")
    stated_purpose = _purpose_text(bundle, resolved_title, cleaned)
    operative_effect = " ".join(point for section in sections[:12] for point in section.operative_points[:1])
    if not operative_effect:
        operative_effect = plain_summary
    glossary_terms = sorted({term for section in sections for term in section.key_terms})
    glossary = {term: LEGAL_GLOSSARY[term] for term in glossary_terms}
    fiscal = _fiscal_analysis(bundle, cleaned, sections)
    flags = _scope_mismatch_flags(sections, resolved_title, official_summary)
    flags.extend(_amendment_scope_flags(bundle, primary_topics=detect_topics(f"{resolved_title} {official_summary}", max_topics=3)))
    timeline = _timeline(bundle, sections)
    metadata = {
        "introduced_date": str(detail.get("introducedDate") or ""),
        "update_date": str(detail.get("updateDate") or ""),
        "origin_chamber": str(detail.get("originChamber") or ""),
        "policy_area": str(safe_get(detail, "policyArea", "name", default="")),
        "constitutional_authority": str(detail.get("constitutionalAuthorityStatementText") or ""),
        "laws": as_list(detail.get("laws"), "item", "laws"),
        "sponsors": as_list(detail.get("sponsors"), "item", "sponsors"),
        "subjects": bundle.get("subjects") if isinstance(bundle.get("subjects"), Mapping) else {},
        "titles": as_list(bundle.get("titles")),
        "source_kind": source_kind,
        "word_count": len(cleaned.split()),
        "section_count": len(sections),
    }
    warnings = [
        "This educational prototype summarizes statutory text; it does not provide legal advice or predict whether a bill will pass.",
        "Scope-mismatch flags are heuristic review prompts, not findings that a provision is an improper rider.",
    ]
    if source_kind != "official Congress.gov text":
        warnings.append("The supplied text was not automatically verified against the current official Congress.gov version.")
    return BillAnalysis(
        title=resolved_title,
        citation=resolved_citation,
        plain_summary=plain_summary,
        sections=sections,
        status=status,
        topics=topics,
        stated_purpose=stated_purpose,
        operative_effect=operative_effect,
        scope_mismatch_flags=flags,
        glossary=glossary,
        fiscal=fiscal,
        timeline=timeline,
        confidence=_analysis_confidence(cleaned, source_kind, sections, bundle),
        metadata=metadata,
        source_links=_source_links(bundle, text_source_url),
        warnings=warnings,
    )


def personalize_impact(analysis: BillAnalysis, profile: Mapping[str, Any]) -> dict[str, Any]:
    age = int(profile.get("age") or 0)
    state = str(profile.get("state") or "").upper()
    income = str(profile.get("income_bracket") or "Not provided")
    occupation = str(profile.get("occupation") or "").strip()
    industry = str(profile.get("industry") or "").strip()
    student = bool(profile.get("student"))
    veteran = bool(profile.get("veteran"))
    business_owner = bool(profile.get("business_owner"))
    caregiver = bool(profile.get("caregiver"))

    corpus = " ".join(
        [analysis.title, analysis.plain_summary, analysis.operative_effect]
        + [section.summary for section in analysis.sections]
    ).lower()
    signals: list[dict[str, str]] = []

    def add(category: str, relevance: str, reason: str, check: str) -> None:
        signals.append({"category": category, "relevance": relevance, "reason": reason, "what_to_verify": check})

    if age:
        if age >= 65 and any(term in corpus for term in ("medicare", "older adult", "senior", "retirement")):
            add("Age", "potentially direct", "The text includes programs or terms commonly tied to older adults.", "Eligibility definitions and effective dates")
        elif age < 26 and any(term in corpus for term in ("student", "youth", "young adult", "dependent")):
            add("Age", "potentially direct", "The text includes youth, student, or dependent-related language.", "Age thresholds and eligibility rules")
        else:
            add("Age", "unclear", "No clear age-specific trigger was detected for the provided age.", "Definitions of eligible or covered individuals")

    if state:
        state_pattern = rf"\b{re.escape(state)}\b"
        state_hit = bool(re.search(state_pattern, corpus, flags=re.I))
        add(
            "State",
            "potentially direct" if state_hit else "generally federal / implementation-dependent",
            f"The supplied state is {state}. " + ("The text contains the state code/name signal." if state_hit else "No state-specific signal was detected in the analyzed text."),
            "Federal preemption, state grants, waivers, and agency implementation",
        )

    if income != "Not provided":
        if any(term in corpus for term in ("tax", "income", "credit", "deduction", "means-tested", "poverty")):
            add("Income", "potentially direct", f"The text contains tax or income-related terms; profile bracket: {income}.", "Thresholds, phase-ins, phase-outs, filing status, and taxable years")
        else:
            add("Income", "unclear", "No obvious income-linked provision was detected.", "Definitions and benefit/tax eligibility tables")

    work_terms = " ".join([occupation, industry]).lower().strip()
    if work_terms:
        overlap = [token for token in re.findall(r"[a-z]{4,}", work_terms) if token in corpus]
        if overlap or "labor & employment" in analysis.topics:
            add("Work", "possibly relevant", f"Work-profile terms may overlap the bill's employment or industry language: {', '.join(overlap[:5]) or 'general labor provisions'}.", "Covered employers, worker classifications, industry definitions, and compliance dates")
        else:
            add("Work", "unclear", "No strong occupation or industry overlap was detected.", "Sector definitions, grant eligibility, and regulatory scope")

    if student and ("education" in analysis.topics or "student" in corpus):
        add("Student status", "potentially direct", "Education or student-related language was detected.", "Institution and student eligibility requirements")
    if veteran and ("defense & veterans" in analysis.topics or "veteran" in corpus):
        add("Veteran status", "potentially direct", "Veterans-related language was detected.", "VA program eligibility and implementation rules")
    if business_owner and any(term in corpus for term in ("small business", "employer", "business", "entity", "tax")):
        add("Business ownership", "possibly relevant", "The text may impose employer/entity duties or create business benefits.", "Employee-count thresholds, entity definitions, credits, penalties, and reporting")
    if caregiver and any(term in corpus for term in ("caregiver", "family leave", "dependent", "home care")):
        add("Caregiving", "possibly relevant", "Caregiving or family-support language was detected.", "Relationship definitions, leave rules, and benefit eligibility")

    if not signals:
        signals.append(
            {
                "category": "Profile match",
                "relevance": "insufficient profile information",
                "reason": "Add optional profile fields to generate a more targeted relevance checklist.",
                "what_to_verify": "Eligibility, definitions, effective dates, and implementing regulations",
            }
        )

    direct = sum(1 for signal in signals if "direct" in signal["relevance"])
    possible = sum(1 for signal in signals if "possible" in signal["relevance"])
    overall = "higher potential relevance" if direct >= 2 else "some potential relevance" if direct + possible >= 1 else "no clear direct match detected"
    return {
        "overall": overall,
        "signals": signals,
        "profile_snapshot": {
            "age": age or None,
            "state": state or None,
            "income_bracket": income,
            "occupation": occupation or None,
            "industry": industry or None,
            "student": student,
            "veteran": veteran,
            "business_owner": business_owner,
            "caregiver": caregiver,
        },
        "disclaimer": "Personalization identifies possible relevance from text and profile fields. It does not determine legal eligibility, financial impact, or how an agency will implement the bill.",
    }


def _flatten_analysis_text(analysis: BillAnalysis) -> str:
    return " ".join([analysis.title, analysis.plain_summary, analysis.operative_effect] + [s.summary for s in analysis.sections])


def compare_analyses(left: BillAnalysis, right: BillAnalysis) -> dict[str, Any]:
    corpus = [_flatten_analysis_text(left), _flatten_analysis_text(right)]
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000).fit_transform(corpus)
        similarity = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
    except ValueError:
        similarity = 0.0
    left_topics, right_topics = set(left.topics), set(right.topics)
    shared = sorted(left_topics & right_topics)
    return {
        "similarity_pct": round(similarity * 100, 1),
        "shared_topics": shared,
        "left_only_topics": sorted(left_topics - right_topics),
        "right_only_topics": sorted(right_topics - left_topics),
        "status": {"left": left.status.label, "right": right.status.label},
        "sections": {"left": len(left.sections), "right": len(right.sections)},
        "fiscal": {
            "left_has_cbo": bool(left.fiscal.get("has_cbo_estimate")),
            "right_has_cbo": bool(right.fiscal.get("has_cbo_estimate")),
            "left_amounts": left.fiscal.get("detected_amounts", []),
            "right_amounts": right.fiscal.get("detected_amounts", []),
        },
        "plain_summary": {"left": left.plain_summary, "right": right.plain_summary},
        "note": "Similarity is a TF-IDF text-overlap measure. It does not determine whether the bills are substitutes, legally equivalent, or politically aligned.",
    }


def version_diff(old_text: str, new_text: str, *, context_lines: int = 3, max_lines: int = 1500) -> dict[str, Any]:
    old_lines = _clean_text(old_text).splitlines()
    new_lines = _clean_text(new_text).splitlines()
    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="older version",
            tofile="newer version",
            lineterm="",
            n=context_lines,
        )
    )
    added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    truncated = len(diff_lines) > max_lines
    visible = diff_lines[:max_lines]
    return {
        "added_lines": added,
        "removed_lines": removed,
        "changed": bool(added or removed),
        "truncated": truncated,
        "unified_diff": "\n".join(visible),
        "summary": f"Detected {added} added and {removed} removed text lines." if added or removed else "No line-level differences were detected after whitespace normalization.",
    }


def extract_votes(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    votes: list[dict[str, Any]] = []
    actions = [item for item in as_list(bundle.get("actions")) if isinstance(item, Mapping)]
    for action in actions:
        action_text = _action_text(action)
        source_system = action.get("sourceSystem") if isinstance(action.get("sourceSystem"), Mapping) else {}
        source_name = str(source_system.get("name") or "")
        recorded_votes = as_list(action.get("recordedVotes"), "recordedVote", "recordedVotes", "item")
        for vote in recorded_votes:
            if not isinstance(vote, Mapping):
                continue
            votes.append(
                {
                    "chamber": vote.get("chamber") or source_name,
                    "date": vote.get("date") or _action_date(action),
                    "roll_number": vote.get("rollNumber") or vote.get("rollCallNumber"),
                    "yeas": vote.get("yeas") or vote.get("yea"),
                    "nays": vote.get("nays") or vote.get("nay"),
                    "present": vote.get("present"),
                    "not_voting": vote.get("notVoting") or vote.get("not_voting"),
                    "result": vote.get("result") or "",
                    "url": vote.get("url") or "",
                    "action": action_text,
                }
            )

        # Some action records place the tally only in prose. Preserve it as a
        # transparent text extraction rather than presenting it as structured
        # official fields.
        tally_match = re.search(
            r"(?:yeas?|ayes?)\s*[-:]?\s*(\d+).*?(?:nays?|noes?)\s*[-:]?\s*(\d+)",
            action_text,
            flags=re.IGNORECASE,
        )
        roll_match = re.search(r"roll(?: call| no\.)?\s*[-#:]?\s*(\d+)", action_text, flags=re.IGNORECASE)
        if tally_match and not recorded_votes:
            votes.append(
                {
                    "chamber": source_name,
                    "date": _action_date(action),
                    "roll_number": roll_match.group(1) if roll_match else "",
                    "yeas": int(tally_match.group(1)),
                    "nays": int(tally_match.group(2)),
                    "present": "",
                    "not_voting": "",
                    "result": "Parsed from official action text",
                    "url": "",
                    "action": action_text,
                }
            )
    return votes


def sponsor_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    detail = bundle.get("detail") if isinstance(bundle.get("detail"), Mapping) else {}
    sponsors = [item for item in as_list(detail.get("sponsors"), "item", "sponsors") if isinstance(item, Mapping)]
    rows = []
    for item in sponsors:
        rows.append(
            {
                "role": "Sponsor",
                "name": item.get("fullName") or item.get("name") or item.get("directOrderName") or "Unknown",
                "party": item.get("party") or "",
                "state": item.get("state") or "",
                "district": item.get("district") or "",
                "bioguide_id": item.get("bioguideId") or "",
                "url": item.get("url") or "",
            }
        )
    for item in [x for x in as_list(bundle.get("cosponsors")) if isinstance(x, Mapping)]:
        rows.append(
            {
                "role": "Cosponsor",
                "name": item.get("fullName") or item.get("name") or item.get("directOrderName") or "Unknown",
                "party": item.get("party") or "",
                "state": item.get("state") or "",
                "district": item.get("district") or "",
                "bioguide_id": item.get("bioguideId") or "",
                "url": item.get("url") or "",
            }
        )
    return rows


def build_contact_message(
    analysis: BillAnalysis,
    *,
    position: str,
    sender_name: str = "",
    state: str = "",
    custom_note: str = "",
) -> str:
    positions = {
        "Request information": "I am writing to request information about",
        "Express support": "I am writing to express support for",
        "Express opposition": "I am writing to express opposition to",
        "Share a concern": "I am writing to share a concern about",
    }
    opening = positions.get(position, positions["Request information"])
    name_line = f"My name is {sender_name.strip()}, and I am a constituent from {state.strip().upper()}." if sender_name.strip() and state.strip() else "I am writing as a constituent."
    note = custom_note.strip() or "Please explain how you understand the bill's practical effects and which provisions you consider most important."
    official_url = str(analysis.source_links.get("official_bill_page") or analysis.source_links.get("official_text") or "").strip()
    source_line = f"\nOfficial bill record: {official_url}\n" if official_url else ""
    return (
        f"Subject: {analysis.citation} — {analysis.title}\n\n"
        f"Dear Representative or Senator,\n\n"
        f"{name_line} {opening} {analysis.citation}, \"{analysis.title}.\"\n\n"
        f"My understanding from the official text is: {analysis.plain_summary}\n\n"
        f"{source_line}\n"
        f"{note}\n\n"
        "Please rely on the latest official bill version and let me know about any amendments, votes, or implementation details that would change this understanding.\n\n"
        "Sincerely,\n"
        f"{sender_name.strip() or '[Your name]'}"
    )


def analysis_to_markdown(analysis: BillAnalysis, personalization: Mapping[str, Any] | None = None) -> str:
    lines = [
        f"# {analysis.citation}: {analysis.title}",
        "",
        f"**Status:** {analysis.status.label.title()} ({analysis.status.confidence} confidence)",
        f"**Overall analysis confidence:** {analysis.confidence.get('label', 'unknown').title()}",
        "",
        "## Plain-English summary",
        analysis.plain_summary,
        "",
        "## What it says it is for",
        analysis.stated_purpose,
        "",
        "## What the operative text appears to do",
        analysis.operative_effect,
        "",
        "## Section-by-section breakdown",
    ]
    for section in analysis.sections:
        lines.extend(
            [
                "",
                f"### Section {section.number}: {section.title}",
                section.summary,
            ]
        )
        if section.operative_points:
            lines.extend(["", "Key operative points:"] + [f"- {item}" for item in section.operative_points])
    lines.extend(["", "## Fiscal and budget signals"])
    if analysis.fiscal.get("has_cbo_estimate"):
        lines.append("- Congress.gov metadata includes one or more CBO cost-estimate records.")
    else:
        lines.append("- No CBO cost-estimate record was present in the retrieved metadata.")
    for amount in analysis.fiscal.get("detected_amounts", []):
        lines.append(f"- Textual dollar reference: {amount}")
    if analysis.scope_mismatch_flags:
        lines.extend(["", "## Possible scope-mismatch review flags"])
        for flag in analysis.scope_mismatch_flags:
            lines.append(f"- Section {flag['section']} ({flag['title']}): {flag['reason']} {flag['review_note']}")
    if personalization:
        lines.extend(["", "## Profile-based relevance checklist", f"**Overall:** {personalization.get('overall', '')}"])
        for signal in personalization.get("signals", []):
            lines.append(f"- **{signal['category']} — {signal['relevance']}:** {signal['reason']} Verify: {signal['what_to_verify']}.")
    lines.extend(["", "## Official source links"])
    for label, url in analysis.source_links.items():
        if url:
            lines.append(f"- {label.replace('_', ' ').title()}: {url}")
    lines.extend(
        [
            "",
            "## Limitations",
            *[f"- {warning}" for warning in analysis.warnings],
            "",
            "---",
            "Author: Claire Yuan  ",
            "Advisor: Dr. Qingyang Xiao  ",
            "Educational prototype; not legal advice.",
        ]
    )
    return "\n".join(lines)


def analysis_to_json(analysis: BillAnalysis, personalization: Mapping[str, Any] | None = None) -> str:
    payload = {"analysis": analysis.to_dict(), "personalization": personalization or {}}
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)
