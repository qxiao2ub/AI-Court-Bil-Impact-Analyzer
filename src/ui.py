from __future__ import annotations

import html
from typing import Any

import streamlit as st


CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

:root {
  --paper: #faf8f3;
  --ink: #1a1a1a;
  --muted: #757168;
  --line: #d7d3c9;
  --soft: #f2ede1;
  --accent: #d07a2d;
  --green: #39724b;
  --red: #9b4438;
  --blue: #3e617d;
  --streamlit-toolbar-safe-area: 5.25rem;
}
html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
}
[data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] { background: var(--paper) !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }

/* Community Cloud places Share/Edit controls in a fixed header. Reserve enough
   real document flow below it; do not use a negative margin on the custom nav. */
.block-container,
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"] {
  max-width: 1180px !important;
  padding-top: var(--streamlit-toolbar-safe-area) !important;
  padding-bottom: 4rem !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
  overflow: visible !important;
}
[data-testid="stHeader"] {
  min-height: 4rem !important;
  background: rgba(250,248,243,.98) !important;
  border-bottom: 1px solid rgba(215,211,201,.88) !important;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  z-index: 999999 !important;
}
html { scroll-padding-top: 5.75rem; }

h1,h2,h3,h4,h5,h6 { color:var(--ink) !important; letter-spacing:-.025em; }
p,li,label,div { color:var(--ink); }
a { color:var(--ink); }

.bb-nav {
  position: relative; z-index: 3; overflow: visible;
  border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
  display:flex; align-items:center; justify-content:space-between;
  min-height:72px; margin:0 -2rem; padding:0 2rem;
  background:var(--paper);
}
.bb-brand { display:flex;align-items:center;gap:.6rem;font-size:.82rem;font-weight:500;letter-spacing:-.02em; }
.bb-square { width:8px;height:8px;background:var(--ink);display:inline-block; }
.bb-version { color:var(--muted);font-weight:400; }
.bb-nav-right { display:flex;align-items:center;gap:1.1rem;font-size:.62rem;text-transform:uppercase;letter-spacing:.17em;color:var(--muted); }
.bb-nav-right a { color:var(--ink);text-decoration:none;border:1px solid var(--ink);padding:.62rem .82rem; }
.bb-counter { border-left:1px solid var(--line);padding-left:1rem;color:var(--muted)!important; }

.bb-ticker { margin:0 -2rem;border-bottom:1px solid var(--line);overflow:hidden;white-space:nowrap;display:flex;align-items:stretch;min-height:40px; }
.bb-ticker-label { background:var(--ink);color:var(--paper);padding:.78rem 1rem;font-size:.59rem;text-transform:uppercase;letter-spacing:.22em;z-index:2; }
.bb-dot { width:6px;height:6px;border-radius:50%;background:var(--accent);display:inline-block;margin-right:.45rem;animation:bbpulse 1.6s ease-in-out infinite; }
.bb-ticker-track { color:var(--muted);font-size:.64rem;text-transform:uppercase;letter-spacing:.18em;padding:.78rem 0;display:inline-block;min-width:max-content;animation:bbmarquee 55s linear infinite; }
.bb-ticker-track span { margin:0 1.4rem;color:var(--muted); }
.bb-ticker:hover .bb-ticker-track { animation-play-state:paused; }
@keyframes bbmarquee { from{transform:translateX(0)} to{transform:translateX(-40%)} }
@keyframes bbpulse { 0%,100%{opacity:.35}50%{opacity:1} }

.bb-hero { border-bottom:1px solid var(--line);margin:0 -2rem;padding:4.8rem 2rem 4.2rem; }
.bb-hero-inner { max-width:1020px;margin:0 auto; }
.bb-eyebrow { color:var(--muted)!important;font-size:.61rem;text-transform:uppercase;letter-spacing:.27em;margin-bottom:1.4rem; }
.bb-display { font-family:"JetBrains Mono",monospace;font-size:clamp(2.9rem,7vw,5.35rem);font-weight:300;line-height:1.03;letter-spacing:-.06em;margin:0;color:var(--ink); }
.bb-display .serif { font-family:"Instrument Serif",Georgia,serif;font-style:italic;font-weight:400;color:var(--muted);letter-spacing:-.02em; }
.bb-display .accent-word { text-decoration:underline;text-decoration-color:var(--accent);text-decoration-thickness:3px;text-underline-offset:11px; }
.bb-lede { max-width:820px;margin-top:2.2rem;color:var(--muted)!important;font-size:.89rem;line-height:1.86; }
.bb-creditline { margin-top:1.6rem;font-size:.65rem;text-transform:uppercase;letter-spacing:.15em;color:var(--muted)!important; }
.bb-creditline strong { color:var(--ink);font-weight:500; }
.bb-warning { border-left:2px solid var(--accent);background:var(--soft);margin:2rem 0 .7rem;padding:1rem 1.15rem;font-family:"Instrument Serif",Georgia,serif;font-style:italic;font-size:1.02rem;line-height:1.55;color:#4c4941!important; }

.bb-section-kicker { margin-top:1.2rem;color:var(--muted)!important;font-size:.61rem;text-transform:uppercase;letter-spacing:.27em; }
.bb-section-title { font-family:"Instrument Serif",Georgia,serif;font-size:clamp(2.1rem,4.2vw,3.45rem);line-height:1.08;margin:.55rem 0 1.1rem;color:var(--ink)!important;font-weight:400; }
.bb-section-copy { max-width:850px;color:var(--muted)!important;font-size:.82rem;line-height:1.72;margin-bottom:1.6rem; }
.bb-rule { border-top:1px solid var(--line);margin:2.5rem 0 2.2rem; }
.bb-smallcaps { color:var(--muted)!important;font-size:.59rem;text-transform:uppercase;letter-spacing:.22em; }

.bb-card-grid { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin:1.2rem 0; }
.bb-card { border-top:1px solid var(--ink);padding:1rem 0 .6rem; }
.bb-card-label { color:var(--muted)!important;font-size:.57rem;text-transform:uppercase;letter-spacing:.2em; }
.bb-card-value { font-family:"Instrument Serif",Georgia,serif;font-size:1.75rem;line-height:1.18;margin:.55rem 0 .3rem; }
.bb-card-sub { color:var(--muted)!important;font-size:.63rem;line-height:1.55; }
.bb-sourcebox { border:1px solid var(--line);padding:1rem 1.1rem;background:#fffdf8;margin:1rem 0; }
.bb-sourcebox h4 { font-family:"JetBrains Mono",monospace!important;font-size:.63rem!important;text-transform:uppercase;letter-spacing:.18em;margin:0 0 .8rem; }
.bb-sourcebox a { word-break:break-word;font-size:.72rem; }
.bb-callout { border-left:2px solid var(--accent);padding:.55rem 0 .55rem 1.1rem;margin:1rem 0; }
.bb-callout-title { font-family:"Instrument Serif",Georgia,serif;font-size:1.45rem;line-height:1.3;margin:.25rem 0; }
.bb-callout-copy { color:var(--muted)!important;font-size:.74rem;line-height:1.65; }
.bb-badge { display:inline-block;border:1px solid var(--line);padding:.3rem .55rem;margin:.15rem .25rem .15rem 0;font-size:.58rem;text-transform:uppercase;letter-spacing:.13em; }
.bb-badge.good { border-color:var(--green);color:var(--green)!important; }
.bb-badge.warn { border-color:var(--accent);color:#8b511b!important; }
.bb-badge.risk { border-color:var(--red);color:var(--red)!important; }
.bb-two-col { display:grid;grid-template-columns:1fr 1fr;gap:1.25rem; }
.bb-note { color:var(--muted)!important;font-size:.68rem;line-height:1.55; }
.bb-footer { border-top:1px solid var(--line);margin:3rem -2rem 0;padding:1.6rem 2rem .5rem;display:flex;justify-content:space-between;gap:1.2rem;flex-wrap:wrap;color:var(--muted)!important;font-size:.58rem;text-transform:uppercase;letter-spacing:.19em; }

.stTabs [data-baseweb="tab-list"] { gap:0!important;border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:var(--paper);margin-top:1.5rem;overflow-x:auto; }
.stTabs [data-baseweb="tab"] { height:54px;padding:0 1.05rem!important;color:var(--muted)!important;font-family:"JetBrains Mono",monospace!important;font-size:.61rem!important;text-transform:uppercase;letter-spacing:.15em;white-space:nowrap; }
.stTabs [aria-selected="true"] { color:var(--ink)!important; }
.stTabs [data-baseweb="tab-highlight"] { background-color:var(--ink)!important;height:1px!important; }
.stTabs [data-baseweb="tab-border"] { display:none; }

.stRadio label,.stCheckbox label,.stSelectbox label,.stNumberInput label,.stTextArea label,.stFileUploader label,.stTextInput label,.stMultiSelect label,.stDateInput label { font-family:"JetBrains Mono",monospace!important;font-size:.65rem!important;letter-spacing:.03em; }
.stTextArea textarea,.stTextInput input,.stNumberInput input,[data-baseweb="select"]>div { border-radius:0!important;border-color:var(--line)!important;background:#fffdf8!important;color:var(--ink)!important;box-shadow:none!important; }
[data-testid="stFileUploaderDropzone"] { border-radius:0!important;border:1px dashed #aaa59a!important;background:#fffdf8!important;padding:1.2rem!important; }
[data-testid="stFileUploaderDropzoneInstructions"] * { color:var(--muted)!important; }
.stButton>button,.stDownloadButton>button,.stLinkButton>a { border-radius:0!important;border:1px solid var(--ink)!important;background:transparent!important;color:var(--ink)!important;font-family:"JetBrains Mono",monospace!important;font-size:.62rem!important;text-transform:uppercase;letter-spacing:.14em;min-height:43px;transition:all .15s ease;text-decoration:none!important; }
.stButton>button:hover,.stDownloadButton>button:hover,.stLinkButton>a:hover { background:var(--ink)!important;color:var(--paper)!important;border-color:var(--ink)!important; }
.stButton>button[kind="primary"] { background:var(--ink)!important;color:var(--paper)!important; }
.stButton>button[kind="primary"]:hover { background:var(--accent)!important;border-color:var(--accent)!important; }
[data-testid="stExpander"] { border:1px solid var(--line)!important;border-radius:0!important;background:transparent!important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); }
[data-testid="stAlert"] { border-radius:0!important; }
[data-testid="stProgress"]>div>div>div>div { background-color:var(--accent)!important; }
[data-testid="stMetric"] { border-top:1px solid var(--ink);padding-top:.8rem; }
[data-testid="stMetricLabel"] { font-size:.61rem!important;text-transform:uppercase;letter-spacing:.12em; }
[data-testid="stMetricValue"] { font-family:"Instrument Serif",Georgia,serif!important;font-size:1.7rem!important; }
[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3 { font-family:"Instrument Serif",Georgia,serif!important;font-weight:400!important; }
[data-testid="stMarkdownContainer"] blockquote { border-left:2px solid var(--accent);color:#5a5650;font-family:"Instrument Serif",Georgia,serif;font-style:italic; }

@media (max-width:900px) {
  :root { --streamlit-toolbar-safe-area: 5rem; }
  .bb-nav-right span,.bb-counter { display:none; }
  .bb-hero { padding-top:3.4rem;padding-bottom:3.5rem; }
  .bb-card-grid,.bb-two-col { grid-template-columns:1fr; }
  .block-container,[data-testid="stAppViewBlockContainer"],[data-testid="stMainBlockContainer"] { padding-top:var(--streamlit-toolbar-safe-area)!important;padding-left:1.15rem!important;padding-right:1.15rem!important; }
  .bb-nav,.bb-ticker,.bb-hero,.bb-footer { margin-left:-1.15rem;margin-right:-1.15rem; }
  .bb-nav,.bb-hero,.bb-footer { padding-left:1.15rem;padding-right:1.15rem; }
}
</style>
"""


def inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def top_chrome(visitor_count: int) -> None:
    st.markdown(
        f"""
<div class="bb-nav">
  <div class="bb-brand"><span class="bb-square"></span><span>bill_lens</span><span class="bb-version">/v3.0</span></div>
  <div class="bb-nav-right">
    <span>neutral legislative research</span>
    <span class="bb-counter">{visitor_count:,} app visitors</span>
    <a href="#analysis-workbench">Analyze a bill</a>
  </div>
</div>
<div class="bb-ticker">
  <div class="bb-ticker-label"><span class="bb-dot"></span>live model notes</div>
  <div class="bb-ticker-track">
    <span>Congress.gov official metadata + text</span> ·
    <span>plain-English extractive AI</span> ·
    <span>section-level analysis</span> ·
    <span>version comparison</span> ·
    <span>uncertainty shown</span> ·
    <span>no passage or election predictions</span> ·
    <span>human verification required</span> ·
    <span>Congress.gov official metadata + text</span> ·
    <span>plain-English extractive AI</span> ·
    <span>section-level analysis</span> ·
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
<section class="bb-hero">
  <div class="bb-hero-inner">
    <div class="bb-eyebrow"><span class="bb-dot"></span>official sources · explainable analysis · civic agency</div>
    <h1 class="bb-display">federal bills,<br><span class="serif">in plain</span> <span class="accent-word">English.</span></h1>
    <div class="bb-lede">
      Search a live House or Senate measure, upload a searchable PDF, or paste legislative text. The app retrieves official metadata,
      maps the legislative stage, summarizes each section, surfaces budget and scope signals, and builds a profile-based relevance checklist.
    </div>
    <div class="bb-creditline"><strong>Author: Claire Yuan</strong> &nbsp;·&nbsp; <strong>Advisor: Dr. Qingyang Xiao</strong> &nbsp;·&nbsp; MIT License</div>
  </div>
</section>
<div class="bb-warning">
  Educational research prototype. It does not provide legal advice, endorse or oppose legislation, determine constitutionality,
  or predict passage. Always read the linked official bill text and verify amended versions.
</div>
<div id="analysis-workbench"></div>
""",
        unsafe_allow_html=True,
    )


def section_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
<div class="bb-section-kicker">{html.escape(kicker)}</div>
<div class="bb-section-title">{html.escape(title)}</div>
<div class="bb-section-copy">{html.escape(copy)}</div>
""",
        unsafe_allow_html=True,
    )


def metric_cards(items: list[tuple[str, str, str]]) -> None:
    cards = []
    for label, value, sub in items:
        cards.append(
            f'<div class="bb-card"><div class="bb-card-label">{html.escape(label)}</div>'
            f'<div class="bb-card-value">{html.escape(value)}</div>'
            f'<div class="bb-card-sub">{html.escape(sub)}</div></div>'
        )
    st.markdown(f'<div class="bb-card-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def callout(title: str, copy: str, label: str = "analysis note") -> None:
    st.markdown(
        f"""
<div class="bb-callout">
  <div class="bb-smallcaps">{html.escape(label)}</div>
  <div class="bb-callout-title">{html.escape(title)}</div>
  <div class="bb-callout-copy">{html.escape(copy)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def badge(text: str, tone: str = "") -> str:
    return f'<span class="bb-badge {html.escape(tone)}">{html.escape(text)}</span>'


def source_box(links: dict[str, str]) -> None:
    rows = []
    for label, url in links.items():
        if url:
            rows.append(f'<div><strong>{html.escape(label.replace("_", " ").title())}:</strong> <a href="{html.escape(url)}" target="_blank" rel="noopener">Open official source ↗</a></div>')
    if rows:
        st.markdown(
            '<div class="bb-sourcebox"><h4>Official verification links</h4>' + "".join(rows) + "</div>",
            unsafe_allow_html=True,
        )


def footer() -> None:
    st.markdown(
        """
<div class="bb-footer">
  <span>Claire Yuan · AI Legislative Bill Impact Analyzer</span>
  <span>Advisor: Dr. Qingyang Xiao · MIT License · educational prototype</span>
</div>
""",
        unsafe_allow_html=True,
    )
