#!/usr/bin/env python3
"""
Quantum Hardware Data Updater

Fetches latest quantum hardware information from public sources,
compares against current data, and regenerates README.md if changes found.

Sources checked:
  - Company hardware/product pages
  - Wikipedia quantum processor list
  - Quantum news aggregators (The Quantum Insider, QC Report)
  - ArXiv recent papers in quant-ph

Usage:
    python scripts/update_hardware_data.py                # Check for updates, regenerate README
    python scripts/update_hardware_data.py --dry-run      # Show what would change without writing
    python scripts/update_hardware_data.py --fetch-only   # Fetch and save raw pages for review
    python scripts/update_hardware_data.py --readme-only  # Regenerate README from current data
"""

import json
import re
import sys
import hashlib
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import URLError, HTTPError
from urllib.request import urlopen, Request

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "data" / "hardware_data.json"
README_FILE = REPO_ROOT / "README.md"
FETCH_DIR = REPO_ROOT / "data" / "fetched"
CHANGELOG_FILE = REPO_ROOT / "data" / "update_log.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Web fetching
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; QuantumHWBot/1.0; "
        "+https://github.com/des137/quantum-hardware-systems)"
    )
}
TIMEOUT = 30  # seconds


def fetch_url(url: str) -> Optional[str]:
    """Fetch URL content. Returns text or None on failure."""
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read()
            # Try utf-8, fall back to latin-1
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1")
    except (URLError, HTTPError, OSError) as exc:
        log.warning("Failed to fetch %s: %s", url, exc)
        return None


def strip_html(html: str) -> str:
    """Rough HTML-to-text conversion for keyword extraction."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Source fetching: company pages, Wikipedia, news
# ---------------------------------------------------------------------------

def fetch_company_pages(sources: dict) -> dict:
    """Fetch all company hardware pages. Returns {company: text}."""
    results = {}
    for company, url in sources.get("company_pages", {}).items():
        log.info("Fetching %s (%s)", company, url)
        html = fetch_url(url)
        if html:
            results[company] = strip_html(html)
    return results


def fetch_wikipedia_processors() -> str:
    """Fetch Wikipedia list of quantum processors via API."""
    api_url = (
        "https://en.wikipedia.org/w/api.php?action=parse"
        "&page=List_of_quantum_processors"
        "&prop=wikitext&format=json&redirects"
    )
    raw = fetch_url(api_url)
    if raw:
        try:
            data = json.loads(raw)
            return data.get("parse", {}).get("wikitext", {}).get("*", "")
        except (json.JSONDecodeError, KeyError):
            pass
    return ""


def fetch_arxiv_recent(max_results: int = 50) -> list:
    """Fetch recent quant-ph papers mentioning quantum hardware keywords."""
    from urllib.parse import quote
    keywords = [
        "quantum processor", "qubit fidelity", "quantum error correction",
        "superconducting qubit",
    ]
    query = "+OR+".join(f"all:%22{quote(kw)}%22" for kw in keywords)
    url = (
        f"http://export.arxiv.org/api/query?search_query={query}"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    )
    raw = fetch_url(url)
    if not raw:
        return []
    # Extract titles and summaries
    papers = []
    for entry in re.findall(r"<entry>(.*?)</entry>", raw, re.S):
        title = re.search(r"<title>(.*?)</title>", entry, re.S)
        summary = re.search(r"<summary>(.*?)</summary>", entry, re.S)
        published = re.search(r"<published>(.*?)</published>", entry)
        if title:
            papers.append({
                "title": re.sub(r"\s+", " ", title.group(1)).strip(),
                "summary": re.sub(r"\s+", " ", summary.group(1)).strip() if summary else "",
                "published": published.group(1) if published else "",
            })
    return papers


def fetch_quantum_news() -> list:
    """Fetch headlines from quantum news aggregators via RSS/Atom feeds."""
    feeds = [
        "https://thequantuminsider.com/feed/",
        "https://quantumcomputingreport.com/feed/",
    ]
    articles = []
    for feed_url in feeds:
        raw = fetch_url(feed_url)
        if not raw:
            continue
        for item in re.findall(r"<item>(.*?)</item>", raw, re.S):
            title = re.search(r"<title>(.*?)</title>", item, re.S)
            link = re.search(r"<link>(.*?)</link>", item, re.S)
            pub_date = re.search(r"<pubDate>(.*?)</pubDate>", item, re.S)
            if title:
                articles.append({
                    "title": re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", title.group(1)).strip(),
                    "link": link.group(1).strip() if link else "",
                    "date": pub_date.group(1).strip() if pub_date else "",
                })
    return articles


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

QUBIT_PATTERN = re.compile(
    r"(\d[\d,]*)\s*(?:physical\s+)?(?:qubit|qbit)s?", re.IGNORECASE
)
QV_PATTERN = re.compile(
    r"quantum\s+volume[:\s]*(\d[\d,]*)", re.IGNORECASE
)


def extract_qubit_mentions(text: str, company: str) -> list:
    """Extract qubit count mentions near a company name."""
    mentions = []
    # Search within 500 chars of company name
    for match in re.finditer(re.escape(company), text, re.IGNORECASE):
        window = text[max(0, match.start() - 200):match.end() + 500]
        for qm in QUBIT_PATTERN.finditer(window):
            count = int(qm.group(1).replace(",", ""))
            if count > 1:  # Skip "1 qubit" noise
                mentions.append(count)
        for qv in QV_PATTERN.finditer(window):
            mentions.append(("QV", int(qv.group(1).replace(",", ""))))
    return mentions


def detect_changes(data: dict, company_texts: dict, wiki_text: str,
                   news: list, arxiv: list) -> list:
    """
    Compare fetched data against current hardware_data.json.
    Returns list of {company, field, current, detected, source} dicts.
    """
    changes = []
    company_name_map = {}
    for sys_entry in data["systems"]:
        company_name_map[sys_entry["company"]] = sys_entry

    all_text = {**company_texts}
    all_text["Wikipedia"] = wiki_text
    for article in news:
        all_text[f"News: {article.get('title', '')[:60]}"] = article.get("title", "")

    for company, entry in company_name_map.items():
        for source_name, text in all_text.items():
            if not text:
                continue
            mentions = extract_qubit_mentions(text, company)
            for m in mentions:
                if isinstance(m, tuple):
                    continue  # Skip QV for now
                # Check if this is a notable increase
                current_str = entry.get("qubits", "0")
                try:
                    current_num = int(re.search(r"\d+", current_str.replace(",", "")).group())
                except (AttributeError, ValueError):
                    current_num = 0
                if m > current_num * 1.2 and m > current_num + 5:
                    changes.append({
                        "company": company,
                        "field": "qubits",
                        "current": current_str,
                        "detected": str(m),
                        "source": source_name,
                    })

    return changes


# ---------------------------------------------------------------------------
# README generation
# ---------------------------------------------------------------------------

def generate_readme(data: dict) -> str:
    """Generate README.md content from hardware_data.json."""
    lines = []
    lines.append("# Quantum Hardware Systems")
    lines.append("Characterize quantum hardware")
    lines.append("")

    # Build table
    lines.append("| Company | Countries | Qubits | Modality | QEC | Status |")
    lines.append("|---------|-----------|--------|----------|-----|--------|")

    # Sort: active first, then by QEC tier, then by qubits descending
    qec_order = {"High": 0, "M-H": 1, "Med": 2, "L-M": 3, "Low": 4}
    status_order = {"active": 0, "acquired": 1, "discontinued": 2}

    def sort_key(s):
        # Extract max numeric qubit count for sorting
        nums = re.findall(r"\d+", s.get("qubits", "0").replace(",", ""))
        max_q = max((int(n) for n in nums), default=0)
        return (
            status_order.get(s.get("status", "active"), 9),
            qec_order.get(s.get("qec", "Med"), 9),
            -max_q,
        )

    sorted_systems = sorted(data["systems"], key=sort_key)

    for sys in sorted_systems:
        company = sys["company"]
        system = sys.get("system")
        if system:
            company_col = f"**{company}** ({system})"
        else:
            company_col = f"**{company}**"

        status = sys.get("status", "active")
        if status == "discontinued":
            status_str = "Discontinued"
        elif status == "acquired":
            status_str = "Acquired"
        else:
            status_str = "Active"

        countries = ", ".join(sys.get("countries", []))
        qubits = sys.get("qubits", "N/A")
        modality = sys.get("modality", "")
        qec = sys.get("qec", "")

        lines.append(
            f"| {company_col} | {countries} | {qubits} | {modality} | {qec} | {status_str} |"
        )

    lines.append("")
    lines.append("## Feature Explanations")
    lines.append("")
    lines.append(
        "**Countries**: ISO 3-letter country codes "
        "(AUT=Austria, AUS=Australia, CAN=Canada, CHN=China, DEU=Germany, "
        "ESP=Spain, FIN=Finland, FRA=France, GBR=United Kingdom, JPN=Japan, "
        "KOR=South Korea, NLD=Netherlands, POL=Poland, SAU=Saudi Arabia, USA=United States)"
    )
    lines.append("")
    lines.append("**Qubits**: Maximum public physical qubits or modes")
    lines.append("")
    lines.append("**Modality**: ")
    lines.append("- SC = Superconducting")
    lines.append("- Trap-ion = Trapped ions")
    lines.append("- Neut-atom = Neutral atoms")
    lines.append("- Photon = Photonic")
    lines.append("- Photon-CV = Photonic continuous variable")
    lines.append("- Si-spin = Silicon spin qubits")
    lines.append("- Si-donor = Silicon donor qubits")
    lines.append("- NV-diamond = NV-center diamond")
    lines.append("- QA = Quantum annealing")
    lines.append("- Spin-photon = Spin-photon hybrid")
    lines.append("- NMR = Nuclear magnetic resonance")
    lines.append("")
    lines.append(
        "**QEC**: Scalability to Quantum Error Correction "
        "(Low/Med/High or L-M/M-H for ranges)"
    )
    lines.append("")
    lines.append(
        "**Status**: Active = currently operating/available; "
        "Acquired = company acquired by another; "
        "Discontinued = quantum program shut down"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Data last verified: {data['metadata']['last_updated']}. "
        "Updated weekly via GitHub Actions — see "
        "[scripts/update_hardware_data.py](scripts/update_hardware_data.py) "
        "and [data/hardware_data.json](data/hardware_data.json) for sources and methodology.*"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Changelog
# ---------------------------------------------------------------------------

def load_changelog() -> list:
    """Load existing changelog entries."""
    if CHANGELOG_FILE.exists():
        try:
            return json.loads(CHANGELOG_FILE.read_text())
        except json.JSONDecodeError:
            return []
    return []


def save_changelog(entries: list):
    """Save changelog entries."""
    CHANGELOG_FILE.write_text(json.dumps(entries, indent=2) + "\n")


def append_changelog(changes: list, source_summary: dict):
    """Append a run entry to the changelog."""
    entries = load_changelog()
    # Keep last 100 entries
    entries = entries[-99:]
    entries.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sources_checked": source_summary,
        "potential_changes": changes[:20],  # Cap to avoid huge logs
    })
    save_changelog(entries)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Update quantum hardware data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show changes without writing files")
    parser.add_argument("--fetch-only", action="store_true",
                        help="Only fetch and cache raw pages")
    parser.add_argument("--readme-only", action="store_true",
                        help="Regenerate README from current data.json (no fetching)")
    args = parser.parse_args()

    # Load current data
    if not DATA_FILE.exists():
        log.error("Data file not found: %s", DATA_FILE)
        sys.exit(1)

    data = json.loads(DATA_FILE.read_text())
    sources = data.get("sources", {})

    if args.readme_only:
        readme_content = generate_readme(data)
        if args.dry_run:
            print(readme_content)
        else:
            README_FILE.write_text(readme_content)
            log.info("README.md regenerated from current data")
        return

    # ----- Fetch from all sources -----
    log.info("=" * 60)
    log.info("Fetching quantum hardware updates...")
    log.info("=" * 60)

    log.info("\n--- Company pages ---")
    company_texts = fetch_company_pages(sources)
    log.info("Fetched %d company pages", len(company_texts))

    log.info("\n--- Wikipedia ---")
    wiki_text = fetch_wikipedia_processors()
    log.info("Wikipedia text length: %d chars", len(wiki_text))

    log.info("\n--- News feeds ---")
    news = fetch_quantum_news()
    log.info("Fetched %d news articles", len(news))

    log.info("\n--- ArXiv ---")
    arxiv = fetch_arxiv_recent(max_results=30)
    log.info("Fetched %d arxiv papers", len(arxiv))

    # Optionally save fetched data
    if args.fetch_only:
        FETCH_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, text in company_texts.items():
            (FETCH_DIR / f"{name.replace(' ', '_')}_{ts}.txt").write_text(text[:50000])
        if wiki_text:
            (FETCH_DIR / f"wikipedia_{ts}.txt").write_text(wiki_text[:100000])
        (FETCH_DIR / f"news_{ts}.json").write_text(json.dumps(news, indent=2))
        (FETCH_DIR / f"arxiv_{ts}.json").write_text(json.dumps(arxiv, indent=2))
        log.info("Raw fetched data saved to %s", FETCH_DIR)
        return

    # ----- Detect changes -----
    log.info("\n--- Detecting changes ---")
    changes = detect_changes(data, company_texts, wiki_text, news, arxiv)

    source_summary = {
        "company_pages": len(company_texts),
        "wikipedia": bool(wiki_text),
        "news_articles": len(news),
        "arxiv_papers": len(arxiv),
    }

    if changes:
        log.info("\nPotential updates detected:")
        for c in changes:
            log.info(
                "  %s: %s %s -> %s (source: %s)",
                c["company"], c["field"], c["current"], c["detected"], c["source"]
            )
        log.info(
            "\nNOTE: These are heuristic detections. Review before applying. "
            "Update data/hardware_data.json manually or via PR review."
        )
    else:
        log.info("No significant qubit count changes detected.")

    # ----- Log recent news headlines for review -----
    if news:
        log.info("\n--- Recent quantum hardware news (for manual review) ---")
        for article in news[:15]:
            log.info("  [%s] %s", article.get("date", "")[:10], article["title"])

    if arxiv:
        log.info("\n--- Recent arxiv papers ---")
        for paper in arxiv[:10]:
            log.info("  [%s] %s", paper.get("published", "")[:10], paper["title"])

    # ----- Save changelog -----
    append_changelog(changes, source_summary)
    log.info("\nUpdate log saved to %s", CHANGELOG_FILE)

    # ----- Update last_updated date -----
    # Always update in-memory so the README preview in dry-run mode reflects
    # the date that would be written.  The file is only written when not dry-run.
    today = datetime.now(timezone.utc).date().isoformat()
    data.setdefault("metadata", {})["last_updated"] = today
    if not args.dry_run:
        DATA_FILE.write_text(json.dumps(data, indent=2) + "\n")
        log.info("hardware_data.json last_updated set to %s", today)

    # ----- Regenerate README -----
    readme_content = generate_readme(data)
    if args.dry_run:
        current_readme = README_FILE.read_text() if README_FILE.exists() else ""
        if readme_content != current_readme:
            log.info("\nREADME.md would be updated (content differs)")
        else:
            log.info("\nREADME.md is already up to date")
    else:
        README_FILE.write_text(readme_content)
        log.info("\nREADME.md regenerated")

    # Exit with code 1 if changes detected (useful for CI to trigger PR)
    if changes:
        sys.exit(1)


if __name__ == "__main__":
    main()
