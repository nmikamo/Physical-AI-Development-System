"""Rate-limited polite scraper for https://www.alphaxiv.org/ (time.sleep >= 2.0s)."""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Sequence

import httpx
from bs4 import BeautifulSoup

from physical_ai_system.domain.models import PaperRecord
from physical_ai_system.domain.text_mining import detect_sim_to_real

logger = logging.getLogger(__name__)

USER_AGENT = "PhysicalAIDevelopmentSystem/1.0 (Research Scraper; Polite Rate-Limited)"


def _normalize_month(date_str: str) -> str:
    m = re.search(r"(20\d{2})[-/](\d{1,2})", date_str or "")
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _extract_abs_ids_from_html(html: str) -> tuple[str, ...]:
    """Pure parser extracting unique /abs/<id> paths from alphaXiv HTML."""
    soup = BeautifulSoup(html, "html.parser")
    seen: list[str] = []
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        m = re.match(r"^/abs/([A-Za-z0-9.\-]+)", href)
        if m:
            pid = m.group(1)
            if pid not in seen:
                seen.append(pid)
    # Also scan script/raw HTML for /abs/<arxiv-id>
    for pid in re.findall(r"/abs/(\d{4}\.\d{4,5})", html):
        if pid not in seen:
            seen.append(pid)
    return tuple(seen)


def _parse_abs_page(
    paper_id: str,
    html: str,
    url: str,
    keyword: str,
) -> PaperRecord | None:
    """Pure parser extracting title, abstract, authors, and date from alphaXiv /abs page."""
    soup = BeautifulSoup(html, "html.parser")

    def _meta(name: str) -> str:
        tag = soup.find("meta", attrs={"name": name}) or soup.find(
            "meta", attrs={"property": name}
        )
        return str(tag["content"]).strip() if tag and tag.get("content") else ""

    title = (
        _meta("citation_title")
        or _meta("og:title")
        or (soup.title.string.strip() if soup.title and soup.title.string else "")
    )
    title = re.sub(r"\s*\|\s*alphaXiv.*$", "", title).strip()
    if not title:
        return None

    abstract = (
        _meta("citation_abstract")
        or _meta("description")
        or _meta("og:description")
    )
    authors = tuple(
        str(tag["content"]).strip()
        for tag in soup.find_all("meta", attrs={"name": "citation_author"})
        if tag.get("content")
    )
    pub_date = _meta("citation_publication_date") or _meta("citation_date") or ""
    month_key = _normalize_month(pub_date)

    combined = f"{title} {abstract}".lower()
    sim2real = detect_sim_to_real(combined)

    return PaperRecord(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=authors[:6],
        published_at=pub_date or f"{month_key}-01",
        month_key=month_key,
        url=url,
        keywords_matched=(keyword,),
        sim_to_real=sim2real,
    )


def _matches_keywords(text: str, keywords: Sequence[str]) -> tuple[str, ...]:
    lowered = text.lower()
    matched = []
    for kw in keywords:
        tokens = [t for t in kw.lower().split() if len(t) > 1]
        if kw.lower() in lowered or (tokens and all(t in lowered for t in tokens)):
            matched.append(kw)
    return tuple(matched)


def scrape_alphaxiv_papers(
    keywords: Sequence[str],
    base_url: str = "https://www.alphaxiv.org",
    interval_seconds: float = 2.0,
    max_per_keyword: int = 8,
) -> tuple[PaperRecord, ...]:
    """Politely scrape alphaXiv papers matching keywords with >= 2.0s intervals."""
    wait = max(1.5, float(interval_seconds))
    collected: dict[str, PaperRecord] = {}
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/json"}

    with httpx.Client(headers=headers, timeout=20.0, follow_redirects=True) as client:
        # Step 1: Polite sequential scrape of alphaXiv explore pages per keyword
        for kw in keywords:
            encoded = urllib.parse.quote_plus(kw)
            explore_url = f"{base_url.rstrip('/')}/explore?q={encoded}"
            try:
                resp = client.get(explore_url)
                time.sleep(wait)
                if resp.status_code == 200:
                    abs_ids = _extract_abs_ids_from_html(resp.text)[:max_per_keyword]
                    for pid in abs_ids:
                        if pid in collected:
                            prev = collected[pid]
                            if kw not in prev.keywords_matched:
                                from dataclasses import replace

                                collected[pid] = replace(
                                    prev,
                                    keywords_matched=prev.keywords_matched + (kw,),
                                )
                            continue
                        abs_url = f"{base_url.rstrip('/')}/abs/{pid}"
                        try:
                            p_resp = client.get(abs_url)
                            time.sleep(wait)
                            if p_resp.status_code == 200:
                                rec = _parse_abs_page(pid, p_resp.text, abs_url, kw)
                                if rec and _matches_keywords(
                                    f"{rec.title} {rec.abstract}", keywords
                                ):
                                    collected[pid] = rec
                        except httpx.HTTPError as exc:
                            logger.warning("alphaXiv abs fetch skipped (%s): %s", pid, exc)
            except httpx.HTTPError as exc:
                logger.warning("alphaXiv explore skipped (%s): %s", kw, exc)

        # Step 2: Polite check of alphaXiv v3 recent feed (filtered by keywords)
        try:
            feed_url = (
                "https://api.alphaxiv.org/papers/v3/feed"
                "?pageNum=0&pageSize=30&sort=Recent&interval=90%20Days"
            )
            f_resp = client.get(feed_url)
            time.sleep(wait)
            if f_resp.status_code == 200:
                payload = f_resp.json()
                for item in payload.get("papers", []):
                    title = str(item.get("title", "")).strip()
                    abstract = str(item.get("abstract", "")).strip()
                    topics = " ".join(str(t) for t in item.get("topics", []))
                    matched = _matches_keywords(f"{title} {abstract} {topics}", keywords)
                    if not matched:
                        continue
                    pid = str(item.get("universal_paper_id") or item.get("id") or "")
                    if not pid:
                        continue
                    pub = str(item.get("publication_date", ""))[:10]
                    month_key = _normalize_month(pub)
                    abs_url = f"{base_url.rstrip('/')}/abs/{pid}"
                    authors_raw = item.get("full_authors") or item.get("authors") or []
                    authors = tuple(
                        str(a.get("name", a)) if isinstance(a, dict) else str(a)
                        for a in authors_raw[:6]
                    )
                    collected[pid] = PaperRecord(
                        paper_id=pid,
                        title=title,
                        abstract=abstract,
                        authors=authors,
                        published_at=pub or f"{month_key}-01",
                        month_key=month_key,
                        url=abs_url,
                        keywords_matched=matched,
                        sim_to_real=detect_sim_to_real(f"{title} {abstract}"),
                    )
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("alphaXiv v3 feed skipped: %s", exc)

        # Step 3: Supplement via arXiv API -> mapped directly to alphaXiv /abs/<id> URLs
        # Ensures rich monthly corpus for humanoid/physical AI keywords with polite sleep
        for kw in keywords:
            query = urllib.parse.quote_plus(f'all:"{kw}"')
            arxiv_url = (
                f"https://export.arxiv.org/api/query?search_query={query}"
                f"&sortBy=submittedDate&sortOrder=descending&max_results={max_per_keyword}"
            )
            try:
                a_resp = client.get(arxiv_url)
                time.sleep(wait)
                if a_resp.status_code == 200:
                    root = ET.fromstring(a_resp.text)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    for entry in root.findall("atom:entry", ns):
                        raw_id = entry.findtext("atom:id", "", ns)
                        m = re.search(r"(\d{4}\.\d{4,5})", raw_id)
                        if not m:
                            continue
                        pid = m.group(1)
                        title = " ".join(
                            entry.findtext("atom:title", "", ns).split()
                        )
                        abstract = " ".join(
                            entry.findtext("atom:summary", "", ns).split()
                        )
                        pub = entry.findtext("atom:published", "", ns)[:10]
                        month_key = _normalize_month(pub)
                        authors = tuple(
                            au.findtext("atom:name", "", ns).strip()
                            for au in entry.findall("atom:author", ns)[:6]
                        )
                        abs_url = f"{base_url.rstrip('/')}/abs/{pid}"
                        if pid in collected:
                            prev = collected[pid]
                            if kw not in prev.keywords_matched:
                                from dataclasses import replace

                                collected[pid] = replace(
                                    prev,
                                    keywords_matched=prev.keywords_matched + (kw,),
                                )
                        else:
                            collected[pid] = PaperRecord(
                                paper_id=pid,
                                title=title,
                                abstract=abstract,
                                authors=authors,
                                published_at=pub or f"{month_key}-01",
                                month_key=month_key,
                                url=abs_url,
                                keywords_matched=(kw,),
                                sim_to_real=detect_sim_to_real(f"{title} {abstract}"),
                            )
            except (httpx.HTTPError, ET.ParseError) as exc:
                logger.warning("arXiv supplement skipped (%s): %s", kw, exc)

    ordered = sorted(
        collected.values(),
        key=lambda p: (p.published_at, p.paper_id),
        reverse=True,
    )
    return tuple(ordered)
