"""Refresh the publication Markdown from the public Google Scholar profile.

No credentials or API key required. A failed/blocked request leaves the last
successful Markdown untouched; deployment can continue using that saved copy.
"""
import argparse
from html import escape
from pathlib import Path
import re
import sys
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
PROFILE_ID = "dq-VzqkAAAAJ"
DESTINATION = ROOT / "research/publications.md"


def parse_profile(html):
    soup = BeautifulSoup(html, "html.parser")
    if soup.select_one("#gsc_prf_in") is None:
        raise ValueError("Google Scholar did not return a public author profile (possibly blocked).")
    papers = []
    for row in soup.select(".gsc_a_tr"):
        title = row.select_one(".gsc_a_at")
        details = row.select(".gs_gray")
        year = row.select_one(".gsc_a_y")
        if title is None or len(details) < 2 or year is None:
            raise ValueError("Google Scholar returned an incomplete publication row.")
        title_text = title.get_text(" ", strip=True)
        author_text = details[0].get_text(" ", strip=True)
        if not title_text or not author_text or not title.get("href"):
            raise ValueError("Publication title, authors, or link is missing.")
        # Scholar repeats the year in a span after the venue; strip that span.
        for extra in details[1].select(".gs_oph"):
            extra.decompose()
        venue = details[1].get_text(" ", strip=True)
        link = urljoin("https://scholar.google.com", title["href"])
        if not link.startswith("https://scholar.google.com/"):
            raise ValueError("Unexpected publication link.")
        arxiv = re.search(r"arXiv\s*:\s*(\d{4}\.\d{4,5}(?:v\d+)?|[a-z.-]+/\d{7})", venue, flags=re.I)
        if arxiv:
            link = "https://arxiv.org/abs/" + arxiv.group(1)
        year_text = year.get_text(strip=True)
        if year_text and not re.fullmatch(r"\d{4}", year_text):
            raise ValueError("Unexpected publication year.")
        papers.append({"title": title_text, "authors": author_text, "venue": venue,
                       "year": year_text, "url": link})
    if not papers:
        raise ValueError("No publications returned; refusing to replace the saved list.")
    more = soup.select_one("#gsc_bpf_more")
    if more is None:
        raise ValueError("Missing pagination control; unable to verify the complete publication list.")
    return papers, not more.has_attr("disabled")


def fetch_publications(fetch=None):
    if fetch is None:
        def fetch(url):
            request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"})
            with urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
    publications = []
    seen = set()
    for start in range(0, 10000, 100):
        # Request the public profile, without the interactive sort endpoint.
        # We collect all pages and sort by publication year locally.
        params = {"user": PROFILE_ID, "hl": "en", "pagesize": 100}
        if start:
            params["cstart"] = start
        url = "https://scholar.google.com/citations?" + urlencode(params)
        batch, more = parse_profile(fetch(url))
        added = 0
        for paper in batch:
            identity = (paper["title"], paper["year"])
            if identity not in seen:
                publications.append(paper)
                seen.add(identity)
                added += 1
        if not more:
            return sorted(publications, key=lambda p: p["year"], reverse=True)
        if not added:
            raise ValueError("Scholar repeated a page; refusing to save a partial list.")
    raise ValueError("Scholar pagination limit reached; refusing to save a partial list.")


def markdown_text(value):
    value = escape(value, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>-])", r"\\\1", value)


def render_publications(papers):
    lines = ["<!-- Automatically refreshed from Google Scholar. Edit research/index.md for your introduction. -->", ""]
    current_year = None
    for paper in papers:
        year = paper["year"] or "Undated"
        if year != current_year:
            lines.extend([f"## {year}", ""])
            current_year = year
        lines.extend([
            f'### [{markdown_text(paper["title"])}]({paper["url"]})', "",
            markdown_text(paper["authors"]), "",
            markdown_text(paper["venue"]) or "Publication", "",
        ])
    return "\n".join(lines)


def sync(destination=DESTINATION, fetch=None):
    papers = fetch_publications(fetch)
    output = render_publications(papers)
    if destination.exists() and destination.read_text(encoding="utf-8") == output:
        print(f"Google Scholar: {len(papers)} publications, no changes.")
        return
    # Only replace the saved list after every page has been fetched and validated.
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(destination)
    print(f"Google Scholar: saved {len(papers)} publications.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cached", action="store_true", help="Keep building from the saved list if Scholar is unavailable.")
    args = parser.parse_args()
    try:
        sync()
    except Exception as error:
        if args.allow_cached and DESTINATION.exists() and "### [" in DESTINATION.read_text():
            print(f"::warning::Google Scholar refresh failed; keeping saved publications. {error}", file=sys.stderr)
        else:
            raise


if __name__ == "__main__":
    main()
