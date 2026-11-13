"""Crawler of https://support.mts.ru/: every page is Next.js SSR, its data is the JSON in <script id="__NEXT_DATA__">.

Home page → products of the groups in GROUPS → sections and articles of a product → article pages. An article becomes one row,
and every «Часто спрашивают» entry of it another one (kind='faq'). Pages are cached in CACHE, so a re-crawl needs no network.
"""

import json
import logging
import re
import time
import urllib.robotparser
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from html_to_markdown import ConversionOptions, convert

log = logging.getLogger(__name__)

BASE = "https://support.mts.ru"
SOURCE = "support.mts.ru"
GROUPS = ("mts_mobile", "mts_home")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) support-agent-kb/0.1"
CACHE = Path("data/mts")

delay = 0.3  # seconds before every request to the site; set by the CLI


def page_props(client: httpx.Client, path: str, *, refresh: bool) -> dict:
    """pageProps of a page, from the cache or the site."""
    cached = CACHE / ((path.strip("/").replace("/", "__") or "index") + ".json")
    if cached.exists() and not refresh:
        return json.loads(cached.read_text(encoding="utf-8"))
    time.sleep(delay)
    response = client.get(BASE + path)
    response.raise_for_status()
    found = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', response.text, re.DOTALL)
    if not found:
        raise ValueError(f"no __NEXT_DATA__ on {path}")
    props = json.loads(found.group(1))["props"]["pageProps"]
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(props, ensure_ascii=False), encoding="utf-8")
    return props


def md(html: str, url: str) -> str:
    options = ConversionOptions(
        base_url=url, extract_images=False, bullets="-", autolinks=False, exclude_selectors=["figure", "img"]
    )
    return convert(html, options).content.strip()


def article_rows(props: dict, url: str, popular_ids: set[int]) -> list[dict]:
    """[the article, *its faq entries] of an article page; rows without a title or text are dropped."""
    art = props["articleData"]
    common = {
        "source": SOURCE,
        "product": props["productData"]["name"],
        "section": props["sectionData"]["name"],
        "modified_at": art.get("lastModifiedAt"),
    }
    soup = BeautifulSoup(art["content"], "html.parser")
    faq = []
    for index, div in enumerate(soup.select("div.small-accordion")):
        header = div.select_one(".small-accordion-header")
        content = div.select_one(".small-accordion-content")
        if header is None or content is None:
            continue
        anchor = div.get("id")
        faq.append(
            {
                **common,
                "id": f"mts-{art['id']}-faq-{anchor or index}",
                "kind": "faq",
                "parent_id": f"mts-{art['id']}",
                "popular": False,
                "url": f"{url}#{anchor}" if anchor else url,
                "title": header.get_text(" ", strip=True),
                "text": md(str(content), url),
            }
        )
    for div in soup.select("div.small-accordion"):
        div.decompose()
    heading = soup.select_one("h2#chasto-sprashivayut")
    if heading is not None:
        heading.decompose()
    article = {
        **common,
        "id": f"mts-{art['id']}",
        "kind": "article",
        "parent_id": None,
        "popular": art["id"] in popular_ids,
        "url": url,
        "title": art["name"],
        "text": md(str(soup), url),
    }
    return [r for r in [article, *faq] if r["text"].strip() and r["title"].strip()]


def crawl(*, refresh: bool) -> tuple[list[dict], list[str]]:
    """(rows, paths that failed). Failing home or product page raises: the article list would be incomplete."""
    robots = urllib.robotparser.RobotFileParser(BASE + "/robots.txt")
    robots.read()
    rows: list[dict] = []
    failed: list[str] = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True) as client:
        home = page_props(client, "/", refresh=refresh)
        products = [
            p["sefUrl"] for group in home["segmentData"]["groups"] if group["code"] in GROUPS for p in group["products"]
        ]
        for product in products:
            try:
                props = page_props(client, f"/{product}", refresh=refresh)
            except (httpx.HTTPError, ValueError) as e:
                raise RuntimeError(f"product page /{product} failed: {e}") from e
            popular_ids = {q["id"] for q in props.get("popularQuestions") or []}
            for section in props["productData"]["sections"]:
                for article in section["articles"]:
                    if article.get("hiddenForSeo") is not False:
                        continue
                    path = f"/{product}/{section['sefUrl']}/{article['sefUrl']}"
                    if not robots.can_fetch(USER_AGENT, BASE + path):
                        continue
                    try:
                        rows += article_rows(page_props(client, path, refresh=refresh), BASE + path, popular_ids)
                    except (httpx.HTTPError, ValueError) as e:
                        log.warning("%s: %s", path, e)
                        failed.append(path)
    return rows, failed
