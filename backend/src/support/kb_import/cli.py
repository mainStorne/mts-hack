"""`support-kb crawl` loads support.mts.ru into the knowledge base; `support-kb analyse` finds what customers must provide and
groups the articles."""

import argparse
import asyncio
import logging
from pathlib import Path

from support import retrieval
from support.db import init, kb, pg
from support.kb_import import clusters, mts


def main() -> None:
    parser = argparse.ArgumentParser(prog="support-kb")
    sub = parser.add_subparsers(dest="command", required=True)
    crawl = sub.add_parser("crawl", help="crawl the site into kb_articles and embed new articles")
    crawl.add_argument("--refresh", action="store_true", help="ignore the page cache")
    crawl.add_argument("--delay", type=float, default=0.3)
    analyse = sub.add_parser("analyse", help="required inputs per article and a report of article clusters")
    analyse.add_argument("--clusters", type=int, default=8)
    analyse.add_argument("--min-share", type=float, default=0.25)
    analyse.add_argument("--min-size", type=int, default=3)
    analyse.add_argument("--parallel", type=int, default=4)
    analyse.add_argument("--refresh-inputs", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    init.init()
    try:
        if args.command == "crawl":
            mts.delay = args.delay
            rows, failed = mts.crawl(refresh=args.refresh)
            kb.upsert_site(rows)
            if failed:
                print(f"kept old articles: {len(failed)} pages failed")
            else:
                kb.delete_site_except(mts.SOURCE, {r["id"] for r in rows})
            embedded = retrieval.sync()
            faq = sum(r["kind"] == "faq" for r in rows)
            print(f"articles: {len(rows) - faq}, faq: {faq}, failed: {len(failed)}, embedded: {embedded}")
        else:
            text = asyncio.run(
                clusters.analyse_site(
                    k=args.clusters,
                    min_share=args.min_share,
                    min_size=args.min_size,
                    parallel=args.parallel,
                    refresh_inputs=args.refresh_inputs,
                )
            )
            print(text)
            out = Path("data/topics_report.md")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
    finally:
        pg.close()


if __name__ == "__main__":
    main()
