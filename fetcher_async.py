#!/usr/bin/env python3
# fetcher_async_typed_fixed.py — typed async fetcher (aiohttp + BeautifulSoup)
from typing import List, Tuple, Optional
import asyncio
import argparse
import os

import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List


async def fetch(session: ClientSession, url: str, timeout: int = 10) -> Tuple[str, Optional[str], Optional[Exception]]:
    """Return (url, content or None, error or None)."""
    try:
        timeout_obj = ClientTimeout(total=timeout)
        async with session.get(url, timeout=timeout_obj) as resp:
            resp.raise_for_status()
            content = await resp.text()
            return (url, content, None)
    except Exception as e:
        return (url, None, e)


def extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    
    # THE FIX: Explicitly declare the type of the list.
    links: List[str] = []
    
    # Iterate through all <a> tags that have an 'href' attribute
    for a in soup.find_all("a", href=True):
        href_val = a.get("href")
        
        # This check confirms href_val is a string
        if isinstance(href_val, str):
            links.append(urljoin(base_url, href_val))
            
    return links


async def bounded_fetch(sem: asyncio.Semaphore, session: ClientSession, url: str, timeout: int):
    async with sem:
        return await fetch(session, url, timeout)


async def run(urls: List[str], concurrency: int, timeout: int, output_dir: str):
    sem = asyncio.Semaphore(concurrency)
    headers = {"User-Agent": "AsyncFetcher/1.0"}
    connector = TCPConnector(limit_per_host=concurrency)
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks = [bounded_fetch(sem, session, u, timeout) for u in urls]
        results = await asyncio.gather(*tasks)

    os.makedirs(output_dir, exist_ok=True)
    for i, (url, content, err) in enumerate(results, start=1):
        if err:
            print(f"[ERR] {url} -> {err}")
            continue

        # Defensive type check: Pylance now sees content is tested before use
        if content is None:
            print(f"[WARN] {url} -> no content")
            continue

        fname = os.path.join(output_dir, f"page_{i}.html")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(content)

        # content is str (guard above) so no Optional[str] leak into extract_links
        links = extract_links(content, url)
        with open(os.path.join(output_dir, f"links_{i}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(links))
        print(f"[OK] {url} -> saved {fname} (+{len(links)} links)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("urls", nargs="+")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--out", default="out_async")
    args = p.parse_args()
    asyncio.run(run(args.urls, args.concurrency, args.timeout, args.out))


if __name__ == "__main__":
    main()
