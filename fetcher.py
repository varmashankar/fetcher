#!/usr/bin/env python3
"""
fetcher.py
A powerful, concurrent web scraper and AI data processor with a rich command-line experience.
- Retries with exponential backoff.
- Multiple modes: save, extract links, extract with CSS selectors, or process with AI.
- Concurrent processing for high performance.
- Rich UI with progress bars and color-coded logging.
- Flexible input: command-line arguments, file, or piped from stdin.
- Gemini AI integration for summarization and intelligent data extraction.
"""

from typing import Optional, List, Tuple, Dict, Any
import argparse
import os
import sys
import logging
import re
import json
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from contextlib import suppress

# Rich for beautiful CLI UI
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.console import Console

# AI Integration
import google.generativeai as genai
import html2text

import time
import undetected_chromedriver as uc # type: ignore
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# --- Global Console Objects for Clean Output Separation ---
# Data goes to stdout (so it can be redirected, e.g., > out.txt)
console = Console()
# Logs and status messages go to stderr (so they appear on screen)
error_console = Console(stderr=True)

logger = logging.getLogger(__name__)


def make_session(user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                 retries: int = 3, backoff_factor: float = 0.3,
                 status_forcelist: Tuple[int, ...] = (500, 502, 503, 504)) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # This now sets a much more believable User-Agent
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Connection": "keep-alive"
    })
    return session



def fetch_text(session: requests.Session, url: str, timeout: int = 10) -> str:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def fetch_json(session: requests.Session, url: str, timeout: int = 10) -> Dict[str, Any]:
    resp = session.get(url, timeout=timeout, headers={"Accept": "application/json"})
    resp.raise_for_status()
    return resp.json()

def extract_links(html: str, base_url: Optional[str] = None) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href_val = a.get("href")
        if isinstance(href_val, str) and (href := href_val.strip()):
            if base_url:
                href = urljoin(base_url, href)
            links.append(href)
    return links

def save_bytes(path: str, data: bytes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def sanitize_filename(url: str) -> str:
    """Creates a safe and descriptive filename from a URL."""
    parsed_url = urlparse(url)
    filename = f"{parsed_url.netloc}{parsed_url.path.replace('/', '_')}"
    filename = re.sub(r'[\\/:*?."<>|]', '_', filename)
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    return filename[:150]



### --- Final, Upgraded fetch_text_with_browser function for Cloudflare ---

def fetch_text_with_browser(url: str, timeout: int = 20) -> str:
    """
    Fetches the full page source using undetected-chromedriver to bypass
    advanced bot detection systems like Cloudflare.
    
    This function handles:
    - Launching a stealth-configured browser.
    - Waiting for security checks to complete.
    - Clicking cookie consent banners.
    - Scrolling down to trigger lazy-loaded content.
    """
    # 1. Set up the browser options. Undetected-chromedriver handles most
    #    of the stealth configurations automatically.
    chrome_options = uc.options.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--start-maximized")

    # 2. Initialize the undetected-chromedriver driver.
    #    This is the key change that provides the anti-detection capabilities.
    driver = uc.Chrome(options=chrome_options, use_subprocess=True)

    # 3. Use a try...finally block to ensure the browser is always closed,
    #    even if errors occur.
    try:
        # Navigate to the URL.
        driver.get(url)
        
        # 4. Wait for Cloudflare or other initial checks to complete.
        logger.info(f"Navigated to {url}. Waiting for security checks...")
        time.sleep(5) # A simple but effective wait.

        # 5. Handle Cookie Banners if they appear after the security check.
        try:
            wait = WebDriverWait(driver, 5) # Wait up to 5 seconds
            accept_button = wait.until(
                EC.element_to_be_clickable((By.ID, "wcp-button-accept"))
            )
            accept_button.click()
            logger.info("Clicked cookie consent button.")
            time.sleep(2) # Wait for any overlays to disappear.
        except Exception:
            # It's okay if this fails; the banner might not be there.
            logger.debug("Cookie consent banner not found or not clickable.")
            pass

        # 6. Handle Lazy-Loading by scrolling down the page.
        logger.info("Scrolling down to load all dynamic content...")
        last_height = driver.execute_script("return document.body.scrollHeight") # type: ignore
        
        while True:
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # type: ignore

            # Wait for new content to load
            time.sleep(2)

            # Check if we've reached the bottom of the page
            new_height = driver.execute_script("return document.body.scrollHeight") # type: ignore
            if new_height == last_height:
                logger.info("Scrolling complete.")
                break
            last_height = new_height
        
        # 7. Get the final, complete page source.
        page_source = driver.page_source

        # 8. Final check to ensure we bypassed the security screen.
        if "Verify you are human" in page_source or "Checking if the site connection is secure" in page_source:
             raise Exception("Failed to bypass Cloudflare even with undetected-chromedriver.")

        return page_source

    finally:
        # 9. Always quit the browser to free up resources.
        with suppress(Exception):
            driver.quit()

### --- Worker Functions ---

def process_url(url: str, session: requests.Session, args: argparse.Namespace) -> str:
    """Worker function for non-AI tasks. Returns a status message."""
    output_dir = args.output_dir or "output"
    base_filename = sanitize_filename(url)
    
    if args.mode == "json":
        data = fetch_json(session, url, timeout=args.timeout)
        out_path = os.path.join(output_dir, f"{base_filename}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            import json
            json.dump(data, f, indent=2, ensure_ascii=False)
        return f"Saved JSON to [cyan]{out_path}[/]"
    else:
        text = fetch_text_with_browser(url, timeout=args.timeout)
        if args.mode == "links":
            links = extract_links(text, base_url=url)
            out_path = os.path.join(output_dir, f"{base_filename}_links.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(links))
            return f"Extracted {len(links)} links to [cyan]{out_path}[/]"
        elif args.mode == "save":
            out_path = os.path.join(output_dir, f"{base_filename}.html")
            save_bytes(out_path, text.encode("utf-8"))
            return f"Saved page to [cyan]{out_path}[/]"
        elif args.mode == "extract":
            text = fetch_text_with_browser(url, timeout=args.timeout) # Use browser for JS-heavy pages
            soup = BeautifulSoup(text, "html.parser")
            elements = soup.select(args.selector) # type: ignore

            # Get all the text from the found elements
            extracted_texts = [el.get_text(strip=True) for el in elements]

            if args.output_file:
                output_dir = os.path.dirname(args.output_file)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                with open(args.output_file, "a", encoding="utf-8") as f:
                    for line in extracted_texts:
                        f.write(line + "\n")
                return f"Appended {len(extracted_texts)} extracted element(s) to [cyan]{args.output_file}[/]"
            else:
                # If no file, print to console
                for line in extracted_texts:
                    console.print(line)
                return f"Extracted {len(extracted_texts)} element(s) with selector '[green]{args.selector}[/]' from {url}"
        else:  # auto or text mode
            console.print(Panel(text[:1200], title=f"Snippet from {url}", border_style="blue"))
            return f"Displayed snippet for {url}"

### --- AI Worker Function with Final Pylance Fix ---

def process_url_ai(url: str, session: requests.Session, args: argparse.Namespace) -> str:
    """Worker function for AI tasks using Gemini. Returns a status message."""
    # --- The first part of the function (fetching, cleaning, calling AI) remains IDENTICAL ---
    html_content = fetch_text_with_browser(url, timeout=args.timeout)
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    text_content = h.handle(html_content)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    genai.configure(api_key=api_key) # type: ignore 
    model = genai.GenerativeModel('gemini-1.5-flash-latest') # type: ignore
    
    system_prompt = "You are an expert data extraction assistant. Given the user prompt, you MUST return ONLY a valid, minified JSON object, with no markdown formatting or explanatory text."
    full_prompt = f"{system_prompt}\n\nUSER PROMPT: '{args.prompt}'\n\nWEBPAGE CONTENT:\n---\n{text_content[:20000]}"
    
    response = model.generate_content(full_prompt) # type: ignore

    if not response.parts:
        block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
        raise ValueError(f"AI response was blocked. Reason: {block_reason}")

    ai_result = response.text
    
    ### --- THIS IS THE MODIFIED SECTION --- ###
    
    # Check if the user wants to save the output to a file.
    if args.output_file:
        try:
            # 1. Parse the AI's string output into a Python dictionary
            parsed_json = json.loads(ai_result)

            # Create the directory for the output file if it doesn't exist
            output_dir = os.path.dirname(args.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 2. Write the Python dictionary to a file with formatting
            #    We use 'w' (write) mode to create a clean new file.
            #    `indent=4` is the key to pretty-printing.
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=4, ensure_ascii=False)
            
            # Return a new status message
            return f"Saved formatted JSON for {url} to [cyan]{args.output_file}[/]"

        except json.JSONDecodeError:
            # If the AI didn't return valid JSON, we log an error and save the raw text for debugging.
            logger.error(f"AI did not return valid JSON for {url}. Saving raw output for review.")
            raw_output_path = args.output_file + ".raw.txt"
            with open(raw_output_path, "w", encoding="utf-8") as f:
                f.write(ai_result)
            raise ValueError(f"AI response was not valid JSON. See {raw_output_path}")

    else:
        # If no output file, print the pretty panel to the console as before.
        console.print(Panel(ai_result, title=f"Gemini AI Result for {url}", border_style="magenta"))
        return f"Successfully processed with Gemini AI: '[yellow]{args.prompt}[/yellow]'"

### --- Main Orchestrator ---

def main():
    epilog_text = """
Examples:
  # Fetch multiple URLs concurrently and save to 'web_pages/' dir
  python fetcher.py --mode save --output-dir web_pages/ https://example.com https://www.python.org

  # Extract all <h2> titles from a list of URLs in a file and save them
  python fetcher.py --input-file urls.txt --mode extract --selector "h2" > titles.txt

  # Use AI to summarize an article
  python fetcher.py --mode ai --prompt "Summarize this article in 3 bullet points" <URL>

  # Get all links from a site and see verbose logging
  cat my_urls.txt | python fetcher.py --mode links --verbose
    """
    p = argparse.ArgumentParser(description=__doc__, epilog=epilog_text, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("urls", nargs="*", default=[], help="One or more URLs to fetch.")
    p.add_argument("--input-file", help="Path to a file containing URLs, one per line.")
    p.add_argument("--mode", choices=["auto", "text", "json", "links", "save", "extract", "ai"], default="auto", help="How to handle the response.")
    p.add_argument("--selector", help="CSS selector to use when mode is 'extract'.")
    p.add_argument("--prompt", help="Natural language prompt for --mode=ai.")
    p.add_argument("-o", "--output-dir", help="Output directory to save files (default: 'output/').")
    p.add_argument("--output-file", help="Path to a single file to save the output of AI or extract modes.")
    p.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds.")
    p.add_argument("--retries", type=int, default=3, help="Number of retry attempts.")
    p.add_argument("--concurrency", type=int, default=10, help="Number of concurrent download threads.")
    p.add_argument("--log-file", help="Path to a file to write logs.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (DEBUG level) logging.")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress informational logging.")
    args = p.parse_args()

    if args.mode == 'extract' and not args.selector:
        p.error("--selector is required when using --mode=extract")
    if args.mode == 'ai' and not args.prompt:
        p.error("--prompt is required when using --mode=ai")

    log_level = "INFO"
    if args.verbose:
        log_level = "DEBUG"
    
    logging.basicConfig(level=log_level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(console=error_console, rich_tracebacks=True, show_path=args.verbose)] if not args.quiet else [])
    logger = logging.getLogger(__name__)

    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(fh)
        
    if not args.verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    urls_to_process = list(args.urls)
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                urls_to_process.extend([line.strip() for line in f if line.strip()])
        except FileNotFoundError:
            logger.critical(f"Input file not found: {args.input_file}"); sys.exit(1)
    
    if not sys.stdin.isatty():
        urls_to_process.extend([line.strip() for line in sys.stdin if line.strip()])

    if not urls_to_process:
        p.error("No URLs provided.")

    urls_to_process = sorted(list(set(urls_to_process)))

    session = make_session(retries=args.retries)
    worker_func = partial(process_url_ai if args.mode == "ai" else process_url, session=session, args=args)
    
    success_count, fail_count = 0, 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), console=error_console, transient=True) as progress:
        task = progress.add_task("[green]Processing URLs...", total=len(urls_to_process))
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(worker_func, url): url for url in urls_to_process}
            for future in futures:
                try:
                    result_message = future.result()
                    logger.info(result_message, extra={"markup": True})
                    success_count += 1
                except Exception as e:
                    logger.error(e, extra={"markup": True})
                    fail_count += 1
                progress.update(task, advance=1)

    error_console.print(Panel(f"[bold green]Successful: {success_count}[/]\n[bold red]Failed: {fail_count}[/]", title="Processing Complete", border_style="blue"))

if __name__ == "__main__":
    main()