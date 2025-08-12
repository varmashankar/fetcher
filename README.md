# Advanced Web Scraper & AI Processor

A powerful Python command-line tool for high-performance, concurrent web scraping, data extraction, and AI-powered analysis using Google's Gemini.

This tool is designed to handle modern web scraping challenges, from simple HTML pages to complex, JavaScript-heavy sites protected by advanced bot detection systems like Cloudflare.

---

## Features

- **Concurrent Fetching:** Uses a thread pool to process many URLs simultaneously, dramatically speeding up large jobs.
- **Multiple Modes:**
  - `save`: Save the full HTML content of webpages.
  - `links`: Extract all hyperlinks from a page.
  - `extract`: Scrape specific data using CSS selectors.
  - `ai`: Use Google's Gemini model to understand, summarize, or extract structured data from a page in natural language.
- **Advanced Scraping Engine:** Employs `undetected-chromedriver` to bypass sophisticated bot detection and CAPTCHA challenges that block standard scrapers.
- **Rich CLI Experience:** Features a live progress bar, color-coded logging, and beautifully formatted output panels powered by the `rich` library.
- **Flexible Input:** Accepts URLs directly from the command line, from a text file (`--input-file`), or piped from standard input (`stdin`).
- **Robust Error Handling:** Automatically retries on transient network errors and handles failures for individual URLs gracefully without crashing the entire process.
- **Professional Logging:** Configurable verbosity (`--verbose`, `--quiet`) and the ability to save detailed logs to a file (`--log-file`).

---

## Setup & Installation

Follow these steps to get the script running.

### 1. Prerequisites

- Python 3.12+ (due to dependencies).
- Google Chrome browser installed.

### 2. Clone or Download

Download the `fetcher.py` script to a local directory.

### 3. Create a Virtual Environment (Recommended)

It's best practice to keep project dependencies isolated.

```bash
python -m venv .venv
# Activate the environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate
```

### 4. Install Dependencies

Create a file named `requirements.txt` in your project directory with the following content:

```
requests
beautifulsoup4
rich
google-generativeai
html2text
selenium
webdriver-manager
undetected-chromedriver
types-beautifulsoup4
setuptools
```

Then, install all the required libraries by running:

```bash
pip install -r requirements.txt
```

### 5. Set Up Your Gemini AI API Key

For any AI-related tasks, you must provide your Google AI API key. The most secure way is to set it as an environment variable.

On Windows (PowerShell):

```powershell
$env:GOOGLE_API_KEY="your-api-key-here"
```

On macOS/Linux:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

**Important:** This key is only set for your current terminal session. You will need to set it again if you open a new terminal.

---

## Usage

The script is run from the command line with various arguments to control its behavior.

### Basic Command Structure

```bash
python fetcher.py --mode <mode> [OPTIONS] [URLS...]
```

### Example 1: Saving Webpages

Download multiple pages concurrently into a directory named `saved_pages`.

```bash
python fetcher.py --mode save --output-dir saved_pages/ https://example.com https://www.python.org
```

### Example 2: Extracting Specific Data with CSS Selectors

Extract all `<h2>` headings from a list of URLs in a file and save them to `headings.txt`.

```bash
# First, create urls.txt with one URL per line.
# Then, run:
python fetcher.py --mode extract --selector "h2" --input-file urls.txt --output-file headings.txt
```

### Example 3: AI-Powered Summarization

Use Gemini AI to summarize a news article. The result will be printed to the console in a decorative panel.

```bash
python fetcher.py --mode ai --prompt "Summarize this article in three key bullet points." "https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/"
```

### Example 4: Advanced AI Extraction to a File

Scrape a tough, JavaScript-heavy product page, extract structured data as JSON, and save it to a file.

```bash
python fetcher.py --mode ai --prompt "Extract the product name, price, and brand. Return as a valid JSON object with keys 'name', 'price', and 'brand'." --output-file product_data.json "https://www.bhphotovideo.com/c/product/1745607-REG/apple_z170_mbp14_19_b_h_14_2_macbook_pro_with.html"
```

### Example 5: Combining Options

Process a large list of URLs from a file with high concurrency, running silently while logging all activity to a file.

```bash
python fetcher.py --input-file big_list.txt --mode links --concurrency 20 --quiet --log-file fetch.log
```

---

## Troubleshooting

- **ModuleNotFoundError: No module named 'distutils':**  
  This means you are using Python 3.12+ and `setuptools` is missing. Fix it by running:

  ```bash
  pip install setuptools
  ```

- **ERROR: GOOGLE_API_KEY environment variable not set:**  
  Your AI API key is not set in the current terminal session. Rerun the export or `$env:` command to set it.

- **ERROR: 403 Client Error: Forbidden or Failed to bypass Cloudflare:**  
  The target website has strong bot protection. The script is designed to handle this with `undetected-chromedriver`, but ensure your libraries are up to date:

  ```bash
  pip install --upgrade undetected-chromedriver
  ```

  This is the most complex part of web scraping, and some sites may still be inaccessible.

---

## License

MIT License © 2024
