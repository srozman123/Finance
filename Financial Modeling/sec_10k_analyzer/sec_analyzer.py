"""
SEC 10-K Analyzer — fetch and analyze company annual filings from SEC EDGAR.

Data sources used:
  - SEC EDGAR Submissions API    : filing metadata (accession numbers, dates)
  - SEC EDGAR XBRL Company Facts : structured, machine-readable financial data
  - SEC EDGAR filing HTML        : Item 1 (business description) plain-text extract

Why XBRL over HTML table parsing?
  Every 10-K filer must tag financial data with standardized US-GAAP taxonomy labels
  (e.g. "NetIncomeLoss"). By reading these tags directly from the XBRL API we skip
  fragile table parsing and handle the vast majority of S&P 500 filings without
  company-specific configuration. The trade-off: some smaller filers or older filings
  may not have XBRL data, in which case we surface a clear error rather than
  silently returning bad numbers.
"""

import sys
import json
import re
import time
import csv
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

console = Console()

# SEC EDGAR requires a descriptive User-Agent header. Requests without one are
# rejected. Format: "<tool name> <contact email>"
EDGAR_HEADERS = {
    "User-Agent": "FinancialModelingTool contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

SEC_BASE        = "https://data.sec.gov"
TICKERS_URL     = "https://www.sec.gov/files/company_tickers.json"


# =============================================================================
# SECTION 1: CIK Lookup
# =============================================================================
# The SEC identifies every registrant by a unique Central Index Key (CIK).
# Before querying any filing data, a ticker symbol must be converted to its CIK.
# The SEC publishes a complete ticker → CIK map as a public JSON file that is
# updated daily. We download it once per run and search it in memory.

def get_cik(ticker: str) -> tuple[str, str]:
    """
    Convert a stock ticker to its SEC CIK number and company name.

    Returns (cik_padded, company_name) where cik_padded is zero-padded to
    10 digits — the format required by all EDGAR API endpoints.
    Raises ValueError if the ticker is not found in the SEC database.
    """
    resp = requests.get(TICKERS_URL, headers=EDGAR_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    ticker_upper = ticker.upper()
    for _, entry in data.items():
        if entry["ticker"].upper() == ticker_upper:
            cik = str(entry["cik_str"])
            name = entry["title"]
            return cik.zfill(10), name

    raise ValueError(
        f"Ticker '{ticker}' not found in SEC EDGAR database. "
        "Check spelling or try the full company name at https://efts.sec.gov"
    )


# =============================================================================
# SECTION 2: 10-K Filing Metadata
# =============================================================================
# After obtaining a CIK, we fetch the company's full submission history from
# the EDGAR Submissions API. This returns metadata for every filing ever made —
# including form type, accession number (unique filing ID), and filing date.
#
# We scan this list for form type "10-K" (annual report) and pick the filing
# that matches the requested fiscal year. If no year is specified, we use the
# most recent 10-K available.
#
# Accession number format: "0000320193-23-000064"  (issuer CIK + year + seq)
# When used in URLs the dashes are removed: "000032019323000064"

def fetch_filing_metadata(cik: str, target_year: int = None) -> dict:
    """
    Fetch 10-K filing metadata for the target fiscal year.

    Returns a dict with keys: accession, date, year, primary_doc (URL to the
    primary 10-K HTML document, if available in the submissions API).
    If target_year is None, returns the most recently filed 10-K.
    """
    url = f"{SEC_BASE}/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    filings  = data.get("filings", {}).get("recent", {})
    forms    = filings.get("form", [])
    accnos   = filings.get("accessionNumber", [])
    dates    = filings.get("filingDate", [])
    # The submissions API also includes the primary document filename for each filing
    primary_docs = filings.get("primaryDocument", [None] * len(forms))

    cik_int = int(cik)
    ten_ks = []
    for i, form in enumerate(forms):
        if form == "10-K":
            acc   = accnos[i]
            pdoc  = primary_docs[i] if i < len(primary_docs) else None
            # Build the primary document URL directly from the accession path
            primary_url = None
            if pdoc:
                acc_nodash  = acc.replace("-", "")
                primary_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_int}/{acc_nodash}/{pdoc}"
                )
            ten_ks.append({
                "accession":   acc,
                "date":        dates[i],
                "year":        int(dates[i][:4]),
                "primary_doc": primary_url,
            })

    if not ten_ks:
        raise ValueError(f"No 10-K filings found for CIK {cik}.")

    if target_year:
        matches = [f for f in ten_ks if f["year"] in (target_year, target_year + 1)]
        if not matches:
            raise ValueError(
                f"No 10-K found for fiscal year {target_year}. "
                f"Available years: {sorted({f['year'] for f in ten_ks}, reverse=True)}"
            )
        return sorted(matches, key=lambda x: x["date"], reverse=True)[0]

    return ten_ks[0]


def fetch_filing_index(cik: str, accession: str) -> list[dict]:
    """
    Fetch the list of documents contained in a single 10-K submission.

    Returns a list of dicts with keys: name, type, href.
    The primary 10-K document is typically the first .htm file with type '10-K'.

    EDGAR hosts the index JSON on both data.sec.gov and www.sec.gov depending on
    filing vintage; we try data.sec.gov first then fall back to www.sec.gov.
    """
    acc_nodash = accession.replace("-", "")
    cik_int    = int(cik)

    # Try both EDGAR hostnames — some filings are only accessible on one
    candidates = [
        f"https://data.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{accession}-index.json",
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{accession}-index.json",
    ]

    last_exc = None
    for url in candidates:
        try:
            resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            base = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/"
            return [
                {
                    "name": doc.get("name", ""),
                    "type": doc.get("type", ""),
                    "href": base + doc.get("name", ""),
                }
                for doc in data.get("directory", {}).get("item", [])
            ]
        except requests.HTTPError as exc:
            last_exc = exc

    raise last_exc


def find_10k_document(documents: list[dict]) -> str:
    """
    Identify the primary 10-K HTML document URL from the filing index.

    Strategy:
      1. Any document explicitly typed '10-K' with an .htm extension.
      2. Fallback: the first .htm file that does not look like an exhibit
         or a taxonomy file (xsd, cal, def, lab, pre).
    """
    for doc in documents:
        if doc["type"] == "10-K" and doc["name"].lower().endswith((".htm", ".html")):
            return doc["href"]

    for doc in documents:
        n = doc["name"].lower()
        if n.endswith((".htm", ".html")) and not any(
            x in n for x in ["ex", "exhibit", "xsd", "cal", "def", "lab", "pre", "r9999"]
        ):
            return doc["href"]

    raise ValueError("Could not locate primary 10-K HTML document in filing index.")


# =============================================================================
# SECTION 3: XBRL Financial Data Extraction
# =============================================================================
# The SEC's XBRL Company Facts API (data.sec.gov/api/xbrl/companyfacts/) returns
# every tagged financial value a company has ever reported, organised by US-GAAP
# taxonomy concept name. This is the most reliable way to extract income statement
# figures because it bypasses HTML layout entirely.
#
# Challenge: companies choose from multiple valid GAAP tag names for the same
# economic concept. For example, "total revenue" might be tagged as:
#   - Revenues
#   - RevenueFromContractWithCustomerExcludingAssessedTax   (post-ASC 606)
#   - SalesRevenueNet   (older filings)
# We handle this by trying a ranked list of tag alternatives per line item.
#
# Annual vs quarterly data: the same tag can appear in both 10-K and 10-Q
# filings. We filter to form="10-K" entries and then require the reporting
# period to span roughly one full year (~330–380 days) to exclude cumulative
# YTD figures that show up in some quarterly filings.

# Ranked tag alternatives for each income statement line item.
# Order matters: more common / modern tags come first.
XBRL_TAGS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
        "NetRevenues",
        "TotalRevenuesAndOtherIncome",
    ],
    "cogs": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfServices",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ],
    "rd_expense": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "sga_expense": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ],
}


def fetch_xbrl_facts(cik: str) -> dict:
    """
    Download all XBRL-tagged financial facts for a company from SEC EDGAR.
    Returns the raw JSON dict from the companyfacts API.
    """
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def extract_annual_values(
    facts: dict, tag_list: list[str], years: list[int]
) -> dict[int, float]:
    """
    Extract annual values for a given list of XBRL tag alternatives.

    Tries each tag in order; returns the first tag that has data.
    Returns {fiscal_year: value_in_dollars} for any years found.

    Filtering logic:
      - form must be "10-K" or "10-K/A"
      - period length must be 330–380 days (full fiscal year, not a quarter)
      - if multiple entries exist for the same year, keep the longest-period one
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for tag in tag_list:
        if tag not in us_gaap:
            continue

        entries = us_gaap[tag].get("units", {}).get("USD", [])
        if not entries:
            continue

        annual: dict[int, tuple[float, int]] = {}   # year → (value, period_days)
        for e in entries:
            if e.get("form") not in ("10-K", "10-K/A"):
                continue
            if "start" not in e or "end" not in e:
                continue

            end_year = int(e["end"][:4])
            if end_year not in years:
                continue

            start_dt = datetime.strptime(e["start"], "%Y-%m-%d")
            end_dt   = datetime.strptime(e["end"],   "%Y-%m-%d")
            days     = (end_dt - start_dt).days

            # Accept entries that span a full fiscal year
            if 330 <= days <= 380:
                if end_year not in annual or days > annual[end_year][1]:
                    annual[end_year] = (e["val"], days)

        if annual:
            return {yr: val for yr, (val, _) in annual.items()}

    return {}   # no data found for any tag variant


def get_financial_data(cik: str, years: list[int]) -> dict:
    """
    Pull structured income statement data from the XBRL API for the given years.

    Returns a dict keyed by metric name, each value a {year: float} dict.
    Gross profit is derived from revenue − COGS when not directly tagged.
    """
    facts = fetch_xbrl_facts(cik)

    result = {}
    for metric, tags in XBRL_TAGS.items():
        result[metric] = extract_annual_values(facts, tags, years)

    # Derive gross profit from revenue − COGS if not directly reported
    if not result["gross_profit"] and result["revenue"] and result["cogs"]:
        result["gross_profit"] = {
            yr: result["revenue"][yr] - result["cogs"][yr]
            for yr in result["revenue"]
            if yr in result["cogs"]
        }

    return result


# =============================================================================
# SECTION 4: Item 1 (Business Description) Extraction
# =============================================================================
# Item 1 of a 10-K is the company's formal, self-authored business overview.
# It covers products/services, operating segments, competition, customers,
# distribution channels, and regulation. Reading it in conjunction with the
# financial metrics gives a qualitative picture that numbers alone can't provide.
#
# Extraction strategy:
#   1. Fetch the primary 10-K HTML document.
#   2. Use BeautifulSoup to navigate the HTML tree and locate the Item 1 region.
#   3. Within that region, identify subsection headings by looking for:
#        - <b> / <strong> elements with short text (likely section titles)
#        - <h2> / <h3> / <h4> heading tags
#        - Short standalone lines that match known section keywords
#   4. Group paragraphs under each detected heading.
#   5. Return a structured dict of {title, text} sections plus a raw fallback.
#
# Why structure matters: displaying a 10,000-character wall of text is not useful.
# Splitting by subsection lets the analyst jump directly to what they need
# (competition, customers, segments) without reading everything linearly.

# Keywords that indicate a named subsection in Item 1.
# Used to distinguish real section headers from inline bold text.
_SECTION_KEYWORDS = {
    "overview", "background", "general", "company overview",
    "products", "product", "services", "service", "solutions",
    "segments", "business segments", "operating segments", "platforms",
    "competition", "competitive", "competitors", "competitive landscape",
    "customers", "customer", "distribution", "channels", "sales channel",
    "employees", "human capital", "workforce", "people", "talent",
    "geographic", "geography", "international", "global", "regions",
    "seasonality", "seasonal",
    "regulation", "regulatory", "government regulation", "legal",
    "research", "development", "r&d", "innovation", "technology",
    "intellectual property", "patents", "trademarks", "proprietary",
    "properties", "facilities", "manufacturing",
    "supply chain", "suppliers", "procurement",
    "strategy", "growth strategy", "business model",
    "available information", "sec filings",
}


def fetch_filing_html(url: str) -> str:
    """Fetch the raw HTML of a 10-K filing document."""
    resp = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def _is_heading_line(text: str) -> bool:
    """
    Heuristic: return True if a short line of text looks like a section heading.
    A heading is typically 1–6 words, matches a known keyword, and is not a
    sentence (no period at the end, not overly long).
    """
    t = text.strip()
    if not t or len(t) > 80 or t.endswith(".") or t.endswith(","):
        return False
    word_count = len(t.split())
    if word_count > 7:
        return False
    return any(kw in t.lower() for kw in _SECTION_KEYWORDS)


def extract_item_1(html: str) -> dict:
    """
    Extract and structure Item 1 content from a 10-K filing HTML document.

    Returns a dict:
      {
        "sections": [{"title": str, "text": str}, ...],   # named subsections
        "raw":      str,                                   # full Item 1 text
      }

    The 'sections' list is populated by scanning for bold/heading elements
    within the Item 1 HTML region that match known subsection keywords.
    If no subsections are found, 'sections' will contain a single entry
    with title "Business Overview" and the full text as the body.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove noise: scripts, styles, inline XBRL metadata tags
    for tag in soup(["script", "style", "ix:header", "ix:nonfraction", "ix:nonNumeric"]):
        tag.decompose()

    # ── Step 1: Isolate the Item 1 region ────────────────────────────────────
    # Use newline-separated text so paragraph/heading structure is preserved.
    # 10-K documents contain a Table of Contents at the top with a short entry
    # like "Item 1. Business ... 3" — the regex will match that first. We use
    # findall and take the longest match to skip the TOC and land on the real
    # body section, which is orders of magnitude longer.
    line_text = soup.get_text(separator="\n", strip=True)

    boundary = re.compile(
        r"Item\s+1[\.\s]+Business\b(.*?)(?=Item\s+1A|Item\s+2\b|ITEM\s+1A|ITEM\s+2\b)",
        re.IGNORECASE | re.DOTALL,
    )
    all_matches = boundary.findall(line_text)
    if all_matches:
        # Take the longest match — that's the actual body, not the TOC entry
        item1_raw = max(all_matches, key=len)
        item1_raw = re.sub(r"\n{3,}", "\n\n", item1_raw).strip()
    else:
        item1_raw = line_text[:25000]

    # ── Step 2: Split into lines and group under detected headings ───────────
    lines = [l.strip() for l in item1_raw.split("\n") if l.strip()]

    sections: list[dict] = []
    current_title = "Business Overview"
    current_body:  list[str] = []

    for line in lines:
        if _is_heading_line(line):
            # Save the section we've been accumulating
            body = " ".join(current_body).strip()
            # Clean up repeated whitespace and registry marks
            body = re.sub(r"\s+", " ", body)
            if body:
                sections.append({"title": current_title, "text": body})
            current_title = line
            current_body  = []
        else:
            current_body.append(line)

    # Flush the final section
    body = re.sub(r"\s+", " ", " ".join(current_body)).strip()
    if body:
        sections.append({"title": current_title, "text": body})

    # ── Step 3: Merge tiny sections into their predecessor ───────────────────
    # Sections shorter than 80 chars are likely mis-detected headings; fold them
    # back into the previous section so we don't show stub entries.
    merged: list[dict] = []
    for sec in sections:
        if merged and len(sec["text"]) < 80:
            merged[-1]["text"] += "  " + sec["title"] + ": " + sec["text"]
        else:
            merged.append(sec)

    # ── Fallback: if no structure was found, return the raw text as one block ─
    if not merged:
        raw_clean = re.sub(r"\s+", " ", item1_raw).strip()
        merged = [{"title": "Business Overview", "text": raw_clean}]

    return {
        "sections": merged,
        "raw":      re.sub(r"\s+", " ", item1_raw).strip(),
    }


# =============================================================================
# SECTION 5: Margin Calculations
# =============================================================================
# Profit margins express how much of each dollar of revenue the company keeps
# at different stages of the income statement — they are the primary lens
# through which profitability is compared across companies and over time.
#
#   Gross Margin     = (Revenue − COGS) / Revenue
#     → What fraction of revenue remains after paying to produce the goods/services.
#     → High gross margins signal pricing power or a capital-light business model.
#
#   Operating Margin = Operating Income / Revenue
#     → What fraction remains after all operating costs (COGS + R&D + SG&A).
#     → Measures how well management controls costs relative to revenue.
#
#   Net Margin       = Net Income / Revenue
#     → The true "bottom line" — what fraction reaches shareholders after
#       interest, taxes, and any non-operating items.
#     → Divergence between operating and net margin often signals debt load
#       or tax strategy changes worth investigating.

def calculate_margins(
    revenue: float,
    cogs: float | None,
    op_income: float | None,
    net_income: float | None,
) -> dict:
    """
    Calculate the three core profitability margins for a single fiscal year.
    All inputs are raw dollar amounts (same unit). Returns percentages (0–100).
    Returns None for any margin where a required input is missing.
    """
    if not revenue or revenue == 0:
        return {"gross_margin": None, "operating_margin": None, "net_margin": None}

    gross_profit = (revenue - cogs) if cogs is not None else None

    return {
        "gross_margin":     (gross_profit / revenue * 100) if gross_profit is not None else None,
        "operating_margin": (op_income    / revenue * 100) if op_income is not None else None,
        "net_margin":       (net_income   / revenue * 100) if net_income is not None else None,
    }


def trend_arrow(values: list[float]) -> str:
    """
    Summarise the direction of a metric over time with a single character.
    Expects values ordered oldest → newest.
    ↑ = >1% improvement  |  ↓ = >1% decline  |  → = stable
    """
    if len(values) < 2 or values[0] == 0:
        return "→"
    pct_change = (values[-1] - values[0]) / abs(values[0]) * 100
    if pct_change > 1:
        return "↑"
    elif pct_change < -1:
        return "↓"
    return "→"


# =============================================================================
# SECTION 6: Full Analysis Pipeline
# =============================================================================
# analyze_company() is the single entry point that orchestrates every step:
#   1. Resolve ticker → CIK
#   2. Find the appropriate 10-K filing
#   3. Pull XBRL-tagged financial data for the 3 most recent fiscal years
#   4. Fetch the 10-K HTML and extract the Item 1 business description
#   5. Calculate margins and package everything into a clean result dict

def analyze_company(ticker: str, target_year: int = None) -> dict:
    """
    Run the full SEC 10-K analysis pipeline for a single ticker.

    Returns a structured dict containing company metadata, per-year financials,
    calculated margins, and the Item 1 business description excerpt.
    """
    console.print(f"[dim]  Resolving ticker → CIK...[/dim]")
    cik, company_name = get_cik(ticker)
    console.print(f"[dim]  Found: {company_name}  (CIK: {int(cik)})[/dim]")

    console.print(f"[dim]  Fetching 10-K filing index...[/dim]")
    filing = fetch_filing_metadata(cik, target_year)

    # Build the range of fiscal years to request from the XBRL API.
    # We ask for base_year and the three years prior to capture 3 years of data.
    base_year    = filing["year"]
    years_to_fetch = list(range(base_year - 4, base_year + 1))

    console.print(f"[dim]  Downloading XBRL financial data...[/dim]")
    financials = get_financial_data(cik, years_to_fetch)

    # Keep only the 3 most recent years that have revenue data
    revenue_years = sorted(financials["revenue"].keys(), reverse=True)[:3]
    if not revenue_years:
        console.print(
            f"[yellow]  Warning: No XBRL financial data found for {ticker}. "
            "Displaying Item 1 (Business Description) only.[/yellow]"
        )

    # Build a per-year summary dict
    yearly = {}
    for yr in revenue_years:
        rev  = financials["revenue"].get(yr)
        cogs = financials["cogs"].get(yr)
        gp   = financials["gross_profit"].get(yr) or ((rev - cogs) if rev and cogs else None)
        op   = financials["operating_income"].get(yr)
        ni   = financials["net_income"].get(yr)

        yearly[yr] = {
            "revenue":          rev,
            "cogs":             cogs,
            "gross_profit":     gp,
            "operating_income": op,
            "net_income":       ni,
            "rd":               financials["rd_expense"].get(yr),
            "sga":              financials["sga_expense"].get(yr),
            "margins":          calculate_margins(rev or 0, cogs, op, ni),
        }

    # Fetch and parse Item 1 from the actual filing document.
    # Strategy: use the primary_doc URL embedded in the submissions API first
    # (most reliable); fall back to walking the filing index JSON if that fails.
    console.print(f"[dim]  Fetching 10-K document for Item 1...[/dim]")
    item1: dict = {"sections": [], "raw": ""}
    try:
        doc_url = filing.get("primary_doc")
        if not doc_url:
            documents = fetch_filing_index(cik, filing["accession"])
            doc_url   = find_10k_document(documents)
        time.sleep(0.15)   # polite delay per SEC crawling guidelines
        html    = fetch_filing_html(doc_url)
        item1   = extract_item_1(html)
    except Exception as exc:
        item1 = {
            "sections": [{"title": "Error", "text": f"Item 1 unavailable: {exc}"}],
            "raw": "",
        }

    return {
        "ticker":           ticker.upper(),
        "company_name":     company_name,
        "cik":              int(cik),
        "filing_date":      filing["date"],
        "years":            revenue_years,   # newest first
        "yearly":           yearly,
        "item1":            item1,
        "has_financials":   bool(revenue_years),
    }


# =============================================================================
# SECTION 7: Key Insights Generator
# =============================================================================
# After calculating metrics we synthesise the numbers into actionable narrative.
# These heuristics are not exhaustive — they are conversation starters that flag
# patterns worth investigating. Each insight is tagged with an icon:
#   ✓ positive signal  |  ✗ concern  |  → neutral / watch

def generate_insights(yearly: dict, years: list[int]) -> list[tuple[str, str]]:
    """
    Produce a ranked list of (icon, message) insight tuples based on
    observed trends in margins, revenue growth, and income quality.
    """
    insights = []

    if len(years) < 2:
        return [("→", "Not enough years of data for trend analysis.")]

    newest, oldest = years[0], years[-1]

    def margin(yr, key):
        return yearly.get(yr, {}).get("margins", {}).get(key)

    def val(yr, key):
        return yearly.get(yr, {}).get(key)

    # ── Gross margin compression / expansion ─────────────────────────────────
    gm_new, gm_old = margin(newest, "gross_margin"), margin(oldest, "gross_margin")
    if gm_new is not None and gm_old is not None:
        if gm_new > gm_old + 1.0:
            insights.append(
                ("[green]✓[/green]",
                 f"Gross margin expanding: {gm_old:.1f}% → {gm_new:.1f}%  "
                 "(stronger pricing power or lower COGS)")
            )
        elif gm_new < gm_old - 1.0:
            insights.append(
                ("[red]✗[/red]",
                 f"Gross margin compressing: {gm_old:.1f}% → {gm_new:.1f}%  "
                 "(investigate input cost pressure or pricing concessions)")
            )
        else:
            insights.append(
                ("[yellow]→[/yellow]",
                 f"Gross margin stable at ~{gm_new:.1f}%")
            )

    # ── Operating leverage ───────────────────────────────────────────────────
    om_new, om_old = margin(newest, "operating_margin"), margin(oldest, "operating_margin")
    if om_new is not None and om_old is not None:
        if om_new > om_old + 0.5:
            insights.append(
                ("[green]✓[/green]",
                 f"Operating leverage improving: {om_old:.1f}% → {om_new:.1f}%  "
                 "(opex growing slower than revenue)")
            )
        elif om_new < om_old - 0.5:
            insights.append(
                ("[red]✗[/red]",
                 f"Operating margin declining: {om_old:.1f}% → {om_new:.1f}%  "
                 "(opex outpacing revenue)")
            )

    # ── Operating vs net margin spread ──────────────────────────────────────
    nm_new = margin(newest, "net_margin")
    if om_new is not None and nm_new is not None:
        spread = om_new - nm_new
        if spread > 8:
            insights.append(
                ("[yellow]→[/yellow]",
                 f"Wide spread between operating ({om_new:.1f}%) and net margin "
                 f"({nm_new:.1f}%)  — review interest expense or effective tax rate")
            )

    # ── Revenue CAGR ─────────────────────────────────────────────────────────
    rev_new, rev_old = val(newest, "revenue"), val(oldest, "revenue")
    if rev_new and rev_old and rev_old != 0:
        n_years = len(years) - 1 or 1
        cagr = ((rev_new / rev_old) ** (1 / n_years) - 1) * 100
        if cagr > 10:
            insights.append(
                ("[green]✓[/green]",
                 f"Strong revenue CAGR of {cagr:.1f}% over the period")
            )
        elif cagr < -2:
            insights.append(
                ("[red]✗[/red]",
                 f"Revenue declining at {cagr:.1f}% CAGR  "
                 "(investigate competitive or structural headwinds)")
            )
        else:
            insights.append(
                ("[yellow]→[/yellow]",
                 f"Modest revenue CAGR of {cagr:.1f}%")
            )

    # ── Net margin trajectory ─────────────────────────────────────────────────
    nm_old = margin(oldest, "net_margin")
    if nm_new is not None and nm_old is not None:
        if nm_new < nm_old - 2:
            insights.append(
                ("[red]✗[/red]",
                 "Net margin declining — check for higher taxes, interest expense, "
                 "or one-time write-downs")
            )
        elif nm_new > nm_old + 2:
            insights.append(
                ("[green]✓[/green]",
                 "Net margin expanding — strong bottom-line leverage")
            )

    return insights or [("→", "No notable trend signals detected.")]


# =============================================================================
# SECTION 8: Display Results (Rich CLI Formatting)
# =============================================================================
# We use the `rich` library to render bordered tables, coloured trend arrows,
# and panels with consistent padding. The layout mirrors the example output
# in the spec, scaled to whatever years of data XBRL provides.

def _fmt_billions(value: float | None) -> str:
    """Format a dollar value in billions, right-aligned."""
    if value is None:
        return "    N/A  "
    return f"${value / 1e9:>8.1f}B"


def _fmt_pct(value: float | None) -> str:
    """Format a percentage to two decimal places."""
    if value is None:
        return "   N/A"
    return f"{value:>6.2f}%"


def display_results(data: dict) -> None:
    """
    Render the full company analysis to the terminal using rich formatting.

    Sections displayed:
      1. Company header panel
      2. Item 1 business summary
      3. Income statement table (FY columns × line-item rows)
      4. Margin analysis table with coloured trend arrows
      5. Key insights
    """
    ticker  = data["ticker"]
    name    = data["company_name"]
    years   = data["years"]       # newest → oldest
    yearly  = data["yearly"]

    # ── 1. Header ─────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold white]TICKER: {ticker}[/bold white]   |   "
        f"[cyan]{name}[/cyan]\n"
        f"[dim]Filed: {data['filing_date']}   ·   CIK: {data['cik']}[/dim]",
        box=box.DOUBLE,
        expand=False,
        padding=(0, 2),
    ))

    # ── 2. Business Summary ────────────────────────────────────────────────────
    # Each subsection from Item 1 is shown under its own labelled header so the
    # reader can jump directly to products, competition, customers, etc.
    console.print()
    console.rule("[bold]BUSINESS SUMMARY  (Item 1)[/bold]", style="bright_black")

    sections = data["item1"].get("sections", [])
    if not sections:
        console.print("[dim]  Business description not available.[/dim]")
    else:
        for sec in sections:
            title = sec["title"]
            text  = sec["text"]
            console.print(f"\n  [bold cyan]{title}[/bold cyan]")
            console.print(f"  {text}")

    # ── 3. Financial Metrics Table ─────────────────────────────────────────────
    console.print()
    console.rule("[bold]FINANCIAL METRICS  (Item 8)[/bold]", style="bright_black")

    if not years:
        console.print("[dim]  Financial data not available (N/A) — no XBRL data found for this filer.[/dim]")
    else:
        tbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan",
                    pad_edge=False, show_edge=True)
        tbl.add_column("Metric",             style="bold",  min_width=26)
        for yr in years:
            tbl.add_column(f"FY {yr}", justify="right", min_width=13)
        tbl.add_column("Trend", justify="center", min_width=6)

        def add_fin_row(label, key, row_style=""):
            vals  = [yearly[yr].get(key) for yr in years]
            fmted = [_fmt_billions(v) for v in vals]
            arrow = trend_arrow([v for v in reversed([v for v in vals if v is not None])])
            tbl.add_row(label, *fmted, arrow, style=row_style)

        add_fin_row("Revenue",                "revenue")
        add_fin_row("Cost of Revenue (COGS)", "cogs")
        add_fin_row("Gross Profit",           "gross_profit", row_style="green")
        add_fin_row("Operating Income",       "operating_income")
        add_fin_row("Net Income",             "net_income")
        tbl.add_section()
        add_fin_row("  R&D Expense",          "rd",  row_style="dim")
        add_fin_row("  SG&A Expense",         "sga", row_style="dim")

        console.print(tbl)

    # ── 4. Margin Analysis Table ───────────────────────────────────────────────
    console.print()
    console.rule("[bold]MARGIN ANALYSIS[/bold]", style="bright_black")

    if not years:
        console.print("[dim]  Margin data not available (N/A).[/dim]")
    else:
        mtbl = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan",
                     pad_edge=False, show_edge=True)
        mtbl.add_column("Margin",  style="bold", min_width=24)
        for yr in years:
            mtbl.add_column(f"FY {yr}", justify="right", min_width=10)
        mtbl.add_column("Trend", justify="center", min_width=6)

        def add_margin_row(label, mkey):
            vals  = [yearly[yr]["margins"].get(mkey) for yr in years]
            fmted = [_fmt_pct(v) for v in vals]
            arrow = trend_arrow([v for v in reversed([v for v in vals if v is not None])])
            color = "green" if arrow == "↑" else ("red" if arrow == "↓" else "yellow")
            mtbl.add_row(label, *fmted, f"[{color}]{arrow}[/{color}]")

        add_margin_row("Gross Margin",     "gross_margin")
        add_margin_row("Operating Margin", "operating_margin")
        add_margin_row("Net Margin",       "net_margin")

        console.print(mtbl)

    # ── 5. Key Insights ────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold]KEY INSIGHTS[/bold]", style="bright_black")

    if not years:
        console.print("[dim]  No financial data available for insight generation (N/A).[/dim]")
    else:
        for icon, msg in generate_insights(yearly, years):
            console.print(f"  {icon}  {msg}")

    console.print()


# =============================================================================
# SECTION 9: CSV / JSON Export
# =============================================================================
# Exporting lets analysts load the data into Excel, Pandas, or a database for
# cross-company comparisons. Each row represents one company × one fiscal year,
# making it trivially filterable and sortable.

def export_csv(results: list[dict], filename: str) -> None:
    """
    Write analysis results for all companies to a flat CSV file.
    Each row = one ticker × one fiscal year.
    """
    rows = []
    for data in results:
        for yr in data["years"]:
            yd = data["yearly"][yr]
            m  = yd["margins"]
            rows.append({
                "ticker":               data["ticker"],
                "company":              data["company_name"],
                "fiscal_year":          yr,
                "revenue":              yd.get("revenue"),
                "cogs":                 yd.get("cogs"),
                "gross_profit":         yd.get("gross_profit"),
                "operating_income":     yd.get("operating_income"),
                "net_income":           yd.get("net_income"),
                "rd_expense":           yd.get("rd"),
                "sga_expense":          yd.get("sga"),
                "gross_margin_pct":     m.get("gross_margin"),
                "operating_margin_pct": m.get("operating_margin"),
                "net_margin_pct":       m.get("net_margin"),
            })

    if not rows:
        console.print("[red]No data to export.[/red]")
        return

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    console.print(
        f"\n[green]✓[/green] Exported {len(rows)} rows → [bold]{filename}[/bold]"
    )


# =============================================================================
# SECTION 10: Configuration — change these variables before running
# =============================================================================

# Ticker symbol(s) to analyze. Add more to the list to compare companies.
#   e.g. TICKERS = ["AAPL", "MSFT", "GOOGL"]
TICKERS     = ["INTU"]

# Target fiscal year. Set to None to use the most recent 10-K available.
#   e.g. TARGET_YEAR = 2022
TARGET_YEAR = None

# Set to a filename (e.g. "results.csv") to export results, or None to skip.
EXPORT_CSV  = None


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    results = []
    for i, ticker in enumerate(TICKERS):
        console.print(f"\n[bold cyan]━━━  Analyzing {ticker.upper()}  ━━━[/bold cyan]")
        try:
            data = analyze_company(ticker, target_year=TARGET_YEAR)
            display_results(data)
            results.append(data)
        except requests.HTTPError as exc:
            console.print(f"[red]  HTTP error for {ticker}: {exc}[/red]")
        except ValueError as exc:
            console.print(f"[red]  Data error for {ticker}: {exc}[/red]")
        except Exception as exc:
            console.print(f"[red]  Unexpected error for {ticker}: {exc}[/red]")

        # Respect SEC's crawling guidelines: ≤10 requests/second per session.
        # A 1-second pause between companies is courteous and keeps us well
        # within the limit while allowing the XBRL cache to warm up.
        if i < len(TICKERS) - 1:
            time.sleep(1)

    if EXPORT_CSV and results:
        export_csv(results, EXPORT_CSV)
