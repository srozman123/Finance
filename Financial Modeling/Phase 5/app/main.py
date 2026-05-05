import sys
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))                            # Phase 5/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "sec_10k_analyzer"))  # sec_10k_analyzer/

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Stock Report API")

_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")


@app.get("/api/report/{ticker}")
def get_report(ticker: str):
    ticker = ticker.upper().strip()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    try:
        from phase5_report_generator import get_report_data
        data = get_report_data(ticker)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        from sec_analyzer import analyze_company
        sec = analyze_company(ticker)
        data["sec"] = {
            "company_name": sec["company_name"],
            "filing_date":  sec["filing_date"],
            "sections":     sec["item1"]["sections"][:6],
        }
    except Exception as exc:
        data["sec"] = {"error": str(exc), "sections": []}

    return data


_STATIC = os.path.join(_HERE, "static")
app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")


if __name__ == "__main__":
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "unknown"

    print("\n  Stock Report App")
    print(f"    Local:   http://127.0.0.1:8000")
    print(f"    Phone:   http://{local_ip}:8000  (same WiFi)\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
