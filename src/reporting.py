from __future__ import annotations

import textwrap
from datetime import datetime
from typing import Dict, Iterable


def fmt_pct(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "N/A"


def fmt_num(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "N/A"


def build_markdown_report(ticker: str, fundamentals: Dict, risk: Dict, signal: Dict, llm_text: str, source: str) -> str:
    lines = [
        f"# AI Trading & Research Copilot Report — {ticker}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Market data source: {source}",
        "",
        "## Executive View",
        f"- Suggested stance: **{signal.get('action','N/A')}**",
        f"- Trend: **{signal.get('trend','N/A')}**",
        f"- Momentum: **{signal.get('momentum','N/A')}**",
        f"- Valuation: **{signal.get('valuation','N/A')}**",
        f"- Risk: **{signal.get('risk_comment','N/A')}**",
        "",
        "## Fundamentals",
        f"- Company: {fundamentals.get('company_name','N/A')}",
        f"- Sector: {fundamentals.get('sector','N/A')}",
        f"- Market cap: {fmt_num(fundamentals.get('market_cap'))}",
        f"- Trailing P/E: {fmt_num(fundamentals.get('trailing_pe'))}",
        f"- Forward P/E: {fmt_num(fundamentals.get('forward_pe'))}",
        f"- Profit margin: {fmt_pct(fundamentals.get('profit_margin'))}",
        f"- Revenue growth: {fmt_pct(fundamentals.get('revenue_growth'))}",
        f"- Debt to equity: {fmt_num(fundamentals.get('debt_to_equity'))}",
        "",
        "## Risk Metrics",
    ]
    for k, v in risk.items():
        if "Return" in k or "Volatility" in k or "Drawdown" in k or "VaR" in k or "CVaR" in k:
            lines.append(f"- {k}: {fmt_pct(v)}")
        else:
            lines.append(f"- {k}: {fmt_num(v)}")
    lines += ["", "## Copilot Commentary", llm_text, "", "## Disclaimer", "This project is for educational and portfolio demonstration purposes only. It is not investment advice."]
    return "\n".join(lines)


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def markdown_to_simple_pdf_bytes(markdown_text: str) -> bytes:
    # Minimal dependency-free PDF writer. Keeps export robust on Streamlit Cloud and local setups.
    cleaned = markdown_text.replace("**", "").replace("#", "")
    lines = []
    for raw in cleaned.splitlines():
        if not raw.strip():
            lines.append("")
            continue
        wrapped = textwrap.wrap(raw, width=88) or [""]
        lines.extend(wrapped)

    pages = [lines[i:i + 44] for i in range(0, len(lines), 44)] or [[]]
    objects = []
    catalog_id = 1
    pages_id = 2
    font_id = 3
    next_id = 4
    page_ids = []
    content_ids = []

    for page_lines in pages:
        page_id, content_id = next_id, next_id + 1
        next_id += 2
        page_ids.append(page_id)
        content_ids.append(content_id)
        stream_lines = ["BT", "/F1 10 Tf", "50 790 Td", "14 TL"]
        for line in page_lines:
            stream_lines.append(f"({_escape_pdf_text(line)}) Tj")
            stream_lines.append("T*")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
        objects.append((content_id, b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"))
        objects.append((page_id, f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()))

    kids = " ".join([f"{pid} 0 R" for pid in page_ids])
    objects.append((catalog_id, f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode()))
    objects.append((pages_id, f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()))
    objects.append((font_id, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))
    objects.sort(key=lambda x: x[0])

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, content in objects:
        offsets.append(len(pdf))
        pdf.extend(f"{obj_id} 0 obj\n".encode())
        pdf.extend(content)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode())
    pdf.extend(f"trailer\n<< /Size {len(objects)+1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return bytes(pdf)
