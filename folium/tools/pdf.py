"""PDF fetch and parse tool using PyMuPDF."""

import io
import urllib.error
import urllib.request

from pydantic import BaseModel, ConfigDict, Field

from .base import Tool, ToolFailure, ToolOutput, tool_failure
from .web import _validate_public_http_url, _SafeRedirectHandler

MAX_PDF_BYTES = 20_000_000  # 20 MB
DEFAULT_MAX_CHARS = 50_000


class PdfFetchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    url: str = Field(description="HTTP or HTTPS URL of the PDF file")
    max_chars: int = Field(default=DEFAULT_MAX_CHARS, description="Maximum characters to return, default 50000")


class PdfFetchTool(Tool):
    name = "pdf_fetch"
    description = (
        "Download a PDF from a URL and extract its text content. "
        "Handles two-column layouts common in academic papers. "
        "Returns cleaned text with page markers."
    )
    args_model = PdfFetchArgs

    def execute(self, url: str, max_chars: int = DEFAULT_MAX_CHARS) -> str | ToolOutput | ToolFailure:
        url = url.strip()
        if not url:
            return tool_failure("missing_url", "validation", "url required")

        error = _validate_public_http_url(url)
        if error:
            return tool_failure("invalid_url", "validation", error)

        limit = max(1000, min(int(max_chars or DEFAULT_MAX_CHARS), 100_000))

        pdf_bytes = _download_pdf(url)
        if isinstance(pdf_bytes, ToolFailure):
            return pdf_bytes

        text = _extract_text(pdf_bytes)
        if isinstance(text, ToolFailure):
            return text
        if not text:
            return tool_failure("no_readable_text", "content", "no readable text found in PDF (may be a scanned document)")

        if len(text) <= limit:
            return text

        model_content = text[:limit] + f"\n\n... truncated ({len(text)} chars total) ..."
        return ToolOutput(content=model_content, raw_content=text)


def _download_pdf(url: str) -> bytes | ToolFailure:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/pdf, */*;q=0.5",
            "User-Agent": "Folium/0.1",
        },
    )
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    try:
        with opener.open(req, timeout=30) as resp:
            final_url = resp.geturl()
            error = _validate_public_http_url(final_url)
            if error:
                return tool_failure("invalid_redirect", "security", f"redirected to disallowed URL: {error}")
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read(MAX_PDF_BYTES + 1)
    except urllib.error.HTTPError as e:
        return tool_failure(
            f"http_{e.code}",
            "network",
            f"PDF download failed with HTTP {e.code}",
            retryable=e.code in {408, 429, 500, 502, 503, 504},
            details={"http_status": e.code},
        )
    except urllib.error.URLError as e:
        return tool_failure("network_error", "network", f"PDF download failed: {e.reason}", retryable=True)
    except TimeoutError:
        return tool_failure("timeout", "timeout", "PDF download timed out", retryable=True)
    except Exception as e:
        return tool_failure("download_failed", "network", f"PDF download failed: {e}")

    if len(body) > MAX_PDF_BYTES:
        return tool_failure("pdf_too_large", "resource", "PDF file too large (>20 MB)")

    return body


def _extract_text(pdf_bytes: bytes) -> str | ToolFailure:
    try:
        import fitz
    except ImportError:
        return tool_failure("missing_dependency", "dependency", "PyMuPDF is not installed. Run: pip install pymupdf")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return tool_failure("invalid_pdf", "content", "invalid or corrupted PDF file")

    if doc.is_encrypted:
        return tool_failure("encrypted_pdf", "permission", "PDF is encrypted or password-protected")

    pages_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        if not blocks:
            continue

        sorted_blocks = _sort_blocks_reading_order(blocks)
        page_text = "\n".join(b[4].strip() for b in sorted_blocks if b[4].strip())
        if page_text:
            pages_text.append(f"--- Page {page_num + 1} ---\n{page_text}")

    doc.close()
    return "\n\n".join(pages_text)


def _sort_blocks_reading_order(blocks: list) -> list:
    """Sort text blocks in reading order, handling two-column layouts."""
    text_blocks = [b for b in blocks if b[6] == 0]
    if not text_blocks:
        return blocks

    x_coords = [b[0] for b in text_blocks]
    page_width = max(b[2] for b in text_blocks)
    mid_x = page_width / 2

    left_blocks = [b for b in text_blocks if b[0] < mid_x]
    right_blocks = [b for b in text_blocks if b[0] >= mid_x]

    is_two_column = len(left_blocks) > 2 and len(right_blocks) > 2
    if not is_two_column:
        return sorted(blocks, key=lambda b: (b[1], b[0]))

    left_blocks.sort(key=lambda b: b[1])
    right_blocks.sort(key=lambda b: b[1])
    return left_blocks + right_blocks
