"""Lightweight web search and fetch tools."""

from html.parser import HTMLParser
import ipaddress
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request

from pydantic import BaseModel, ConfigDict, Field

from .base import Tool, ToolOutput


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_FETCH_BYTES = 2_000_000
DEFAULT_FETCH_CHARS = 12_000
HTML_SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside"}


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum number of results to return, default 5")


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the general web for news, blogs, documentation, tutorials, or "
        "non-academic content. Returns title, URL, and content snippet. "
        "Do NOT use this for academic paper searches — use paper_search instead."
    )
    args_model = WebSearchArgs

    def execute(self, query: str, max_results: int = 5) -> str:
        query = query.strip()
        if not query:
            return "Error: query required"

        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return "Error: TAVILY_API_KEY is required for web_search"

        count = max(1, min(int(max_results or 5), 10))
        payload = json.dumps({
            "api_key": api_key,
            "query": query,
            "max_results": count,
            "include_answer": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            TAVILY_SEARCH_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read(1_000_000).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return f"Error: Tavily request failed with HTTP {e.code}: {_read_error(e)}"
        except urllib.error.URLError as e:
            return f"Error: Tavily request failed: {e.reason}"
        except TimeoutError:
            return "Error: Tavily request timed out"
        except Exception as e:
            return f"Error: Tavily request failed: {e}"

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            return f"Error: Tavily returned invalid JSON: {e}"

        results = data.get("results") or []
        if not results:
            return f"No web search results for: {query}"

        lines = [f"Search results for: {query}"]

        answer = data.get("answer")
        if answer:
            lines.append(f"\nAnswer: {_clean(answer)}")

        for index, item in enumerate(results[:count], 1):
            title = _clean(item.get("title") or "Untitled")
            url = _clean(item.get("url") or "")
            content = _clean(item.get("content") or "")
            lines.append(f"\n{index}. {title}")
            if url:
                lines.append(f"   URL: {url}")
            if content:
                lines.append(f"   Snippet: {content}")
        return "\n".join(lines)


class WebFetchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    url: str = Field(description="HTTP or HTTPS URL to fetch")
    max_chars: int = Field(default=DEFAULT_FETCH_CHARS, description="Maximum characters of cleaned text to return, default 12000")


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch a single HTTP(S) URL and return cleaned title and text excerpt. "
        "Use this after web_search when a specific result needs to be read."
    )
    args_model = WebFetchArgs

    def execute(self, url: str, max_chars: int = DEFAULT_FETCH_CHARS) -> str | ToolOutput:
        url = url.strip()
        if not url:
            return "Error: url required"

        error = _validate_public_http_url(url)
        if error:
            return f"Error: {error}"

        limit = max(1000, min(int(max_chars or DEFAULT_FETCH_CHARS), 30_000))
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "text/html, text/plain;q=0.9, */*;q=0.5",
                "Accept-Encoding": "identity",
                "User-Agent": "Folium/0.1",
            },
        )

        opener = urllib.request.build_opener(_SafeRedirectHandler())
        try:
            with opener.open(req, timeout=15) as resp:
                final_url = resp.geturl()
                error = _validate_public_http_url(final_url)
                if error:
                    return f"Error: redirected to disallowed URL: {error}"
                content_type = resp.headers.get("Content-Type", "")
                body = resp.read(MAX_FETCH_BYTES + 1)
        except urllib.error.HTTPError as e:
            return f"Error: web_fetch failed with HTTP {e.code}: {_read_error(e)}"
        except urllib.error.URLError as e:
            return f"Error: web_fetch failed: {e.reason}"
        except TimeoutError:
            return "Error: web_fetch timed out"
        except Exception as e:
            return f"Error: web_fetch failed: {e}"

        truncated_bytes = len(body) > MAX_FETCH_BYTES
        body = body[:MAX_FETCH_BYTES]
        text = _decode_body(body, content_type)
        title, cleaned = _html_to_text(text) if _looks_like_html(content_type, text) else ("", _clean_text(text))
        if not cleaned:
            return f"No readable text found at: {url}"

        excerpt = cleaned[:limit]
        truncated_chars = len(cleaned) > limit
        lines = [f"Fetched: {url}"]
        if title:
            lines.append(f"Title: {title}")
        lines.append("")
        lines.append(excerpt)
        if truncated_chars or truncated_bytes:
            lines.append(
                f"\n... truncated ({len(cleaned)} chars"
                + (f", read first {MAX_FETCH_BYTES} bytes" if truncated_bytes else "")
                + ") ..."
            )
        model_content = "\n".join(lines)
        if not truncated_chars:
            return model_content

        raw_lines = [f"Fetched: {url}"]
        if title:
            raw_lines.append(f"Title: {title}")
        raw_lines.extend(["", cleaned])
        if truncated_bytes:
            raw_lines.append(f"\n... source truncated after {MAX_FETCH_BYTES} bytes ...")
        return ToolOutput(content=model_content, raw_content="\n".join(raw_lines))


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        error = _validate_public_http_url(newurl)
        if error:
            raise urllib.error.URLError(f"redirected to disallowed URL: {error}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in HTML_SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in {"p", "br", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in HTML_SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        else:
            self.text_parts.append(data)


def _read_error(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read(500).decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def _validate_public_http_url(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "only http:// and https:// URLs are allowed"
    if not parsed.hostname:
        return "URL hostname required"
    host = parsed.hostname.strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        return "localhost URLs are not allowed"
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        return f"could not resolve hostname: {e}"
    for entry in addresses:
        ip = ipaddress.ip_address(entry[4][0])
        if not _is_public_ip(ip):
            return f"private or local address is not allowed: {ip}"
    return None


def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _encoding_from_content_type(content_type: str) -> str | None:
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            return part.split("=", 1)[1].strip()
    return None


def _decode_body(body: bytes, content_type: str) -> str:
    encoding = _encoding_from_content_type(content_type) or "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _looks_like_html(content_type: str, text: str) -> bool:
    return "html" in content_type.lower() or "<html" in text[:500].lower()


def _html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    title = _clean(" ".join(parser.title_parts))
    return title, _clean_text(" ".join(parser.text_parts))


def _clean_text(text: str) -> str:
    lines = [_clean(line) for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line)
