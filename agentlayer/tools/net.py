"""Read-only network lookups: HTTP GET only, re-authorised on every redirect."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from agentlayer.broker import Requirement, Verdict
from agentlayer.tools.base import Clip, Tool, ToolArgError, ToolContext, ToolOutput, fmt_bytes, param, register

_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "pre", "table"}
_SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "template"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    text = "\n".join(line for line in lines if line)
    return parser.title.strip(), text


def _requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    url = args["url"].strip()
    if "://" not in url:
        url = "https://" + url
    if not urlsplit(url).hostname:
        raise ToolArgError(f"not a valid URL: {args['url']}")
    return [Requirement("net.read", url=url)]


def _fetch(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    grant = decision.grant("net.read")
    url = _requires(args, ctx)[0].url
    headers = {"User-Agent": "agentlayer/1.0 (read-only lookup)", "Accept": "text/html,application/json,text/plain,*/*"}
    with httpx.Client(timeout=grant.timeout, follow_redirects=False, headers=headers) as client:
        for _hop in range(4):
            with client.stream("GET", url) as resp:
                if resp.is_redirect:
                    target = urljoin(url, resp.headers.get("location", ""))
                    verdict = ctx.broker.evaluate([Requirement("net.read", url=target)])
                    if verdict.verdict is not Verdict.ALLOW:
                        return ToolOutput.fail(f"{url} redirects to {target}, which was not approved: {verdict.reason}",
                                               status=resp.status_code)
                    url = target
                    continue
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) > grant.max_bytes:
                        break
                status, ctype = resp.status_code, resp.headers.get("content-type", "")
                break
        else:
            return ToolOutput.fail("too many redirects")

    limit = min(args["max_chars"], ctx.policy.max_tool_output - 400)
    data: dict = {"url": url, "status": status, "content_type": ctype.split(";")[0], "bytes": len(body)}
    text = body.decode(resp.encoding or "utf-8", errors="replace") if body else ""
    if "html" in ctype:
        title, text = html_to_text(text)
        data["title"] = title
    elif "json" in ctype:
        try:
            text = json.dumps(json.loads(text), indent=1, ensure_ascii=False)
        except ValueError:
            pass
    elif text and "\0" in text[:2048]:
        data["note"] = "binary content not shown"
        text = ""
    clip = Clip.of(text, limit)
    data["text"] = clip.text
    if clip.truncated:
        data["truncated"] = True
    summary = f"HTTP {status} from {urlsplit(url).hostname} ({fmt_bytes(len(body))})"
    return ToolOutput(200 <= status < 400, data, summary, clip.text[:2000] or None)


register(Tool(
    name="fetch_url",
    description="Fetch a web page or API response with an HTTP GET (read-only). HTML is converted to plain text.",
    rights=("net.read",),
    params={
        "url": param("string", "http(s) URL"),
        "max_chars": param("integer", "Maximum characters of text to return", default=4000, minimum=200, maximum=20000),
    },
    required=("url",),
    requires=_requires,
    run=_fetch,
    preview=lambda a, c: f"GET {_requires(a, c)[0].url}",
))
