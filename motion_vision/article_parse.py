"""Extract article bodies from Bilibili HTML and embedded JSON without executing scripts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

BODY_CLASSES = (
    "opus-module-content",
    "article-holder",
    "read-article-holder",
    "article-content",
    "opus-paragraph-children",
)


@dataclass(frozen=True)
class ParsedArticle:
    title: str = ""
    author: str = ""
    summary: str = ""
    content: str = ""
    cover_url: str = ""
    source: str = "网页正文"


def html_to_text(source: str) -> str:
    """Preserve paragraphs and nested content while discarding scripts and controls."""
    soup = BeautifulSoup(source, "html.parser")
    for node in soup.select("script, style, svg, iframe, form, button, noscript"):
        node.decompose()
    for node in soup.find_all("img"):
        node.replace_with(f"[文章图片：{node.get('alt') or '无文字说明'}]")
    for node in soup.find_all("br"):
        node.replace_with("\n")
    for node in soup.find_all(
        [
            "p",
            "div",
            "section",
            "article",
            "li",
            "blockquote",
            "pre",
            *[f"h{i}" for i in range(1, 7)],
        ]
    ):
        node.append("\n")
    return _lines(soup.get_text())


def parse_article_page(source: str) -> ParsedArticle:
    """Use explicit article regions or known initial-state fields; never the page shell."""
    soup = BeautifulSoup(source, "html.parser")

    def meta(name: str) -> str:
        node = soup.find("meta", attrs={"property": name}) or soup.find(
            "meta", attrs={"name": name}
        )
        return str(node.get("content") or "").strip() if node else ""

    title = meta("og:title") or (soup.title.get_text(strip=True) if soup.title else "")
    summary = meta("og:description") or meta("description")
    cover = meta("og:image")
    author_node = soup.select_one(".opus-module-author__name, .up-name, .author-name")
    author = author_node.get_text(strip=True) if author_node else ""
    body = ""
    for marker in BODY_CLASSES:
        nodes = soup.select(f".{marker}")
        if nodes:
            # A fallback class can occur on several sibling paragraphs. Keep every one.
            body = "\n".join(html_to_text(str(node)) for node in nodes)
            if body.strip():
                break
    parsed = ParsedArticle(title, author, summary, _lines(body), cover)
    for script in soup.find_all("script"):
        text = script.string or script.get_text()
        match = re.search(r"(?:window\.)?__(?:INITIAL_STATE|NEXT_DATA)__\s*=\s*", text)
        if match:
            text = text[match.end() :].lstrip()
        elif script.get("id") not in {"__NEXT_DATA__", "__INITIAL_STATE__"}:
            continue
        try:
            state, _end = json.JSONDecoder().raw_decode(text)
        except (ValueError, TypeError, RecursionError):
            continue
        candidate = _state_article(state)
        if candidate:
            parsed = ParsedArticle(
                candidate.title or parsed.title,
                candidate.author or parsed.author,
                candidate.summary or parsed.summary,
                candidate.content or parsed.content,
                candidate.cover_url or parsed.cover_url,
                candidate.source if candidate.content else parsed.source,
            )
            break
    return parsed


def _state_article(state: Any) -> ParsedArticle | None:
    if not isinstance(state, dict):
        return None
    detail = state.get("detail")
    if isinstance(detail, dict) and isinstance(detail.get("modules"), list):
        title = author = summary = cover = ""
        paragraphs: list[str] = []
        for module in detail["modules"]:
            if not isinstance(module, dict):
                continue
            title_data = module.get("module_title") or {}
            author_data = module.get("module_author") or {}
            content_data = module.get("module_content") or {}
            bottom = module.get("module_bottom")
            share = bottom.get("share_info") or {} if isinstance(bottom, dict) else {}
            if isinstance(title_data, dict):
                title = title or str(title_data.get("text") or "")
            if isinstance(author_data, dict):
                author = author or str(author_data.get("name") or "")
            if isinstance(share, dict):
                summary = summary or str(share.get("summary") or "")
                cover = cover or str(share.get("pic") or "")
            if isinstance(content_data, dict):
                paragraphs.extend(_paragraph_text(content_data.get("paragraphs")))
        return ParsedArticle(
            title, author, summary, "\n".join(paragraphs), cover, "Opus 结构化正文"
        )
    # Older read pages carry article data rather than Opus modules.
    for data in (state.get("readInfo"), state.get("article"), state.get("articleInfo")):
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            author = data.get("author") or {}
            return ParsedArticle(
                str(data.get("title") or ""),
                str(author.get("name") or "") if isinstance(author, dict) else "",
                str(data.get("summary") or ""),
                html_to_text(data["content"]),
                str(data.get("banner_url") or ""),
                "专栏结构化正文",
            )
    props = state.get("props")
    if isinstance(props, dict):
        return _state_article(props.get("pageProps"))
    return None


def _paragraph_text(value: Any, depth: int = 0) -> list[str]:
    if depth > 12 or not isinstance(value, list):
        return []
    result = []
    for paragraph in value[:10000]:
        if not isinstance(paragraph, dict):
            continue
        text = _node_text(paragraph, depth + 1).strip()
        if text:
            result.append(text)
    return result


def _node_text(value: Any, depth: int = 0) -> str:
    if depth > 16:
        return ""
    if isinstance(value, list):
        return "".join(_node_text(node, depth + 1) for node in value[:10000])
    if not isinstance(value, dict):
        return ""
    if isinstance(value.get("words"), str):
        return value["words"]
    if isinstance(value.get("text"), str):
        return value["text"]
    parts = []
    # Read only content-bearing fields, not style, metrics, avatars or recommendations.
    for key in ("text", "nodes", "word", "rich", "heading", "blockquote", "list", "items"):
        parts.append(_node_text(value.get(key), depth + 1))
    if isinstance(value.get("paragraphs"), list):
        parts.append("\n".join(_paragraph_text(value["paragraphs"], depth + 1)))
    formula = value.get("formula")
    if isinstance(formula, dict):
        parts.append(str(formula.get("latex") or formula.get("text") or ""))
    code = value.get("code")
    if isinstance(code, dict):
        parts.append(str(code.get("content") or code.get("code") or ""))
    if value.get("pic"):
        parts.append("[文章图片：无文字说明]")
    return "".join(parts)


def _lines(value: str) -> str:
    return "\n".join(
        line for raw in value.splitlines() if (line := re.sub(r"[ \t\r\f\v]+", " ", raw).strip())
    )[:100000]
