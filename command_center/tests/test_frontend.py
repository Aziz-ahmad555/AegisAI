"""Stage 6 frontend guarantees: every page is self-contained (no CDN/third-party
assets), shares the design system and status bar, and assets are cacheable."""
import re

import pytest

import app as aegis
from helpers import login

PAGES = ["/", "/twin", "/sensors", "/nlp", "/chat", "/aerial", "/vision"]


@pytest.fixture
def authed(client):
    assert login(client).status_code == 302
    return client


@pytest.mark.parametrize("path", PAGES)
def test_pages_use_only_first_party_assets(authed, path):
    html = authed.get(path).get_data(as_text=True)
    refs = re.findall(r'<(?:script|link|img)\b[^>]*?(?:src|href)="([^"]+)"', html)
    external = [r for r in refs if r.startswith(("http://", "https://", "//"))]
    assert external == [], f"{path} loads third-party assets: {external}"
    assert "<style>" not in html, f"{path} still has an inline <style> block"


@pytest.mark.parametrize("path", PAGES)
def test_pages_share_design_system_and_status_bar(authed, path):
    html = authed.get(path).get_data(as_text=True)
    assert "css/aegis.css" in html
    assert 'id="threat-level"' in html and "js/status.js" in html


@pytest.mark.parametrize("path", PAGES)
def test_pages_show_the_emergency_nexus_brand(authed, path):
    html = authed.get(path).get_data(as_text=True)
    assert "<title>Emergency Nexus</title>" in html
    assert "<h1>Emergency Nexus</h1>" in html and "AI COMMAND CENTER" in html
    assert ">Aegis<" not in html and "EMERGENCY INTELLIGENCE" not in html


def test_login_page_shows_the_brand(client):
    html = client.get("/login").get_data(as_text=True)
    assert "<title>Emergency Nexus - Sign in</title>" in html
    assert ">Emergency Nexus</h1>" in html and "AI COMMAND CENTER" in html


def test_login_page_is_self_contained_too(client):
    html = client.get("/login").get_data(as_text=True)
    assert "css/aegis.css" in html and "fonts.googleapis" not in html


def test_csp_allows_no_third_party_origins(client):
    csp = client.get("/login").headers["Content-Security-Policy"]
    assert "http" not in csp.replace("'self'", "")          # no https://... origins at all
    assert "font-src 'self'" in csp and "script-src 'self'" in csp


@pytest.mark.parametrize("asset", [
    "css/aegis.css", "js/status.js", "js/incidents.js",
    "vendor/socket.io-4.7.2.min.js", "vendor/chart-4.4.0.umd.min.js", "vendor/three-0.160.0.min.js",
    "fonts/ibm-plex-sans-latin-var.woff2", "fonts/ibm-plex-mono-latin-400.woff2",
])
def test_vendored_assets_are_served_and_cacheable(client, asset):
    r = client.get(f"/static/{asset}")
    assert r.status_code == 200 and len(r.data) > 1000
    assert "max-age=3600" in r.headers.get("Cache-Control", "")
    r.close()


def test_red_is_reserved_for_danger_in_the_stylesheet():
    css = open(aegis.app.static_folder + "/css/aegis.css", encoding="utf-8").read()
    # Every rule that uses the danger colour must be a danger-state selector.
    allowed = ("danger", "critical", "fire", "blocked", "CRITICAL", "FIRE", ":root", "action")
    offenders = []
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if "var(--danger" in body and not any(word in selector for word in allowed):
            offenders.append(selector.strip())
    assert offenders == []


def test_static_urls_are_versioned_for_cache_busting(authed):
    html = authed.get("/").get_data(as_text=True)
    assert re.search(r'/static/css/aegis\.css\?v=\d+', html)
    assert re.search(r'/static/js/status\.js\?v=\d+', html)
