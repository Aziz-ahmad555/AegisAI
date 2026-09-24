"""Stage 4 security behaviour: cloud startup refusal, credentials, login rate
limiting, CSRF, input size limits, cookies and response headers."""
import os
import subprocess
import sys

import pytest
from werkzeug.security import generate_password_hash

import app as aegis
import security
from conftest import TEST_PASSWORD
from helpers import api_post, csrf_token, login

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOD_KEY = "k" * 64
GOOD_HASH = generate_password_hash("s3cret")


# --- cloud-mode startup refusal ------------------------------------------------------

def test_valid_cloud_config_has_no_problems():
    assert security.cloud_config_problems({"AEGISAI_SECRET_KEY": GOOD_KEY, "AEGISAI_PASSWORD_HASH": GOOD_HASH}) == []


@pytest.mark.parametrize("env, expected", [
    ({"AEGISAI_PASSWORD_HASH": GOOD_HASH}, "AEGISAI_SECRET_KEY is not set"),
    ({"AEGISAI_SECRET_KEY": security.DEFAULT_SECRET_KEY, "AEGISAI_PASSWORD_HASH": GOOD_HASH}, "AEGISAI_SECRET_KEY is not set"),
    ({"AEGISAI_SECRET_KEY": "short", "AEGISAI_PASSWORD_HASH": GOOD_HASH}, "too short"),
    ({"AEGISAI_SECRET_KEY": GOOD_KEY}, "AEGISAI_PASSWORD_HASH is not set"),
    ({"AEGISAI_SECRET_KEY": GOOD_KEY, "AEGISAI_PASSWORD": "hunter2"}, "AEGISAI_PASSWORD_HASH is not set"),
    ({"AEGISAI_SECRET_KEY": GOOD_KEY, "AEGISAI_PASSWORD_HASH": "plaintext"}, "doesn't look like"),
])
def test_insecure_cloud_configs_are_named(env, expected):
    problems = security.cloud_config_problems(env)
    assert any(expected in p for p in problems)


def _import_app(extra_env):
    env = {k: v for k, v in os.environ.items() if not k.startswith("AEGISAI_")}
    env.update(extra_env)
    return subprocess.run([sys.executable, "-c", "import app"], cwd=APP_DIR, env=env,
                          capture_output=True, text=True, timeout=120)


def test_cloud_mode_refuses_to_start_without_secrets():
    r = _import_app({"AEGISAI_CLOUD_MODE": "true"})
    assert r.returncode != 0
    assert "Refusing to start in cloud mode" in r.stderr
    assert "AEGISAI_SECRET_KEY" in r.stderr and "AEGISAI_PASSWORD_HASH" in r.stderr
    assert "generate_password_hash" in r.stderr          # tells you how to fix it


def test_cloud_mode_starts_with_proper_config():
    r = _import_app({"AEGISAI_CLOUD_MODE": "true", "AEGISAI_SECRET_KEY": GOOD_KEY,
                     "AEGISAI_PASSWORD_HASH": GOOD_HASH})
    assert r.returncode == 0, r.stderr


# --- credentials -------------------------------------------------------------------

def test_local_defaults_work_with_no_configuration():
    creds = security.Credentials.from_env({})
    assert creds.uses_default_password
    assert creds.verify("operator", "aegisai2026")
    assert not creds.verify("operator", "wrong")


def test_hash_takes_precedence_over_plaintext():
    creds = security.Credentials.from_env({"AEGISAI_PASSWORD_HASH": GOOD_HASH, "AEGISAI_PASSWORD": "other"})
    assert creds.verify("operator", "s3cret")
    assert not creds.verify("operator", "other")
    assert not creds.uses_default_password


def test_wrong_username_or_malformed_hash_fails_closed():
    assert not security.Credentials.from_env({"AEGISAI_PASSWORD_HASH": GOOD_HASH}).verify("admin", "s3cret")
    assert not security.Credentials("operator", password_hash="scrypt:bogus$x$y").verify("operator", "x")


def test_documented_hash_command_produces_a_working_hash():
    # The exact one-liner from the README / .env.example, fed a password on stdin.
    code = security.HASH_COMMAND.split('"', 1)[1].rsplit('"', 1)[0].replace("getpass.getpass()", "input()")
    r = subprocess.run([sys.executable, "-c", code], input="my password\n", capture_output=True, text=True)
    hashed = r.stdout.strip()
    assert security.Credentials("operator", password_hash=hashed).verify("operator", "my password")
    assert security.cloud_config_problems({"AEGISAI_SECRET_KEY": GOOD_KEY, "AEGISAI_PASSWORD_HASH": hashed}) == []


# --- login rate limiting ------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_limiter_locks_after_max_failures_and_unlocks_after_window():
    clock = FakeClock()
    lim = security.LoginRateLimiter(max_failures=3, window=60, clock=clock)
    for _ in range(3):
        assert lim.retry_after("1.2.3.4") == 0
        lim.record_failure("1.2.3.4")
    assert lim.retry_after("1.2.3.4") > 0
    assert lim.retry_after("5.6.7.8") == 0           # other clients unaffected
    clock.t += 61
    assert lim.retry_after("1.2.3.4") == 0


def test_login_is_rate_limited_even_for_the_right_password(client):
    for _ in range(security.LoginRateLimiter.MAX_FAILURES):
        assert login(client, password="wrong").status_code == 401
    r = login(client)                                  # correct password, still locked out
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0
    assert b"Too many failed attempts" in r.data


def test_successful_login_clears_failure_history(client):
    for _ in range(security.LoginRateLimiter.MAX_FAILURES - 1):
        login(client, password="wrong")
    assert login(client).status_code == 302
    for _ in range(security.LoginRateLimiter.MAX_FAILURES - 1):
        assert login(client, password="wrong").status_code == 401   # counter restarted


# --- CSRF ----------------------------------------------------------------------------

def test_login_form_without_csrf_token_is_rejected(client):
    r = client.post("/login", data={"username": "operator", "password": TEST_PASSWORD})
    assert r.status_code == 400
    with client.session_transaction() as s:
        assert not s.get("logged_in")


def test_api_post_without_or_with_wrong_token_is_rejected(client):
    assert login(client).status_code == 302
    r = client.post("/api/chat", json={"question": "fire?"})
    assert r.status_code == 400 and "CSRF" in r.get_json()["error"]
    r = client.post("/api/chat", json={"question": "fire?"}, headers={"X-CSRF-Token": "forged"})
    assert r.status_code == 400
    assert api_post(client, "/api/chat", {"question": "fire?"}).status_code == 200


def test_csrf_token_rotates_on_login(client):
    before = csrf_token(client)
    login(client)
    client.get("/")
    with client.session_transaction() as s:
        assert s["csrf_token"] != before


def test_pages_embed_the_token_for_fetch(client):
    login(client)
    html = client.get("/nlp").get_data(as_text=True)
    with client.session_transaction() as s:
        assert f'<meta name="csrf-token" content="{s["csrf_token"]}">' in html


# --- input limits ------------------------------------------------------------------------

def test_oversized_report_and_question_are_rejected(client):
    login(client)
    r = api_post(client, "/api/analyze_report", {"text": "fire " * 500})
    assert r.status_code == 413 and "too long" in r.get_json()["error"]
    r = api_post(client, "/api/chat", {"question": "x" * (security.MAX_QUESTION_CHARS + 1)})
    assert r.status_code == 413
    assert api_post(client, "/api/chat", {"question": "x" * security.MAX_QUESTION_CHARS}).status_code == 200


def test_oversized_request_body_is_rejected_before_parsing(client):
    login(client)
    r = api_post(client, "/api/analyze_report", {"text": "a" * (security.MAX_REQUEST_BYTES + 1)})
    assert r.status_code == 413


def test_malformed_json_is_a_400_not_a_crash(client):
    login(client)
    client.get("/")          # login clears the session; the next page issues the token
    with client.session_transaction() as s:
        token = s["csrf_token"]
    r = client.post("/api/chat", data="{not json", content_type="application/json", headers={"X-CSRF-Token": token})
    assert r.status_code == 400
    r = api_post(client, "/api/analyze_report", {"text": ["not", "a", "string"]})
    assert r.status_code == 400


# --- cookies & headers -------------------------------------------------------------------

def test_session_cookie_flags_in_cloud_mode(client):
    r = login(client)
    cookie = r.headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie and "Secure" in cookie


def test_security_headers_on_every_response(client):
    for r in (client.get("/login"), client.get("/static/js/incidents.js")):
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]


def test_default_login_hint_hidden_when_not_using_defaults(client):
    assert b"aegisai2026" not in client.get("/login").data
    assert aegis.credentials.uses_default_password is False
