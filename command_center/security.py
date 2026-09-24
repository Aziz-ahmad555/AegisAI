"""
Security policy for the Command Center.

Local mode keeps simple defaults so the app runs and demos with zero setup.
Cloud mode (AEGISAI_CLOUD_MODE=true) refuses to start unless it has a real
secret key and a hashed operator password.
"""
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque

from flask import abort, request, session
from werkzeug.security import check_password_hash

DEFAULT_USERNAME = "operator"
DEFAULT_PASSWORD = "aegisai2026"
DEFAULT_SECRET_KEY = "aegisai-command-center-demo-key"
MIN_SECRET_KEY_LENGTH = 32

# One-liner shown in errors and docs (werkzeug ships with Flask). getpass
# keeps the password off the screen and out of shell history.
HASH_COMMAND = (
    'python -c "import getpass; from werkzeug.security import generate_password_hash as h; '
    'print(h(getpass.getpass()))"'
)
SECRET_COMMAND = 'python -c "import secrets; print(secrets.token_hex(32))"'

# Request size limits.
MAX_REQUEST_BYTES = 64 * 1024        # whole request body (Flask MAX_CONTENT_LENGTH)
MAX_REPORT_CHARS = 2000              # /api/analyze_report text
MAX_QUESTION_CHARS = 1000            # /api/chat question


# ----- credentials -------------------------------------------------------------

class Credentials:
    """Operator login. Prefers a password hash; falls back to a plaintext
    password (env or the documented local default) - local mode only."""

    def __init__(self, username, password_hash=None, password=None):
        self.username = username
        self.password_hash = password_hash
        self.password = password

    @classmethod
    def from_env(cls, env):
        return cls(
            username=env.get("AEGISAI_USERNAME") or DEFAULT_USERNAME,
            password_hash=env.get("AEGISAI_PASSWORD_HASH") or None,
            password=env.get("AEGISAI_PASSWORD") or DEFAULT_PASSWORD,
        )

    @property
    def uses_default_password(self):
        return self.password_hash is None and self.password == DEFAULT_PASSWORD

    def verify(self, username, password):
        # Constant-time comparisons; always evaluate both parts so timing
        # doesn't reveal whether the username was right.
        user_ok = hmac.compare_digest(username.encode(), self.username.encode())
        if self.password_hash:
            try:
                pass_ok = check_password_hash(self.password_hash, password)
            except ValueError:          # malformed hash in the env var
                pass_ok = False
        else:
            pass_ok = hmac.compare_digest(password.encode(), self.password.encode())
        return user_ok and pass_ok


def cloud_config_problems(env):
    """Everything that makes a hosted deployment unsafe to start. Empty = OK."""
    problems = []
    key = env.get("AEGISAI_SECRET_KEY", "")
    if not key or key == DEFAULT_SECRET_KEY:
        problems.append(f"AEGISAI_SECRET_KEY is not set. Generate one with:  {SECRET_COMMAND}")
    elif len(key) < MIN_SECRET_KEY_LENGTH:
        problems.append(f"AEGISAI_SECRET_KEY is too short ({len(key)} chars, need {MIN_SECRET_KEY_LENGTH}+).")
    pw_hash = env.get("AEGISAI_PASSWORD_HASH", "")
    if not pw_hash:
        problems.append(
            "AEGISAI_PASSWORD_HASH is not set (plaintext/default passwords are refused in cloud mode). "
            f"Generate one with:  {HASH_COMMAND}"
        )
    elif ":" not in pw_hash or "$" not in pw_hash:
        problems.append("AEGISAI_PASSWORD_HASH doesn't look like a werkzeug password hash.")
    return problems


class InsecureCloudConfig(RuntimeError):
    pass


# ----- login rate limiting ----------------------------------------------------------

class LoginRateLimiter:
    """
    In-memory failed-login limiter per client address: after MAX_FAILURES
    failures within WINDOW seconds the address is locked out until the oldest
    failure ages out. A successful login clears that address's failures.

    In-memory is correct here because the app runs as exactly one process
    (see "Production server" in ROADMAP.md).
    """

    MAX_FAILURES = 5
    WINDOW = 300  # seconds

    def __init__(self, max_failures=MAX_FAILURES, window=WINDOW, clock=time.monotonic):
        self.max_failures = max_failures
        self.window = window
        self.clock = clock
        self._failures = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key, now):
        q = self._failures[key]
        while q and now - q[0] >= self.window:
            q.popleft()
        if not q:
            self._failures.pop(key, None)
        return q

    def retry_after(self, key):
        """Seconds until `key` may try again, or 0 if not locked out."""
        with self._lock:
            now = self.clock()
            q = self._prune(key, now)
            if len(q) >= self.max_failures:
                return max(1, int(self.window - (now - q[0])) + 1)
            return 0

    def record_failure(self, key):
        with self._lock:
            self._failures[key].append(self.clock())

    def reset(self, key=None):
        with self._lock:
            if key is None:
                self._failures.clear()
            else:
                self._failures.pop(key, None)


# ----- CSRF ----------------------------------------------------------------------

CSRF_SESSION_KEY = "csrf_token"
CSRF_HEADER = "X-CSRF-Token"
CSRF_FIELD = "csrf_token"


def csrf_token():
    """Per-session token, created on first use; exposed to templates."""
    token = session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


def verify_csrf():
    """before_request hook: every state-changing request must echo the
    session's token (form field for the login form, header for fetch())."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    expected = session.get(CSRF_SESSION_KEY)
    sent = request.headers.get(CSRF_HEADER) or request.form.get(CSRF_FIELD)
    if not expected or not sent or not hmac.compare_digest(expected, sent):
        abort(400, description="Missing or invalid CSRF token. Reload the page and try again.")


# ----- response headers -------------------------------------------------------------

CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    # Every script, stylesheet and font is self-hosted (static/vendor, static/
    # fonts): no third-party origins at all. 'unsafe-inline' remains for the
    # templates' inline page scripts and style attributes.
    "script-src 'self' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    "font-src 'self'",
    "img-src 'self' data: blob:",
    "connect-src 'self' ws: wss:",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
])


def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
    return response
