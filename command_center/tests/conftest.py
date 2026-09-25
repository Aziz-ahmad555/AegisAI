import os
import sys

# Tests run without a camera: cloud mode keeps the vision stack (torch,
# ultralytics, webcam) out of the app entirely.
os.environ["AEGISAI_CLOUD_MODE"] = "true"
os.environ.pop("ANTHROPIC_API_KEY", None)  # always exercise the offline coordinator path
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("AEGISAI_LLM_PROVIDER", None)
# Never call the real Groq models API from tests.
os.environ["AEGISAI_GROQ_CHECK_MODELS"] = "false"
# Play the guided scenario instantly in tests.
os.environ["AEGISAI_SCENARIO_TIME_SCALE"] = "0"
os.environ.pop("AEGISAI_GROQ_MODEL", None)

# Cloud mode refuses to start without a real secret key and a hashed password,
# so the test app is configured the way a proper deployment would be.
import secrets  # noqa: E402

from werkzeug.security import generate_password_hash  # noqa: E402

TEST_USERNAME = "operator"
TEST_PASSWORD = "correct horse battery staple"
os.environ["AEGISAI_SECRET_KEY"] = secrets.token_hex(32)
os.environ["AEGISAI_USERNAME"] = TEST_USERNAME
os.environ["AEGISAI_PASSWORD_HASH"] = generate_password_hash(TEST_PASSWORD)
os.environ.pop("AEGISAI_PASSWORD", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest  # noqa: E402


@pytest.fixture
def client():
    """Flask test client on a clean slate: no rate-limit history, empty twin."""
    from aegis_core import coordinator

    import app as aegis

    # Other test modules point the (module-global) agents at their own
    # systems; the app's agents must read the app's system.
    coordinator.init_agents(aegis.system)
    aegis.app.config["TESTING"] = True
    aegis.login_limiter.reset()
    c = aegis.app.test_client()
    yield c
    aegis.login_limiter.reset()
    for zone in aegis.system.zones():
        aegis.system.clear_zone(zone)
    for p in aegis.system.pending_actions():
        aegis.system.decide(p["id"], approve=False)
