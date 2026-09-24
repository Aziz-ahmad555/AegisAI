import os
import sys

# Tests run without a camera: cloud mode keeps the vision stack (torch,
# ultralytics, webcam) out of the app entirely.
os.environ["AEGISAI_CLOUD_MODE"] = "true"
os.environ.pop("ANTHROPIC_API_KEY", None)  # always exercise the offline coordinator path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
