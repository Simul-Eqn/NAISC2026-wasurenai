"""WSGI entry point for production deployment."""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_inference_server.app import app

if __name__ == "__main__":
    app.run()
