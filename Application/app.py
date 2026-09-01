import os
import sys
from pathlib import Path

# Allow the web application to use the shared development helpers while the
# repository keeps application and model-development code in separate folders.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DEVELOPMENT_DIR = PROJECT_ROOT / "Model Development"
if str(MODEL_DEVELOPMENT_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DEVELOPMENT_DIR))

from tsdr_app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
        threaded=True,
    )
