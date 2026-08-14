"""Start the dashboard. One command, opens in the browser.

    .venv/bin/python scripts/dashboard.py

Serves the built React app plus the JSON API from a single origin on localhost, so there is
no CORS setup and nothing to configure. If the frontend has not been built yet, this builds
it automatically (requires node and npm).

For frontend development, run the backend and the Vite dev server separately:

    .venv/bin/python scripts/dashboard.py --api-only
    cd dashboard/frontend && npm run dev      # http://localhost:5173, proxies /api
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND = REPO_ROOT / "dashboard" / "frontend"
DIST = FRONTEND / "dist"


def ensure_frontend_built(force: bool = False) -> bool:
    """Build the React app if needed. Returns True if a usable bundle exists."""
    if DIST.is_dir() and (DIST / "index.html").exists() and not force:
        return True

    npm = shutil.which("npm")
    if npm is None:
        print("npm not found, so the built UI cannot be produced.")
        print("The JSON API will still work at http://127.0.0.1:8000/api/runs")
        return False

    if not (FRONTEND / "node_modules").is_dir():
        print("installing frontend dependencies (first run only)...")
        subprocess.run([npm, "install"], cwd=FRONTEND, check=True)

    print("building frontend...")
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True)
    return (DIST / "index.html").exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-only", action="store_true", help="skip the frontend build")
    ap.add_argument("--rebuild", action="store_true", help="force a frontend rebuild")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not args.api_only:
        try:
            ensure_frontend_built(force=args.rebuild)
        except subprocess.CalledProcessError as exc:
            print(f"frontend build failed ({exc}), serving the API only")

    sys.path.insert(0, str(REPO_ROOT / "dashboard" / "backend"))
    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"\ndashboard: {url}")
    print(f"runs dir : {REPO_ROOT / 'runs'}\n")

    if not args.no_browser and not args.api_only:
        # Delay slightly so the server is accepting connections before the tab opens.
        threading.Thread(
            target=lambda: (time.sleep(1.2), webbrowser.open(url)), daemon=True
        ).start()

    uvicorn.run("app:app", host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
