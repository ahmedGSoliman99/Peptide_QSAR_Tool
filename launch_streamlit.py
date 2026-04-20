"""Launcher script for packaging the Streamlit app as a Windows executable."""

from __future__ import annotations

import subprocess
import sys
import webbrowser
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    app_path = root / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"app.py not found at: {app_path}")

    webbrowser.open("http://localhost:8501", new=2)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode",
        "false",
        "--server.headless",
        "true",
        "--server.port",
        "8501",
        "--browser.gatherUsageStats",
        "false",
    ]
    subprocess.run(cmd, cwd=root, check=False)


if __name__ == "__main__":
    main()
