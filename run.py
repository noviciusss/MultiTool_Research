"""Entry point — launches the Streamlit UI."""

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "streamlit", "run", "src/ui/streamlit.py"],
    check=True,
)