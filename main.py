"""Convenience launcher so the app can be run with ``python main.py``.

Adds the ``src`` directory to the import path so the ``virtual_mouse`` package
can be found without installing it first.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from virtual_mouse.app import main  # noqa: E402

if __name__ == "__main__":
    main()
