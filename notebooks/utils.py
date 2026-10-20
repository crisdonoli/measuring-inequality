"""Locate the project root and make its source packages importable."""

from pathlib import Path
import sys


def load_root():
    ROOT = next(
        (p for p in (Path.cwd(), *Path.cwd().parents)
         if (p / 'source').is_dir() and (p / 'data').is_dir()),
        None,
    )
    if ROOT is None:
        raise RuntimeError(
            f"No project root found: no ancestor of {Path.cwd()} "
            "contains both 'source/' and 'data/'."
        )

    SOURCE = str(ROOT)
    if SOURCE not in sys.path:
        sys.path.insert(0, SOURCE)

    print(f"[utils] Project root loaded successfully: {ROOT}")
    return ROOT
