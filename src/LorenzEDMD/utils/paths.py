import subprocess
from pathlib import Path


def get_project_root() -> Path:
    """
    Returns the project root directory.

    Tries:
    1. Git-based detection
    2. Fallback: relative to this file (3 levels up)
    3. Raises error if neither works
    """
    # 1. Try Git
    try:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        return Path(root.decode("utf-8").strip())
    except Exception:
        pass

    # 2. Fallback based on known file location
    fallback_root = Path(__file__).resolve().parents[3]
    if (fallback_root / "pyproject.toml").exists():
        return fallback_root

    # 3. Fail
    raise RuntimeError(
        "Could not determine project root. "
        "Check if you're in a Git repo or if the utils/ folder has moved."
    )


def get_data_folder_path() -> Path:
    """
    Returns the path of the data folder
    """
    root_path = get_project_root()
    return root_path / "data"
