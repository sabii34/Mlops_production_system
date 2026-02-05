import os
from pathlib import Path

# This makes the script always work relative to where template.py exists (repo root)
BASE_DIR = Path(__file__).resolve().parent

list_of_paths = [
    # directories
    BASE_DIR / "data" / "raw",
    BASE_DIR / "data" / "processed",
    BASE_DIR / "models",
    BASE_DIR / "reports",
    BASE_DIR / "src" / "config",
    BASE_DIR / "src" / "data",
    BASE_DIR / "src" / "features",
    BASE_DIR / "src" / "models",
    BASE_DIR / "src" / "utils",
    BASE_DIR / "tests",
    BASE_DIR / "notebooks",
    BASE_DIR / "scripts",

    # files
    BASE_DIR / "requirements.txt",
    # BASE_DIR / "README.md",
    BASE_DIR / ".gitignore",
]

def create_structure(paths: list[Path]) -> None:
    for path in paths:
        # If path has a suffix, it's a file
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()
        else:
            path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_structure(list_of_paths)
    print("\n✅ Project structure created in repo root.\n")

