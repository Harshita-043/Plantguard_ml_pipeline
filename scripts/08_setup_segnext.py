"""Safely clone and inspect the official SegNeXt repository for PlantGuard.

Default behavior only clones (if needed) and reports repository state. It does NOT
install unpinned MM* dependencies into the active environment. Once the project
has selected a compatible legacy MMSegmentation stack, use --install-repo to run
`pip install -e external/SegNeXt` explicitly.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/Visual-Attention-Network/SegNeXt.git"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default="external/SegNeXt", help="Local SegNeXt checkout path")
    parser.add_argument("--install-repo", action="store_true", help="Explicitly pip-install the cloned repo in the active environment")
    parser.add_argument("--pull", action="store_true", help="Update an existing checkout with git pull (changes local checkout)")
    args = parser.parse_args()

    if shutil.which("git") is None:
        print("ERROR: Git was not found on PATH. Install Git or activate a terminal where git is available.", file=sys.stderr)
        return 2

    repo = Path(args.repo_dir).resolve()
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        run(["git", "clone", REPO_URL, str(repo)])
    elif not (repo / ".git").exists():
        print(f"ERROR: {repo} exists but is not a Git checkout. No files were changed.", file=sys.stderr)
        return 2
    elif args.pull:
        run(["git", "pull", "--ff-only"], cwd=repo)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    except subprocess.CalledProcessError:
        commit, branch = "unknown", "unknown"

    print(f"\nSegNeXt checkout: {repo}")
    print(f"Branch: {branch or '(detached)'}")
    print(f"Commit: {commit}")
    print("\nDependency check (current Python):")
    for module in ("torch", "mmengine", "mmcv", "mmseg"):
        try:
            mod = __import__(module)
            print(f"  {module}: {getattr(mod, '__version__', 'importable; version unavailable')}")
        except Exception as exc:
            print(f"  {module}: NOT IMPORTABLE ({type(exc).__name__}: {exc})")

    if args.install_repo:
        print("\nInstalling the repository in editable mode into the CURRENT Python environment.")
        print("This does not guarantee MM* dependency compatibility; use only after pinning and verifying versions.")
        run([sys.executable, "-m", "pip", "install", "-e", str(repo)])
    else:
        print("\nClone/inspection complete. No pip installation was performed.")
        print("Next: choose a compatible SegNeXt + MMSegmentation dependency set before installation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
