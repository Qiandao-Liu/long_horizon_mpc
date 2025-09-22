#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clone/update MuJoCo Menagerie into third_party/mujoco_menagerie,
and print the resolved UR5e XML path & export hint.

Usage:
  python scripts/fetch_menagerie.py
  python scripts/fetch_menagerie.py --dest third_party/mujoco_menagerie --force
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import zipfile
import io
import urllib.request

DEFAULT_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"

def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def have_git():
    code, _ = run(["git", "--version"])
    return code == 0

def download_zip_and_extract(repo_url: str, dest_dir: Path):
    """
    Fallback when git is not available. Downloads the default branch as zip and extracts.
    """
    # GitHub 'archive/refs/heads/main.zip' usually works; if default branch is 'master', this may need change.
    zip_url = repo_url.rstrip(".git") + "/archive/refs/heads/main.zip"
    print(f"[fetch] Downloading zip from {zip_url} ...")
    with urllib.request.urlopen(zip_url) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # Extract to temp then move content into dest_dir
        tmp_extract = dest_dir.parent / (dest_dir.name + "_tmp_download")
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract)
        zf.extractall(tmp_extract)
        # Find single top-level folder
        subdirs = [p for p in tmp_extract.iterdir() if p.is_dir()]
        if not subdirs:
            raise RuntimeError("Zip does not contain a top-level directory.")
        top = subdirs[0]
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.move(str(top), str(dest_dir))
        shutil.rmtree(tmp_extract)
    print(f"[fetch] Extracted to {dest_dir}")

def ensure_repo(dest_dir: Path, repo_url: str, force: bool):
    if dest_dir.exists():
        if force:
            print(f"[fetch] Removing existing directory: {dest_dir}")
            shutil.rmtree(dest_dir)
        else:
            # Try to `git pull` if it's a git repo
            if (dest_dir / ".git").exists() and have_git():
                print(f"[fetch] Updating existing repo at {dest_dir} ...")
                code, out = run(["git", "pull", "--ff-only"], cwd=str(dest_dir))
                if code != 0:
                    print(out)
                    raise RuntimeError("git pull failed.")
                return
            else:
                print(f"[fetch] {dest_dir} exists (non-git). Use --force to replace, or keep as-is.")
                return

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if have_git():
        print(f"[fetch] Cloning {repo_url} into {dest_dir} ...")
        code, out = run(["git", "clone", "--depth", "1", repo_url, str(dest_dir)])
        if code != 0:
            print(out)
            raise RuntimeError("git clone failed.")
    else:
        download_zip_and_extract(repo_url, dest_dir)

def find_ur5e_xml(repo_root: Path) -> Path:
    p = repo_root / "universal_robots_ur5e" / "ur5e.xml"
    if not p.exists():
        raise FileNotFoundError(f"UR5e XML not found at {p}")
    return p

def main():
    project_root = Path(__file__).resolve().parents[1]  # .../long_horizon_mpc
    default_dest = project_root / "third_party" / "mujoco_menagerie"

    ap = argparse.ArgumentParser(description="Fetch MuJoCo Menagerie into third_party/")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Menagerie git repo URL")
    ap.add_argument("--dest", default=str(default_dest), help="Destination directory")
    ap.add_argument("--force", action="store_true", help="Remove existing dest and refetch")
    args = ap.parse_args()

    dest_dir = Path(args.dest)
    ensure_repo(dest_dir, args.repo, args.force)

    ur5e_xml = find_ur5e_xml(dest_dir)
    print("\n[fetch] Success:")
    print(f"  Menagerie path: {dest_dir}")
    print(f"  UR5e XML     : {ur5e_xml}")

    # Helpful hint for environment variable
    print("\nExport this environment variable so your scene XML can reference ${UR5E_XML}:")
    print(f"  export UR5E_XML='{ur5e_xml}'")
    print("\nOr pass --xml-path to your loader if you prefer not to use env vars.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[fetch] ERROR: {e}")
        sys.exit(1)
