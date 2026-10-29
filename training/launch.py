#!/usr/bin/env python3
"""Dispatch to a model family's launcher in the active Python environment."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch a documented training preset for SD1.4, DDPM, or EDM.",
        epilog="Example: python training/launch.py ddpm --list-configs. "
        "Arguments following the family are passed to its launcher (see FAMILY --help).",
    )
    parser.add_argument("family", choices=("sd14", "ddpm", "edm"))
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.parse_args(sys.argv[1:])
        return
    args = parser.parse_args(sys.argv[1:2])
    launcher = Path(__file__).resolve().parent / args.family / "launch.py"
    result = subprocess.run([sys.executable, str(launcher), *sys.argv[2:]])
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
