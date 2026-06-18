"""
CLI for matching an image's color palette against known colorschemes.

Usage:
    floppy-hue path/to/image.png
    floppy-hue path/to/image.png --top 5
    floppy-hue path/to/image.png --themes-file my_themes.json
"""

import argparse
import json
import sys
from pathlib import Path
import importlib.resources

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from floppy_hue.core import match_themes


def load_themes(themes_file):
    if themes_file is None:
        ref = importlib.resources.files("floppy_hue.data").joinpath("themes.toml")
        with ref.open("rb") as f:
            return tomllib.load(f)

    path = Path(themes_file)
    if not path.exists():
        print(f"error: themes file not found: {themes_file}", file=sys.stderr)
        sys.exit(1)

    with open(path, "rb") as f:
        custom = tomllib.load(f)

    for name, theme in custom.items():
        for key in ("background", "foreground", "accents"):
            if key not in theme:
                print(f"error: theme '{name}' is missing required key '{key}'", file=sys.stderr)
                sys.exit(1)

    return custom


def build_parser():
    parser = argparse.ArgumentParser(
        prog="floppy-hue",
        description="Match an image's color palette against known colorschemes.",
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to the image file to analyze.",
    )

    parser.add_argument(
        "--top", "-n",
        type=int,
        default=None,
        help="Only show the top N matches (default: show all configured themes).",
    )

    parser.add_argument(
        "--colors", "-c",
        type=int,
        default=10,
        help="Number of dominant colors to extract from the image (default: 10).",
    )

    parser.add_argument(
        "--themes-file",
        type=str,
        default=None,
        help="Path to a JSON file of custom themes, same shape as the built-in set "
             "If omitted, uses the built-in theme set.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    themes = load_themes(args.themes_file)

    try:
        ranked = match_themes(str(image_path), themes, n_colors=args.colors)
    except Exception as e:
        print(f"error: failed to process image: {e}", file=sys.stderr)
        sys.exit(1)

    if args.top is not None:
        ranked = ranked[:args.top]

    # name_width = max(len(name) for name, _ in ranked) if ranked else 0
    # for name, pct in ranked:
    #     print(f"{name.ljust(name_width)}  {pct:5.1f}%")

    print(ranked)

if __name__ == "__main__":
    main()
