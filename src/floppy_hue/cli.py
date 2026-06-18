"""
CLI for matching an image's color palette against known colorschemes.

Usage:
    floppy-hue path/to/image.png
    floppy-hue path/to/image.png --top 5
    floppy-hue path/to/image.png --themes-file my_themes.json
"""

import argparse
import sys
import importlib.resources
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from floppy_hue.core import (
    load_image_sample,
    extract_image_colors,
    score_base,
    score_weighted,
    score_structural,
    visualize_palette,
    visualize_3d_clusters,
)


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
                print(
                    f"error: theme '{name}' is missing required key '{key}'",
                    file=sys.stderr,
                )
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
        "--top",
        "-n",
        type=int,
        default=1,
        help="Only show the top N matches (default: show all configured themes).",
    )

    parser.add_argument(
        "--colors",
        "-c",
        type=int,
        default=100,
        help="Number of dominant colors to extract from the image (default: 16).",
    )

    parser.add_argument(
        "--themes-file",
        type=str,
        default=None,
        help="Path to a TOML file of custom themes, same shape as the built-in set. "
        "If omitted, uses the built-in theme set.",
    )

    parser.add_argument(
        "--strategy",
        choices=["base", "weighted", "structural"],
        default="structural",
        help="Scoring math to use (default: structural)",
    )

    parser.add_argument(
        "--show-palette",
        action="store_true",
        help="Show 2D color bar of extracted palette",
    )

    parser.add_argument(
        "--show-3d", action="store_true", help="Show 3D Oklab point cloud"
    )

    parser.add_argument(
        "--manual-theme",
        type=str,
        default=None,
        help="Force a specific theme to render in 3D (e.g., 'nord')",
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

    strategy_map = {
        "base": score_base,
        "weighted": score_weighted,
        "structural": score_structural,
    }
    scoring_function = strategy_map[args.strategy]

    try:
        print(f"Analyzing {image_path.name}...")
        raw_pixels, oklab_pixels = load_image_sample(str(image_path))
        centers_oklab, weights = extract_image_colors(
            oklab_pixels, n_colors=args.colors
        )

        results = []
        for name, theme in themes.items():
            score = scoring_function(centers_oklab, weights, theme)
            results.append({"theme": name, "score": score})

        results.sort(key=lambda x: x["score"])

    except Exception as e:
        print(f"error: failed to process image: {e}", file=sys.stderr)
        sys.exit(1)

    if args.top is not None:
        results = results[: args.top]

    print(f"\nTop Matches (Strategy: {args.strategy}):")
    print("-" * 40)
    name_width = max(len(res["theme"]) for res in results) if results else 0

    for res in results:
        print(f"{res['theme'].ljust(name_width)} | Score: {res['score']:6.2f}")

    if args.show_palette:
        visualize_palette(centers_oklab, weights)

    if args.show_3d:
        visualize_3d_clusters(
            raw_pixels=raw_pixels,
            centers_oklab=centers_oklab,
            weights=weights,
            themes=themes,
            matches=results,
            top_k=len(results),
            manual_theme=args.manual_theme,
        )


if __name__ == "__main__":
    main()
