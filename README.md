# Floppy-Hue: Usage Guide

This is a hobby project for my niri rice and to learn and implement a Machine Learning project. It uses K-Means clustering to obtain dominant colors in an image, convert colors of OKLab colorspace, and rank which colorscheme suits it the best.

In future, I would create templating engines to automatically generate color config files for programs (foot terminal, niri, waybar, etc) and dynamically change colorschemes when a wallpaper change occurs.

Hopefully, this is not too resource intensive. I will be trying to reduce resource usage.


## Installation
It is highly recommended to install `floppy-hue` globally using `pipx` to keep its data-science dependencies isolated:

```bash
pipx install floppy-hue
```

## Basic Usage

Point the CLI at any image to find the closest matching standard theme:
Bash

```bash
floppy-hue wallpaper.jpg
```

Advanced Commands

View Top N Matches:

```bash
floppy-hue wallpaper.jpg --top 5
```

Change the Scoring Strategy:
Available strategies: structural (default, penalizes missing BG/FG), weighted (prioritizes hue over lightness), or base (pure Oklab distance).

```bash
floppy-hue wallpaper.jpg --strategy weighted
```

Adjust Color Extraction Resolution:
Increase the number of K-Means clusters extracted (default is 16).

```bash
floppy-hue wallpaper.jpg --colors 24
```

Load Custom Themes:
Pass your own TOML file formatted identically to the default themes.

```bash
floppy-hue wallpaper.jpg --themes-file ~/.config/floppy-hue/themes.toml
```

Visualizations

View 2D Palette Bar:
Pops up a proportional color bar showing the dominant colors extracted from the image.

```bash
floppy-hue wallpaper.jpg --show-palette
```

Interactive 3D Oklab Comparison:
Pops up an interactive 3D scatter plot comparing your image's pixel cloud against the top matching themes. Highlights background (Cyan) and foreground (Magenta) anchor points.

```bash
floppy-hue wallpaper.jpg --show-3d
```

Debug a Failing Theme:
Force a specific theme (e.g., nord) to render in the 3D plot so you can visually see why the math rejected it.

```bash
floppy-hue wallpaper.jpg --show-3d --manual-theme nord
```
