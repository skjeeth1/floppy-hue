import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Oklab Math Constants
# ------------------------------------------------------------------
M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
M1_INV = np.linalg.inv(M1)
M2_INV = np.linalg.inv(M2)

# ------------------------------------------------------------------
# Color Space Conversions
# ------------------------------------------------------------------
def rgb_to_oklab(rgb_array: np.ndarray) -> np.ndarray:
    linear = np.where(rgb_array <= 0.04045, rgb_array / 12.92, ((rgb_array + 0.055) / 1.055) ** 2.4)
    lms = np.dot(linear, M1.T)
    lms_non_linear = np.sign(lms) * (np.abs(lms) ** (1.0 / 3.0))
    return np.dot(lms_non_linear, M2.T)

def hex_to_oklab(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0
    return rgb_to_oklab(np.array([rgb]))[0]

def oklab_to_rgb(oklab_array: np.ndarray) -> np.ndarray:
    lms_non_linear = np.dot(oklab_array, M2_INV.T)
    lms = lms_non_linear ** 3
    linear = np.clip(np.dot(lms, M1_INV.T), 0.0, 1.0)
    rgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * (linear ** (1.0 / 2.4)) - 0.055)
    return np.clip(rgb, 0.0, 1.0)

# ------------------------------------------------------------------
# Centralized Image Processing
# ------------------------------------------------------------------
def load_image_sample(image_path: str, sample_size: int = 150):
    """Loads and downsamples the image ONCE for the entire pipeline."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((sample_size, sample_size), resample=Image.Resampling.NEAREST)
    pixels = np.array(img).reshape(-1, 3) / 255.0
    oklab_pixels = rgb_to_oklab(pixels)
    return pixels, oklab_pixels

def extract_image_colors(oklab_pixels: np.ndarray, n_colors: int = 16):
    """Runs KMeans on pre-loaded pixels."""
    kmeans = MiniBatchKMeans(n_clusters=n_colors, n_init="auto", random_state=42, batch_size=2048, max_iter=50)
    labels = kmeans.fit_predict(oklab_pixels)
    counts = np.bincount(labels, minlength=n_colors)
    weights = counts / counts.sum()
    return kmeans.cluster_centers_, weights

# ------------------------------------------------------------------
# Scoring Strategies
# ------------------------------------------------------------------
def _get_theme_oklab_array(theme: dict) -> np.ndarray:
    theme_hexes = [theme["background"], theme["foreground"], *theme["accents"]]
    return np.array([hex_to_oklab(c) for c in theme_hexes])

def score_base(image_oklab: np.ndarray, image_weights: np.ndarray, theme: dict) -> float:
    """Standard Euclidean distance in Oklab space."""
    theme_oklab = _get_theme_oklab_array(theme)
    dist_matrix = cdist(image_oklab, theme_oklab, metric="euclidean") * 100
    best_match_distances = np.min(dist_matrix, axis=1)
    return float(np.sum(best_match_distances * image_weights))

def score_weighted(image_oklab: np.ndarray, image_weights: np.ndarray, theme: dict) -> float:
    """Penalizes Lightness differences so Hue matters more."""
    theme_oklab = _get_theme_oklab_array(theme)
    weights_array = np.array([0.5, 1.0, 1.0]) # L, a, b
    
    dist_matrix = cdist(image_oklab * weights_array, theme_oklab * weights_array, metric="euclidean") * 100
    best_match_distances = np.min(dist_matrix, axis=1)
    return float(np.sum(best_match_distances * image_weights))

def score_structural(image_oklab: np.ndarray, image_weights: np.ndarray, theme: dict) -> float:
    """Uses the weighted score, but heavily penalizes if the Theme BG/FG is missing."""
    base_w_score = score_weighted(image_oklab, image_weights, theme)
    weights_array = np.array([0.5, 1.0, 1.0])
    weighted_image = image_oklab * weights_array
    
    bg_oklab = hex_to_oklab(theme["background"]) * weights_array
    fg_oklab = hex_to_oklab(theme["foreground"]) * weights_array

    bg_dist = np.min(cdist([bg_oklab], weighted_image, metric="euclidean")) * 100
    fg_dist = np.min(cdist([fg_oklab], weighted_image, metric="euclidean")) * 100

    # 1.5x penalty for missing BG, 0.5x penalty for missing FG
    return float(base_w_score + (bg_dist * 2) + (fg_dist * 0.5))

# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------
def visualize_palette(centers_oklab: np.ndarray, weights: np.ndarray):
    """Plots the extracted palette directly from pre-computed centers."""
    valid_indices = weights > 0
    c_oklab, w = centers_oklab[valid_indices], weights[valid_indices]
    
    order = np.argsort(w)[::-1]
    centers_rgb = oklab_to_rgb(c_oklab[order])
    w = w[order]
    
    fig, ax = plt.subplots(figsize=(10, 2))
    left = 0 
    for rgb, weight in zip(centers_rgb, w):
        ax.barh(0, weight, left=left, color=rgb, height=1.0, edgecolor='none')
        left += weight
        
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off') 
    plt.title(f"Extracted Palette ({len(w)} active colors)")
    plt.tight_layout()
    plt.show()

def visualize_3d_clusters(raw_pixels: np.ndarray, centers_oklab: np.ndarray, weights: np.ndarray, themes: dict, matches: list[dict], max_points: int = 2000, top_k: int = 2, manual_theme: str = None):
    """Plots the 3D space with explicitly separated BG (Cyan) and FG (Magenta) anchors."""
    valid = weights > 0
    c_oklab = centers_oklab[valid]
    centers_rgb = oklab_to_rgb(c_oklab)
    
    np.random.shuffle(raw_pixels)
    sample_pixels = raw_pixels[:max_points]
    sample_oklab = rgb_to_oklab(sample_pixels)
    
    matches = matches[:top_k]
    themes_to_plot = []
    for i, match in enumerate(matches):
        t_name = match["theme"]
        themes_to_plot.append({
            "name": t_name,
            "label": f"#{i+1}: {t_name}",
            "hexes": [themes[t_name]["background"], themes[t_name]["foreground"], *themes[t_name]["accents"]]
        })
        
    if manual_theme and manual_theme in themes:
        if not any(t["name"] == manual_theme for t in themes_to_plot):
            themes_to_plot.append({
                "name": manual_theme,
                "label": f"Manual Input: {manual_theme}",
                "hexes": [themes[manual_theme]["background"], themes[manual_theme]["foreground"], *themes[manual_theme]["accents"]]
            })

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot Raw Pixels
    ax.scatter(sample_oklab[:, 1], sample_oklab[:, 2], sample_oklab[:, 0], c=sample_pixels, s=10, alpha=0.15, label="Raw Pixels")
    
    # 2. Plot Image Dominant Colors
    ax.scatter(c_oklab[:, 1], c_oklab[:, 2], c_oklab[:, 0], c=centers_rgb, s=200, marker='o', edgecolor='black', linewidth=2.0, alpha=1.0, label="Dominant Colors")
    
    marker_shapes = ['s', '^', 'D', 'P', 'X', 'v', '*'] 
    
    # 3. Plot Themes
    for i, t_info in enumerate(themes_to_plot):
        marker = marker_shapes[i % len(marker_shapes)]
        
        # Isolate hexes
        bg_hex = t_info["hexes"][0]
        fg_hex = t_info["hexes"][1]
        accent_hexes = t_info["hexes"][2:]

        # Convert BG to Oklab/RGB
        bg_oklab = hex_to_oklab(bg_hex)
        bg_rgb = [int(bg_hex.lstrip('#')[j:j+2], 16) / 255.0 for j in (0, 2, 4)]

        # Convert FG to Oklab/RGB
        fg_oklab = hex_to_oklab(fg_hex)
        fg_rgb = [int(fg_hex.lstrip('#')[j:j+2], 16) / 255.0 for j in (0, 2, 4)]

        # Convert Accents to Oklab/RGB
        accents_oklab = np.array([hex_to_oklab(h) for h in accent_hexes])
        accents_rgb = np.array([[int(h.lstrip('#')[j:j+2], 16) / 255.0 for j in (0, 2, 4)] for h in accent_hexes])

        # A. Plot Accents (Smaller, standard white border)
        ax.scatter(accents_oklab[:, 1], accents_oklab[:, 2], accents_oklab[:, 0], 
                   c=accents_rgb, s=80, marker=marker, edgecolor='white', linewidth=0.5, alpha=0.7, 
                   label=f"{t_info['label']} (Accents)")
        
        # B. Plot BG Anchor (Cyan border, slightly larger)
        # We wrap the coordinates in lists so scatter knows it's a single point sequence
        ax.scatter([bg_oklab[1]], [bg_oklab[2]], [bg_oklab[0]], 
                   c=[bg_rgb], s=200, marker=marker, edgecolor='cyan', linewidth=2.5, alpha=1.0, 
                   label=f"{t_info['label']} (BG)")

        # C. Plot FG Anchor (Magenta border, slightly larger)
        ax.scatter([fg_oklab[1]], [fg_oklab[2]], [fg_oklab[0]], 
                   c=[fg_rgb], s=200, marker=marker, edgecolor='magenta', linewidth=2.5, alpha=1.0, 
                   label=f"{t_info['label']} (FG)")

    ax.set_xlabel('a (Green <-> Red)')
    ax.set_ylabel('b (Blue <-> Yellow)')
    ax.set_zlabel('L (Lightness)')
    plt.title("Oklab 3D Color Space Comparison")
    
    # Legend adjustments to keep it clean
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), scatterpoints=1)
    plt.tight_layout()
    plt.show()
