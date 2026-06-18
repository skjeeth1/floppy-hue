import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

# Oklab Math Constants

# M1 Matrix: Maps Linear sRGB to cone responses (LMS)
M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005]
])

# M2 Matrix: Maps non-linear cone responses to Oklab axes
M2 = np.array([
    [ 0.2104542553,  0.7936177850, -0.0040720468],
    [ 1.9779984951, -2.4285922050,  0.4505937099],
    [ 0.0259040371,  0.7827717662, -0.8086757660]
])

def rgb_to_oklab(rgb_array: np.ndarray) -> np.ndarray:
    """Vectorized conversion from sRGB (0.0 - 1.0) to Oklab using raw NumPy."""
    # 1. Undo sRGB gamma compression (Linearize)
    linear = np.where(
        rgb_array <= 0.04045, 
        rgb_array / 12.92, 
        ((rgb_array + 0.055) / 1.055) ** 2.4
    )

    # 2. Convert Linear sRGB to LMS cone space
    lms = np.dot(linear, M1.T)

    # 3. Apply non-linearity (cube root). 
    # Using np.sign ensures safety against floating point inaccuracies near zero.
    lms_non_linear = np.sign(lms) * (np.abs(lms) ** (1.0 / 3.0))

    # 4. Convert LMS to Oklab space
    oklab = np.dot(lms_non_linear, M2.T)
    return oklab


def hex_to_oklab(hex_color: str) -> np.ndarray:
    """Convert a '#rrggbb' string into a 1D Oklab array."""
    hex_color = hex_color.lstrip("#")
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0

    # Wrap in array to utilize the vectorized function, then extract the single item
    return rgb_to_oklab(np.array([rgb]))[0]


def extract_image_colors(image_path: str, n_colors: int = 8, sample_size: int = 150):
    """
    Cluster the image's pixels into n_colors dominant colors directly in Oklab space.
    Returns Oklab centers and their relative weights (fraction of pixels).
    """
    img = Image.open(image_path).convert("RGB")
    # Downsample for performance
    # img = img.resize((sample_size, sample_size))
    img.thumbnail((sample_size, sample_size), resample=Image.Resampling.NEAREST)
    pixels = np.array(img).reshape(-1, 3) / 255.0

    # Convert all pixels to Oklab using our custom vectorized NumPy function
    oklab_pixels = rgb_to_oklab(pixels)

    kmeans = MiniBatchKMeans(n_clusters=n_colors, n_init="auto", random_state=42, batch_size=2048, max_iter=50)
    labels = kmeans.fit_predict(oklab_pixels)

    counts = np.bincount(labels, minlength=n_colors)
    weights = counts / counts.sum()
    centers_oklab = kmeans.cluster_centers_

    return centers_oklab, weights


def score_theme(image_oklab: np.ndarray, image_weights: np.ndarray, theme: dict) -> float:
    """
    Calculates the weighted many-to-one perceptual distance between image colors
    and theme colors using Oklab Euclidean distance. Lower score = better match.
    """
    # Pool all theme colors into a single palette array
    theme_hexes = [theme["background"], theme["foreground"]] + theme["accents"]
    theme_oklab = np.array([hex_to_oklab(c) for c in theme_hexes])

    # scipy's cdist calculates the Euclidean distance between EVERY image color 
    # and EVERY theme color instantly in C.
    dist_matrix = cdist(image_oklab, theme_oklab, metric='euclidean')

    # Scale the Oklab distance (usually 0.0 to 1.0) by 100 to match our familiar threshold scale
    dist_matrix *= 100

    # For each image cluster, find the minimum perceptual distance to ANY theme color.
    # This naturally handles gradients without strict index matching.
    best_match_distances = np.min(dist_matrix, axis=1)

    # The final score is the weighted average of these minimum distances
    final_score = np.sum(best_match_distances * image_weights)

    return float(final_score)


def match_themes(image_path: str, themes: dict, n_colors: int = 10) -> list:
    """
    Extracts colors, scores against all themes, and returns the top K matches
    with their perceptual tiers.
    """
    image_lab, image_weights = extract_image_colors(image_path, n_colors=n_colors)

    results = []
    for name, theme in themes.items():
        score = score_theme(image_lab, image_weights, theme)
        results.append((name, score))

    # Sort by lowest Delta-E score (best match)
    results.sort(key=lambda x: x[1])

    matches = []
    for name, score in results:
        matches.append({
            "theme": name, 
            "score": round(score, 2), 
        })

    return matches

