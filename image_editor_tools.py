import os
import uuid
import cv2
import numpy as np
from PIL import Image

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
PROCESSED_DIR = os.path.join(UPLOAD_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def apply_color_match(target_path: str, reference_path: str) -> str:
    """Transfers color atmosphere from reference image to target image in LAB space."""
    target = cv2.imread(target_path).astype(np.float32)
    reference = cv2.imread(reference_path).astype(np.float32)

    if target is None or reference is None:
        raise ValueError("Could not read target or reference image")

    # Convert RGB to LAB color space
    target_lab = cv2.cvtColor(target / 255.0, cv2.COLOR_BGR2Lab)
    ref_lab = cv2.cvtColor(reference / 255.0, cv2.COLOR_BGR2Lab)

    # Calculate statistics (mean and std dev) for both
    (t_mean, t_std) = cv2.meanStdDev(target_lab)
    (r_mean, r_std) = cv2.meanStdDev(ref_lab)

    t_mean = np.hstack([t_mean[0][0], t_mean[1][0], t_mean[2][0]])
    t_std = np.hstack([t_std[0][0], t_std[1][0], t_std[2][0]]) + 1e-5
    r_mean = np.hstack([r_mean[0][0], r_mean[1][0], r_mean[2][0]])
    r_std = np.hstack([r_std[0][0], r_std[1][0], r_std[2][0]])

    # Color transfer algorithm
    res_lab = target_lab - t_mean
    res_lab = (r_std / t_std) * res_lab
    res_lab += r_mean

    # Clip values and convert back to BGR
    res_lab = np.clip(res_lab * 255.0, 0, 255).astype(np.uint8)
    res_bgr = cv2.cvtColor(res_lab, cv2.COLOR_Lab2BGR)

    out_name = f"colormatch_{uuid.uuid4().hex[:6]}.png"
    out_path = os.path.join(PROCESSED_DIR, out_name)
    cv2.imwrite(out_path, res_bgr)
    return f"/uploads/processed/{out_name}"


def apply_liquify_bloat(image_path: str, center_x: int, center_y: int, radius: int = 100, strength: float = 0.5) -> str:
    """Applies a localized 'Bloat' (expand) distortion warp inside brush radius."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file")
    h, w = img.shape[:2]

    # Create meshgrid for coordinate mapping
    grid_y, grid_x = np.indices((h, w), dtype=np.float32)

    # Distance calculation from brush center
    dx = grid_x - center_x
    dy = grid_y - center_y
    distance = np.sqrt(dx**2 + dy**2)

    # Mask region within brush radius
    mask = distance < radius

    # Calculate radial distortion factor
    r_norm = distance[mask] / max(1.0, radius)
    distortion = np.power(r_norm, strength)  # Non-linear expansion

    map_x = grid_x.copy()
    map_y = grid_y.copy()

    map_x[mask] = center_x + dx[mask] * distortion
    map_y[mask] = center_y + dy[mask] * distortion

    # Remap pixels locally using bilinear interpolation
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    out_name = f"liquify_{uuid.uuid4().hex[:6]}.png"
    out_path = os.path.join(PROCESSED_DIR, out_name)
    cv2.imwrite(out_path, warped)
    return f"/uploads/processed/{out_name}"


def apply_tilt_shift_blur(
    image_path: str,
    center_y_ratio: float = 0.5,
    focus_bandwidth_ratio: float = 0.2,
    feather_ratio: float = 0.2,
    blur_strength: int = 25
) -> str:
    """
    Creates a Tilt-Shift DSLR Bokeh Depth Blur using linear transition alpha gradient mask
    and Gaussian blur blending. Focus band remains 100% sharp while top and bottom ramp to full blur.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file")

    h, w = img.shape[:2]

    # 1. Generate Gaussian blurred version
    ksize = max(3, int(blur_strength) | 1)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # 2. Generate linear transition alpha mask
    center_y = int(center_y_ratio * h)
    half_band = int((focus_bandwidth_ratio * h) / 2)
    feather = int(max(1, feather_ratio * h))

    y_coords = np.arange(h, dtype=np.float32)

    # Compute distance from focal sharp band
    top_edge = center_y - half_band
    bottom_edge = center_y + half_band

    alpha = np.zeros(h, dtype=np.float32)

    # Above focal band
    above_mask = y_coords < top_edge
    alpha[above_mask] = np.clip((top_edge - y_coords[above_mask]) / feather, 0.0, 1.0)

    # Below focal band
    below_mask = y_coords > bottom_edge
    alpha[below_mask] = np.clip((y_coords[below_mask] - bottom_edge) / feather, 0.0, 1.0)

    # Cosine smooth transition ramp
    alpha = 0.5 * (1.0 - np.cos(alpha * np.pi))

    # Expand to 2D image mask (h, w, 3)
    alpha_2d = np.tile(alpha[:, np.newaxis, np.newaxis], (1, w, 3))

    # Alpha blending: 0 = completely sharp in focus band, 1 = full blur
    res = (img * (1.0 - alpha_2d) + blurred * alpha_2d).astype(np.uint8)

    out_name = f"tiltshift_{uuid.uuid4().hex[:6]}.png"
    out_path = os.path.join(PROCESSED_DIR, out_name)
    cv2.imwrite(out_path, res)
    return f"/uploads/processed/{out_name}"
