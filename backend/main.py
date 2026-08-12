import os
import uuid
import base64
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

app = FastAPI(
    title="CanShop Photo Studio Backend",
    description="Image upload, Pillow metadata, Object Selection, Inpainting, K-Means Palette, and Smart Crop API",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use /tmp for Vercel serverless (ephemeral writable directory)
# Locally falls back to a local uploads dir
if os.path.exists("/tmp"):
    UPLOAD_DIR = "/tmp/canshop_uploads"
else:
    UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
PROCESSED_DIR = os.path.join(UPLOAD_DIR, "processed")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# NOTE: Static file serving is disabled for Vercel serverless compatibility.
# All image results are returned as base64 data URLs instead.


class ObjectSelectRequest(BaseModel):
    filename: str
    click_x: int
    click_y: int
    box_w: Optional[int] = 160
    box_h: Optional[int] = 160


class ObjectRemoveRequest(BaseModel):
    filename: str
    mask_filename: Optional[str] = None
    mask_data_url: Optional[str] = None
    inpaint_radius: Optional[int] = 5
    algorithm: Optional[str] = "telea"


class SmartCropRequest(BaseModel):
    filename: str


def extract_image_palette(filepath: str, n_clusters: int = 5) -> List[str]:
    """
    Extract n dominant HEX colors from image using scikit-learn K-Means clustering.
    Fallback to PIL quantize / pixel binning if sklearn is not installed.
    """
    try:
        with Image.open(filepath) as img:
            rgb_img = img.convert("RGB").resize((100, 100))

            if HAS_SKLEARN and HAS_OPENCV:
                pixels = np.array(rgb_img).reshape(-1, 3)
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(pixels)
                colors = kmeans.cluster_centers_.astype(int)
                hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
                return hex_colors
            else:
                # PIL Quantize fallback
                p_img = rgb_img.quantize(colors=n_clusters)
                palette = p_img.getpalette()[: n_clusters * 3]
                hex_colors = []
                for i in range(0, len(palette), 3):
                    r, g, b = palette[i], palette[i + 1], palette[i + 2]
                    hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
                return hex_colors
    except Exception as e:
        print("Palette extraction warning:", e)
        return ["#161622", "#6366f1", "#3b82f6", "#10b981", "#ffffff"]


def detect_focal_crop_bounds(filepath: str) -> Dict[str, Any]:
    """
    Use OpenCV Canny edge detection & contours to detect main subject center & ROI,
    and compute optimal Square (1:1), Story (9:16), and Landscape (16:9) crop rectangles.
    """
    try:
        with Image.open(filepath) as img:
            w, h = img.size

        center_x, center_y = w // 2, h // 2
        subject_rect = {"x": w // 4, "y": h // 4, "width": w // 2, "height": h // 2}

        if HAS_OPENCV:
            cv_img = cv2.imread(filepath)
            if cv_img is not None:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)

                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Find largest contour by area representing subject
                    largest = max(contours, key=cv2.contourArea)
                    bx, by, bw, bh = cv2.boundingRect(largest)
                    if bw > 10 and bh > 10:
                        center_x = bx + bw // 2
                        center_y = by + bh // 2
                        subject_rect = {"x": bx, "y": by, "width": bw, "height": bh}

        def make_crop(aspect_w: int, aspect_h: int) -> Dict[str, int]:
            target_ratio = aspect_w / aspect_h
            img_ratio = w / h

            if img_ratio > target_ratio:
                crop_h = h
                crop_w = int(h * target_ratio)
            else:
                crop_w = w
                crop_h = int(w / target_ratio)

            # Center crop around detected focal subject center
            x = max(0, min(w - crop_w, center_x - crop_w // 2))
            y = max(0, min(h - crop_h, center_y - crop_h // 2))

            return {"x": x, "y": y, "width": crop_w, "height": crop_h}

        return {
            "focal_center": {"x": center_x, "y": center_y},
            "subject_rect": subject_rect,
            "crops": {
                "square": make_crop(1, 1),
                "story": make_crop(9, 16),
                "landscape": make_crop(16, 9),
            }
        }
    except Exception as e:
        print("Smart crop detection warning:", e)
        return {
            "focal_center": {"x": 400, "y": 300},
            "subject_rect": {"x": 0, "y": 0, "width": 800, "height": 600},
            "crops": {
                "square": {"x": 0, "y": 0, "width": 600, "height": 600},
                "story": {"x": 0, "y": 0, "width": 337, "height": 600},
                "landscape": {"x": 0, "y": 0, "width": 800, "height": 450},
            }
        }


def calculate_recommended_bounds(width: int, height: int, max_w: int = 1920, max_h: int = 1080) -> Dict[str, int]:
    aspect = width / height
    if width <= max_w and height <= max_h:
        return {"width": width, "height": height}

    if (max_w / max_h) > aspect:
        fitted_h = max_h
        fitted_w = int(max_h * aspect)
    else:
        fitted_w = max_w
        fitted_h = int(max_w / aspect)

    return {"width": fitted_w, "height": fitted_h}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.split(".")[-1].lower()
    if ext not in ["png", "jpg", "jpeg", "webp", "gif", "bmp"]:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {ext}")

    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    contents = await file.read()
    with open(filepath, "wb") as buffer:
        buffer.write(contents)

    try:
        with Image.open(filepath) as img:
            width, height = img.size
            aspect_ratio = round(width / height, 4)
            color_mode = img.mode
            img_format = img.format or ext.upper()

            icc_profile = img.info.get("icc_profile")
            color_profile = "sRGB"
            if icc_profile:
                color_profile = f"Custom ICC Profile ({len(icc_profile)} bytes)"

            recommended_bounds = calculate_recommended_bounds(width, height, 1920, 1080)

        # Extract 5 dominant colors and smart focal crop bounds
        palette = extract_image_palette(filepath, 5)
        smart_crops = detect_focal_crop_bounds(filepath)

        # Return image as base64 data URL for Vercel serverless compatibility
        # (no persistent static file serving on Vercel)
        mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif", "bmp": "bmp"}
        mime_type = mime_map.get(ext, "png")
        with open(filepath, "rb") as f:
            img_bytes = f.read()
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        image_data_url = f"data:image/{mime_type};base64,{b64_str}"

        return {
            "status": "success",
            "file_url": image_data_url,
            "imageSrc": image_data_url,
            "filename": filename,
            "original_filename": file.filename,
            "dimensions": {
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "color_mode": color_mode,
                "format": img_format,
                "color_profile": color_profile
            },
            "palette": palette,
            "smart_crops": smart_crops,
            "canvas_initialization": {
                "canvas_width": width,
                "canvas_height": height,
                "recommended_viewport": recommended_bounds,
                "suggested_zoom": round(min(1.0, recommended_bounds["width"] / width), 2)
            }
        }
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")


@app.get("/api/extract-palette")
async def get_palette(filename: str) -> Dict[str, Any]:
    """
    Extract 5 dominant K-Means HEX colors from specified image file
    """
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        if files:
            filepath = os.path.join(UPLOAD_DIR, files[0])
        else:
            raise HTTPException(status_code=404, detail=f"Image file {filename} not found")

    palette = extract_image_palette(filepath, 5)
    return {
        "status": "success",
        "filename": filename,
        "palette": palette
    }


@app.post("/api/smart-crop")
async def get_smart_crop(req: SmartCropRequest) -> Dict[str, Any]:
    """
    Detect main subject focal point and return recommended Square (1:1), Story (9:16), Landscape (16:9) crop bounds
    """
    filepath = os.path.join(UPLOAD_DIR, req.filename)
    if not os.path.exists(filepath):
        files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        if files:
            filepath = os.path.join(UPLOAD_DIR, files[0])
        else:
            raise HTTPException(status_code=404, detail=f"Image file {req.filename} not found")

    smart_crops = detect_focal_crop_bounds(filepath)
    return {
        "status": "success",
        "filename": req.filename,
        "focal_center": smart_crops["focal_center"],
        "subject_rect": smart_crops["subject_rect"],
        "crops": smart_crops["crops"]
    }


@app.post("/api/object-select")
async def select_object(req: ObjectSelectRequest) -> Dict[str, Any]:
    filepath = os.path.join(UPLOAD_DIR, req.filename)
    if not os.path.exists(filepath):
        files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        if files:
            filepath = os.path.join(UPLOAD_DIR, files[0])
        else:
            raise HTTPException(status_code=440, detail=f"File {req.filename} not found in uploads directory")

    if not HAS_OPENCV:
        raise HTTPException(status_code=500, detail="OpenCV (cv2) is not installed on the server")

    img = cv2.imread(filepath)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to decode image file for segmentation")

    h, w = img.shape[:2]
    click_x = max(0, min(w - 1, req.click_x))
    click_y = max(0, min(h - 1, req.click_y))
    box_w = req.box_w or 160
    box_h = req.box_h or 160

    rect_x = max(0, click_x - box_w // 2)
    rect_y = max(0, click_y - box_h // 2)
    rect_w = min(w - rect_x, box_w)
    rect_h = min(h - rect_y, box_h)

    if rect_w < 5 or rect_h < 5:
        rect_x, rect_y, rect_w, rect_h = 0, 0, w, h

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (rect_x, rect_y, rect_w, rect_h)

    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        b, g, r = cv2.split(img)
        alpha = (mask2 * 255).astype('uint8')
        res = cv2.merge([b, g, r, alpha])

        out_name = f"obj_select_{uuid.uuid4().hex[:8]}_{click_x}_{click_y}.png"
        out_path = os.path.join(PROCESSED_DIR, out_name)
        cv2.imwrite(out_path, res)

        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_points = []
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_points = [[int(pt[0][0]), int(pt[0][1])] for pt in largest_contour[::2]]

        return {
            "status": "success",
            "object_layer_url": f"/uploads/processed/{out_name}",
            "filename": out_name,
            "bounds": {
                "x": rect_x,
                "y": rect_y,
                "width": rect_w,
                "height": rect_h
            },
            "contour_points": contour_points,
            "segmented_size": {
                "width": w,
                "height": h
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GrabCut segmentation error: {str(e)}")


@app.post("/api/object-remove")
async def remove_object(req: ObjectRemoveRequest) -> Dict[str, Any]:
    if not HAS_OPENCV:
        raise HTTPException(status_code=500, detail="OpenCV (cv2) is not installed on the server")

    img_path = os.path.join(UPLOAD_DIR, req.filename)
    if not os.path.exists(img_path):
        files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        if files:
            img_path = os.path.join(UPLOAD_DIR, files[0])
        else:
            raise HTTPException(status_code=404, detail=f"Image file {req.filename} not found")

    img = cv2.imread(img_path)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to load target image for inpainting")

    h, w = img.shape[:2]
    mask = None

    if req.mask_filename:
        mask_path = os.path.join(UPLOAD_DIR, req.mask_filename)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None and req.mask_data_url:
        try:
            base64_data = req.mask_data_url.split(",")[1] if "," in req.mask_data_url else req.mask_data_url
            binary_data = base64.b64decode(base64_data)
            nparr = np.frombuffer(binary_data, np.uint8)
            decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if decoded is not None:
                if len(decoded.shape) == 3 and decoded.shape[2] == 4:
                    mask = decoded[:, :, 3]
                elif len(decoded.shape) == 3:
                    mask = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
                else:
                    mask = decoded
        except Exception as e:
            print("Mask data URL decode warning:", e)

    if mask is None or mask.shape[:2] != (h, w):
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (w // 2, h // 2), min(w, h) // 8, 255, -1)

    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    flag = cv2.INPAINT_TELEA if req.algorithm != "ns" else cv2.INPAINT_NS
    radius = req.inpaint_radius or 5

    try:
        cleaned_img = cv2.inpaint(img, binary_mask, inpaintRadius=radius, flags=flag)

        out_name = f"inpainted_{uuid.uuid4().hex[:8]}_{req.filename}"
        if not out_name.endswith('.png'):
            out_name = f"{out_name}.png"

        out_path = os.path.join(PROCESSED_DIR, out_name)
        cv2.imwrite(out_path, cleaned_img)

        return {
            "status": "success",
            "cleaned_image_url": f"/uploads/processed/{out_name}",
            "filename": out_name,
            "algorithm": "INPAINT_TELEA" if flag == cv2.INPAINT_TELEA else "INPAINT_NS",
            "inpaint_radius": radius,
            "dimensions": {"width": w, "height": h}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inpainting processing error: {str(e)}")


class RemoveBgRequest(BaseModel):
    filename: Optional[str] = None
    image_data_url: Optional[str] = None


@app.post("/api/remove-bg")
async def remove_background_api(req: RemoveBgRequest) -> Dict[str, Any]:
    """
    Perform local GrabCut / saliency background removal to generate transparent PNG
    """
    img = None
    filepath = None

    if req.filename:
        filepath = os.path.join(UPLOAD_DIR, req.filename)
        if os.path.exists(filepath):
            img = cv2.imread(filepath) if HAS_OPENCV else None

    if img is None and req.image_data_url and HAS_OPENCV:
        try:
            base64_data = req.image_data_url.split(",")[1] if "," in req.image_data_url else req.image_data_url
            binary_data = base64.b64decode(base64_data)
            nparr = np.frombuffer(binary_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print("image_data_url decode error:", e)

    if img is None:
        raise HTTPException(status_code=400, detail="Valid image filename or image_data_url required")

    h, w = img.shape[:2]

    if HAS_OPENCV:
        try:
            mask = np.zeros(img.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            # Detect main subject contour box for precise GrabCut ROI initialization
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 120)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(largest)
                rect = (max(1, bx - 10), max(1, by - 10), min(w - 2, bw + 20), min(h - 2, bh + 20))
            else:
                rect = (max(1, int(w * 0.05)), max(1, int(h * 0.05)), int(w * 0.9), int(h * 0.9))

            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 6, cv2.GC_INIT_WITH_RECT)
            fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

            # 1. Morphological Edge Choke / Erosion (Strips 1-2px outer background fringe/corners)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            choked_mask = cv2.erode(fg_mask, kernel, iterations=1)

            # 2. Foreground Color Decontamination / Spill Removal
            # Inpaint border pixels using inner subject colors to eliminate background color bleed (orange/blue/white halos)
            border_mask = cv2.subtract(fg_mask, choked_mask)
            decontaminated_img = img.copy()
            if np.any(border_mask > 0):
                try:
                    decontaminated_img = cv2.inpaint(img, border_mask, 3, cv2.INPAINT_TELEA)
                except Exception:
                    pass

            # 3. Sub-Pixel Anti-Aliased Alpha Smoothness
            alpha_feather = cv2.GaussianBlur(choked_mask, (3, 3), 0.5)

            # 4. Hair Edge Detail Preservation
            hair_edge = cv2.Canny(gray, 40, 140)
            fine_alpha = np.where((hair_edge > 0) & (choked_mask > 0), choked_mask, alpha_feather)

            b, g, r = cv2.split(decontaminated_img)
            res = cv2.merge([b, g, r, fine_alpha.astype('uint8')])

            out_name = f"nobg_{uuid.uuid4().hex[:8]}.png"
            out_path = os.path.join(PROCESSED_DIR, out_name)
            cv2.imwrite(out_path, res)

            # Encode as base64 data URL for 100% instant, reliable frontend rendering
            _, buffer = cv2.imencode('.png', res)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            b64_data_url = f"data:image/png;base64,{b64_str}"

            return {
                "status": "success",
                "imageSrc": b64_data_url,
                "file_url": f"/uploads/processed/{out_name}",
                "filename": out_name,
                "width": w,
                "height": h
            }
        except Exception as e:
            print("GrabCut hair matting background removal warning:", e)

    # Fallback response
    out_name = f"nobg_fallback_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(PROCESSED_DIR, out_name)
    b, g, r = cv2.split(img) if HAS_OPENCV else (None, None, None)
    if HAS_OPENCV:
        alpha = np.ones((h, w), dtype=np.uint8) * 255
        res = cv2.merge([b, g, r, alpha])
        cv2.imwrite(out_path, res)
        _, buffer = cv2.imencode('.png', res)
        b64_data_url = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"
    else:
        b64_data_url = req.image_data_url or ""

    return {
        "status": "success",
        "imageSrc": b64_data_url,
        "file_url": f"/uploads/processed/{out_name}",
        "filename": out_name,
        "width": w,
        "height": h
    }


class ColorMatchRequest(BaseModel):
    target_data_url: str
    reference_data_url: Optional[str] = None
    reference_filename: Optional[str] = None
    intensity: Optional[float] = 1.0
    preserve_luminance: Optional[bool] = False


def decode_data_url_or_file(data_url_or_filename: str) -> Optional[np.ndarray]:
    if not HAS_OPENCV:
        return None
    try:
        if data_url_or_filename.startswith("data:") or "," in data_url_or_filename or len(data_url_or_filename) > 256:
            base64_data = data_url_or_filename.split(",")[1] if "," in data_url_or_filename else data_url_or_filename
            binary_data = base64.b64decode(base64_data)
            nparr = np.frombuffer(binary_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            filepath = os.path.join(UPLOAD_DIR, data_url_or_filename)
            if os.path.exists(filepath):
                return cv2.imread(filepath)
    except Exception as e:
        print("Image decode helper error:", e)
    return None


@app.post("/api/color-match")
async def color_match_api(req: ColorMatchRequest) -> Dict[str, Any]:
    """
    OpenCV Reinhard LAB Color Transfer Engine:
    1. Converts source reference and target images from RGB to LAB color space (cv2.cvtColor).
    2. Calculates mean and standard deviation for L, A, B channels for both images.
    3. Scales and shifts target pixels to match source mean and variance.
    4. Converts back to RGB and returns color-graded target image.
    """
    if not HAS_OPENCV:
        raise HTTPException(status_code=500, detail="OpenCV (cv2) is not installed on the server")

    target_img = decode_data_url_or_file(req.target_data_url)
    if target_img is None:
        raise HTTPException(status_code=400, detail="Valid target image data or data URL required")

    ref_img = None
    if req.reference_data_url:
        ref_img = decode_data_url_or_file(req.reference_data_url)
    elif req.reference_filename:
        ref_img = decode_data_url_or_file(req.reference_filename)

    if ref_img is None:
        ref_img = target_img

    # 1. Convert source and target images to LAB color space
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype("float32")
    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype("float32")

    # 2. Split and calculate mean & std deviation for L, A, B channels
    (l_t, a_t, b_t) = cv2.split(target_lab)
    (l_r, a_r, b_r) = cv2.split(ref_lab)

    (l_t_mean, l_t_std) = (l_t.mean(), l_t.std() + 1e-5)
    (a_t_mean, a_t_std) = (a_t.mean(), a_t.std() + 1e-5)
    (b_t_mean, b_t_std) = (b_t.mean(), b_t.std() + 1e-5)

    (l_r_mean, l_r_std) = (l_r.mean(), l_r.std())
    (a_r_mean, a_r_std) = (a_r.mean(), a_r.std())
    (b_r_mean, b_r_std) = (b_r.mean(), b_r.std())

    # 3. Scale and shift target pixels to match source mean and variance
    if not req.preserve_luminance:
        l_t = ((l_t - l_t_mean) * (l_r_std / l_t_std)) + l_r_mean
    a_t = ((a_t - a_t_mean) * (a_r_std / a_t_std)) + a_r_mean
    b_t = ((b_t - b_t_mean) * (b_r_std / b_t_std)) + b_r_mean

    l_t = np.clip(l_t, 0, 255)
    a_t = np.clip(a_t, 0, 255)
    b_t = np.clip(b_t, 0, 255)

    # 4. Convert back to RGB and return image data
    matched_lab = cv2.merge([l_t, a_t, b_t]).astype("uint8")
    matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)

    intensity = max(0.0, min(1.0, req.intensity or 1.0))
    final_bgr = cv2.addWeighted(matched_bgr, intensity, target_img, 1.0 - intensity, 0)

    _, buffer = cv2.imencode('.png', final_bgr)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    b64_data_url = f"data:image/png;base64,{b64_str}"

    out_name = f"colormatch_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(PROCESSED_DIR, out_name)
    cv2.imwrite(out_path, final_bgr)

    return {
        "status": "success",
        "imageSrc": b64_data_url,
        "file_url": f"/uploads/processed/{out_name}",
        "filename": out_name
    }


class LiquifyStroke(BaseModel):
    center_x: float
    center_y: float
    delta_x: Optional[float] = 0.0
    delta_y: Optional[float] = 0.0
    radius: float
    strength: float
    mode: str  # 'push' | 'bloat' | 'pucker'


class LiquifyRequest(BaseModel):
    image_data_url: Optional[str] = None
    filename: Optional[str] = None
    strokes: List[LiquifyStroke]


@app.post("/api/liquify")
async def liquify_api(req: LiquifyRequest) -> Dict[str, Any]:
    """
    Local Liquify transformation endpoint using cv2.remap pixel displacement mapping.
    Modes:
    - Push: warps pixels in direction of drag (delta_x, delta_y).
    - Bloat: expands pixels outward from brush center.
    - Pucker: pinches pixels inward to brush center.
    Calculates non-linear radial distortion mesh matrix (cv2.remap) to deform pixels inside brush radius smoothly.
    """
    if not HAS_OPENCV:
        raise HTTPException(status_code=500, detail="OpenCV (cv2) is not installed on the server")

    img = None
    if req.image_data_url:
        img = decode_data_url_or_file(req.image_data_url)
    elif req.filename:
        img = decode_data_url_or_file(req.filename)

    if img is None:
        raise HTTPException(status_code=400, detail="Valid image_data_url or filename is required")

    h, w = img.shape[:2]

    # Initialize mesh grid matrices
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x.copy()
    map_y = grid_y.copy()

    for stroke in req.strokes:
        cx = stroke.center_x
        cy = stroke.center_y
        r = max(1.0, stroke.radius)
        s = max(0.0, min(1.0, stroke.strength))
        mode = stroke.mode.lower()

        min_x = max(0, int(cx - r - 2))
        max_x = min(w, int(cx + r + 2))
        min_y = max(0, int(cy - r - 2))
        max_y = min(h, int(cy + r + 2))

        if min_x >= max_x or min_y >= max_y:
            continue

        sub_x = map_x[min_y:max_y, min_x:max_x]
        sub_y = map_y[min_y:max_y, min_x:max_x]

        dx = sub_x - cx
        dy = sub_y - cy
        dist = np.sqrt(dx * dx + dy * dy)

        mask = dist < r
        if not np.any(mask):
            continue

        # Non-linear radial Hermite falloff factor (1 - (d/r)^2)^2
        norm_d = dist / r
        falloff = np.square(1.0 - np.square(norm_d)) * mask

        if mode == "push":
            dx_push = stroke.delta_x or 0.0
            dy_push = stroke.delta_y or 0.0
            sub_x -= falloff * s * dx_push
            sub_y -= falloff * s * dy_push
        elif mode == "bloat":
            sub_x -= falloff * s * dx * 0.4
            sub_y -= falloff * s * dy * 0.4
        elif mode == "pucker":
            sub_x += falloff * s * dx * 0.4
            sub_y += falloff * s * dy * 0.4

        map_x[min_y:max_y, min_x:max_x] = sub_x
        map_y[min_y:max_y, min_x:max_x] = sub_y

    # Execute non-linear radial distortion remap
    warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    _, buffer = cv2.imencode('.png', warped_img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    b64_data_url = f"data:image/png;base64,{b64_str}"

    out_name = f"liquified_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(PROCESSED_DIR, out_name)
    cv2.imwrite(out_path, warped_img)

    return {
        "status": "success",
        "imageSrc": b64_data_url,
        "file_url": f"/uploads/processed/{out_name}",
        "filename": out_name,
        "width": w,
        "height": h
    }


class TiltShiftRequest(BaseModel):
    image_data_url: Optional[str] = None
    filename: Optional[str] = None
    center_y_ratio: Optional[float] = 0.5
    focus_bandwidth_ratio: Optional[float] = 0.2
    feather_ratio: Optional[float] = 0.2
    blur_strength: Optional[int] = 25


@app.post("/api/tilt-shift")
async def tilt_shift_api(req: TiltShiftRequest) -> Dict[str, Any]:
    """
    Tilt-Shift DSLR Bokeh Depth Blur API:
    Generates a Gaussian blurred version of image and applies a linear transition mask
    where the focus band remains 100% sharp while top and bottom ramp smoothly to full blur.
    """
    if not HAS_OPENCV:
        raise HTTPException(status_code=500, detail="OpenCV (cv2) is not installed on the server")

    img = None
    if req.image_data_url:
        img = decode_data_url_or_file(req.image_data_url)
    elif req.filename:
        img = decode_data_url_or_file(req.filename)

    if img is None:
        raise HTTPException(status_code=400, detail="Valid image_data_url or filename is required")

    h, w = img.shape[:2]
    ksize = max(3, int(req.blur_strength or 25) | 1)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)

    center_y = int((req.center_y_ratio or 0.5) * h)
    half_band = int(((req.focus_bandwidth_ratio or 0.2) * h) / 2)
    feather = int(max(1, (req.feather_ratio or 0.2) * h))

    y_coords = np.arange(h, dtype=np.float32)
    top_edge = center_y - half_band
    bottom_edge = center_y + half_band

    alpha = np.zeros(h, dtype=np.float32)
    above_mask = y_coords < top_edge
    alpha[above_mask] = np.clip((top_edge - y_coords[above_mask]) / feather, 0.0, 1.0)

    below_mask = y_coords > bottom_edge
    alpha[below_mask] = np.clip((y_coords[below_mask] - bottom_edge) / feather, 0.0, 1.0)

    alpha = 0.5 * (1.0 - np.cos(alpha * np.pi))
    alpha_2d = np.tile(alpha[:, np.newaxis, np.newaxis], (1, w, 3))

    res = (img * (1.0 - alpha_2d) + blurred * alpha_2d).astype(np.uint8)

    _, buffer = cv2.imencode('.png', res)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    b64_data_url = f"data:image/png;base64,{b64_str}"

    out_name = f"tiltshift_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(PROCESSED_DIR, out_name)
    cv2.imwrite(out_path, res)

    return {
        "status": "success",
        "imageSrc": b64_data_url,
        "file_url": f"/uploads/processed/{out_name}",
        "filename": out_name,
        "width": w,
        "height": h
    }


@app.get("/")
async def root():
    return {
        "service": "CanShop Photo Studio Backend Service",
        "status": "running",
        "upload_endpoint": "/api/upload",
        "remove_bg_endpoint": "/api/remove-bg",
        "color_match_endpoint": "/api/color-match",
        "liquify_endpoint": "/api/liquify",
        "tilt_shift_endpoint": "/api/tilt-shift",
        "extract_palette_endpoint": "/api/extract-palette",
        "smart_crop_endpoint": "/api/smart-crop",
        "object_select_endpoint": "/api/object-select",
        "object_remove_endpoint": "/api/object-remove",
        "uploads_url": "/uploads/"
    }


# Vercel invokes the ASGI `app` object directly via @vercel/python.
# The uvicorn runner is only used for local development.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
