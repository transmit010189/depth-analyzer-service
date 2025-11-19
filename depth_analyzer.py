# /app/depth_analyzer.py

import os
import logging
import time
import cv2
import numpy as np
from PIL import Image
import io
import re
from typing import Optional, Tuple, List, Dict, Any, TYPE_CHECKING
from statistics import mode, StatisticsError

# --- START: 新增的程式碼 (第一處) ---
# 自動偵測 CUDA GPU 並設定運算裝置
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = "cpu"  # 如果 torch 沒安裝，預設為 cpu

logging.info(f"--- Global computation device set to: {DEVICE} ---")
# --- END: 新增的程式碼 (第一處) ---

# --- Type Hinting for Conditionally Imported Modules ---
if TYPE_CHECKING:
    from ultralytics import YOLO as UltralyticsYOLO_for_typehinting
    from ultralytics.engine.results import Results as UltralyticsResults_for_typehinting
    import google.generativeai as genai_for_typehinting
    import google.generativeai.types as genai_types_for_typehinting
    GenerativeModel_Type = genai_for_typehinting.GenerativeModel
else:
    UltralyticsYOLO_for_typehinting = Any
    UltralyticsResults_for_typehinting = Any
    GenerativeModel_Type = Any
    genai_types_for_typehinting = Any

# --- YOLO Model Import ---
_YOLO_CLASS_ACTUAL = None
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO as _YOLO_CLASS_ACTUAL
    YOLO_AVAILABLE = True
except ImportError:
    logging.warning("ultralytics library not found. YOLO object detection/segmentation will be unavailable.")

# --- Gemini (Generative AI) Imports ---
genai_models = None
genai_configure = None
try:
    import google.generativeai.generative_models as genai_models
    from google.generativeai.client import configure as genai_configure
    from google.generativeai import types as genai_types
    GEMINI_API_AVAILABLE = True
except ImportError:
    logging.warning("google-generativeai library not found. Gemini API processing will be unavailable.")
    genai_models = None
    genai_types = None
    GEMINI_API_AVAILABLE = False

# --- Configuration ---
MODEL_PATH_ENV_VAR = "MODEL_PATH_ENV"
MODEL_PATH = os.environ.get(MODEL_PATH_ENV_VAR)

SEG_MODEL_PATH_ENV_VAR = "SEG_MODEL_PATH_ENV" # New for segmentation model
SEG_MODEL_PATH = os.environ.get(SEG_MODEL_PATH_ENV_VAR) # New

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Global variables for loaded models
yolo_model: Optional[UltralyticsYOLO_for_typehinting] = None
yolo_seg_model: Optional[UltralyticsYOLO_for_typehinting] = None # New for segmentation model
gemini_model: Optional[GenerativeModel_Type] = None

MAX_SEG_SIDE = 1024

# --- START: 新增的程式碼 (第二處) ---
def _download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    """從 GCS 下載檔案到本地路徑"""
    if os.path.exists(destination_file_name):
        logging.info(f"'{destination_file_name}' 已存在，跳過下載。")
        return True

    try:
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        logging.info(
            f"開始從 gs://{bucket_name}/{source_blob_name} 下載模型至 '{destination_file_name}'..."
        )
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        blob.download_to_filename(destination_file_name)
        logging.info(f"模型成功下載至 '{destination_file_name}'.")
        return True
    except Exception as e:
        logging.error(f"從 GCS 下載模型失敗: {e}", exc_info=True)
        return False
# --- END: 新增的程式碼 (第二處) ---

def load_models(gemini_api_key_from_env: Optional[str] = None):
    """
    從 GCS 下載模型 (如果本地不存在)，然後載入 YOLO detection, YOLO segmentation, 和 Gemini 模型。
    模型載入後會被移至指定的運算裝置 (GPU/CPU)。
    """
    global yolo_model, yolo_seg_model, gemini_model, MODEL_PATH, SEG_MODEL_PATH

    # --- Step 1: 從 GCS 下載模型 ---
    gcs_bucket_name = os.environ.get("MODEL_BUCKET")  # Cloud Run 會設定此環境變數
    if gcs_bucket_name:
        if MODEL_PATH:
            _download_from_gcs(gcs_bucket_name, os.path.basename(MODEL_PATH), MODEL_PATH)
        if SEG_MODEL_PATH:
            _download_from_gcs(gcs_bucket_name, os.path.basename(SEG_MODEL_PATH), SEG_MODEL_PATH)
    else:
        logging.warning("環境變數 MODEL_BUCKET 未設定，將嘗試直接從本地路徑載入模型。")

    # --- Step 2: 載入 YOLO detection 模型並移至 GPU/CPU ---
    if YOLO_AVAILABLE and MODEL_PATH and _YOLO_CLASS_ACTUAL is not None:
        if os.path.exists(MODEL_PATH):
            try:
                yolo_model = _YOLO_CLASS_ACTUAL(MODEL_PATH)
                try:
                    yolo_model.to(DEVICE)
                except AttributeError:
                    logging.warning("YOLO detection model object lacks '.to()'. Skipping device move.")
                logging.info(f"YOLO detection model loaded from {MODEL_PATH} and moved to {DEVICE}.")
            except Exception as e:
                logging.error(f"Error loading YOLO detection model from {MODEL_PATH}: {e}", exc_info=True)
                yolo_model = None
        else:
            logging.error(f"YOLO detection model file not found at path: {MODEL_PATH}")
    elif not YOLO_AVAILABLE:
        logging.error("YOLO (ultralytics) library not installed. Cannot load detection model.")
    elif not MODEL_PATH:
        logging.error(f"YOLO detection model path environment variable {MODEL_PATH_ENV_VAR} not set.")

    # --- Step 3: 載入 YOLO segmentation 模型並移至 GPU/CPU ---
    if YOLO_AVAILABLE and SEG_MODEL_PATH and _YOLO_CLASS_ACTUAL is not None:
        if os.path.exists(SEG_MODEL_PATH):
            try:
                yolo_seg_model = _YOLO_CLASS_ACTUAL(SEG_MODEL_PATH)
                try:
                    yolo_seg_model.to(DEVICE)
                except AttributeError:
                    logging.warning("YOLO segmentation model object lacks '.to()'. Skipping device move.")
                logging.info(f"YOLO segmentation model loaded from {SEG_MODEL_PATH} and moved to {DEVICE}.")
                if yolo_seg_model and hasattr(yolo_seg_model, 'names'):
                    logging.info(f"Segmentation model classes: {yolo_seg_model.names}")
                else:
                    logging.warning("Segmentation model loaded but has no 'names' attribute.")
            except Exception as e:
                logging.error(f"Error loading YOLO segmentation model from {SEG_MODEL_PATH}: {e}", exc_info=True)
                yolo_seg_model = None
        else:
            logging.error(f"YOLO segmentation model file not found at path: {SEG_MODEL_PATH}")
    elif not YOLO_AVAILABLE:
        logging.error("YOLO (ultralytics) library not installed. Cannot load segmentation model.")
    elif not SEG_MODEL_PATH:
        logging.info(f"YOLO segmentation model path environment variable {SEG_MODEL_PATH_ENV_VAR} not set. Segmentation features will be unavailable.")

    # --- Step 4: 載入 Gemini 模型 ---
    if GEMINI_API_AVAILABLE and genai_models is not None and genai_types is not None and genai_configure is not None and gemini_api_key_from_env:
        try:
            genai_configure(api_key=gemini_api_key_from_env)
            gemini_model = genai_models.GenerativeModel('gemini-2.0-flash')
            logging.info("Gemini API configured successfully with gemini-2.0-flash.")
        except Exception as e:
            logging.error(f"Failed to configure Gemini API: {e}", exc_info=True)
            gemini_model = None
    elif GEMINI_API_AVAILABLE and genai_types is not None and not gemini_api_key_from_env:
        logging.warning("Gemini API key not provided. Gemini features will be unavailable.")
    elif not GEMINI_API_AVAILABLE:
        logging.warning("google-generativeai library not found. Gemini features unavailable.")

# --- Helper Functions (Internal to this module, prefixed with _) ---

def _log_message(log_list: List[str], message: str):
    """Helper to append to a log list and also log via logging module."""
    logging.info(message)
    log_list.append(message)

def _read_image_from_bytes(image_bytes: bytes, log_list: List[str]) -> Optional[np.ndarray]:
    try:
        image_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            _log_message(log_list, "ERROR: cv2.imdecode returned None. Image format might be unsupported or data corrupted.")
            return None
        return img
    except Exception as e:
        _log_message(log_list, f"ERROR: Could not read image from bytes: {e}")
        return None

def _detect_objects_yolo(img: np.ndarray, log_list: List[str]) -> Any:
    if yolo_model is None:
        _log_message(log_list, "ERROR: YOLO detection model not loaded, cannot detect objects.")
        return None
    try:
        results = yolo_model(img, verbose=False)
        return results[0] if results else None
    except Exception as e:
        _log_message(log_list, f"ERROR: YOLO detection inference error: {e}")
        return None

def _extract_numeric_data_from_gemini(ocr_text: str, log_list: List[str]) -> Optional[float]:
    _log_message(log_list, f"Parsing Gemini output for depth: '{ocr_text}'")
    if not ocr_text:
        _log_message(log_list, "Input Gemini text is empty for parsing.")
        return None

    text_to_parse = ocr_text.strip()
    if text_to_parse.lower() == "無法判斷":
        _log_message(log_list, "Gemini reported '無法判斷'.")
        return None

    cm_numbers_found_by_regex = re.findall(r'^\s*(\d+)\s*$', text_to_parse)
    _log_message(log_list, f"Regex (pure cm digits) matches: {cm_numbers_found_by_regex}")

    converted_value_meters: Optional[float] = None
    if cm_numbers_found_by_regex:
        num_cm_str = cm_numbers_found_by_regex[0]
        try:
            cm_value_int = int(num_cm_str)
            meters_value = float(cm_value_int) / 100.0
            min_depth_m, max_depth_m = 0.01, 7.00
            if min_depth_m <= meters_value <= max_depth_m:
                _log_message(log_list, f"  - Parsed from pure digits: '{num_cm_str}cm' -> {meters_value:.3f}m [VALID]")
                converted_value_meters = meters_value
            else:
                _log_message(log_list, f"  - Parsed from pure digits: '{num_cm_str}cm' -> {meters_value:.3f}m [INVALID RANGE {min_depth_m}-{max_depth_m}m]")
        except ValueError:
            _log_message(log_list, f"  - Could not convert pure digits output '{num_cm_str}' to integer cm value.")
    else:
        fallback_numbers = re.findall(r'\b(\d+(?:\.\d{1,3})?)\b', text_to_parse)
        _log_message(log_list, f"Fallback: Regex (number with optional decimal) matches: {fallback_numbers}")
        if fallback_numbers:
            for num_str_fallback in fallback_numbers:
                try:
                    val_fallback = float(num_str_fallback.replace(',', '.'))
                    temp_val_meters = val_fallback
                    interpretation = "Assumed meters (due to decimal or general number)"
                    if '.' not in num_str_fallback and len(num_str_fallback) <= 3:
                        temp_val_meters = val_fallback / 100.0
                        interpretation = f"Assumed cm, converted to meters"
                    elif '.' not in num_str_fallback and len(num_str_fallback) > 3:
                        _log_message(log_list, f"  - [Fallback] Skipping ambiguous long integer '{num_str_fallback}'.")
                        continue

                    min_depth_m, max_depth_m = 0.01, 7.00
                    if min_depth_m <= temp_val_meters <= max_depth_m:
                        _log_message(log_list, f"  - [Fallback] Parsed '{num_str_fallback}' -> {temp_val_meters:.3f}m ({interpretation}) [VALID]")
                        if converted_value_meters is None:
                            converted_value_meters = temp_val_meters
                    else:
                        _log_message(log_list, f"  - [Fallback] Parsed '{num_str_fallback}' -> {temp_val_meters:.3f}m ({interpretation}) [INVALID RANGE]")
                except ValueError:
                    _log_message(log_list, f"  - [Fallback] Could not convert '{num_str_fallback}' to float.")
        if not converted_value_meters:
            _log_message(log_list, f"Warning: Gemini did not strictly return pure cm digits ('{text_to_parse}'), and fallback parsing also failed.")

    if converted_value_meters is None:
        _log_message(log_list, "- Failed to determine depth from extracted numbers.")
        return None

    return round(converted_value_meters, 2)


def _extract_depth_crop_logic(img: np.ndarray, vertical_boxes: List[List[float]], horizontal_boxes: List[List[float]], log_list: List[str]) -> Optional[np.ndarray]:
    """
    Extracts the depth reading crop area from the image based on detected vertical and horizontal rulers.
    This logic is adapted from the more robust GUI version for higher accuracy.
    It scores V-H pairs and uses dynamic margins for cropping.
    """
    _log_message(log_list, "[Crop Logic] Starting revised depth extraction logic.")
    img_height, img_width = img.shape[:2]
    image_center_x, image_center_y = img_width // 2, img_height // 2
    selected_crop: Optional[np.ndarray] = None

    # In the web version, we don't have a GUI toggle for this. Defaulting to False.
    force_v_only_crop = False

    if force_v_only_crop:
        _log_message(log_list, "[Crop Logic] V-only crop mode is forced by configuration.")
        horizontal_boxes = []

    if not vertical_boxes:
        _log_message(log_list, "[Crop Logic] Cropping failed: No vertical rulers detected.")
        return None

    if horizontal_boxes:
        _log_message(log_list, f"[Crop Logic] Found {len(horizontal_boxes)} horizontal boxes. Finding best V-H pair.")
        best_pair_score = -float('inf')
        selected_v_box_data_pair: Optional[List[float]] = None
        selected_h_box_data_pair: Optional[List[float]] = None

        for v_box_data in vertical_boxes:
            vx1, vy1, vx2, vy2 = map(int, v_box_data[:4])
            v_conf = v_box_data[4] if len(v_box_data) > 4 else 0.7

            for h_box_data in horizontal_boxes:
                hx1, hy1, hx2, hy2 = map(int, h_box_data[:4])
                h_conf = h_box_data[4] if len(h_box_data) > 4 else 0.7

                inter_x1, inter_y1 = max(vx1, hx1), max(vy1, hy1)
                inter_x2, inter_y2 = min(vx2, hx2), min(vy2, hy2)
                intersection_width, intersection_height = inter_x2 - inter_x1, inter_y2 - inter_y1
                current_score = 0
                is_significant_intersection = False

                if intersection_width > 0 and intersection_height > 0:
                    v_ruler_width_at_h_level = vx2 - vx1
                    # A significant intersection is when the horizontal ruler overlaps a good portion of the vertical one.
                    if (intersection_width >= v_ruler_width_at_h_level * 0.3 and
                        hy1 < vy2 and hy2 > vy1 and hx1 < vx2 and hx2 > vx1):
                        is_significant_intersection = True
                        current_score += 100  # Base score for significant intersection
                        _log_message(log_list, f"  [V-H Pair] V({vx1}-{vx2},{vy1}-{vy2}) H({hx1}-{hx2},{hy1}-{hy2}) -> Significant intersection.")
                    else:
                         _log_message(log_list, f"  [V-H Pair] V({vx1}-{vx2},{vy1}-{vy2}) H({hx1}-{hx2},{hy1}-{hy2}) -> Insignificant intersection (W:{intersection_width}, H:{intersection_height}).")

                if is_significant_intersection:
                    intersection_center_x = inter_x1 + intersection_width // 2
                    intersection_center_y = inter_y1 + intersection_height // 2
                    dist_to_center = np.sqrt((intersection_center_x - image_center_x)**2 + (intersection_center_y - image_center_y)**2)
                    max_dist = np.sqrt(image_center_x**2 + image_center_y**2) if image_center_x > 0 and image_center_y > 0 else 1.0
                    proximity_score = (1 - (dist_to_center / max_dist)) * 50
                    current_score += proximity_score
                    current_score += (v_conf + h_conf) * 10
                    _log_message(log_list, f"    Score details: Proximity={proximity_score:.1f}, Confidence={(v_conf + h_conf) * 10:.1f}, Total={current_score:.1f}")

                    if current_score > best_pair_score:
                        best_pair_score = current_score
                        selected_v_box_data_pair, selected_h_box_data_pair = v_box_data, h_box_data

        if selected_v_box_data_pair and selected_h_box_data_pair:
            vx1_sel, vy1_sel, vx2_sel, vy2_sel = map(int, selected_v_box_data_pair[:4])
            hx1_sel, hy1_sel, hx2_sel, hy2_sel = map(int, selected_h_box_data_pair[:4])

            _log_message(log_list, f"[Crop Logic] Selected best V-H pair with score {best_pair_score:.1f}. V:({vx1_sel}-{vx2_sel},{vy1_sel}-{vy2_sel}), H:({hx1_sel}-{hx2_sel},{hy1_sel}-{hy2_sel})")

            crop_x_start, crop_x_end = vx1_sel, vx2_sel
            y_ref_h_bottom = hy2_sel
            v_ruler_width = vx2_sel - vx1_sel

            # These values are from the GUI defaults, ensuring consistent behavior.
            margin_above_ratio = 0.7
            margin_below_ratio = 0.35
            min_pixels = 30
            _log_message(log_list, f"  [Crop Logic] Using V+H params: above_ratio={margin_above_ratio}, below_ratio={margin_below_ratio}, min_pixels={min_pixels}")

            margin_y_above = int(v_ruler_width * margin_above_ratio)
            margin_y_below = int(v_ruler_width * margin_below_ratio)
            margin_y_above = max(margin_y_above, min_pixels)
            margin_y_below = max(margin_y_below, min_pixels)

            crop_y_start = max(0, y_ref_h_bottom - margin_y_above)
            crop_y_end = min(img_height, y_ref_h_bottom + margin_y_below)
            # Clamp crop to be within the vertical ruler's bounds (+ a small buffer)
            crop_y_start = max(crop_y_start, vy1_sel - 5)
            crop_y_end = min(crop_y_end, vy2_sel + 5)

            if crop_x_start < crop_x_end and crop_y_start < crop_y_end:
                selected_crop = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                _log_message(log_list, f"[Crop Logic] V+H crop successful. Region size: {selected_crop.shape[1]}W x {selected_crop.shape[0]}H")
            else:
                _log_message(log_list, f"[Crop Logic] V+H crop failed: Invalid coordinates {crop_x_start, crop_y_start, crop_x_end, crop_y_end}. Fallback to V-only.")
                selected_crop = None
        else:
            if horizontal_boxes:
                _log_message(log_list, "[Crop Logic] No significant V-H pair found. Fallback to V-only.")
            else:
                _log_message(log_list, "[Crop Logic] No horizontal boxes detected. Using V-only logic.")

    if selected_crop is None:
        _log_message(log_list, "[Crop Logic] Executing V-only fallback crop logic.")
        if not vertical_boxes:
            _log_message(log_list, "[Crop Logic] V-only fallback failed: No vertical boxes available.")
            return None

        # Select the tallest vertical box for V-only mode
        primary_v_box_data = max(vertical_boxes, key=lambda b: (b[3] - b[1]))
        vx1, vy1, vx2, vy2 = map(int, primary_v_box_data[:4])
        _log_message(log_list, f"  [Crop Logic] V-only using tallest ruler: ({vx1},{vy1})-({vx2},{vy2})")

        target_crop_height_v_only = 300  # Match GUI's V-only target height
        v_box_height = vy2 - vy1

        if v_box_height <= target_crop_height_v_only:
            crop_y_start, crop_y_end = vy1, vy2
        else:
            v_box_center_y = vy1 + v_box_height // 2
            crop_y_start = max(0, v_box_center_y - target_crop_height_v_only // 2)
            crop_y_end = min(img_height, v_box_center_y + target_crop_height_v_only // 2)
            # Ensure crop does not extend beyond the original V-box boundaries
            crop_y_start = max(vy1, crop_y_start)
            crop_y_end = min(vy2, crop_y_end)

        crop_x_start, crop_x_end = vx1, vx2
        if crop_x_start < crop_x_end and crop_y_start < crop_y_end:
            selected_crop = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            _log_message(log_list, f"[Crop Logic] V-only crop successful. Region size: {selected_crop.shape[1]}W x {selected_crop.shape[0]}H")
        else:
            _log_message(log_list, f"[Crop Logic] V-only crop failed: Invalid coordinates {crop_x_start, crop_y_start, crop_x_end, crop_y_end}")
            selected_crop = None

    if selected_crop is None:
        _log_message(log_list, "[Crop Logic] All cropping strategies failed.")

    return selected_crop

def _generate_gemini_prompt_text(prompt_type: str, vertical_boxes_orig_coords: List[List[float]], horizontal_boxes_orig_coords: List[List[float]], log_list: List[str]) -> str:
    _log_message(log_list, f"Generating Gemini prompt for type: {prompt_type}")
    prompt_parts = []

    prompt_parts.append("你是一位高度精確的工程測量儀器讀數專家。你的唯一任務是從提供的圖像中識別並報告一個精確的深度讀數。")
    prompt_parts.append("工程箱尺通常是長條形的，顏色鮮明（例如黃色或白色），上面印有黑色的刻度和數字。厘米刻度是密集的、連續的標記。")

    if prompt_type == "cropped_image":
        prompt_parts.append("\n這張裁切圖片是從原始照片中針對箱尺最相關的區域進行的裁切。")
        prompt_parts.append("儘管這是裁切後的圖像，但請記住箱尺通常是垂直放置的，其上的厘米數字是連續排列的。在原始照片中，一個垂直的箱尺被放置在挖掘管線旁。水平參考桿（如果存在）用於精確讀取箱尺上的刻度。")

        prompt_parts.append("\n**圖像內容描述與讀數規則（針對已裁切圖像）：**")
        if horizontal_boxes_orig_coords and vertical_boxes_orig_coords:
            prompt_parts.append("1. 圖片中主要包含一個垂直的工程箱尺和一個水平的參考橫桿的交會部分。")
            prompt_parts.append("2. **主要目標：** 找到水平參考橫桿的**底部邊緣**。")
            prompt_parts.append("3. **讀數位置：** 確定該底部邊緣在垂直工程箱尺上所**精確對齊或緊鄰其正下方**的**厘米 (cm) 刻度數字**。")
        elif vertical_boxes_orig_coords:
            prompt_parts.append("1. 圖片中主要包含一個垂直的工程箱尺的一部分。可能沒有清晰的水平參考。")
            prompt_parts.append("2. **主要目標：** 嘗試從圖像中心區域找到最清晰可辨的**厘米 (cm) 刻度數字**。")
            prompt_parts.append("3. **讀數位置：** 由於缺乏水平參考，請直接報告你看到的、最可能代表當前主要讀數的厘米數。")
        else:
            prompt_parts.append("警告：裁切後的圖像內容可能不清晰或不包含箱尺（基於原始檢測）。請仔細檢查。")
    else:  # prompt_type == "full_image"
        prompt_parts.append("\n這張圖片是完整的原始照片。")
        if not vertical_boxes_orig_coords:
            prompt_parts.append("系統的初步物件偵測未能找到清晰的垂直箱尺。")
        elif not horizontal_boxes_orig_coords and vertical_boxes_orig_coords:
            prompt_parts.append("系統的初步物件偵測找到了垂直箱尺，但未找到清晰的水平參考。")
        elif vertical_boxes_orig_coords and horizontal_boxes_orig_coords:
            prompt_parts.append("系統的初步物件偵測找到了垂直箱尺和水平參考。")

        prompt_parts.append("\n**圖像內容描述與讀數規則（針對完整圖像）：**")
        prompt_parts.append("1. **首要任務：** 請先仔細在整個圖像中定位一個垂直的工程箱尺（通常背景為黃色或白色，帶有黑色刻度和數字）和一個（如果存在）橫跨其上的水平參考橫桿（通常為銀色或淺色）。")
        prompt_parts.append("2. 如果找到了箱尺和橫桿：")
        prompt_parts.append("   a. **主要目標：** 找到水平參考橫桿的**底部邊緣**。")
        prompt_parts.append("   b. **讀數位置：** 確定該底部邊緣在垂直工程箱尺上所**精確對齊或緊鄰其正下方**的**厘米 (cm) 刻度數字**。")
        prompt_parts.append("3. 如果只找到垂直箱尺但沒有清晰的水平橫桿：")
        prompt_parts.append("   a. **主要目標：** 嘗試從圖像中可見的最清晰的垂直箱尺部分讀取一個**厘米 (cm) 刻度數字**，該數字最能代表一個可能的深度值。")

    prompt_parts.append("\n**通用讀數提示（請在分析時應用）：**")
    prompt_parts.append("- **厘米刻度外觀與優先級：** 箱尺上主要的測量單位是厘米 (cm)，它們通常以較小的黑色數字連續標示，並且間距均勻。這些數字通常是兩位數或三位數（例如：05, 10, 15 ... 100, 105 ...）。請務必優先識別這些連續且較小的數字。")
    prompt_parts.append("- **區分不同刻度：** 注意箱尺上可能存在較大的、獨立的黑色數字（例如 1, 2, 6）。這些通常表示分米或米，是為了方便快速參考，**不是主要的厘米讀數**。除非厘米刻度完全不可見，否則請忽略這些較大的獨立數字。")
    prompt_parts.append("- **避免組合：** 絕對不要將大的風格化數字與附近的小的厘米刻度進行任何形式的數學加總或組合來產生一個新的數字。例如，如果看到大 '2' 和小 '15', 不要輸出 '215'，除非水平橫桿明確指向 '215' 這個厘米刻度本身。")
    prompt_parts.append("- **避免單個數字誤判：** 不要將單個的數字（例如 '1', '2', '3'）直接作為厘米讀數輸出。這些很可能不是完整的厘米讀數，而是輔助的分米/米標記。")
    prompt_parts.append("- **預期的讀數長度：** 你要尋找的厘米讀數通常是兩位數（例如 '05', '90'）或三位數（例如 '125'）。單個數字極不可能是完整的厘米讀數。")
    prompt_parts.append("- **關於圓點（米標記輔助）：**")
    prompt_parts.append("  - **觀察：** 在某些大號風格化數字（通常是 '1' 到 '9'，代表分米）的**正上方**，有時會出現一個或多個小圓點。這些圓點是判斷**整數米**的重要線索。")
    prompt_parts.append("  - **含義：** 1個圓點通常表示讀數在 **1米以上** 的範圍，2個圓點表示在 **2米以上** 的範圍，以此類推。")
    prompt_parts.append("  - **結合上下文判斷：** 當你看到一個兩位數的厘米讀數（例如 '54'）時，請檢查其上方是否有圓點或者其他可能指示米數的標記（例如箱尺的顏色分段）。如果存在這樣的米標記（例如1個圓點），那麼完整的讀數很可能是\"米數+看到的厘米數\"（例如 '154' 厘米）。")
    prompt_parts.append("  - **優先報告直接看到的、最完整的厘米數：**")
    prompt_parts.append("    - **如果水平參考線清晰指向一個三位數的厘米刻度（例如 '154'），請直接報告這個三位數。這是最理想的情況。**")
    prompt_parts.append("    - 如果水平參考線指向一個兩位數的厘米刻度（例如 '54'），並且你根據上下文（如圓點）判斷出它前面應該有一個米數（例如1米），**但你沒有在圖像中直接看到完整的 '154'，那麼此時你仍然應該報告你直接看到的兩位數 '54'（或 '054'）。你的主要任務是報告直接視覺確認的厘米數。**")
    prompt_parts.append("    - 圓點信息主要幫助你理解讀數的量級，並在可能的情況下尋找更完整的厘米讀數。")

    prompt_parts.append(
"""
**輸出要求（至關重要）：**
 - 你的輸出**必須且只能是**一個代表你從【指示的區域或圖像中】直接看到的那個最精確的**厘米數**的數字字符串。
 - **正確範例：** '058', '58', '115', '021', '123', '008', '154'.
 - **錯誤範例：** '1', '2', '3' (單獨的分米/米標記), '2.13', '讀數是 58 厘米', '58 cm', '58厘米', '1米58厘米', '1.58m'.
 - **重點是報告你直接看到的那個精確的厘米刻度數字。**
 - **不要進行任何單位轉換（例如，不要將厘米轉為米）。不要輸出任何米值或帶小數點的數值。**
 - 不要包含任何其他文字、單位 ('cm', 'm')、解釋或標點符號。
 - 如果根據上述規則無法找到清晰的兩位或三位連續數字作為厘米讀數，或者圖片中沒有可讀的箱尺深度，請輸出 '無法判斷'。
"""
    )

    full_prompt = "\n".join(prompt_parts)
    # _log_message(log_list, "Generated Gemini Prompt.") # Less verbose, actual prompt logged elsewhere if needed
    return full_prompt

def _call_gemini_api(image_to_send_to_gemini: np.ndarray, prompt_text: str, log_list: List[str]) -> Optional[str]:
    if gemini_model is None or not GEMINI_API_AVAILABLE or genai_models is None or genai_types is None or genai_configure is None:
        _log_message(log_list, "ERROR: Gemini model/library not available for generating response.")
        return None
    try:
        _log_message(log_list, f"Preparing image for Gemini. Image shape: {image_to_send_to_gemini.shape}")
        img_rgb = cv2.cvtColor(image_to_send_to_gemini, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')  # Still keep bytes for potential debugging
        image_bytes_for_gemini = img_byte_arr.getvalue()
        image_part = {"mime_type": "image/png", "data": image_bytes_for_gemini}

        generation_config = genai_types.GenerationConfig(temperature=0.1, max_output_tokens=50)

        # Updated safety settings based on Gemini 1.5 flash
        safety_settings_list = [
            {"category": genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": genai_types.HarmBlockThreshold.BLOCK_NONE},
            {"category": genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": genai_types.HarmBlockThreshold.BLOCK_NONE},
            {"category": genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": genai_types.HarmBlockThreshold.BLOCK_NONE},
            {"category": genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": genai_types.HarmBlockThreshold.BLOCK_NONE},
        ]
        # Check for CIVIC_INTEGRITY availability as it might not be in all genai lib versions
        if hasattr(genai_types.HarmCategory, 'HARM_CATEGORY_CIVIC_INTEGRITY'):
            civic_integrity_category = getattr(genai_types.HarmCategory, 'HARM_CATEGORY_CIVIC_INTEGRITY')
            safety_settings_list.append({
                 "category": civic_integrity_category,
                 "threshold": genai_types.HarmBlockThreshold.BLOCK_NONE,
            })
        else:
            _log_message(log_list, "Warning: types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY not found. Skipping this safety setting for Gemini.")


        _log_message(log_list, "Sending request to AI...")
        response = gemini_model.generate_content(
            [pil_image, prompt_text],
            generation_config=generation_config,
            safety_settings=safety_settings_list,
            stream=False
        )

        gemini_output_text = ""
        if hasattr(response, 'text') and response.text:
            gemini_output_text = response.text.strip()
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
            reason_obj = response.prompt_feedback.block_reason
            reason_message = response.prompt_feedback.block_reason_message if hasattr(response.prompt_feedback, 'block_reason_message') else str(reason_obj)
            gemini_output_text = f"Blocked: {reason_message}"
            _log_message(log_list, f"ERROR: Gemini API request blocked. Reason: {reason_message}")
        elif not hasattr(response, 'text') and not (hasattr(response, 'prompt_feedback') and response.prompt_feedback):
             _log_message(log_list, f"Warning: Gemini response structure unexpected. Response: {type(response)}")
             if hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text'):
                 gemini_output_text = response.parts[0].text.strip()
             else:
                 gemini_output_text = "無法判斷"

        _log_message(log_list, f"AI raw response: '{gemini_output_text}'")
        return gemini_output_text

    except Exception as e:
        if "prompt was blocked" in str(e).lower() or "safety settings" in str(e).lower():
            _log_message(log_list, f"ERROR: Gemini API request was blocked: {str(e)}")
            return f"Blocked: {str(e)}"
        _log_message(log_list, f"ERROR: Error during Gemini API call: {str(e)}")
        return None

def _get_dominant_color_by_segmentation(
    image_crop_for_segmentation: np.ndarray,
    seg_log_list: List[str]
) -> Tuple[Optional[str], Optional[np.ndarray], float, float]:
    """
    Segments the provided image crop using yolo_seg_model and determines its dominant color
    (yellow/white) based on 'yellow_ruler_segment' and 'white_ruler_segment' classes.
    Returns the dominant color string, an annotated image of the segmented crop, the white pixel ratio, and the yellow pixel ratio.
    """
    _log_message(seg_log_list, "[SegColor] Starting segmentation-based color determination on crop...")
    FIXED_SEG_CONFIDENCE = 0.35  # As used in GUI

    # Initialize return values for error cases
    initial_crop_copy = image_crop_for_segmentation.copy() if image_crop_for_segmentation is not None else None
    white_ratio, yellow_ratio = 0.0, 0.0

    if yolo_seg_model is None:
        _log_message(seg_log_list, "[SegColor] Segmentation model not loaded. Cannot perform color determination.")
        return None, initial_crop_copy, white_ratio, yellow_ratio

    if not hasattr(yolo_seg_model, 'names') or not yolo_seg_model.names:
        _log_message(seg_log_list, "[SegColor] Segmentation model loaded but has no 'names' attribute. Cannot map class IDs.")
        return None, initial_crop_copy, white_ratio, yellow_ratio

    seg_class_names = yolo_seg_model.names

    if image_crop_for_segmentation is None or image_crop_for_segmentation.size == 0:
        _log_message(seg_log_list, "[SegColor] Input image crop for segmentation is empty.")
        return None, None, white_ratio, yellow_ratio

    # --- Resize large crops to speed up CPU inference ---
    orig_h, orig_w = image_crop_for_segmentation.shape[:2]
    resize_scale = 1.0
    if max(orig_h, orig_w) > MAX_SEG_SIDE:
        resize_scale = MAX_SEG_SIDE / float(max(orig_h, orig_w))
        new_w, new_h = int(orig_w * resize_scale), int(orig_h * resize_scale)
        resized_crop_for_seg = cv2.resize(image_crop_for_segmentation, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _log_message(seg_log_list, f"[SegColor] Resized crop for segmentation: {orig_w}x{orig_h} -> {new_w}x{new_h} (scale={resize_scale:.3f})")
    else:
        resized_crop_for_seg = image_crop_for_segmentation

    _log_message(seg_log_list, f"[SegColor] Segmenting crop (size {resized_crop_for_seg.shape[1]}W x {resized_crop_for_seg.shape[0]}H) with confidence {FIXED_SEG_CONFIDENCE}...")

    annotated_crop_image: Optional[np.ndarray] = resized_crop_for_seg.copy()
    dominant_color_overall: Optional[str] = None

    try:
        # Ultralytics Results object type hint for clarity
        results: Optional[List[UltralyticsResults_for_typehinting]] = yolo_seg_model(
            resized_crop_for_seg,
            verbose=False,
            conf=FIXED_SEG_CONFIDENCE,
            imgsz=max(64, resized_crop_for_seg.shape[0]) # Ensure imgsz is reasonable
        )

        if results and results[0].masks is not None and len(results[0].masks) > 0:
            try:
                # .plot() returns a new image, doesn't modify in place
                plotted_image = results[0].plot(boxes=False, labels=True, conf=False)
                if plotted_image is not None and plotted_image.size > 0:
                    annotated_crop_image = plotted_image
                _log_message(seg_log_list, "[SegColor] Generated preview image with segmentation masks.")
            except Exception as e_plot:
                _log_message(seg_log_list, f"[SegColor] Error generating segmentation preview image: {e_plot}")

            # Ensure masks and boxes are available and not None
            masks_obj = results[0].masks
            boxes_obj = results[0].boxes

            if masks_obj is None or boxes_obj is None or masks_obj.data is None or boxes_obj.cls is None:
                 _log_message(seg_log_list, "[SegColor] Segmentation results missing masks.data or boxes.cls.")
                 return None, annotated_crop_image, white_ratio, yellow_ratio


            masks_data_cpu = masks_obj.data.cpu().numpy()
            class_indices = boxes_obj.cls.cpu().numpy().astype(int)

            yellow_class_idx, white_class_idx = -1, -1
            for k_idx, v_name_str in seg_class_names.items(): # seg_model.names is Dict[int, str]
                if "yellow_ruler_segment" in v_name_str.lower(): yellow_class_idx = int(k_idx)
                elif "white_ruler_segment" in v_name_str.lower(): white_class_idx = int(k_idx)

            _log_message(seg_log_list, f"[SegColor] Determined class indices -> Yellow: {yellow_class_idx}, White: {white_class_idx} from {seg_class_names}")

            total_yellow_pixels, total_white_pixels = 0, 0
            for i, mask_instance_roi in enumerate(masks_data_cpu):
                if i < len(class_indices): # Boundary check
                    class_id = class_indices[i]
                    num_pixels_in_mask = np.sum(mask_instance_roi > 0.5) # Threshold mask
                    if class_id == yellow_class_idx:
                        total_yellow_pixels += num_pixels_in_mask
                    elif class_id == white_class_idx:
                        total_white_pixels += num_pixels_in_mask
                else:
                    _log_message(seg_log_list, f"[SegColor] Warning: Mask index {i} out of bounds for class_indices length {len(class_indices)}.")

            _log_message(seg_log_list, f"[SegColor] Total segmented pixels -> Yellow: {total_yellow_pixels}, White: {total_white_pixels}")

            crop_area = image_crop_for_segmentation.shape[0] * image_crop_for_segmentation.shape[1]
            white_ratio = total_white_pixels / crop_area if crop_area > 0 else 0.0
            yellow_ratio = total_yellow_pixels / crop_area if crop_area > 0 else 0.0
            _log_message(seg_log_list, f"[SegColor] Pixel ratios -> White: {white_ratio:.2%}, Yellow: {yellow_ratio:.2%}")

            min_pixel_threshold_ratio = 0.05 # For simple dominance check

            if total_white_pixels > total_yellow_pixels and white_ratio >= min_pixel_threshold_ratio:
                dominant_color_overall = "white"
            elif total_yellow_pixels > total_white_pixels and yellow_ratio >= min_pixel_threshold_ratio:
                dominant_color_overall = "yellow"
            else:
                dominant_color_overall = None
                _log_message(seg_log_list, f"[SegColor] Neither color met dominance criteria (min ratio: {min_pixel_threshold_ratio:.2%}).")

            if dominant_color_overall:
                _log_message(seg_log_list, f"[SegColor] Determined dominant color in crop: {dominant_color_overall}")

        else:
            _log_message(seg_log_list, "[SegColor] Segmentation model did not return any valid masks or detections for the crop.")

    except Exception as e:
        _log_message(seg_log_list, f"[SegColor] Critical error during segmentation: {e}")
        import traceback
        _log_message(seg_log_list, traceback.format_exc())
        return None, annotated_crop_image, white_ratio, yellow_ratio

    return dominant_color_overall, annotated_crop_image, white_ratio, yellow_ratio

# --- Main Analysis Function (Exported) ---

def analyze_image_depth(image_data: bytes, filename_for_log: str) -> Dict[str, Any]:
    start_time_total = time.time()
    internal_log: List[str] = []
    _log_message(internal_log, f"--- Starting analysis for image: {filename_for_log} ---")

    analysis_result: Dict[str, Any] = {
        "filename": filename_for_log,
        "placement_status": "無法判斷",
        "depth_value_meters": None,
        "raw_gemini_output": None,
        "error_message": None,
        "yolo_log_details": [],
        "crop_log_details": [],
        "segmentation_log_details": [], # New for segmentation logs
        "gemini_log_details": [],
        "image_with_yolo_boxes_bytes": None,
        "cropped_image_for_gemini_bytes": None,
        "segmentation_preview_image_bytes": None, # New for segmentation preview
        "was_color_corrected": False, # Will indicate if segmentation-based correction applied
    }
    seg_log = analysis_result["segmentation_log_details"]

    original_cv2_image = _read_image_from_bytes(image_data, analysis_result["gemini_log_details"])
    if original_cv2_image is None:
        analysis_result["error_message"] = "Failed to read or decode image."
        _log_message(internal_log, f"--- Analysis for {filename_for_log} ended due to image read error ---")
        return analysis_result

    yolo_annotated_image = original_cv2_image.copy()
    vertical_boxes_coords: List[List[float]] = []
    horizontal_boxes_coords: List[List[float]] = []

    if yolo_model:
        start_time_yolo_det = time.time()
        yolo_results_obj = _detect_objects_yolo(original_cv2_image, analysis_result["yolo_log_details"])
        logging.info(f"[TIMER] YOLO Detection took: {time.time() - start_time_yolo_det:.4f}s")
        if yolo_results_obj and hasattr(yolo_results_obj, 'boxes') and yolo_results_obj.boxes:
            if hasattr(yolo_results_obj.boxes, 'data') and hasattr(yolo_results_obj.boxes.data, 'cpu') and hasattr(yolo_results_obj.boxes.data.cpu(), 'numpy'):
                boxes_data_raw = yolo_results_obj.boxes.data.cpu().numpy()
            else:
                boxes_data_raw = np.array(yolo_results_obj.boxes) if yolo_results_obj.boxes else []

            conf_threshold = 0.45
            filtered_boxes_list = [box_row for box_row in boxes_data_raw if len(box_row) > 4 and float(box_row[4]) >= conf_threshold]
            _log_message(analysis_result["yolo_log_details"], f"YOLO: Raw detections {len(boxes_data_raw)}, filtered (conf >= {conf_threshold}): {len(filtered_boxes_list)}")

            if filtered_boxes_list:
                for box_item in filtered_boxes_list:
                    try:
                        x1, y1, x2, y2 = map(int, box_item[:4])
                        conf = float(box_item[4])
                        current_box_coords: List[float] = [float(c) for c in box_item[:5]]

                        if (y2 - y1) > (x2 - x1):
                            vertical_boxes_coords.append(current_box_coords)
                            cv2.rectangle(yolo_annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(yolo_annotated_image, f"V {conf:.2f}", (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            horizontal_boxes_coords.append(current_box_coords)
                            cv2.rectangle(yolo_annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(yolo_annotated_image, f"H {conf:.2f}", (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    except (ValueError, IndexError) as e_box:
                        _log_message(analysis_result["yolo_log_details"], f"YOLO: Skipping malformed box item {box_item}: {e_box}")
            else:
                _log_message(analysis_result["yolo_log_details"], "YOLO: No boxes found after confidence filtering.")
        else:
            _log_message(analysis_result["yolo_log_details"], "YOLO: No detections or invalid results object.")

        # --- New Logic: Determine main vertical ruler color from the whole ruler ---
        main_vbox_dominant_color: Optional[str] = None
        main_vbox_seg_preview_img: Optional[np.ndarray] = None
        if vertical_boxes_coords and yolo_seg_model:
            _log_message(seg_log, "[Main Ruler Seg] Finding largest vertical ruler for global color analysis.")
            largest_v_box = max(vertical_boxes_coords, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            vx1, vy1, vx2, vy2 = map(int, largest_v_box[:4])
            _log_message(seg_log, f"[Main Ruler Seg] Largest V-box found: ({vx1}, {vy1}, {vx2}, {vy2})")
            vbox_full_crop = original_cv2_image[vy1:vy2, vx1:vx2]
            if vbox_full_crop.size > 0:
                _log_message(seg_log, "[Main Ruler Seg] Running segmentation on the full vertical ruler crop.")
                # We only need the dominant color and preview from this step. Ratios are not used here.
                main_vbox_dominant_color, main_vbox_seg_preview_img, _, _ = _get_dominant_color_by_segmentation(
                    vbox_full_crop,
                    seg_log
                )
                _log_message(seg_log, f"[Main Ruler Seg] Determined global ruler color: {main_vbox_dominant_color}")
            else:
                _log_message(seg_log, "[Main Ruler Seg] Cropping the full vertical ruler failed (size is zero).")
        elif not vertical_boxes_coords:
            _log_message(seg_log, "[Main Ruler Seg] Skipping global color analysis: No vertical rulers detected.")
        elif not yolo_seg_model:
            _log_message(seg_log, "[Main Ruler Seg] Skipping global color analysis: Segmentation model not loaded.")

        is_success_yolo_img, buffer_yolo_img = cv2.imencode(".jpg", yolo_annotated_image)
        if is_success_yolo_img:
            analysis_result["image_with_yolo_boxes_bytes"] = buffer_yolo_img.tobytes()
    else:
        _log_message(analysis_result["yolo_log_details"], "YOLO detection model not loaded. Skipping detection.")

    num_v, num_h = len(vertical_boxes_coords), len(horizontal_boxes_coords)
    if not yolo_model: analysis_result["placement_status"] = "YOLO偵測模型未載入"
    elif num_v == 0: analysis_result["placement_status"] = "未偵測到垂直箱尺"
    elif num_v >= 1 and num_h == 0: analysis_result["placement_status"] = f"僅偵測到垂直箱尺 ({num_v}) (建議加入水平參考)"
    elif num_v >= 1 and num_h >= 1: analysis_result["placement_status"] = f"正確 (垂直:{num_v}, 水平:{num_h})"
    else: analysis_result["placement_status"] = f"偵測情況複雜 (垂直:{num_v}, 水平:{num_h})"
    _log_message(analysis_result["yolo_log_details"], f"Placement status: {analysis_result['placement_status']}")


    image_to_send_to_gemini: Optional[np.ndarray] = None
    prompt_type = "full_image"

    # Placeholder for color derived from cropped (golden) region – will be set after cropping logic
    dominant_color_crop: Optional[str] = None
    seg_crop_preview_img: Optional[np.ndarray] = None

    cropped_image_cv = _extract_depth_crop_logic(original_cv2_image, vertical_boxes_coords, horizontal_boxes_coords, analysis_result["crop_log_details"])
    if cropped_image_cv is not None and cropped_image_cv.size > 0:
        image_to_send_to_gemini = cropped_image_cv
        prompt_type = "cropped_image"
        is_success_crop_img, buffer_crop_img = cv2.imencode(".jpg", cropped_image_cv)
        if is_success_crop_img:
            analysis_result["cropped_image_for_gemini_bytes"] = buffer_crop_img.tobytes()
        _log_message(analysis_result["gemini_log_details"], "Using cropped image for AI.")

        # --- New: Determine dominant color based on the cropped (golden) region ---
        if yolo_seg_model is not None:
            _log_message(seg_log, "[Crop Seg] Running segmentation on cropped image for local color determination.")
            dominant_color_crop, seg_crop_preview_img, _, _ = _get_dominant_color_by_segmentation(
                cropped_image_cv,
                seg_log
            )
            _log_message(seg_log, f"[Crop Seg] Determined dominant color in cropped region: {dominant_color_crop}")

            # Prefer showing the cropped-region segmentation preview (overrides global preview later)
            if seg_crop_preview_img is not None and seg_crop_preview_img.size > 0:
                is_success_seg_crop_prev, buffer_seg_crop_prev = cv2.imencode(".jpg", seg_crop_preview_img)
                if is_success_seg_crop_prev:
                    analysis_result["segmentation_preview_image_bytes"] = buffer_seg_crop_prev.tobytes()
    else:
        image_to_send_to_gemini = original_cv2_image
        _log_message(analysis_result["gemini_log_details"], "Cropping failed or not applicable. Using full image for Gemini.")
        if vertical_boxes_coords and not analysis_result["error_message"] and (cropped_image_cv is None or cropped_image_cv.size == 0):
             analysis_result["error_message"] = "偵測到垂直箱尺但裁切失敗，深度分析可能不準確。"
        elif not vertical_boxes_coords and not analysis_result["error_message"]:
            analysis_result["error_message"] = "未偵測到垂直箱尺且無法裁切，深度分析可能不準確。"

    gemini_output_text_final = None
    depth_value_meters_from_gemini: Optional[float] = None
    if gemini_model and image_to_send_to_gemini is not None:
        prompt_text = _generate_gemini_prompt_text(prompt_type, vertical_boxes_coords, horizontal_boxes_coords, analysis_result["gemini_log_details"])
        start_time_gemini = time.time()
        gemini_output_text_final = _call_gemini_api(image_to_send_to_gemini, prompt_text, analysis_result["gemini_log_details"])
        logging.info(f"[TIMER] Gemini API call took: {time.time() - start_time_gemini:.4f}s")
        analysis_result["raw_gemini_output"] = gemini_output_text_final

        if gemini_output_text_final and "Blocked" not in gemini_output_text_final:
            depth_m_temp = _extract_numeric_data_from_gemini(gemini_output_text_final, analysis_result["gemini_log_details"])
            if depth_m_temp is not None:
                depth_value_meters_from_gemini = depth_m_temp
            elif "無法判斷" not in gemini_output_text_final.lower() and not analysis_result["error_message"]:
                 analysis_result["error_message"] = "Gemini 未能提供有效深度值或回應解析失敗。"
        elif gemini_output_text_final and "Blocked" in gemini_output_text_final and not analysis_result["error_message"]:
            analysis_result["error_message"] = f"Gemini 請求處理失敗 ({gemini_output_text_final})"
        elif not gemini_output_text_final and not analysis_result["error_message"]:
             analysis_result["error_message"] = "與 Gemini API 通訊時發生錯誤。"
    elif not gemini_model:
        _log_message(analysis_result["gemini_log_details"], "Gemini model not available. Skipping Gemini processing.")
        if not analysis_result["error_message"]: analysis_result["error_message"] = "Gemini 服務未配置。"
    elif image_to_send_to_gemini is None:
         _log_message(analysis_result["gemini_log_details"], "No valid image to send to Gemini.")
         if not analysis_result["error_message"]: analysis_result["error_message"] = "圖像準備失敗，無法送往 Gemini。"

    # --- Initialize correction-related variables to ensure they are always defined ---
    final_depth_to_display = depth_value_meters_from_gemini  # may be None at this stage
    was_segmentation_corrected_this_time = False

    # --- NEW: 3-digit cm safeguard (e.g., '154') ---
    raw_gemini_text_safe = (gemini_output_text_final or "").strip()
    is_three_digit_cm = bool(re.fullmatch(r"\d{3}", raw_gemini_text_safe))
    if is_three_digit_cm:
        _log_message(seg_log, "[SegColorCorrect] Detected 3-digit cm output from Gemini; skipping ±1 m color correction.")

    # If we still don't have a segmentation preview image from the cropped region, use the global ruler preview
    if analysis_result["segmentation_preview_image_bytes"] is None and main_vbox_seg_preview_img is not None and main_vbox_seg_preview_img.size > 0:
        is_success_seg_preview, buffer_seg_preview = cv2.imencode(".jpg", main_vbox_seg_preview_img)
        if is_success_seg_preview:
            analysis_result["segmentation_preview_image_bytes"] = buffer_seg_preview.tobytes()
            _log_message(seg_log, "[SegColor] Stored global ruler segmentation preview image (fallback).")

    # --- Correction logic: Prefer cropped-region color; fallback to global ---
    color_ref_for_correction = dominant_color_crop if dominant_color_crop is not None else main_vbox_dominant_color

    if (not is_three_digit_cm) and depth_value_meters_from_gemini is not None and color_ref_for_correction is not None:
        region_tag = "CROPPED" if dominant_color_crop is not None else "GLOBAL"
        _log_message(seg_log, f"[SegColorCorrect] Applying correction logic based on {region_tag} ruler color: '{color_ref_for_correction}'")
        apply_correction_flag = 0  # 0: no change, 1: add 1m, -1: subtract 1m

        if color_ref_for_correction == "white":
            _log_message(seg_log, f"[SegColorCorrect] {region_tag} color: White. Depth from Gemini: {depth_value_meters_from_gemini:.3f}m")
            if depth_value_meters_from_gemini < 0.90:
                apply_correction_flag = 1
                _log_message(seg_log, "[SegColorCorrect] Condition met (depth < 0.90m): Applying +1.0m correction.")
            else:
                _log_message(seg_log, "[SegColorCorrect] Condition NOT met for +1.0m (depth >= 0.90m).")

        elif color_ref_for_correction == "yellow":
            _log_message(seg_log, f"[SegColorCorrect] {region_tag} color: Yellow. Depth from Gemini: {depth_value_meters_from_gemini:.3f}m")
            if depth_value_meters_from_gemini >= 1.10:
                apply_correction_flag = -1
                _log_message(seg_log, "[SegColorCorrect] Condition met (depth >= 1.10m for yellow ruler): Applying -1.0m correction.")
            else:
                _log_message(seg_log, "[SegColorCorrect] Condition NOT met for -1.0m correction.")
         
        if apply_correction_flag == 1:
            final_depth_to_display = depth_value_meters_from_gemini + 1.0
            was_segmentation_corrected_this_time = True
        elif apply_correction_flag == -1:
            final_depth_to_display = depth_value_meters_from_gemini - 1.0
            was_segmentation_corrected_this_time = True

        if was_segmentation_corrected_this_time:
            _log_message(seg_log, f"[SegColorCorrect] Final depth after seg-correction: {final_depth_to_display:.3f}m")

    elif is_three_digit_cm:
        pass  # already logged above; no correction applied
    elif depth_value_meters_from_gemini is None:
        _log_message(seg_log, "[SegColorCorrect] Skipping correction: No valid depth from Gemini.")
    elif color_ref_for_correction is None:
        _log_message(seg_log, "[SegColorCorrect] Skipping correction: Ruler color could not be determined.")

    analysis_result["depth_value_meters"] = final_depth_to_display
    analysis_result["was_color_corrected"] = was_segmentation_corrected_this_time

    _log_message(internal_log, f"--- Analysis for {filename_for_log} completed ---")
    logging.info(f"[TIMER] Total analysis took: {time.time() - start_time_total:.4f}s")
    return analysis_result

if __name__ == '__main__':
    print("Depth Analyzer Module - Direct Run Test")

    test_gemini_key = os.environ.get("GEMINI_API_KEY_TEST")

    if not MODEL_PATH:
        print(f"Skipping YOLO detection model loading: {MODEL_PATH_ENV_VAR} environment variable not set.")
    if not SEG_MODEL_PATH: # New check
        print(f"Skipping YOLO segmentation model loading: {SEG_MODEL_PATH_ENV_VAR} environment variable not set.")
    if not test_gemini_key:
        print(f"Skipping Gemini model loading: GEMINI_API_KEY_TEST environment variable not set.")

    print(f"Attempting to load models. YOLO Det Path: {MODEL_PATH if MODEL_PATH else 'Not set'}, YOLO Seg Path: {SEG_MODEL_PATH if SEG_MODEL_PATH else 'Not set'}")
    load_models(gemini_api_key_from_env=test_gemini_key)

    if not yolo_model: print("YOLO detection model failed to load or not configured for test.")
    if not yolo_seg_model: print("YOLO segmentation model failed to load or not configured for test.") # New check
    if not gemini_model : print("Gemini model failed to load or not configured for test.")

    test_image_file = "test_image_analyzer.jpg"
    img_h, img_w = 400, 600
    dummy_array = np.full((img_h, img_w, 3), (200, 200, 200), dtype=np.uint8)

    v_ruler_x1, v_ruler_y1 = 280, 50
    v_ruler_x2, v_ruler_y2 = 320, 350
    cv2.rectangle(dummy_array, (v_ruler_x1, v_ruler_y1), (v_ruler_x2, v_ruler_y2), (0, 230, 230), -1)
    for i in range(7):
        y_mark = v_ruler_y1 + 40 + i * 40
        cv2.line(dummy_array, (v_ruler_x1, y_mark), (v_ruler_x2, y_mark), (0,0,0), 2)
        cv2.putText(dummy_array, str(50 + i*10), (v_ruler_x1 - 30, y_mark + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)


    h_ruler_x1, h_ruler_y1 = 150, 100 # Moved H-ruler up to interact with "60"
    h_ruler_x2, h_ruler_y2 = 450, 120
    cv2.rectangle(dummy_array, (h_ruler_x1, h_ruler_y1), (h_ruler_x2, h_ruler_y2), (192, 192, 192), -1)

    cv2.imwrite(test_image_file, dummy_array)
    print(f"Created/Replaced dummy test image: {test_image_file}")


    if os.path.exists(test_image_file):
        with open(test_image_file, "rb") as f:
            test_image_bytes = f.read()

        print(f"\nAnalyzing test image: {test_image_file}...")

        if (MODEL_PATH and not yolo_model) or \
           (SEG_MODEL_PATH and not yolo_seg_model) or \
           (test_gemini_key and not gemini_model):
             print("Re-attempting model load for test run...")
             load_models(gemini_api_key_from_env=test_gemini_key)

        if not yolo_model and MODEL_PATH: print("Warning: YOLO detection model path set but model not loaded for test.")
        if not yolo_seg_model and SEG_MODEL_PATH: print("Warning: YOLO segmentation model path set but model not loaded for test.")
        if not gemini_model and test_gemini_key: print("Warning: Gemini API key set but model not loaded for test.")


        results = analyze_image_depth(test_image_bytes, test_image_file)

        print("\n--- Test Analysis Results ---")
        print(f"Filename: {results.get('filename')}")
        print(f"Placement Status: {results.get('placement_status')}")
        print(f"Depth (meters): {results.get('depth_value_meters')}")
        print(f"Was Color Corrected (Seg): {results.get('was_color_corrected')}")
        print(f"Raw Gemini Output: {results.get('raw_gemini_output')}")
        print(f"Error Message: {results.get('error_message')}")

        print("\n--- Log Details ---")
        print("\nYOLO Det Log:")
        for log_entry in results.get("yolo_log_details", []): print(f"  {log_entry}")
        print("\nCrop Log:")
        for log_entry in results.get("crop_log_details", []): print(f"  {log_entry}")
        print("\nSegmentation Log:") # New
        for log_entry in results.get("segmentation_log_details", []): print(f"  {log_entry}")
        print("\nGemini Log:")
        for log_entry in results.get("gemini_log_details", []): print(f"  {log_entry}")

        if results.get("image_with_yolo_boxes_bytes"):
            with open("test_yolo_output.jpg", "wb") as f_out:
                f_out.write(results["image_with_yolo_boxes_bytes"])
            print(f"\nSaved YOLO annotated image to: test_yolo_output.jpg")
        if results.get("cropped_image_for_gemini_bytes"):
            with open("test_gemini_crop_output.jpg", "wb") as f_out:
                f_out.write(results["cropped_image_for_gemini_bytes"])
            print(f"Saved cropped image for Gemini to: test_gemini_crop_output.jpg")
        if results.get("segmentation_preview_image_bytes"): # New
            with open("test_segmentation_preview.jpg", "wb") as f_out:
                f_out.write(results["segmentation_preview_image_bytes"])
            print(f"Saved segmentation preview image to: test_segmentation_preview.jpg")
    else:
        print(f"Test image {test_image_file} not found for direct run test, and dummy creation failed.")