# /app/main_web.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import logging
import os
from typing import Dict, Any, Optional 
import base64
from uuid import uuid4
import asyncio
from pathlib import Path  # 新增: 使用絕對路徑需要 Path 模組
from fastapi.middleware.cors import CORSMiddleware  # 新增: CORS middleware import

# --- 從 depth_analyzer 導入核心邏輯和配置 ---
try:
    from depth_analyzer import (
        analyze_image_depth,    
        load_models,            
        MODEL_PATH,             
        SEG_MODEL_PATH,         # Path for Segmentation model
        GEMINI_API_AVAILABLE,   
        MODEL_PATH_ENV_VAR,     
        SEG_MODEL_PATH_ENV_VAR  # Env var name for Seg model path
    )
    DEPTH_ANALYZER_LOADED = True
except ImportError as e:
    logging.error(f"無法從 depth_analyzer 導入模組: {e}. 應用程式將以受限模式運行或無法啟動。")
    DEPTH_ANALYZER_LOADED = False
    MODEL_PATH_ENV_VAR = "MODEL_PATH_ENV_NOT_LOADED" 
    SEG_MODEL_PATH_ENV_VAR = "SEG_MODEL_PATH_ENV_NOT_LOADED"
    MODEL_PATH = os.environ.get(MODEL_PATH_ENV_VAR, "未設定(核心模組錯誤)") 
    SEG_MODEL_PATH = os.environ.get(SEG_MODEL_PATH_ENV_VAR, "未設定(核心模組錯誤)")
    GEMINI_API_AVAILABLE = False
    def load_models(gemini_api_key_from_env: Optional[str] = None): # type: ignore
        logging.error("depth_analyzer.py 載入失敗，無法載入模型。")
    def analyze_image_depth(image_data: bytes, filename_for_log: str) -> Dict[str, Any]: # type: ignore
        logging.error("depth_analyzer.py 載入失敗，無法分析圖像。")
        return {
            "filename": filename_for_log,
            "placement_status": "錯誤",
            "depth_value_meters": None,
            "raw_gemini_output": None,
            "error_message": "核心分析模組載入失敗。",
            "yolo_log_details": ["核心分析模組載入失敗。"],
            "crop_log_details": ["核心分析模組載入失敗。"],
            "segmentation_log_details": ["核心分析模組載入失敗。"], # Added
            "gemini_log_details": ["核心分析模組載入失敗。"],
            "image_with_yolo_boxes_bytes": None,
            "cropped_image_for_gemini_bytes": None,
            "segmentation_preview_image_bytes": None, # Added
            "was_color_corrected": False,
        }

app = FastAPI(
    title="箱尺埋深檢測 API",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS middleware setup (equivalent to previous Nginx CORS settings) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- End of CORS setup ---

BASE_DIR = Path(__file__).resolve().parent  # 新增: 取得此檔案所在資料夾
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))  # 修改: 使用絕對路徑

GEMINI_API_KEY_FROM_ENVIRONMENT: Optional[str] = None

MAX_FILE_SIZE = 10 * 1024 * 1024          # 10 MB

@app.on_event("startup")
async def startup_event():
    import asyncio
    app.state.loop = asyncio.get_event_loop()
    global GEMINI_API_KEY_FROM_ENVIRONMENT

    if not DEPTH_ANALYZER_LOADED:
        app.state.models_loaded = False 
        app.state.startup_error = "核心分析模組 (depth_analyzer.py) 導入失敗。" 
        logging.critical(app.state.startup_error) 
        return

    GEMINI_API_KEY_FROM_ENVIRONMENT = os.environ.get("GEMINI_API_KEY") 

    try:
        if GEMINI_API_AVAILABLE and not GEMINI_API_KEY_FROM_ENVIRONMENT:
            logging.warning("環境變數 GEMINI_API_KEY 未設定。Gemini 功能可能受限或不可用。")
        
        load_models(gemini_api_key_from_env=GEMINI_API_KEY_FROM_ENVIRONMENT) 
        app.state.models_loaded = True  
        app.state.yolo_model_path = MODEL_PATH  
        app.state.yolo_seg_model_path = SEG_MODEL_PATH # Store seg model path
        logging.info(f"模型載入流程完成。YOLO Det模型路徑 (來自 {MODEL_PATH_ENV_VAR}): {MODEL_PATH}")
        logging.info(f"YOLO Seg模型路徑 (來自 {SEG_MODEL_PATH_ENV_VAR}): {SEG_MODEL_PATH}")


    except Exception as e:
        app.state.models_loaded = False 
        app.state.startup_error = str(e) 
        logging.error(f"啟動時模型加載失敗: {e}", exc_info=True)


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def read_root(request: Request):
    error_msg_display = None
    startup_error_attr = getattr(app.state, 'startup_error', None)
    models_loaded_attr = getattr(app.state, 'models_loaded', False)

    if startup_error_attr:
        error_msg_display = f"服務啟動錯誤: {startup_error_attr}"
    elif not models_loaded_attr and DEPTH_ANALYZER_LOADED :
        error_msg_display = "模型正在加載中或加載失敗，請稍後再試。詳情請查看服務器日誌。"
    elif not DEPTH_ANALYZER_LOADED:
        error_msg_display = "核心分析模組載入失敗，服務不可用。"

    context = {
        "request": request,
        "error_message": error_msg_display,
        "result": None,
    }
    try:
        return templates.TemplateResponse("index.html", context)
    except Exception as e:
        # 若模板找不到，嘗試直接讀取磁碟上的 index.html 或回傳簡易訊息
        logging.error(f"Template index.html 載入失敗: {e}")
        try:
            import pathlib
            html_content = pathlib.Path("index.html").read_text(encoding="utf-8")
            return HTMLResponse(content=html_content, status_code=200)
        except Exception:
            return HTMLResponse(content="<h2>Service is running but index.html not found.</h2>", status_code=200)


@app.post("/analyze-depth", tags=["Analysis"], response_class=JSONResponse)
@app.post("/analyze-depth/", tags=["Analysis"], response_class=JSONResponse)
async def analyze_depth_endpoint(
    file: UploadFile = File(..., description="上傳的管線挖掘埋深圖片"),
):
    """
    只返回核心結果，不包含任何詳細日誌或影像資料。
    """
    # --- 服務狀態驗證 ---
    if not DEPTH_ANALYZER_LOADED:
        raise HTTPException(status_code=503, detail="服務核心模組錯誤，無法處理請求。")

    if not getattr(app.state, 'models_loaded', False):
        startup_err = getattr(app.state, 'startup_error', '未知啟動錯誤或模型正在加載')
        raise HTTPException(status_code=503, detail=f"服務模型未成功加載: {startup_err}")

    # --- 檔案驗證 ---
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="上傳的文件必須是圖片類型。")

    image_data = await file.read()
    if len(image_data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="圖片不可超過 10 MB。")

    try:
        # analyze_image_depth 函數會返回包含所有詳細信息的字典
        analysis_result = analyze_image_depth(
            image_data=image_data,
            filename_for_log=file.filename or "uploaded_file"
        )

        # --- 僅保留核心欄位 ---
        api_response = {
            "filename": analysis_result.get("filename"),
            "placement_status": analysis_result.get("placement_status"),
            "depth_value_meters": analysis_result.get("depth_value_meters"),
            "raw_gemini_output": analysis_result.get("raw_gemini_output"),
            "error_message": analysis_result.get("error_message"),
        }

        return JSONResponse(content=api_response)

    except HTTPException:
        # 重新拋出 FastAPI HTTP 異常
        raise
    except Exception as e:
        logging.error(
            f"API 端點 (/analyze-depth/) 處理圖像 '{file.filename}' 時發生意外錯誤: {e}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤，無法處理圖像: {str(e)}")


@app.post("/analyze-depth-form/", response_class=HTMLResponse, tags=["Frontend"])
async def analyze_depth_form_endpoint(request: Request, file: UploadFile = File(...)):
    result_for_template: Dict[str, Any] = {} 
    error_for_template: Optional[str] = None

    if not DEPTH_ANALYZER_LOADED:
        error_for_template = "服務核心模組錯誤，無法處理請求。"
    elif not getattr(app.state, 'models_loaded', False) :
        error_for_template = getattr(app.state, 'startup_error', '模型未成功加載或正在加載中。')
    elif not file.content_type or not file.content_type.startswith("image/"):
        error_for_template = "上傳的文件必須是圖片類型。"
    
    if error_for_template:
        try:
            return templates.TemplateResponse("index.html", {"request": request, "result": None, "error_message": error_for_template})
        except Exception as e_tmpl:
            logging.error(f"Template index.html 載入失敗(表單): {e_tmpl}")
            return HTMLResponse(content=f"<h2>{error_for_template}</h2>", status_code=500)
    
    try:
        image_data_bytes = await file.read()
        analysis_result = analyze_image_depth(image_data=image_data_bytes, filename_for_log=file.filename or "uploaded_file_form")

        result_for_template = {
            "filename": analysis_result.get("filename"),
            "placement_status": analysis_result.get("placement_status", "未知"),
            "depth_value_meters": analysis_result.get("depth_value_meters"),
            "raw_gemini_output": analysis_result.get("raw_gemini_output"),
            "error_message": analysis_result.get("error_message"),
            "was_color_corrected": analysis_result.get("was_color_corrected", False), # Updated field
            "yolo_log_display": "\n".join(analysis_result.get("yolo_log_details", [])),
            "crop_log_display": "\n".join(analysis_result.get("crop_log_details", [])),
            "segmentation_log_display": "\n".join(analysis_result.get("segmentation_log_details", [])), # New
            "gemini_log_display": "\n".join(analysis_result.get("gemini_log_details", []))
        }

        if analysis_result.get("image_with_yolo_boxes_bytes"):
            result_for_template["yolo_image_base64"] = base64.b64encode(analysis_result["image_with_yolo_boxes_bytes"]).decode()
        
        if analysis_result.get("cropped_image_for_gemini_bytes"):
            result_for_template["gemini_crop_image_base64"] = base64.b64encode(analysis_result["cropped_image_for_gemini_bytes"]).decode()

        if analysis_result.get("segmentation_preview_image_bytes"): # New
            result_for_template["segmentation_preview_image_base64"] = base64.b64encode(analysis_result["segmentation_preview_image_bytes"]).decode()

        result_for_template["original_image_base64"] = base64.b64encode(image_data_bytes).decode()
        
    except Exception as e:
        logging.error(f"HTML 表單 (/analyze-depth-form/) 處理圖像 '{file.filename}' 時發生意外錯誤: {e}", exc_info=True)
        error_for_template = f"伺服器內部錯誤: {str(e)}"
        if not result_for_template: 
            result_for_template = {"filename": file.filename if file else "N/A"}
        result_for_template["error_message"] = error_for_template # Ensure error is passed even if result_for_template was partially built
    
    try:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": result_for_template, 
            "error_message": error_for_template 
        })
    except Exception as e_tmpl_ret:
        logging.error(f"Template index.html 載入失敗(表單回傳): {e_tmpl_ret}")
        import json
        return HTMLResponse(content=json.dumps(result_for_template, ensure_ascii=False, indent=2), media_type="application/json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    if GEMINI_API_AVAILABLE and not os.environ.get("GEMINI_API_KEY"):
        logging.warning("環境變數 GEMINI_API_KEY 未設定。Gemini 功能可能無法使用。")
    
    if not os.environ.get(MODEL_PATH_ENV_VAR) and DEPTH_ANALYZER_LOADED: 
        logging.warning(f"環境變數 {MODEL_PATH_ENV_VAR} 未設定。YOLO 偵測模型可能無法載入。")
    if not os.environ.get(SEG_MODEL_PATH_ENV_VAR) and DEPTH_ANALYZER_LOADED: # New check
        logging.warning(f"環境變數 {SEG_MODEL_PATH_ENV_VAR} 未設定。YOLO 分割模型可能無法載入。")
    elif not DEPTH_ANALYZER_LOADED:
        logging.critical("depth_analyzer.py 載入失敗，應用程式無法正確運行。請檢查日誌。")

    uvicorn.run("main_web:app", host="0.0.0.0", port=8000, reload=True)