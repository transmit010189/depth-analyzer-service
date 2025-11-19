# 箱尺埋深檢測服務 (Depth Analyzer Service)

本專案為道路挖掘使用的箱尺埋深判讀工具，  
使用 Ultralytics YOLO (YOLO11) 與 YOLO segmentation 模型，搭配 Google Gemini 進行刻度讀取與黃/白顏色判斷。

## 功能

- 偵測影像中的箱尺位置。
- 裁切讀值區域並交給 Gemini 讀取深度。
- 使用 segmentation 模型判斷箱尺顏色（黃色/白色），作為 ±1 公尺校正依據。
- 提供：
  - `POST /analyze-depth` API
  - `/` / `/analyze-depth-form` 網頁上傳介面

## 安裝與執行

```bash
pip install -r requirements.txt

# 設定必要環境變數（範例）
export GEMINI_API_KEY="你的金鑰"
export MODEL_PATH_ENV="/app/models/yolo_det.pt"
export SEG_MODEL_PATH_ENV="/app/models/yolo_seg.pt"

uvicorn main_web:app --host 0.0.0.0 --port 8000

授權

本專案以 AGPL-3.0 授權釋出，詳細條款請參考 LICENSE。
本專案使用 Ultralytics YOLO11，Ultralytics 之原始碼與授權資訊請見官方專案。


