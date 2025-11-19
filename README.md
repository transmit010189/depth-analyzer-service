# depth-analyzer-service

道路挖掘工程用的 **箱尺埋深檢測服務**。  
使用 Ultralytics YOLO (YOLO11) 與 segmentation 模型辨識箱尺位置與顏色，並搭配 Google Gemini 讀取刻度，提供管線埋深與黃/白線 ±1 公尺校正資訊。

本專案是以 FastAPI 建立的雲端服務，可部署於 Cloud Run，供「道路挖掘管理系統」或其他系統透過 API 呼叫使用。

---

## 功能簡介

- 物件偵測：偵測影像中的箱尺與相關區域。
- 實例分割：使用 YOLO segmentation 模型判斷箱尺顏色（黃色 / 白色），作為埋深 ±1m 判定依據。
- 刻度讀取：裁切箱尺讀值區域，交由 Gemini 進行文字/數值辨識。
- 校正邏輯：依顏色判斷與讀值，計算最終埋深結果。
- 提供：
  - JSON API：`POST /analyze-depth` / `/analyze-depth/`
  - HTML 表單頁面：`/`（上傳圖片立刻在網頁看到結果）

---

## 專案結構

```text
depth-analyzer-service/
  main_web.py            # FastAPI 入口與路由
  depth_analyzer.py      # YOLO / segmentation / Gemini 分析核心邏輯
  requirements.txt       # Python 套件依賴
  openapi-spec.yaml      # API Gateway / OpenAPI 規格 (選用)
  templates/
    index.html           # 前端上傳與結果顯示頁面
  LICENSE                # AGPL-3.0 授權條款
  README.md
  .gitignore

注意：本 repo 不包含實際的 YOLO 權重檔與測試影像，請依下方說明自行準備。

執行前置需求

Python 3.10 或以上

已安裝 git、pip

若要使用 GPU，需有對應的 CUDA / 驅動環境

需要一組可用的 Google Gemini API Key

安裝與本機執行

1.取得原始碼：
git clone https://github.com/transmit010189/depth-analyzer-service.git
cd depth-analyzer-service

2.建立並啟用虛擬環境（建議）：
python -m venv .venv
source .venv/bin/activate   # Windows 改用: .venv\Scripts\activate

3.安裝依賴套件：
pip install -r requirements.txt

4.準備 YOLO 權重檔並設定環境變數：

將你訓練好的模型放到指定路徑，例如：
models/
  yolo_det.pt   # YOLO detection 模型
  yolo_seg.pt   # YOLO segmentation 模型

設定環境變數（可以用 .env 或在終端機直接輸入）：
# 給 depth_analyzer.py 使用的模型路徑
export MODEL_PATH_ENV="/absolute/path/to/models/yolo_det.pt"
export SEG_MODEL_PATH_ENV="/absolute/path/to/models/yolo_seg.pt"

# Gemini 金鑰（請換成你自己的）
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

若你是從 GCS 抓模型，可另外設定：
export MODEL_BUCKET="your-gcs-bucket-name"

5.啟動服務：
uvicorn main_web:app --host 0.0.0.0 --port 8000

使用方式：

開啟瀏覽器造訪：
http://localhost:8000/
上傳一張箱尺 / 管線挖掘照片即可看到分析結果與詳細日誌。

以 JSON API 呼叫：
curl -X POST "http://localhost:8000/analyze-depth" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg"

API 端點概觀
GET /

類型：HTML

內容：上傳圖片的表單頁與分析結果圖 / 日誌。

POST /analyze-depth 及 /analyze-depth/

類型：multipart/form-data

參數：

file：要分析的圖片檔案。

回傳：application/json，內容包含：

基本資訊：檔名、推論時間等

偵測結果：箱尺位置、顏色判斷

埋深判讀：原始讀值、校正後深度

raw_gemini_output：Gemini 的原始文字輸出

gemini_log_details：分析過程日誌（方便除錯）

POST /analyze-depth-form/

類型：HTML（表單送出）

與 / 搭配使用，主要給使用者在網頁介面操作。

更完整的 Request / Response 定義可參考 openapi-spec.yaml。

部署到 Cloud Run（簡略說明）

建立 Docker 映像或使用 Cloud Run 直接從原始碼部署。

在 Cloud Run 服務中設定與本機相同的環境變數：

GEMINI_API_KEY

MODEL_PATH_ENV

SEG_MODEL_PATH_ENV

（選用）MODEL_BUCKET

若搭配 API Gateway，可將 openapi-spec.yaml 匯入，將外部 URL 代理到 Cloud Run 服務。

實際部署步驟可依各自環境調整。

安全性與機密資料

本專案程式碼 不包含任何真實 API Key 或 GCP 憑證。

請務必透過環境變數或 Secret Manager 管理：

GEMINI_API_KEY

任何 GCP service account JSON 等敏感資訊。

請勿將 .env、金鑰檔、實際工地影像資料 commit 到公開 repo。

授權 / License

本專案程式碼以 AGPL-3.0 授權釋出，詳細條款請參閱本庫中的 LICENSE 檔案。

本專案使用：

Ultralytics YOLO11 模型與程式庫

其他開源套件（FastAPI、OpenCV、NumPy、google-generativeai 等）

上述第三方軟體皆依各自授權條款使用，相關著作權與商標權屬原權利人所有。

聯絡方式

如有錯誤回報、功能建議或協助需求，歡迎於 GitHub Issues 建立問題。

Repository：
https://github.com/transmit010189/depth-analyzer-service
