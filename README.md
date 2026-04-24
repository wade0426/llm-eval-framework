# LLM Capability Evaluation Framework

一套以 CLI 驅動的 Python 應用程式，專為**系統性能力邊界評測**設計。支援任意 OpenAI 相容的 LLM API。
使用者只需準備一份含有標準答案的 Golden Dataset（CSV 格式），並透過 YAML 設定檔控制所有評測參數，程式即可自動呼叫 LLM、記錄回答，並可選擇性啟用「LLM-as-a-Judge」二次評判機制，最後輸出結構化結果報告。

## 專案亮點

- **設定驅動（Config-Driven）：** 所有的 API 參數、欄位對應、執行策略皆由 `config.yaml` 控制，程式碼內不寫死業務邏輯。
- **標準答案絕對隔離：** 嚴格確保 Golden Dataset 中的標準答案不會在任何情況下洩漏至受測 LLM 的 Prompt 中。
- **冪等性與斷點續跑：** 自動記錄已處理進度（Checkpoint），中斷後重跑不會重複呼叫 API 或造成資料汙染。
- **[V2] 多線程並行推論：** 支援 ThreadPoolExecutor 並行處理，透過 `max_workers` 大幅提升評測吞吐量。
- **[V2] 對話紀錄持久化：** 支援輸出 JSONL 格式的完整對話日誌，便於後續分析每一次的 Prompt 與 Response。
- **[V3] 多模態圖片支援（Vision）：** 支援載入本地圖片並以 Data URI (Base64) 格式與文字 Prompt 整合，向後相容純文字評測。支援多張圖片（分號分隔）。

---

## 1. 系統需求與安裝

### 環境需求
- Python 3.10+
- 一組 OpenAI 相容的 API Endpoint 與 API Key（支援環境變數注入）

### 安裝步驟

```bash
# 1. 建立虛擬環境
python -m venv .venv

# 2. 啟動虛擬環境 (Windows)
.venv\Scripts\activate
# 或者 (Linux / macOS)
source .venv/bin/activate

# 3. 安裝依賴套件
pip install -r requirements.txt
```

### 設定環境變數
系統支援在 YAML 設定檔中使用 `${ENV_VAR}` 語法。請先在終端機中設定對應的環境變數：

```bash
export OPENAI_API_KEY="sk-..."
export JUDGE_API_KEY="sk-..."  # 若有啟用 Judge 功能
```

---

## 2. 設定檔說明 (`config.yaml`)

核心行為完全由 YAML 定義，並透過 Pydantic V2 進行嚴格的結構與防呆驗證。

### 基礎結構範例

```yaml
dataset:
  input_path: "data/golden/golden_dataset.csv"      # 輸入的 CSV 檔案
  output_path: "data/output/result.csv"             # 輸出的 CSV 檔案
  output_mode: "new"                                # "new" 或 "overwrite"
  encoding: "utf-8"

column_mapping:
  answer_column: "expected_output"                  # 標準答案欄位（受隔離保護）
  input_columns:
    merge: true                                     # 是否合併多個欄位作為 Prompt
    columns: ["input"]
    merge_template: "{input}"                       # merge 為 true 時的 Prompt 樣板
  output_column: "llm_answer"                       # 存放受測 LLM 回答的欄位名稱

primary_llm:
  base_url: "https://api.openai.com/v1"             # 支援任意 OpenAI 相容 API
  api_key: "${OPENAI_API_KEY}"                      # 讀取環境變數
  model: "gpt-4o"
  system_prompt: "你是一個專業的問答助手..."
  temperature: 0.0
  max_tokens: 1024
  timeout_seconds: 30

execution:
  max_retries: 3                                    # 失敗重試次數（指數退避）
  retry_delay_seconds: 2
  rate_limit_rpm: 60                                # 全局速率限制（Requests Per Minute）
  checkpoint_path: "data/checkpoints/.checkpoint.json"
  batch_log_interval: 10
  max_workers: 4                                    # [V2] 並行處理的執行緒數量

judge:
  enabled: false                                    # 是否啟用 LLM-as-a-Judge
  # ... 其他 LLM 設定與 primary_llm 類似 ...

conversation_log:                                   # [V2] 對話紀錄設定
  enabled: true
  log_path: "data/output/conversation.jsonl"
  log_mode: "append"                                # "append" 或 "latest_only"

image:                                              # [V3] 多模態圖片設定
  enabled: true                                     # 是否傳送圖片（預設 false 為純文字模式）
  image_column: "image_path"                        # CSV 中的圖片欄位名稱
  base_dir: "./data/golden/images"                  # 本地圖片根目錄
  detail: "auto"                                    # OpenAI Vision 參數: "auto" | "low" | "high"
  separator: ";"                                    # 單一欄位中多張圖片的切分符號（不可為逗號）
```

---

## 3. CLI 指令與使用方式

程式進入點為 `src/main.py`，所有日誌均統一輸出至終端機及 `data/output/run.log`。

### 乾跑驗證 (Dry-run)
強烈建議在實際耗費 API 額度前，先進行乾跑驗證。
此模式會**載入並驗證 Config**、**檢查 CSV 欄位**，並**印出前 3 筆組合好的 Prompt** 供人工確認，**不會**呼叫任何 API 且**不會**寫入任何檔案。

```bash
python -m src.main --dry-run -c config.yaml
```

### 正式執行
依照設定檔開始逐筆（或並行）評測：

```bash
python -m src.main -c config.yaml
```

### 重新執行（清除斷點）
若需要忽視之前的執行進度並從頭開始，可加上 `--reset-checkpoint`：

```bash
python -m src.main --reset-checkpoint -c config.yaml
```

---

## 4. V2 特性說明 (Phase 6-7)

本專案已升級至 V2 架構，針對大規模評測提供了效能與觀測性的改進：

### 🚀 多線程並行推論 (Multi-threaded Inference)
透過設定 `execution.max_workers`，程式會啟動 `ThreadPoolExecutor` 進行平行 API 呼叫。
- **線程安全保證**：CSV 的更新 (`CsvHandler`) 與斷點寫入 (`CheckpointManager`) 均已加入 `threading.Lock` 確保多執行緒安全。
- **全域速率限制**：`RateLimiter` 跨執行緒共享，確保整體 API 呼叫頻率不會超過 `rate_limit_rpm` 限制。
- **優雅中斷**：支援 `Ctrl+C` (KeyboardInterrupt)。觸發時會中斷排隊中的任務、等待運行中任務完成，並安全地儲存當前的 DataFrame 與 Checkpoint，確保下次能完美續跑。

### 📝 對話紀錄持久化 (Conversation Logger)
透過設定 `conversation_log` 區塊，系統可以即時將每一筆推論細節寫入 JSONL 檔案。
- 單行 JSON 包含：`index`, `timestamp`, `system_prompt`, `user_prompt`, `image_paths` (V3 支援), `llm_response`, `status`, `duration_seconds`。
- **Lock-protected Flush**：即使在多線程高併發下，寫入依然保證行級別的完整性，並在每次寫入後呼叫 `flush()` 立即落盤，確保程式崩潰時不掉 Log。

---

## 5. V3 多模態圖片支援 (Phase 8-10)

V3 支援 OpenAI 的 Vision API 格式。系統會讀取本地圖片，將其轉為 Base64 (Data URI) 格式後與 Prompt 並列送入 LLM。

### 📌 向後相容與開關設定
- 當 `image.enabled: false` 時，系統回歸純文字模式（V2 行為），不受任何圖片影響。
- 當啟用圖片時，CSV 中**允許該欄位留空**。對於沒有圖片的橫列，LLM 仍會接收單純的文字 Prompt，不會因為沒有圖片而報錯。

### 🖼️ 多張圖片與路徑解析
為避免與 CSV 的 `,` 逗號格式衝突，請在 CSV 儲存格中使用**分號** `;` (由 `image.separator` 決定) 串接多張圖片。
- 舉例：`1.png;2.jpg;folder/3.webp`
- 系統會透過 `image.base_dir` 將相對路徑拼接為絕對路徑進行讀取。

### ⚠️ 圖片錯誤處理 (`__IMAGE_ERROR__`)
若某列資料的圖片檔案不存在、或副檔名不支援，**該單筆資料**的評測結果會直接被標記為 `__IMAGE_ERROR__`，不會發送請求至 LLM（節省 Token 成本），且**不影響其他並發 Thread** 的繼續執行，確保大規模評測的穩定度。

---

## 6. 測試 (Testing)

專案包含完整的 Pytest 單元與整合測試，覆蓋了邊界防護、Retry 機制與多線程安全。

```bash
# 執行完整測試套件
pytest -q
```

**測試涵蓋範圍包含：**
- **Config Loader**：環境變數插值與 Pydantic 防呆驗證（例如答案欄位洩漏檢查）。
- **Prompt Builder**：多欄位合併模板與強隔離斷言、V3 多模態 Payload (Content Parts)。
- **LLM Client**：Tenacity 指數退避 Retry 與 RateLimiter 邏輯。
- **Thread Safety**：在多線程併發環境下的 `Checkpoint.mark_done` 與 `CsvHandler.update_row` 不會產生 Race Condition。
- **Conversation Logger**：`append` 與 `latest_only` 模式以及並發寫入完整性。
- **Image Loader**：Base64/MIME 解析與錯誤處理（File Not Found）。

---

## 7. 專案架構 (Project Structure)

```text
llm-eval-framework/
├── src/
│   ├── main.py                # CLI 進入點
│   ├── core/
│   │   ├── config_loader.py       # YAML 載入與環境變數解析
│   │   ├── csv_handler.py         # CSV 讀寫 (Thread-safe)
│   │   ├── prompt_builder.py      # Prompt 組裝與答案隔離
│   │   ├── llm_client.py          # API 客戶端與重試機制
│   │   ├── judge.py               # LLM-as-a-Judge 邏輯
│   │   ├── image_loader.py        # [V3] 圖片載入與多模態 Base64 解析
│   │   ├── checkpoint.py          # 斷點管理 (Thread-safe)
│   │   ├── conversation_logger.py # JSONL 對話紀錄 (Thread-safe)
│   │   └── runner.py              # 並行調度器 Orchestrator
│   ├── models/
│   │   └── config_schema.py       # Pydantic 模型定義
│   └── utils/
│       └── logger.py              # 日誌格式統一定義
├── data/
│   ├── golden/                # 評測用 CSV 輸入檔
│   ├── output/                # 結果 CSV 與 Log 輸出
│   └── checkpoints/           # 斷點紀錄檔
├── tests/                     # Pytest 測試目錄
├── config.yaml                # 設定檔範本
└── requirements.txt           # 依賴套件清單
```

