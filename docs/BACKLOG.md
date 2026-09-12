# Backlog

尚未排入 Epic 的待辦項目,以及 CLAUDE.md §6 各 Epic 尚未完成的任務卡。
開工前依 §7 流程先提設計選項。

最後盤點:2026-09-12。

## Epic 進度

E1–E9 全部任務卡已完成(2026-09-12)。以下為各 Epic 收尾紀錄與尚未排入 Epic 的項目:

### E6 — 評估與測試(P3)

全部任務卡已完成(2026-09-10 補齊 E6-T4/T5/T6/T8)。設計決定記錄在
`horos/api/error_analysis.py` 與 `horos/api/visualize.py` 的模組 docstring:
評估時保存原始偵測、類別無關的貪婪 IoU 配對、錯誤數排序、伺服器端 Pillow 疊圖。

### E7 — 實驗管理(P4)

全部任務卡已完成(2026-09-12 補齊 E7-T1~T7)。設計決定記錄在
`horos/api/experiment.py` 與 `horos/core/fingerprint.py` 的模組 docstring:
使用者備註/標籤與快取放在 `<run>/experiment.json` sidecar(避免與 worker 改寫
`run.json` 競爭)、指紋以資料內容(每 split 的檔名/尺寸/類別名/框/多邊形)雜湊、
mosaic 合成圖不計入、可比較性以指紋差異判定並指出是哪個 split 變了。
UI 為獨立的 `/experiments` 頁;匯出流程由該頁深連結到 `/train#<run_id>`。

### E8 — 匯出與部署(P4)

全部任務卡已完成(2026-09-12 補齊 E8-T7、E8-T3)。

已完成:E8-T1、T2、T4、T5、T6、T8(測試集中在 `tests/api/test_export_model.py`
與 `tests/api/test_export_e2e.py`,未依 CLAUDE.md 逐卡命名)。

E8-T7 完成(2026-09-12):`horos serve` 獨立服務、`horos/backends/runtime/` 免框架 ONNX
執行器、Lab 頁 Serve 區塊;設計決定見 `horos/api/serve.py` 模組 docstring。
E8-T3 完成(2026-09-12):ONNX → onnx2tf → TFLite(float32 + float16,輸入維持 NCHW),
工具鏈為 `horos install --tflite` 選配,parity 以 ai-edge-litert 比對;見 `horos/backends/convert/tflite.py`。
`horos serve` 自 Serve-T1(2026-09-12)起也執行 TensorRT engine 與 TFLite,見下節。

## `horos serve` 執行 TensorRT engine 與 TFLite(Serve)

**完成(2026-09-12)。** 設計決定見 `horos/backends/runtime/__init__.py` 與 `_graphs.py` 的
模組 docstring:三種成品共用同一套前處理與 model card 輸出契約解碼,只有「graph runner」
分格式(onnxruntime / tensorrt runtime + cuda-python 或 torch 的 device memory / LiteRT);
engine 只能在 CUDA 執行、TFLite 只在 CPU 執行,要求做不到的裝置一律明確報錯(R7);
啟動前以 import-free 探測拒絕缺少 runtime 或平台不支援(macOS + engine)的來源,不會先
spawn 子行程再失敗。任務卡:

| 卡 | 內容 | 完成定義 |
|---|---|---|
| Serve-T1 | runtime 執行器支援 TensorRT engine(.trt/.engine/.plan)與 TFLite;`resolve_source` 接受 engine / tflite bundle 與裸檔;`start_server` 啟動前檢查 runtime 與平台能力;`/health` 回報 runtime;CLI `--format` 補齊 | `tests/web/test_serve.py`(合成模型實跑 engine / tflite)、`tests/api/test_serve_artifacts_e2e.py`(真實 RF-DETR 三格式一致) |
| Serve-T2 | Lab 頁 Source 下拉列出 TensorRT / TFLite,依 `/api/v1/capabilities` 灰掉;執行中顯示 runtime | 介面情境 `tests/ui_scenarios/E8-T7.md` A/C 節 |

未做:TFLite int8 量化、`horos serve` 的多請求併發(仍一次一個請求)。

## 點/框 prompt 的互動式標註輔助(SAM 2.1)

**完成(2026-09-12)。設計決定見 `horos/api/segment.py` 與 `horos/backends/sam2/__init__.py` 的模組 docstring。任務卡:**

| 卡 | 內容 | 完成定義 |
|---|---|---|
| SAM-T1 | `PromptableSegmenter` 介面(embed 一次、segment 多次)、SAM 2.1 backend、SAM v1 補實作、registry | `tests/api/test_backend_sam2.py` |
| SAM-T2 | embedding LRU 快取、`segment_image` / `prefetch_embedding` API(只回候選、不寫入) | `tests/api/test_segment_cache.py`、`tests/api/test_segment_interactive.py` |
| SAM-T3 | Web API `POST /images/<id>/segment`、`/segment/prefetch`;能力清單 `assist_interactive` | `tests/web/test_segment_routes.py` |
| SAM-T4 | 標註頁「SAM」工具:點/負點/框、即時預覽、Enter 接受、輸出 polygon/bbox | 介面情境 `tests/ui_scenarios/SAM-T4.md` |


**現況(2026-09-10)**:`horos/backends/sam/` 已有 SAM v1(`facebook/sam-vit-base`,
Apache 2.0)作為**框轉 polygon 的精修器**,供 E3 autolabel 的 polygon 輸出與標註頁的
`POST /images/<id>/assist` 使用。這是批次式、以框為 prompt 的單次呼叫,每次都重跑
整個模型。以下所述的互動式點擊與 embedding 快取尚未實作。

**需求**:輔助標記除了 OWLv2 文字 prompt 之外,增加「點一下」與「畫粗框」兩種
prompt 方式,即時產生 mask / polygon / bbox。

**方案結論**(2026-09 調研):

- 採 **SAM 2.1**(Apache 2.0,程式碼與權重皆是),`transformers` 原生支援
  `Sam2Model` / `Sam2Processor` — horos 已依賴 `transformers>=5.1`,零新相依
- 實作位置:沿用或改寫 `horos/backends/sam/`,遵守 R1 隔離、R1b 延遲載入、E3-T7 權重快取
- 模型變體:`facebook/sam2.1-hiera-tiny`(~150MB,Jetson 首選)/ `hiera-small`
- **關鍵架構點**:image encoder 每張影像只跑一次並快取 embedding,每次點擊只跑
  輕量 prompt decoder(毫秒級)。每次點擊重跑整個模型的體驗不可接受。
  現有的 `BoxToMaskBackend` 介面沒有 embedding 快取的概念,需要擴充 `backends/base.py`
- 定位:互動式輔助貼近 E2 標註畫布(一次一張、即時回饋),與 E3 批次自動標記互補。
  自然流程:OWLv2 批次預標 → 標註頁用 SAM 點/框修正補框

**授權上要避開**:SAM 3(自訂 SAM License,非 Apache,需比照 XL/2XL 阻擋機制)、
FastSAM(AGPL)、EdgeSAM(S-Lab 僅研究用)、ultralytics 的 SAM 封裝(AGPL)。
合規備案:MobileSAM / EfficientSAM(Apache 2.0,但不在 transformers 內,划算度低)。

參考:
- https://huggingface.co/docs/transformers/model_doc/sam2
- https://huggingface.co/facebook/sam2.1-hiera-tiny

## CI(R7:Ubuntu + Windows runner)

**完成(2026-09-12)。** `.github/workflows/ci.yml`:

| 卡 | 內容 | 完成定義 |
|---|---|---|
| CI-T1 | GitHub Actions:`invariants` job 先跑 `tests/test_invariants.py` 與 ruff;`core` 矩陣 Ubuntu × Windows × Python 3.10 / 3.12,torch-free 安裝(加 onnx / onnxruntime / matplotlib / openpyxl)跑整套測試;`ml` job(CPU torch 全棧)僅每週排程或手動觸發;`.gitattributes` 固定 LF | Actions 兩個 OS 綠燈;README 徽章 |

設計決定:每次 push 的矩陣刻意不裝 torch —— 那正是 `pip install horos` 使用者(只標註)的環境,
需要 ML stack 的測試自行 skip、假 backend 覆蓋訓練 / 匯出 / 服務流程;全棧測試太慢太大,留給排程。

## SAM-T5 — SAM 工具多物體批次接受(2026-09-12 完成)

Space(或「Next object」)把目前候選連同當時的類別入列,prompt 清空繼續點下一個;拖新框自動入列;
Enter 一次寫入全部(單一 undo 步);Esc 兩段式清除;切工具/切圖清空隊列。介面情境
`tests/ui_scenarios/SAM-T4.md` D 節。

