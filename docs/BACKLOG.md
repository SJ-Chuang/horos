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
`horos serve` 目前仍不執行 TensorRT engine 與 TFLite(明確拒絕),可作後續項目。

## 點/框 prompt 的互動式標註輔助(SAM 2.1)

**進行中(2026-09-12,設計選項已確認「照建議」)。任務卡:**

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
