# Backlog

尚未排入 Epic 的待辦項目,以及 CLAUDE.md §6 各 Epic 尚未完成的任務卡。
開工前依 §7 流程先提設計選項。

最後盤點:2026-09-10。

## Epic 進度

E1、E2、E3、E4、E5、E6、E9 全部任務卡已完成。其餘尚未完成的任務卡:

### E6 — 評估與測試(P3)

全部任務卡已完成(2026-09-10 補齊 E6-T4/T5/T6/T8)。設計決定記錄在
`horos/api/error_analysis.py` 與 `horos/api/visualize.py` 的模組 docstring:
評估時保存原始偵測、類別無關的貪婪 IoU 配對、錯誤數排序、伺服器端 Pillow 疊圖。

### E7 — 實驗管理(P4,幾乎未做)

| 任務 | 狀態 | 備註 |
|---|---|---|
| E7-T1 run metadata schema 與儲存 | 部分 | `RunRecord` 已存在(`horos/api/train.py`),缺 `tests/api/test_run_store.py` |
| E7-T2 資料集指紋 | 部分 | `_dataset_fingerprint` 在 `horos/api/export.py`,只寫進 model card;缺獨立模組與 `tests/api/test_dataset_fingerprint.py` |
| E7-T3 run 查詢與排序 | 部分 | 有 `list_runs`,沒有依指標排序 |
| E7-T4 不可比較性警告 | 未做 | 指紋不同時要標示 |
| E7-T5 備註與標籤 | 未做 | train.html 的 notes 是超參數推導說明,不是使用者備註 |
| E7-T6 比較 UI | 未做 | |
| E7-T7 Web API endpoints | 未做 | 缺 `tests/web/test_experiment_routes.py` |

### E8 — 匯出與部署(P4,部分完成)

| 任務 | 狀態 | 備註 |
|---|---|---|
| E8-T3 TFLite 匯出 | 未做 | `MODEL_FORMATS` 目前只有 pytorch / onnx / tensorrt |
| E8-T7 本地推論服務 | 未做 | 已有 `/train/runs/<id>/infer` 端點,但沒有獨立的 serve 服務與 CLI 子命令;缺 `tests/web/test_serve.py` |

已完成:E8-T1、T2、T4、T5、T6、T8(測試集中在 `tests/api/test_export_model.py`
與 `tests/api/test_export_e2e.py`,未依 CLAUDE.md 逐卡命名)。

## 點/框 prompt 的互動式標註輔助(SAM 2.1)

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
