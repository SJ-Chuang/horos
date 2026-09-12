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

## E8-T3b — TFLite int8(2026-09-12 完成:int8 權重的 dynamic-range 變體)

`start_model_export(..., format="tflite", options={"int8": True})` 在 float32 / float16 之外多產出
`<model>_int8.tflite`:**dynamic-range 量化(int8 權重、float32 activation 與 I/O,不需校準)**,
檔案約為 float32 的 1/4(nano:106 MB → 29 MB)。走 onnx2tf 的 legacy `tf_converter` 後端、
Erf 以 tanh 近似取代(TFLite 沒有內建 Erf,否則會變成需要 TF runtime 的 Flex op;近似後 float32
與原圖差 ~4e-5),多花約 4 分鐘,輸入為 NHWC(執行器兩種版面都吃,model card `variants.int8`
記錄 `input_layout`)。float32 仍是主成品;int8 變體有獨立 parity(容差 0.1),失敗只警告不阻擋。
Train 頁 Model 下拉多一項「TFLite + int8 weights」。測試:`tests/api/test_tflite_convert.py`(合成模型,
靜態 int8 與 dynamic-range 兩條路)、`tests/api/test_export_tflite.py`(真實 RF-DETR)、
`tests/api/test_export_model.py`(選項傳遞與 card 記錄)。

**已驗證不可行(2026-09-12,onnx2tf 2.6.8 / TF 2.21 / LiteRT 2.1.2)— 靜態 int8(權重+activation)for RF-DETR:**
- `flatbuffer_direct` 後端 `-oiqt`:轉換直接失敗(`flatbuffer_direct fast path failed`),加 pseudo-Erf 亦同
- `tf_converter` 後端 `-oiqt` + pseudo-Erf:能產出 `_integer_quant.tflite`,但 LiteRT 執行時
  `tflite/kernels/div.cc:242 data[i] != 0 was not true`(LayerNorm 的分母被量化為 0),隨機與真實校準資料皆同
- 不加 pseudo-Erf:校準器無法執行 Flex Erf
轉換器層保留了靜態 int8 的支援(`precisions=("int8",)` + `calibration`,小圖有測試),日後 onnx2tf /
TFLite 修好 transformer 的量化再開給 RF-DETR。未做:full-integer(int8 I/O)、`horos serve --format tflite`
直接選 int8 變體(裸檔路徑可以)。

## SAM-T5 — SAM 工具多物體批次接受(2026-09-12 完成)

Space(或「Next object」)把目前候選連同當時的類別入列,prompt 清空繼續點下一個;拖新框自動入列;
Enter 一次寫入全部(單一 undo 步);Esc 兩段式清除;切工具/切圖清空隊列。介面情境
`tests/ui_scenarios/SAM-T4.md` D 節。

## SAM-T6 — box 當作 hint：box → polygon（2026-09-12 完成）

既有的 box 標註本身就是好的 SAM prompt。三個入口共用同一個 API 家族（`horos/api/segment.py`）：
- 單一 box：Shapes 列表的「⬠」或選取後按 P（前端直接呼叫既有 `POST /images/<id>/segment` 的 box prompt，
  對未儲存的 box 也有效，結果走一般儲存）
- 整張圖：SAM 面板「Boxes → polygons (this image)」（同上，逐 box 呼叫，embedding 快取）
- 整個專案、可指定類別：Auto-label 對話框「Boxes → polygons (SAM)」→ `POST /api/v1/segment/boxes`
  背景 job（`segment.boxes_to_polygons_batch`；CLI `horos boxes-to-polygons --class X`）；
  單張版 `POST /api/v1/images/<id>/segment/boxes`（`segment.boxes_to_polygons`）
幾何以外的欄位（id、類別、pending 狀態、分數、來源）保持不變；SAM 找不到 mask 的 box 維持 box 並計入 skipped。
測試：`tests/api/test_segment_boxes.py`、`tests/web/test_segment_routes.py`、`tests/api/test_cli.py`；
介面情境 `tests/ui_scenarios/SAM-T4.md` E 節、`E3-autolabel.md` D 節。

## 標註流程調整：畫完不切回 Select、沒有預設 `object`（2026-09-12 完成）

矩形 / 多邊形完成後停留在原工具（連續畫）。Object Class 留空時不再默默套上 `object`：
形狀先灰色虛線顯示，跳「Which class is this?」對話框（輸入 / datalist / chips），確認後成為
當前類別;取消則丟棄形狀。SAM 的 Enter / Space、相同規則;拖新框自動入列只在已有類別時發生。
儲存路徑遇到空類別會明確報錯而不是建立類別。情境 `tests/ui_scenarios/E2-annotator.md` B/C/D。


## 標記 QC：SAM polygon 只描最大 blob、validator 檢查 polygon 與 bbox 一致（2026-09-13 完成）

對 demo_project（Roboflow「Box detection」，10,831 張、23,506 筆，全部經 SAM-T6 批次 box → polygon）做 QC
時發現 4,489 筆 polygon 的範圍與 bbox 不一致，其中 1,024 筆 polygon 只是幾個像素的碎片。根因在
`horos/backends/sam/polygonize.py`：只描「最上/最左那個前景像素所在的 blob」，而 bbox 用整張 mask 的邊界；
Roboflow 的雜訊增強讓 SAM mask 常帶零星小島，小島在物體上方時就被當成 polygon。修法：

- polygonizer 先以 row-run union-find 找出最大 8-連通 blob（不依賴 OpenCV/numpy），polygon、bbox、area
  一律由同一個 blob 計算（`mask_to_shape`）；`TransformersPromptableMixin._result_from_mask` 走這條路。
- validator 新增 `polygon_bbox_mismatch`（容差 2 px）：polygon 覆蓋 bbox 寬高各 ≥ 50% 視為 bbox 漂移，
  `horos validate --fix` / Dataset 頁 Fix 以 polygon 重算 bbox（warning、fixable）；覆蓋更少即碎片，
  只能重畫或移除 polygon 後再跑 boxes-to-polygons（error）。`invalid_polygon` 另外抓零面積（共線）polygon。
- demo_project 修復：備份至 `~/research/demo_project_annotations_backup_20260913.tar.gz`；1,024 筆碎片改回
  原始人工 box（1,010 筆比對到來源 IoU ≥ 0.5）後以修正後的 polygonizer 重跑，先前 SAM 略過的 1,326 個 box
  一併重試（共 2,346 筆轉成 polygon，5 筆仍為 box）；3,110 筆漂移 bbox 以 `--fix` 重算；來源有、專案中
  遺失的 12479 那筆 box 已還原（SAM 對它切到背景，保留人工 box）。`horos validate` 回到 0 issue。
- 尚待人工複核：444 筆 SAM polygon 的外框與原始人工 box 的 IoU < 0.5（清單
  `~/research/demo_project_qc_review_20260913.txt`）。抽樣目視兩種情況都有：人工 box 畫得鬆、SAM 反而更準
  （不能自動退回），以及 SAM 切到相鄰物件或背景（該退回人工 box），所以不做自動處理。

測試：`tests/unit/test_polygonize.py`、`tests/api/test_backend_sam2.py`、`tests/api/test_dataset_validate.py`、
`tests/api/test_validate_fix.py`。
