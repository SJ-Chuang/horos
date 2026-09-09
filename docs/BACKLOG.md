# Backlog

尚未排入 Epic 的待辦項目。開工前依 §7 流程先提設計選項。

## 點/框 prompt 的互動式標註輔助（SAM 2.1）

**需求**：輔助標記除了 OWLv2 文字 prompt 之外，增加「點一下」與「畫粗框」兩種
prompt 方式，即時產生 mask / polygon / bbox。

**方案結論**（2026-09 調研）：

- 採 **SAM 2.1**（Apache 2.0，程式碼與權重皆是），`transformers` 原生支援
  `Sam2Model` / `Sam2Processor` — horos 已依賴 `transformers>=5.1`，零新相依
- 實作位置：`horos/backends/sam2/`，沿用 R1 隔離、R1b 延遲載入、E3-T7 權重快取
- 模型變體：`facebook/sam2.1-hiera-tiny`（~150MB，Jetson 首選）/ `hiera-small`
- **關鍵架構點**：image encoder 每張影像只跑一次並快取 embedding，每次點擊只跑
  輕量 prompt decoder（毫秒級）。每次點擊重跑整個模型的體驗不可接受
- 定位：互動式輔助貼近 E2 標註畫布（一次一張、即時回饋），與 E3 批次自動標記互補。
  自然流程：OWLv2 批次預標 → 標註頁用 SAM 點/框修正補框

**授權上要避開**：SAM 3（自訂 SAM License，非 Apache，需比照 XL/2XL 阻擋機制）、
FastSAM（AGPL）、EdgeSAM（S-Lab 僅研究用）、ultralytics 的 SAM 封裝（AGPL）。
合規備案：MobileSAM / EfficientSAM（Apache 2.0，但不在 transformers 內，划算度低）。

參考：
- https://huggingface.co/docs/transformers/model_doc/sam2
- https://huggingface.co/facebook/sam2.1-hiera-tiny
