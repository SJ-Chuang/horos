# E3 — 自動標記與審核（E3-T8 審核 UI、E3-S1/S2/S3/S5/S6）

## 怎麼啟動

```bash
# 需要安裝完整相依（含 torch/transformers）；首次執行會下載 OWLv2 權重
horos ui --project ./demo_project
# 瀏覽器開 http://localhost:5000/annotate
```

## 測試步驟

### A. 批次自動標記（E3-S1）
1. 網格頁右上按「Auto-label…」：彈窗出現，一列＝一個類別（類別名＋逗號分隔的提示詞，
   提示詞留空則用類別名）
2. 填入 `helmet`、`vest` 兩列，Confidence 0.10，勾選 only unannotated，按 Start
3. 進度條隨影像數推進，文字顯示「N/M — 檔名: K pre-label(s)」；首次執行顯示
   權重下載提示（E3-S6，下載中斷可續）
4. 完成後 toast 顯示產生的預標數，佇列自動切到「Needs review (uncertain first)」
5. 執行中按 Cancel：已完成的影像保留預標，job 狀態顯示 cancelled

### B. 審核（E3-S2/S3/S5）
1. 「Needs review」模式下網格依不確定性排序（模型最沒把握的排前面），
   角標顯示黃色 ◌N（N＝待審數）
2. 點入影像：預標以「虛線框＋類別名與信心度」呈現，與人工標註（實線）可區分
3. 右欄「Review pre-labels」卡片：拉動門檻滑桿，低於門檻的預標即時淡化、
   計數顯示「N pending · K above threshold」（E3-S2）
4. 按「Accept ≥ t」：門檻以上轉正式標註（變實線）、以下刪除；「Reject all」全部刪除
5. Shapes 清單中單一預標可按 ✓ 個別接受、× 個別刪除；拖曳修正預標的框
   再接受，修正會保留（E3-S5）
6. 被接受的預標在資料上仍標記 source=auto，與人工標註可追溯區分

### C. 編輯器單張 AI 輔助
1. 按 `4`（或點綠色 T 工具）：出現提示詞輸入列
2. 輸入 `helmet, person` 按 Apply（或 Enter）：當前影像即時產生虛線預標，
   走同一套審核流程；新類別自動建立
3. 找不到物件時顯示「no objects found — lower the confidence?」

### D. CLI 對等（E9-S3）
```bash
horos autolabel --project ./demo_project --prompt "helmet=helmet,hard hat" --prompt vest
# 逐行輸出 JSON events（started/prediction/progress/completed）
```

## 預期結果

標註者從「修正模型的預標」開始工作而不是從空白開始；批次跑在背景、
進度即時可見；沒把握的影像優先送到眼前。

## 已知限制

- 同時只允許一個背景 job，第二個會被明確拒絕
- 單張 AI 輔助的首次呼叫需載入模型（數秒到數十秒），期間 Apply 按鈕停用
- dev 環境未裝 transformers 時，job 會以明確的 failed 事件結束（不會靜默）
- 批次進度輪詢間隔 0.8 秒，進度最多慢一拍
