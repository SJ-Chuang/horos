# 上傳擴充 — VOC/Darknet 匯入與檔名衝突確認

## 怎麼啟動

```bash
horos init ./demo_project
horos ui ./demo_project
# 瀏覽器開 http://localhost:5000
```

## 測試步驟

### A. VOC 與 Darknet 上傳

1. 準備 Roboflow 匯出的 Pascal VOC zip（影像旁有同名 `.xml`），拖進虛線框
2. 應直接匯入成功，摘要列顯示 `(VOC)`；split 依 train/valid/test 目錄歸類
3. 準備 Roboflow 匯出的 Darknet zip（影像旁有同名 `.txt` 與 `_darknet.labels`），拖入
4. 應直接匯入成功，摘要列顯示 `(DARKNET)`，類別名取自 `_darknet.labels`

### B. Darknet 缺類別名清單

1. 準備一包 Darknet zip，但**移除所有 `_darknet.labels`**，拖入
2. 應彈出「Class names needed」對話框，逐類一個輸入框，預設值為類別索引（0、1、…）
3. 修改名稱（如 `helmet`、`vest`）後按「Import with these names」
4. 匯入成功，「Instances per class」表格顯示的是輸入的名稱
5. 重做步驟 1 但按 Cancel：狀態列顯示「Import cancelled — nothing was changed.」，摘要數字不變

### C. 檔名衝突確認

1. 上傳任一資料集 zip，成功後**再上傳同一包**
2. 不應彈窗：狀態列顯示 `Imported 0 images, … — N identical duplicate(s) skipped`（內容相同自動跳過）
3. 修改 zip 裡任一張影像的內容（檔名不變）後重新上傳
4. 應彈出「Some file names already exist」對話框，列出衝突檔名
5. 按「Overwrite」：狀態列顯示 `1 overwritten`，該影像與其標註被新版取代
6. 重做步驟 3–4 改按「Skip them」：顯示 `1 conflict(s) skipped`，專案維持原樣
7. 重做步驟 3–4 改按「Import renamed」：顯示 `1 renamed`，出現 `xxx_1.jpg` 新檔
8. 重做步驟 3–4 改按「Cancel」：狀態列顯示 cancelled，任何東西都沒被改動

## 預期結果

- 四種格式（COCO / YOLO / VOC / Darknet）拖進去都直接匯入，格式自動偵測
- 內容相同的重複上傳永不彈窗；真正的衝突一定先問、未確認前不寫入任何資料

## 已知限制

- VOC 只讀 bbox（VOC 無標準 polygon 表示法）；Darknet 只讀 bbox
- VOC / Darknet 為僅匯入格式，匯出仍為 COCO / YOLO
- 衝突對話框一次套用同一策略到整批衝突，不支援逐張選擇
