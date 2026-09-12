# SAM-T4 — 標註頁「SAM」工具(點/框互動式輔助,SAM 2.1)

工具列第 5 個按鈕「✦」(快捷鍵 5)。第一次點某張圖會跑一次 image encoder(桌機 GPU 約 0.2 s,
首次使用會先下載約 150 MB 的權重,狀態列會說明),之後每一次點擊只跑 prompt decoder(數十 ms)。
候選形狀只在畫面上預覽,按 Enter 才寫入,走的是一般的儲存路徑(版本鎖、多人衝突邏輯不變)。

## 啟動

```
horos ui <project>
瀏覽器開 http://localhost:5000/annotate,點任一張圖進入編輯器
```

## 測試步驟

### A. 點一下就有 mask

1. 按 5 或點「✦」。側欄出現 SAM 面板,狀態列顯示「preparing the image…」,數秒內變成「ready (N ms)」
2. 在 Object Class 輸入類別(例如 `forklift`),面板的「Enter accepts it as forklift」跟著變
3. 左鍵點物體:畫布出現半透明的類別色 mask 加白色虛線邊,點位是綠色「+」;
   狀態列顯示「mask N px² · confidence 0.9x · NN ms」
4. 再點同一物體的另一處:mask 擴大修正,第二次起狀態列不再有「(encoder)」字樣
5. 右鍵點(不拖曳)被誤含進來的區域:紅色「−」點,mask 收回;Shift+左鍵效果相同。
   右鍵按住拖曳仍是平移畫布,不會落點
6. Backspace 移除最後一個點並重算;Esc 清空全部

### B. 粗框

7. 按住左鍵拖一個大概的框放開:藍色虛線框為 prompt,mask 貼合框內物體;可再加正/負點修正

### C. 接受與輸出型別

8. Enter(或按「Accept」):形狀以當前類別加入(polygon),工具維持在 SAM,狀態清空,
   可立刻點下一個物體;Auto Save 開著會在 0.5 s 後儲存,Ctrl+Z 可還原
9. Output 切成 Box:預覽變成 mask 的外接矩形,Enter 加入的是 rectangle
10. 右上 model 下拉可切 SAM 2.1 Tiny / Small / SAM ViT-B;切換後重新 prefetch,選擇會記住

### D. 邊界

11. 只放負點:狀態列黃字「no mask for this prompt — add a point inside the object」,Accept 停用
12. 切到別張圖再切回:prompt 清空;若工具仍是 SAM,新圖自動 prefetch
13. macOS 上能力清單 `assist_interactive` 為 limited(較慢),功能不被封鎖

## 預期結果

一張圖的第一次點擊付一次 encoder 成本,之後每次點擊都是即時回饋;所有寫入都經由既有的儲存流程。

## 已知限制

- 一次一張圖、一個物體;沒有跨影格追蹤(SAM 2 的 video 功能未使用)
- 候選 polygon 由 mask 邊界簡化而來,細碎孔洞不保留
- 觸控裝置沒有右鍵與 Shift,負點需以滑鼠操作
