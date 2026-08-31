# E2 — 標註介面（E2-T1 畫布、E2-T2 bbox、E2-T3 polygon、E2-T5 快捷鍵）

互動模式參照 exVision 的標記頁面：縮圖網格 → 編輯器兩階段、工具鍵 1/2/3、
自由文字類別欄＋chips、Auto Save 開關、右鍵平移。

## 怎麼啟動

```bash
horos init ./demo_project
horos import <資料集目錄或 zip 解壓後路徑> --project ./demo_project
horos ui --project ./demo_project
# 瀏覽器開 http://localhost:5000 → 右上「Open annotator →」，或直接開 /annotate
```

## 快捷鍵對照表（頁面內按右上 ⓘ 資訊圖示亦可查看）

| 鍵 | 動作 |
|---|---|
| `1` / `2` / `3` | Select ✎ / Rectangle ▭ / Polygon ⬠ 工具 |
| `A` / `D` | 上一張 / 下一張 |
| `Ctrl+S` | 儲存標註 |
| `Ctrl+Z` | 復原 |
| `Ctrl+C` / `Ctrl+V` | 複製 / 貼上選取的形狀（貼上偏移 20px） |
| `Delete` / `Backspace` | 刪除選取的形狀 |
| `Esc` | 取消繪製 / 取消選取 |
| 雙擊 | 閉合 polygon（或點回起點） |
| 滾輪 | 以游標為中心縮放（1×–16×） |
| 右鍵拖曳 | 平移放大後的影像 |

快捷鍵以 `e.code`（實體鍵）比對且忽略 IME 組字中的按鍵——中文輸入法下不失效。

## 測試步驟

### A. 網格（入口）
1. 開 /annotate：縮圖網格顯示、頂欄有進度（N/M images annotated）
2. 佇列模式切「Only unannotated」、split 切「valid」：網格即時過濾
3. 每頁數量切 20：分頁按鈕出現；縮圖角標 ✓ 已標、🔒 他人標註中
4. 點任一縮圖進入編輯器，影像自動縮放至符合畫面（小圖也放大適配）；「← Back to Grid」返回且網格刷新

### B. 編輯器 — 矩形（E2-T2）
1. 按 `2`（或點 ▭），拖曳畫框：畫完自動選取、右上顯示綠色 saved（Auto Save 預設開）
2. 按 `1` 回 Select：hover 形狀會加亮、游標變 move；hover 任一控制點（不需先選取）該點放大變白、游標變 pointer，可直接拖曳
3. 拖曳角點改大小、拖曳內部移動（拖不出影像邊界）；放開後自動儲存
4. `Ctrl+C` → `Ctrl+V`：複製出偏移 20px 的新框

### C. 編輯器 — 多邊形（E2-T3）
1. 按 `3`，逐點點擊；游標移近起點時起點放大變白，點下即閉合；或雙擊閉合
2. Select 模式下 hover 任一頂點即可直接拖曳改形狀（不需先點選該形狀）
3. 繪製中（矩形拖曳、多邊形逐點）控制點即時顯示，不是完成後才出現

### D. 類別（自由輸入 + chips）
1. 在 Object Class 輸入框輸入新名稱（如 `crane`）再畫框：儲存後自動建立新類別，
   chips 出現 crane
2. 選取一個形狀：輸入框顯示其類別；改字時畫布標籤即時變，blur/Enter 才寫入
3. 點 chips 直接改選取形狀的類別
4. 「Manage Classes」：改色（色塊）、Rename、Del（有標註引用會先確認再連同刪除）

### E. 儲存模式（E2-T6）
1. Auto Save 開（預設）：每個動作 500ms 去抖自動存，手動 Save 鈕反灰
2. 關 Auto Save：動作後右上顯示黃色 unsaved；按 `Ctrl+S` 或 Save 鈕儲存；
   未存就按 Prev/Next/Back 會跳確認框；直接關頁籤觸發瀏覽器離開警告
3. 換張再回來、重新整理頁面：標註完整保留

### F. 多人（E2-S6，開兩個瀏覽器視窗模擬）
1. 視窗 1 開啟某張後，視窗 2 網格中該張顯示 🔒，開啟時頂欄顯示
   「⚠ someone else is annotating this image」
2. 兩邊同時改同一張：後儲存者收到紅色 toast 並自動載入對方版本

### G. 空類別
1. 專案裡沒有任何標註的舊類別不出現在 Object Class chips
2. 本次 session 新建或使用過的類別立即出現在 chips（即使還沒存檔）
3. Manage Classes 彈窗仍列出全部類別（含空的），供改名/刪除清理

## 預期結果

從網格挑圖 → 全鍵盤畫框改框換類別換頁 → 返回網格，進度與角標即時反映；
不會因輸入法或未儲存離開而遺失標註。

## 已知限制

- polygon 不支援插入新頂點（僅拖曳既有頂點）；無 redo（僅 undo 50 步）
- 縮圖直接載原圖（lazy loading），數千張以上首屏較慢，縮圖端點留待後續
- 矩形以兩角點編輯（exVision 模式），無邊中點控制
