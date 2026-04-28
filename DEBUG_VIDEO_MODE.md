# 調試 Video Hands Detection 模式

## 已修正的問題

1. **switchMode 函數改為使用直接 style 改變**
   - 從 CSS 類切換改為直接 `style.display` 操作
   - 更可靠，避免 CSS 特異性問題

2. **添加了安全檢查**
   - Canvas context 初始化時檢查 null
   - 所有 DOM 操作前檢查元素是否存在
   - 防止未定義的元素導致 JavaScript 錯誤

3. **添加了調試日誌**
   - 控制台會輸出所有重要操作日誌
   - 方便追蹤問題

## 測試步驟

### 1. 打開瀏覽器開發者工具
- 按 `F12` 打開開發者工具
- 點擊 "Console" 標籤查看日誌

### 2. 刷新頁面
- 查看 Console 中是否有 "UI Elements loaded" 的信息
- 應該顯示所有元素都已找到 (true)

### 3. 點擊 "🎥 Video Hands Detection" 按鈕
期望看到的結果:
```
Video Mode button clicked
Switched to Video Mode
Status: Camera: Ready | Detection: Idle
```

如果沒有看到這些日誌，表示：
- 按鈕點擊事件沒有觸發
- 需要檢查 HTML 是否正確加載

### 4. 點擊 "📸 Image Mode" 按鈕
期望看到的結果:
```
Image Mode button clicked
Switched to Image Mode
```

## 預期行為

### Image Mode 時
- 看到上傳/拍照按鈕
- 看到分析模式和任務類型選擇框
- **看不到**視頻相關控制按鈕

### Video Mode 時
- **看不到**上傳/拍照按鈕
- **看不到**分析模式選擇
- 看到視頻相關的 4 個按鈕：
  - 📷 Start Camera (可點擊)
  - ⏹️ Stop Camera (灰色，禁用)
  - 🔍 Start Detection (灰色，禁用)
  - ⏹️ Stop Detection (灰色，禁用)

## 常見問題

### 問題1: 點擊後沒有反應
**解決方案:**
- 打開 Console，查看是否有錯誤信息
- 檢查 "UI Elements loaded" 日誌中是否所有元素都是 true
- 如果有 false，表示元素未找到，HTML 結構可能有問題

### 問題2: 按鈕按下但頁面沒有變化
**解決方案:**
- 檢查 Console 中的日誌，確認 switchMode 是否被呼叫
- 確認 CSS 沒有覆蓋 style.display 的改變

### 問題3: 視頻按鈕按不了
**解決方案:**
- 確保已切換到 Video Mode
- 檢查 HTML 中按鈕是否都有正確的 id
- 檢查 JavaScript 是否正確初始化了 button 變量

## 完整的元素檢查清單

在 Console 中運行以下命令檢查所有元素：

```javascript
console.log({
    imageModeBtn: document.getElementById('imageModeBtn'),
    videoModeBtn: document.getElementById('videoModeBtn'),
    videoModeContainer: document.getElementById('videoModeContainer'),
    imageModeCtrls: document.querySelectorAll('.image-mode-controls'),
    startCameraBtn: document.getElementById('startCameraBtn'),
    videoPreview: document.getElementById('videoPreview'),
    videoCanvas: document.getElementById('videoCanvas')
});
```

所有元素應該都不是 null 或 undefined。

## HTML 結構驗證

HTML 應該包含：

```html
<!-- Mode Toggle Buttons -->
<button id="imageModeBtn">📸 Image Mode</button>
<button id="videoModeBtn">🎥 Video Hands Detection</button>

<!-- Image Mode Controls -->
<div class="image-mode-controls"><!-- Image upload/camera buttons --></div>
<div class="image-mode-controls"><!-- Analysis options --></div>

<!-- Video Mode Container -->
<div id="videoModeContainer" class="video-mode-container">
    <video id="videoPreview"></video>
    <canvas id="videoCanvas"></canvas>
    <button id="startCameraBtn">Start Camera</button>
    <button id="stopCameraBtn">Stop Camera</button>
    <button id="startDetectionBtn">Start Detection</button>
    <button id="stopDetectionBtn">Stop Detection</button>
</div>
```

## CSS 驗證

打開 DevTools 的 "Inspector" 標籤，選擇各個元素檢查：

1. **Mode toggle buttons** 應該能看到 `.mode-toggle-btn` 和 `.active` 類
2. **videoModeContainer** 應該有 `display: none`（切換前）或 `display: block`（切換後）
3. **imageModeCtrls** 應該有 `display: block`（Image Mode）或 `display: none`（Video Mode）

## 最後步驟

如果以上都沒問題，但功能仍然不工作：

1. 檢查 HTML 文件是否被正確保存
2. 硬刷新瀏覽器 (Ctrl+Shift+R 或 Cmd+Shift+R)
3. 清除瀏覽器緩存
4. 在隱私/無痕窗口中打開頁面重新測試
