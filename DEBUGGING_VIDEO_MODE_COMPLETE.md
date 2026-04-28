# Video Hands Detection - 完整調試指南

## 🔧 已修復的問題

### 1. **switchMode 函數的關鍵 Bug**
**問題**: 在切換到 Image Mode 時，視頻容器沒有被隱藏
```javascript
// ❌ 錯誤 (舊代碼):
} else {
    videoModeContainer.style.display = 'block';  // 應該是 'none'!
}

// ✅ 正確 (新代碼):
} else {
    videoModeContainer.style.display = 'none';
}
```

### 2. **RemoteCheckForm 的空引用**
**問題**: HTML 中沒有這個表單元素，導致 JavaScript 錯誤
**解決**: 添加條件檢查 `if (remoteCheckForm) { ... }`

### 3. **視頻幀捕獲的防守檢查**
**問題**: 視頻可能還未加載時嘗試捕獲幀，導致 0x0 尺寸
**解決**: 檢查 `videoWidth > 0 && videoHeight > 0` 再繼續

---

## ✅ 完整測試步驟

### 步驟 1: 打開 DevTools 並清空 Console

```
按 F12 → Console 標籤 → 清空控制台
```

### 步驟 2: 刷新頁面

看 Console 中應該出現：
```
UI Elements loaded: {
    videoCanvas: true,
    videoModeContainer: true,
    imageModeBtn: true,
    videoModeBtn: true,
    imageModeCtrls: 2
}
```

**如果有 false，表示元素未找到 ⚠️**

### 步驟 3: 點擊 "🎥 Video Hands Detection" 按鈕

Console 應該顯示：
```
Video Mode button clicked
Switched to Video Mode
Status: Camera: Ready | Detection: Idle
```

**如果沒有這些日誌：**
- 確認按鈕能點擊
- 檢查按鈕是否有正確的 id 和事件監聽器
- 打開 DevTools → Elements 標籤，搜索 `videoModeBtn`

### 步驟 4: 檢查視頻容器是否顯示

用 DevTools Elements 檢查：
```html
<div class="video-mode-container" id="videoModeContainer">
```

應該看到 `style="display: block;"` (不能是 none)

### 步驟 5: 點擊 "📷 Start Camera"

期望結果：
- 瀏覽器彈出攝像頭權限提示
- Console 顯示 `Status: Camera: Running | Detection: Ready to start`
- 視頻預覽出現（黑色背景，如果無攝像頭）

**常見問題：**
- ❌ 沒有權限提示 → 檢查瀏覽器是否支持 getUserMedia
- ❌ 沒有日誌 → 按鈕事件監聽器未正確設置
- ❌ 視頻黑色 → 正常，等待一秒鐘

### 步驟 6: 點擊 "🔍 Start Detection"

期望結果：
- Console 顯示 `Status: Camera: Running | Detection: Active | Last: XXs ago`
- 每 300ms 發送一次檢測請求（可在 Network 標籤看到）

**常見問題：**
- ❌ 按鈕仍未啟用 → 確認攝像頭已成功啟動
- ❌ 沒有請求發送 → 檢查後端是否在運行

---

## 🐛 故障排查

### 問題 1: 視頻模式按鈕按不了

**檢查清單：**
```javascript
// 在 Console 中運行：
console.log({
    imageModeBtn: document.getElementById('imageModeBtn'),
    videoModeBtn: document.getElementById('videoModeBtn'),
    isVideoMode: isVideoMode
});
```

**預期輸出：**
- `imageModeBtn`: HTMLButtonElement (不是 null)
- `videoModeBtn`: HTMLButtonElement (不是 null)
- `isVideoMode`: false (初始值)

### 問題 2: 切換到視頻模式後容器仍看不到

**在 Console 中檢查：**
```javascript
console.log({
    containerDisplay: document.getElementById('videoModeContainer').style.display,
    imageControlsDisplay: Array.from(document.querySelectorAll('.image-mode-controls')).map(el => el.style.display)
});
```

**預期輸出 (在視頻模式中):**
- `containerDisplay`: 'block'
- `imageControlsDisplay`: ['none', 'none']

### 問題 3: Start Camera 按鈕按了沒反應

**執行以下診斷：**
```javascript
// 在 Console 中運行
console.log('Testing camera access:');
navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: 'environment' } }
}).then(stream => {
    console.log('✓ Camera access granted!');
    console.log('Track count:', stream.getTracks().length);
    stream.getTracks().forEach(track => track.stop());
}).catch(err => {
    console.error('✗ Camera access failed:', err.message);
});
```

### 問題 4: Start Detection 按鈕無效

**檢查：**
```javascript
console.log({
    cameraRunning: isCameraRunning,
    detectionRunning: !!detectionIntervalId,
    videoReadiness: {
        videoWidth: videoPreview.videoWidth,
        videoHeight: videoPreview.videoHeight,
        srcObject: !!videoPreview.srcObject
    }
});
```

**預期值 (按下 Start Detection 前):**
- `cameraRunning`: true
- `detectionRunning`: false
- `videoWidth`: > 0 (例如 640)
- `videoHeight`: > 0 (例如 480)
- `srcObject`: true

---

## 🔧 快速修復命令

如果遇到問題，在 Console 中運行這些命令：

### 重置視頻模式
```javascript
// 回到圖像模式並重置狀態
isVideoMode = false;
videoModeContainer.style.display = 'none';
document.querySelectorAll('.image-mode-controls').forEach(el => el.style.display = 'block');
cleanupVideoMode();
console.log('Video mode reset');
```

### 強制刷新視頻模式
```javascript
// 清空所有視頻狀態並重新初始化
cleanupVideoMode();
isVideoMode = true;
videoModeContainer.style.display = 'block';
document.querySelectorAll('.image-mode-controls').forEach(el => el.style.display = 'none');
console.log('Video mode reinitialized');
```

---

## 📋 硬件/瀏覽器兼容性

### 必需功能
- ✅ `navigator.mediaDevices.getUserMedia()` - 現代瀏覽器都支持
- ✅ Canvas 2D Context - 標準功能
- ✅ Fetch API - 現代瀏覽器都支持

### 已知問題
- 🔴 **舊 iOS Safari** (< 14.5): getUserMedia 可能不支持
- 🟡 **Firefox**: 某些設置下需要 HTTPS 才能訪問攝像頭
- 🟢 **Chrome/Edge**: 完全支持，HTTPS 推薦但不必須（localhost 除外）

### 測試瀏覽器
推薦在以下瀏覽器測試：
- Chrome 90+
- Firefox 89+
- Edge 90+
- Safari 14.5+

---

## 🚀 如果還是不行

1. **硬刷新瀏覽器**: `Ctrl+Shift+R` (Windows) 或 `Cmd+Shift+R` (Mac)
2. **清除瀏覽器緩存**: DevTools → Application → Storage → Clear Site Data
3. **用隱私窗口測試**: 避免擴展插件干擾
4. **檢查 HTML 文件是否正確保存**: 在編輯器中確認所有更改都已保存
5. **重啟後端服務器**: 確保 Flask 服務器在運行

---

## 📞 最後一步

如果以上都檢查過還是不行，請提供以下信息：

1. 瀏覽器版本：`console.log(navigator.userAgent)`
2. Console 中的完整錯誤信息（紅色警告）
3. 按下視頻按鈕後 Console 的完整輸出
4. Network 標籤中是否有 `/hand_detection` 的請求

