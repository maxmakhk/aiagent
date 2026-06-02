const tabs = document.querySelectorAll('.tab');
const panels = {
    user: document.getElementById('tab-user'),
    editor: document.getElementById('tab-editor')
};

const userImageCamera = document.getElementById('userImageCamera');
const userImageAlbum = document.getElementById('userImageAlbum');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultJson = document.getElementById('resultJson');
const summary = document.getElementById('summary');
const readableSummary = document.getElementById('readableSummary');
const refObjectCheck = document.getElementById('refObjectCheck');
const qrcodeCheck = document.getElementById('qrcodeCheck');
const barcodeCheck = document.getElementById('barcodeCheck');
const localCodeHint = document.getElementById('localCodeHint');

const newRefLabel = document.getElementById('newRefLabel');
const createRefBtn = document.getElementById('createRefBtn');
const refLabelSelect = document.getElementById('refLabelSelect');
const refFiles = document.getElementById('refFiles');
const uploadRefBtn = document.getElementById('uploadRefBtn');
const refList = document.getElementById('refList');

const dbType = document.getElementById('dbType');
const dbCode = document.getElementById('dbCode');
const dbImageCamera = document.getElementById('dbImageCamera');
const dbImageAlbum = document.getElementById('dbImageAlbum');
const dbName = document.getElementById('dbName');
const dbCodeNumber = document.getElementById('dbCodeNumber');
const dbSimple = document.getElementById('dbSimple');
const dbDesc = document.getElementById('dbDesc');
const saveDbBtn = document.getElementById('saveDbBtn');
const reloadDbBtn = document.getElementById('reloadDbBtn');
const deleteDbBtn = document.getElementById('deleteDbBtn');
const dbJson = document.getElementById('dbJson');

const dbSearch = document.getElementById('dbSearch');
const dbRecordsList = document.getElementById('dbRecordsList');
const dbRecordCount = document.getElementById('dbRecordCount');
const clearDbFormBtn = document.getElementById('clearDbFormBtn');

let currentDbRecords = [];

const CODE_FORMAT_MAP = {
    qrcode: ['qr_code'],
    barcode: ['ean_13', 'ean_8', 'code_128', 'code_39', 'code_93', 'upc_a', 'upc_e', 'itf', 'codabar']
};

tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
        tabs.forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');
        const key = tab.dataset.tab;
        Object.keys(panels).forEach((k) => {
            panels[k].classList.toggle('active', k === key);
        });
    });
});

function setLocalCodeHint(message, isError = false) {
    if (!localCodeHint) return;
    localCodeHint.textContent = message || '';
    localCodeHint.style.color = isError ? 'var(--danger)' : 'var(--muted)';
}

function ensureFileInstance(input, fallbackName = 'capture.jpg', fallbackType = 'image/jpeg') {
    if (!input) return input;
    if (input instanceof File) return input;

    const type = input.type || fallbackType;
    return new File([input], fallbackName, {
        type,
        lastModified: Date.now()
    });
}

const previewImage = (file) => {
    if (!file) {
        preview.style.display = 'none';
        return;
    }
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
};

function getSelectedUserFile() {
    return (userImageCamera.files && userImageCamera.files[0]) ||
        (userImageAlbum.files && userImageAlbum.files[0]) ||
        null;
}

function getSelectedDbFile() {
    return (dbImageCamera.files && dbImageCamera.files[0]) ||
        (dbImageAlbum.files && dbImageAlbum.files[0]) ||
        null;
}

if (userImageCamera) {
    userImageCamera.addEventListener('change', () => {
        const file = userImageCamera.files && userImageCamera.files[0];
        if (file) {
            userImageAlbum.value = '';
            previewImage(file);
        }
    });
}

if (userImageAlbum) {
    userImageAlbum.addEventListener('change', () => {
        const file = userImageAlbum.files && userImageAlbum.files[0];
        if (file) {
            userImageCamera.value = '';
            previewImage(file);
        }
    });
}

if (dbImageCamera && dbImageAlbum) {
    dbImageCamera.addEventListener('change', () => {
        if (dbImageCamera.files && dbImageCamera.files[0]) {
            dbImageAlbum.value = '';
        }
    });
    dbImageAlbum.addEventListener('change', () => {
        if (dbImageAlbum.files && dbImageAlbum.files[0]) {
            dbImageCamera.value = '';
        }
    });
}

/**
 * Downscale an image file if its long edge exceeds maxLongEdge, always returning a File.
 * Runs in the browser using HTML5 Canvas.
 */
function downscaleImage(file, maxLongEdge = 1000) {
    return new Promise((resolve) => {
        if (!file || !file.type.startsWith('image/')) {
            resolve(file);
            return;
        }

        const reader = new FileReader();
        reader.onload = function (event) {
            const img = new Image();
            img.onload = function () {
                const currentLongEdge = Math.max(img.width, img.height);
                if (currentLongEdge <= maxLongEdge) {
                    resolve(file);
                    return;
                }

                const canvas = document.createElement('canvas');
                const scale = maxLongEdge / currentLongEdge;
                canvas.width = Math.round(img.width * scale);
                canvas.height = Math.round(img.height * scale);

                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                canvas.toBlob((blob) => {
                    if (blob) {
                        resolve(ensureFileInstance(blob, file.name || 'capture.jpg', file.type || 'image/jpeg'));
                    } else {
                        resolve(file);
                    }
                }, file.type || 'image/jpeg', 0.85);
            };
            img.onerror = function () {
                resolve(file);
            };
            img.src = event.target.result;
        };
        reader.onerror = function () {
            resolve(file);
        };
        reader.readAsDataURL(file);
    });
}

async function detectWithBarcodeDetector(file, requestedKind = 'all') {
    if (!('BarcodeDetector' in window)) return null;

    try {
        let formats = [];
        const supported = await BarcodeDetector.getSupportedFormats();

        if (requestedKind === 'qrcode') {
            formats = CODE_FORMAT_MAP.qrcode.filter((f) => supported.includes(f));
        } else if (requestedKind === 'barcode') {
            formats = CODE_FORMAT_MAP.barcode.filter((f) => supported.includes(f));
        } else {
            formats = [...CODE_FORMAT_MAP.qrcode, ...CODE_FORMAT_MAP.barcode]
                .filter((f) => supported.includes(f));
        }

        if (!formats.length) return null;

        const detector = new BarcodeDetector({ formats });
        const img = new Image();
        const objectUrl = URL.createObjectURL(file);

        try {
            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
                img.src = objectUrl;
            });

            const results = await detector.detect(img);
            if (!results || !results.length) return null;

            const first = results[0];
            return {
                text: first.rawValue || '',
                format: first.format || 'unknown',
                kind: first.format === 'qr_code' ? 'qrcode' : 'barcode',
                engine: 'BarcodeDetector',
                raw: results
            };
        } finally {
            URL.revokeObjectURL(objectUrl);
        }
    } catch (err) {
        console.warn('BarcodeDetector failed:', err);
        return null;
    }
}

async function detectWithHtml5Qrcode(file, requestedKind = 'all') {
    console.log("requestedKind", requestedKind);
    if (!window.Html5Qrcode) return null;

    try {
        const decodeFile = ensureFileInstance(file, file && file.name ? file.name : 'capture.jpg', file && file.type ? file.type : 'image/jpeg');
        if (!(decodeFile instanceof File)) return null;

        const tempId = 'reader-temp';
        const tempElement = document.getElementById(tempId);
        if (!tempElement) {
            console.warn('Missing #reader-temp element for html5-qrcode fallback');
            return null;
        }

        const scanner = new Html5Qrcode(tempId);
        
        // 1. 定義要支援的格式（包含你需要的單行條碼）
        let formatsToSupport = [];
        
        if (requestedKind === 'qrcode') {
            formatsToSupport = [Html5QrcodeSupportedFormats.QR_CODE];
        } else if (requestedKind === 'barcode') {
            // 這裡全是單行一維條碼
            formatsToSupport = [
                Html5QrcodeSupportedFormats.EAN_13,
                Html5QrcodeSupportedFormats.EAN_8,
                Html5QrcodeSupportedFormats.CODE_128,
                Html5QrcodeSupportedFormats.CODE_39,
                Html5QrcodeSupportedFormats.CODE_93,
                Html5QrcodeSupportedFormats.UPC_A,
                Html5QrcodeSupportedFormats.UPC_E,
                Html5QrcodeSupportedFormats.ITF
            ];
        } else {
            // all: 條碼與 QR Code 統統打開
            formatsToSupport = [
                Html5QrcodeSupportedFormats.QR_CODE,
                Html5QrcodeSupportedFormats.EAN_13,
                Html5QrcodeSupportedFormats.EAN_8,
                Html5QrcodeSupportedFormats.CODE_128,
                Html5QrcodeSupportedFormats.CODE_39,
                Html5QrcodeSupportedFormats.CODE_93,
                Html5QrcodeSupportedFormats.UPC_A,
                Html5QrcodeSupportedFormats.UPC_E,
                Html5QrcodeSupportedFormats.ITF
            ];
        }

        // 2. 使用 scanFileV2 並帶入格式設定，同時開啟瀏覽器加速優化
        const scanResult = await scanner.scanFileV2(decodeFile, {
            formatsToSupport: formatsToSupport,
            experimentalFeatures: {
                useBarCodeDetectorIfSupported: true
            }
        });

        // scanFileV2 回傳的是結果物件，從中提取文字
        const resultText = scanResult.decodedText || '';

        let inferredKind = 'unknown';
        if (requestedKind === 'qrcode') inferredKind = 'qrcode';
        if (requestedKind === 'barcode') inferredKind = 'barcode';
        
        // 自動識別判斷：如果套件有回傳格式名稱，只要不是 QR_CODE，通通歸類為單行條碼 (barcode)
        if (scanResult.result && scanResult.result.format) {
            const fmtName = scanResult.result.format.formatName;
            inferredKind = fmtName === 'QR_CODE' ? 'qrcode' : 'barcode';
        }

        return {
            text: resultText,
            format: scanResult.result?.format?.formatName || 'unknown',
            kind: inferredKind,
            engine: 'html5-qrcode'
        };
    } catch (err) {
        const msg = String((err && err.message) || err || '');
        if (msg.includes('No MultiFormat Readers were able to detect the code') || msg.includes('NotFoundException')) {
            console.debug('html5-qrcode: no code detected in image');
        } else {
            console.warn('html5-qrcode scanFile failed:', err);
        }
        return null;
    }
}

async function decodeCodeLocally(file, requestedKind = 'all') {
    if (!file) return null;

    const nativeResult = await detectWithBarcodeDetector(file, requestedKind);
    if (nativeResult && nativeResult.text) return nativeResult;

    const fallbackResult = await detectWithHtml5Qrcode(file, requestedKind);
    if (fallbackResult && fallbackResult.text) return fallbackResult;

    return null;
}

function getRequestedKindFromChecks() {
    const qrOn = qrcodeCheck && qrcodeCheck.value === '1';
    const barOn = barcodeCheck && barcodeCheck.value === '1';

    if (qrOn && !barOn) return 'qrcode';
    if (!qrOn && barOn) return 'barcode';
    return 'all';
}

analyzeBtn.addEventListener('click', async () => {
    let file = getSelectedUserFile();
    if (!file) {
        alert('Please capture/select an image first.');
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Preparing image...';
    setLocalCodeHint('');

    if (readableSummary) {
        readableSummary.innerHTML = `
            <div class="readable-title">Analyzing...</div>
            <div class="readable-body" style="text-align:center; padding: 25px 10px;">
                <div class="spinner" style="margin: 0 auto 12px auto;"></div>
                <div style="color:var(--muted); font-size:13px; font-weight:500;">
                    Uploading and running AI multi-engine analysis. Please wait...
                </div>
            </div>
        `;
    }

    const originalName = file.name || 'capture.jpg';

    try {
        file = await downscaleImage(file, 1000);
    } catch (e) {
        console.warn('Downscaling failed, using original:', e);
    }

    let localDecoded = null;
    try {
        const shouldTryLocalCode =
            (qrcodeCheck && qrcodeCheck.value === '1') ||
            (barcodeCheck && barcodeCheck.value === '1');

        if (shouldTryLocalCode) {
            analyzeBtn.textContent = 'Decoding code locally...';
            const requestedKind = getRequestedKindFromChecks();
            localDecoded = await decodeCodeLocally(file, requestedKind);

            if (localDecoded) {
                setLocalCodeHint(`Local decode: ${localDecoded.text} (${localDecoded.engine})`);
            } else {
                setLocalCodeHint('Local decode not found, server will continue image analysis.');
            }
        }
    } catch (e) {
        console.warn('Local decode failed:', e);
        setLocalCodeHint('Local decode failed, server will continue image analysis.', true);
    }

    const fd = new FormData();
    fd.append('image', file, originalName);
    fd.append('keywords', document.getElementById('keywords').value || '');
    fd.append('threshold', document.getElementById('threshold').value || '0.75');
    fd.append('target_label', document.getElementById('targetLabel').value || '');
    fd.append('model_keyword_check', document.getElementById('modelKeywordCheck').value || '1');
    fd.append('ocr_check', document.getElementById('ocrCheck').value || '1');
    fd.append('ref_object_check', refObjectCheck ? refObjectCheck.value : '1');
    fd.append('qrcode_check', qrcodeCheck ? qrcodeCheck.value : '1');
    fd.append('barcode_check', barcodeCheck ? barcodeCheck.value : '1');

    if (localDecoded) {
        fd.append('local_decoded_text', localDecoded.text || '');
        fd.append('local_decoded_format', localDecoded.format || '');
        fd.append('local_decoded_kind', localDecoded.kind || '');
        fd.append('local_decoded_engine', localDecoded.engine || '');
    }

    analyzeBtn.textContent = 'Analyzing...';

    try {
        const res = await fetch('/vision2/api/analyze', {
            method: 'POST',
            body: fd
        });
        const data = await res.json();

        if (!res.ok || !data.success) {
            throw new Error(data.error || 'Analyze failed');
        }

        const ref = data.user_view.reference_detection || {};
        const hand = data.user_view.hand_tracking || {};
        const code = data.user_view.code_detection || {};
        const keywordInfo = data.user_view.keyword_tags || {};
        const matched = keywordInfo.matched || [];
        const detectorOnly = keywordInfo.detector_only_matched || [];
        const modelOnly = keywordInfo.model_matched || [];
        const modelContext = data.user_view.model_understanding || {};
        const ocrEnabled = !!data.input?.ocr_check;
        const ocrText = data.user_view?.ocr_check?.text || '';
        const namedCodes = (code.all_codes || []).filter((c) => c.db_record && c.db_record.name);
        const perf = data.performance || {};

        summary.innerHTML = [
            `<span>Ref Objects: ${Object.keys(ref.counts || {}).length}</span>`,
            `<span>Hands: ${hand.hands_detected || 0}</span>`,
            `<span>Codes: ${code.counts?.total || 0}</span>`,
            `<span>Code Names: ${namedCodes.length}</span>`,
            `<span>Obj ms: ${perf.reference_detection_ms ?? '-'}</span>`,
            `<span>QR ms: ${perf.qrcode_detection_ms ?? '-'}</span>`,
            `<span>Barcode ms: ${perf.barcode_detection_ms ?? '-'}</span>`,
            `<span>Keyword Matched: ${matched.length}</span>`,
            `<span>Detector Match: ${detectorOnly.length}</span>`,
            `<span>Model Match: ${modelOnly.length}</span>`,
            `<span>OCR: ${ocrEnabled ? (ocrText ? 'Text Found' : 'No Text') : 'Disabled'}</span>`,
            `<span>Model Context: ${modelContext.available === false ? 'Unavailable' : 'Ready'}</span>`
        ].join('');

        renderReadableSummary(data);
        resultJson.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
        renderReadableError(err.message);
        resultJson.textContent = `Error: ${err.message}`;
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze Unified Result';
    }
});

function formatCountItems(counts) {
    const keys = Object.keys(counts || {});
    if (keys.length === 0) return 'None';
    return keys.map((k) => `${k}: ${counts[k]}`).join(', ');
}

function renderReadableError(message) {
    if (!readableSummary) return;
    readableSummary.innerHTML = `
        <div class="readable-title">Readable Result</div>
        <div class="readable-body readable-error">Analyze failed: ${message}</div>
    `;
}

function renderReadableSummary(data) {
    if (!readableSummary) return;

    const ref = data.user_view?.reference_detection || {};
    const hand = data.user_view?.hand_tracking || {};
    const code = data.user_view?.code_detection || {};
    const keywordInfo = data.user_view?.keyword_tags || {};
    const modelInfo = data.user_view?.model_understanding || {};
    const ocrText = data.user_view?.ocr_check?.text || '';
    const perf = data.performance || {};

    const handCount = hand.hands_detected || 0;
    const graspedCount = Array.isArray(hand.grasped_objects) ? hand.grasped_objects.length : 0;
    const codeTotal = code.counts?.total || 0;
    const matchedKeywords = keywordInfo.matched || [];
    const matchedLabel = matchedKeywords.length > 0 ? matchedKeywords.join(', ') : 'None';
    const refLine = formatCountItems(ref.counts || {});
    const codeList = (code.all_codes || []).map((c) => `${c.kind}:${c.code}`).join(', ') || 'None';
    const namedCodeList = (code.all_codes || [])
        .filter((c) => c.db_record && c.db_record.name)
        .map((c) => `${c.db_record.name} (${c.kind}:${c.code})`)
        .join(', ') || 'None';
    const modelStatus = modelInfo.available === false ? 'Not available' : 'Available';
    const captionPreview = modelInfo.caption ? String(modelInfo.caption).slice(0, 220) : 'N/A';
    const ocrPreview = ocrText ? String(ocrText).slice(0, 180) : 'N/A';

    readableSummary.innerHTML = `
        <div class="readable-title">Readable Result</div>
        <div class="readable-grid">
            <div class="readable-item"><span class="k">Detected Objects</span><span class="v">${refLine}</span></div>
            <div class="readable-item"><span class="k">Hands / Grasp</span><span class="v">${handCount} hand(s), ${graspedCount} grasp candidate(s)</span></div>
            <div class="readable-item"><span class="k">Codes</span><span class="v">${codeTotal} found (${codeList})</span></div>
            <div class="readable-item"><span class="k">Code Names</span><span class="v">${namedCodeList}</span></div>
            <div class="readable-item"><span class="k">Object Speed</span><span class="v">${perf.reference_detection_ms ?? '-'} ms</span></div>
            <div class="readable-item"><span class="k">QRCode Speed</span><span class="v">${perf.qrcode_detection_ms ?? '-'} ms</span></div>
            <div class="readable-item"><span class="k">Barcode Speed</span><span class="v">${perf.barcode_detection_ms ?? '-'} ms</span></div>
            <div class="readable-item"><span class="k">Matched Keywords</span><span class="v">${matchedLabel}</span></div>
            <div class="readable-item"><span class="k">Model Understanding</span><span class="v">${modelStatus}</span></div>
            <div class="readable-item"><span class="k">OCR Check</span><span class="v">${data.input?.ocr_check ? 'Enabled' : 'Disabled'}</span></div>
        </div>
        <div class="readable-note"><strong>Caption:</strong> ${captionPreview}</div>
        <div class="readable-note"><strong>OCR:</strong> ${ocrPreview}</div>
    `;
}

async function refreshReferences() {
    try {
        const res = await fetch('/api/reference/list');
        const refs = await res.json();
        const options = ['<option value="">Select label</option>'];
        refs.forEach((r) => {
            options.push(`<option value="${r.label}">${r.label}</option>`);
        });
        refLabelSelect.innerHTML = options.join('');

        if (refs.length === 0) {
            refList.textContent = 'No references yet.';
            return;
        }

        refList.innerHTML = refs
            .map((r) => `<div><strong>${r.label}</strong> (${r.photo_count} photos)</div>`)
            .join('');
    } catch (err) {
        refList.textContent = `Load reference failed: ${err.message}`;
    }
}

createRefBtn.addEventListener('click', async () => {
    const label = newRefLabel.value.trim();
    if (!label) {
        alert('Please input label');
        return;
    }

    try {
        const res = await fetch('/api/reference/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label })
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.error || 'Create label failed');
        }
        newRefLabel.value = '';
        await refreshReferences();
    } catch (err) {
        alert(err.message);
    }
});

uploadRefBtn.addEventListener('click', async () => {
    const label = refLabelSelect.value;
    const files = refFiles.files;
    if (!label) {
        alert('Please select label');
        return;
    }
    if (!files || files.length === 0) {
        alert('Please choose image files');
        return;
    }

    const fd = new FormData();
    fd.append('label', label);
    Array.from(files).forEach((f) => fd.append('files', f));

    uploadRefBtn.disabled = true;
    uploadRefBtn.textContent = 'Uploading...';
    try {
        const res = await fetch('/api/reference/upload', {
            method: 'POST',
            body: fd
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        refFiles.value = '';
        await refreshReferences();
    } catch (err) {
        alert(err.message);
    } finally {
        uploadRefBtn.disabled = false;
        uploadRefBtn.textContent = 'Upload Reference Images';
    }
});

function collectDbPayload() {
    return {
        code: dbCode.value.trim(),
        name: dbName.value.trim(),
        code_number: dbCodeNumber.value.trim(),
        simple_description: dbSimple.value.trim(),
        description: dbDesc.value.trim()
    };
}

function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function renderDbRecordsList() {
    const query = dbSearch ? dbSearch.value.trim().toLowerCase() : '';
    const filtered = currentDbRecords.filter((r) => {
        const code = String(r.code || '').toLowerCase();
        const name = String(r.name || '').toLowerCase();
        const desc = String(r.description || '').toLowerCase();
        const simple = String(r.simple_description || '').toLowerCase();
        const num = String(r.code_number || '').toLowerCase();
        const kind = String(r.kind || '').toLowerCase();
        return code.includes(query) || name.includes(query) || desc.includes(query) || simple.includes(query) || num.includes(query) || kind.includes(query);
    });

    if (dbRecordCount) {
        dbRecordCount.textContent = filtered.length;
    }

    if (!dbRecordsList) return;

    if (filtered.length === 0) {
        dbRecordsList.innerHTML = `<div style="color:var(--muted); text-align:center; padding:20px; font-size:13px;">No records found</div>`;
        return;
    }

    dbRecordsList.innerHTML = filtered.map((r) => {
        const kind = r.kind || dbType.value;
        const badgeClass = kind === 'qrcode' ? 'qrcode' : 'barcode';
        const badgeLabel = kind === 'qrcode' ? 'QR' : 'Bar';

        return `
            <div class="db-record-item">
                <div class="db-record-info">
                    <div class="db-record-title">
                        <span class="db-record-badge ${badgeClass}">${badgeLabel}</span>
                        <span>${escapeHtml(r.name || 'Unnamed')}</span>
                    </div>
                    <div class="db-record-meta">
                        <strong>Code:</strong> <code>${escapeHtml(r.code)}</code>
                        ${r.code_number ? `| <strong>No:</strong> ${escapeHtml(r.code_number)}` : ''}
                    </div>
                    ${r.simple_description ? `<div class="db-record-meta"><strong>Summary:</strong> ${escapeHtml(r.simple_description)}</div>` : ''}
                    ${r.description ? `<div class="db-record-desc">${escapeHtml(r.description)}</div>` : ''}
                </div>
                <div class="db-record-actions">
                    <button class="db-btn-icon db-btn-edit" data-code="${escapeHtml(r.code)}">Edit</button>
                    <button class="db-btn-icon db-btn-delete" data-code="${escapeHtml(r.code)}">Delete</button>
                </div>
            </div>
        `;
    }).join('');
}

async function loadDb() {
    const kind = dbType.value;
    try {
        const res = await fetch(`/vision2/api/db/${kind}`);
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.error || 'Load db failed');
        }
        dbJson.textContent = JSON.stringify(data.db, null, 2);
        currentDbRecords = (data.db?.records || []).map((r) => ({
            ...r,
            kind: r.kind || kind
        }));
        renderDbRecordsList();
    } catch (err) {
        dbJson.textContent = `Error: ${err.message}`;
        if (dbRecordsList) {
            dbRecordsList.innerHTML = `<div style="color:var(--danger); padding:10px; font-size:13px;">Load DB failed: ${err.message}</div>`;
        }
        if (dbRecordCount) {
            dbRecordCount.textContent = '0';
        }
    }
}

async function tryDecodeDbImageOnSelect() {
    const file = getSelectedDbFile();
    if (!file) return;

    try {
        const result = await decodeCodeLocally(file, dbType.value);
        if (result && result.text) {
            dbCode.value = result.text;
        }
    } catch (err) {
        console.warn('DB local decode on select failed:', err);
    }
}

if (dbImageCamera) {
    dbImageCamera.addEventListener('change', tryDecodeDbImageOnSelect);
}
if (dbImageAlbum) {
    dbImageAlbum.addEventListener('change', tryDecodeDbImageOnSelect);
}

saveDbBtn.addEventListener('click', async () => {
    let payload = collectDbPayload();
    let imageFile = getSelectedDbFile();

    if (!payload.code && !imageFile) {
        alert('Please enter a code or upload a QR/barcode image.');
        return;
    }

    saveDbBtn.disabled = true;
    const originalText = saveDbBtn.textContent;
    saveDbBtn.textContent = 'Saving...';

    try {
        let localDecodedDb = null;
        let res;

        if (imageFile) {
            saveDbBtn.textContent = 'Preparing image...';
            const originalName = imageFile.name || 'code-image.jpg';

            try {
                imageFile = await downscaleImage(imageFile, 1000);
            } catch (e) {
                console.warn('Downscaling failed, using original:', e);
            }

            saveDbBtn.textContent = 'Decoding locally...';
            localDecodedDb = await decodeCodeLocally(imageFile, dbType.value);

            if (localDecodedDb && localDecodedDb.text && !payload.code) {
                payload.code = localDecodedDb.text;
                dbCode.value = localDecodedDb.text;
            }

            if (localDecodedDb && localDecodedDb.kind && localDecodedDb.kind !== 'unknown') {
                if (localDecodedDb.kind !== dbType.value) {
                    dbType.value = localDecodedDb.kind;
                    await loadDb();
                }
            }

            saveDbBtn.textContent = 'Uploading...';
            const fd = new FormData();
            fd.append('code', payload.code);
            fd.append('name', payload.name);
            fd.append('code_number', payload.code_number);
            fd.append('simple_description', payload.simple_description);
            fd.append('description', payload.description);
            fd.append('image', imageFile, originalName);

            if (localDecodedDb) {
                fd.append('local_decoded_text', localDecodedDb.text || '');
                fd.append('local_decoded_format', localDecodedDb.format || '');
                fd.append('local_decoded_kind', localDecodedDb.kind || '');
                fd.append('local_decoded_engine', localDecodedDb.engine || '');
            }

            res = await fetch(`/vision2/api/db/${dbType.value}`, {
                method: 'POST',
                body: fd
            });
        } else {
            res = await fetch(`/vision2/api/db/${dbType.value}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
        }

        const data = await res.json();
        if (!res.ok || !data.success) {
            if (data.suggested_kind) {
                dbType.value = data.suggested_kind;
                if (data.detected_code) {
                    dbCode.value = data.detected_code;
                }
                await loadDb();
                throw new Error(`${data.error}. Switched Type to ${data.suggested_kind} and filled detected code.`);
            }
            throw new Error(data.error || 'Save failed');
        }

        if (data.record?.code) {
            dbCode.value = data.record.code;
        }

        if (dbImageCamera) dbImageCamera.value = '';
        if (dbImageAlbum) dbImageAlbum.value = '';

        await loadDb();
    } catch (err) {
        alert(err.message);
    } finally {
        saveDbBtn.disabled = false;
        saveDbBtn.textContent = originalText;
    }
});

reloadDbBtn.addEventListener('click', loadDb);

deleteDbBtn.addEventListener('click', async () => {
    const code = dbCode.value.trim();
    if (!code) {
        alert('Input code to delete');
        return;
    }

    try {
        const res = await fetch(`/vision2/api/db/${dbType.value}?code=${encodeURIComponent(code)}`, {
            method: 'DELETE'
        });
        const data = await res.json();
        if (!res.ok || !data.success) {
            throw new Error(data.error || 'Delete failed');
        }
        await loadDb();
    } catch (err) {
        alert(err.message);
    }
});

if (clearDbFormBtn) {
    clearDbFormBtn.addEventListener('click', () => {
        dbCode.value = '';
        dbName.value = '';
        dbCodeNumber.value = '';
        dbSimple.value = '';
        dbDesc.value = '';
        if (dbImageCamera) dbImageCamera.value = '';
        if (dbImageAlbum) dbImageAlbum.value = '';
    });
}

if (dbSearch) {
    dbSearch.addEventListener('input', renderDbRecordsList);
}

if (dbRecordsList) {
    dbRecordsList.addEventListener('click', async (e) => {
        const editBtn = e.target.closest('.db-btn-edit');
        const deleteBtn = e.target.closest('.db-btn-delete');

        if (editBtn) {
            const code = editBtn.dataset.code;
            const record = currentDbRecords.find((r) => r.code === code);
            if (record) {
                dbCode.value = record.code || '';
                dbName.value = record.name || '';
                dbCodeNumber.value = record.code_number || '';
                dbSimple.value = record.simple_description || '';
                dbDesc.value = record.description || '';
                if (dbImageCamera) dbImageCamera.value = '';
                if (dbImageAlbum) dbImageAlbum.value = '';
                dbCode.focus();
            }
        }

        if (deleteBtn) {
            const code = deleteBtn.dataset.code;
            if (confirm(`Are you sure you want to delete record: ${code}?`)) {
                try {
                    const kind = dbType.value;
                    const res = await fetch(`/vision2/api/db/${kind}?code=${encodeURIComponent(code)}`, {
                        method: 'DELETE'
                    });
                    const data = await res.json();
                    if (!res.ok || !data.success) {
                        throw new Error(data.error || 'Delete failed');
                    }
                    await loadDb();
                } catch (err) {
                    alert(err.message);
                }
            }
        }
    });
}

dbType.addEventListener('change', loadDb);

(async function init() {
    await refreshReferences();
    await loadDb();
})();