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

const previewImage = (file) => {
    if (!file) {
        preview.style.display = 'none';
        return;
    }
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
};

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
 * Downscale an image file if its long edge exceeds maxLongEdge, returning a Blob/File.
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
                    resolve(file); // No downscaling needed
                    return;
                }

                const canvas = document.createElement('canvas');
                const scale = maxLongEdge / currentLongEdge;
                canvas.width = img.width * scale;
                canvas.height = img.height * scale;

                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                canvas.toBlob((blob) => {
                    if (blob) {
                        resolve(blob); // Return Blob directly, avoiding File constructor error on iOS
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

analyzeBtn.addEventListener('click', async () => {
    let file = (userImageCamera.files && userImageCamera.files[0]) || (userImageAlbum.files && userImageAlbum.files[0]);
    if (!file) {
        alert('Please capture/select an image first.');
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Preparing image...';

    // Show a beautiful loading status spinner in the summary card
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

    analyzeBtn.disabled = true;
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
            `<span>Obj ms: ${perf.reference_detection_ms ?? '-'} </span>`,
            `<span>QR ms: ${perf.qrcode_detection_ms ?? '-'} </span>`,
            `<span>Barcode ms: ${perf.barcode_detection_ms ?? '-'} </span>`,
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
    if (keys.length === 0) {
        return 'None';
    }
    return keys
        .map((k) => `${k}: ${counts[k]}`)
        .join(', ');
}

function renderReadableError(message) {
    if (!readableSummary) {
        return;
    }
    readableSummary.innerHTML = `
        <div class="readable-title">Readable Result</div>
        <div class="readable-body readable-error">Analyze failed: ${message}</div>
    `;
}

function renderReadableSummary(data) {
    if (!readableSummary) {
        return;
    }

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

const dbSearch = document.getElementById('dbSearch');
const dbRecordsList = document.getElementById('dbRecordsList');
const dbRecordCount = document.getElementById('dbRecordCount');
const clearDbFormBtn = document.getElementById('clearDbFormBtn');

let currentDbRecords = [];

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
    const filtered = currentDbRecords.filter(r => {
        const code = String(r.code || '').toLowerCase();
        const name = String(r.name || '').toLowerCase();
        const desc = String(r.description || '').toLowerCase();
        const simple = String(r.simple_description || '').toLowerCase();
        const num = String(r.code_number || '').toLowerCase();
        return code.includes(query) || name.includes(query) || desc.includes(query) || simple.includes(query) || num.includes(query);
    });

    if (dbRecordCount) {
        dbRecordCount.textContent = filtered.length;
    }

    if (!dbRecordsList) return;

    if (filtered.length === 0) {
        dbRecordsList.innerHTML = `<div style="color:var(--muted); text-align:center; padding:20px; font-size:13px;">No records found</div>`;
        return;
    }

    dbRecordsList.innerHTML = filtered.map(r => {
        const kind = dbType.value;
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
        currentDbRecords = data.db?.records || [];
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

saveDbBtn.addEventListener('click', async () => {
    const payload = collectDbPayload();
    let imageFile = (dbImageCamera.files && dbImageCamera.files[0]) || (dbImageAlbum.files && dbImageAlbum.files[0]);
    if (!payload.code && !imageFile) {
        alert('Please enter a code or upload a QR/barcode image.');
        return;
    }

    saveDbBtn.disabled = true;
    const originalText = saveDbBtn.textContent;
    saveDbBtn.textContent = 'Saving...';

    try {
        let res;
        if (imageFile) {
            saveDbBtn.textContent = 'Preparing image...';
            const originalName = imageFile.name || 'code-image.jpg';
            try {
                imageFile = await downscaleImage(imageFile, 1000);
            } catch (e) {
                console.warn('Downscaling failed, using original:', e);
            }

            saveDbBtn.textContent = 'Uploading...';
            const fd = new FormData();
            fd.append('code', payload.code);
            fd.append('name', payload.name);
            fd.append('code_number', payload.code_number);
            fd.append('simple_description', payload.simple_description);
            fd.append('description', payload.description);
            fd.append('image', imageFile, originalName);

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
            const record = currentDbRecords.find(r => r.code === code);
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
