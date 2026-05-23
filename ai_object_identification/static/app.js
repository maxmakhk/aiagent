// Global State
let selectedRefFiles = [];
let selectedSceneFile = null;
let activeLabels = [];

// On Load
document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

// Initialize App Data
async function initApp() {
    showToast("Initializing Vision Engine...");
    await refreshCatalogs();
    await checkBackendHealth();
}

// Check Backend Connection and Pre-warm
async function checkBackendHealth() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        console.log("[*] Backend Status:", data);
        if (data.status === "healthy") {
            showToast(`Connected! Device: ${data.device.toUpperCase()}`, 3000);
        } else {
            showToast("System loading. Please wait...", 4000);
        }
    } catch (e) {
        showToast("Error: Cannot connect to Flask Backend!", 5000);
    }
}

// Switch Tabs Panel
function switchTab(tabId) {
    // Hide all contents
    document.querySelectorAll(".tab-content").forEach(el => {
        el.classList.remove("active-content");
    });
    // Remove active button classes
    document.querySelectorAll(".tab-btn").forEach(el => {
        el.classList.remove("active");
    });
    
    // Show selected
    document.getElementById(`tab-${tabId}`).classList.add("active-content");
    document.getElementById(`tab-btn-${tabId}`).classList.add("active");
}

// Switch Active Catalog galleries
async function refreshCatalogs() {
    try {
        const res = await fetch("/api/reference/list");
        const data = await res.json();
        activeLabels = data;
        
        renderCatalogList(data);
        populateDropdowns(data);
    } catch (e) {
        console.error("Failed to fetch references list:", e);
        showToast("Failed to reload target catalog.");
    }
}

// Render Tab 1 Object List
function renderCatalogList(labels) {
    const listContainer = document.getElementById("catalog-list");
    listContainer.innerHTML = "";
    
    if (labels.length === 0) {
        listContainer.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">📭</span>
                <p>No reference objects defined yet.</p>
                <p class="sub">Create a label and upload photos to get started.</p>
            </div>
        `;
        return;
    }
    
    labels.forEach(item => {
        const card = document.createElement("div");
        card.className = "catalog-item";
        card.onclick = () => loadReferenceGallery(item.label);
        
        let thumbHtml = "";
        if (item.thumbnail) {
            thumbHtml = `<img class="catalog-thumb" src="${item.thumbnail}" alt="${item.label}">`;
        } else {
            thumbHtml = `<div class="catalog-thumb-empty">📷</div>`;
        }
        
        card.innerHTML = `
            <div class="catalog-info">
                ${thumbHtml}
                <div class="catalog-name">
                    <h4>${item.label}</h4>
                    <p>${item.photo_count} Reference photos</p>
                </div>
            </div>
            <button class="catalog-action-btn">View Gallery</button>
        `;
        
        listContainer.appendChild(card);
    });
}

// Populate target dropdown selectors
function populateDropdowns(labels) {
    const uploadSelect = document.getElementById("upload-target-select");
    const filterSelect = document.getElementById("filter-label-select");
    
    // Keep initial choices
    uploadSelect.innerHTML = `<option value="" disabled selected>-- Choose an object --</option>`;
    filterSelect.innerHTML = `<option value="all">Search all reference objects</option>`;
    
    labels.forEach(item => {
        // Upload form dropdown
        const opt1 = document.createElement("option");
        opt1.value = item.label;
        opt1.textContent = item.label;
        uploadSelect.appendChild(opt1);
        
        // Recognition filter dropdown
        const opt2 = document.createElement("option");
        opt2.value = item.label;
        opt2.textContent = `Search only: ${item.label}`;
        filterSelect.appendChild(opt2);
    });
}

// Form: Teach new target label
async function handleCreateLabel(event) {
    event.preventDefault();
    const labelInput = document.getElementById("new-label-name");
    const rawLabel = labelInput.value.trim();
    
    if (!rawLabel) return;
    
    showToast("Registering label...");
    try {
        const res = await fetch("/api/reference/create", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ label: rawLabel })
        });
        
        const data = await res.json();
        
        if (res.ok) {
            showToast(data.message, 3000);
            labelInput.value = "";
            await refreshCatalogs();
            // Select the newly created label automatically in dropdown
            document.getElementById("upload-target-select").value = data.label;
        } else {
            showToast(`Error: ${data.error}`, 4000);
        }
    } catch (e) {
        showToast("Error creating label", 4000);
    }
}

// Files Selection: References
function handleRefFileSelect(event) {
    const files = event.target.files;
    selectedRefFiles = Array.from(files);
    
    const countEl = document.getElementById("selected-ref-files-count");
    const btnUpload = document.getElementById("btn-upload-references");
    
    if (selectedRefFiles.length > 0) {
        countEl.textContent = `${selectedRefFiles.length} photos selected`;
        countEl.classList.remove("hidden");
        btnUpload.disabled = false;
    } else {
        countEl.classList.add("hidden");
        btnUpload.disabled = true;
    }
}

// Upload Reference files
async function handleUploadReferences() {
    const labelSelect = document.getElementById("upload-target-select");
    const selectedLabel = labelSelect.value;
    
    if (!selectedLabel) {
        showToast("Please choose an object label first!", 4000);
        return;
    }
    
    if (selectedRefFiles.length === 0) {
        showToast("Please select reference photos first!", 4000);
        return;
    }
    
    const formData = new FormData();
    formData.append("label", selectedLabel);
    selectedRefFiles.forEach(file => {
        formData.append("files", file);
    });
    
    setUploadState(true);
    showToast(`Uploading references for '${selectedLabel}' & generating CLIP embeddings...`, 15000);
    
    try {
        const res = await fetch("/api/reference/upload", {
            method: "POST",
            body: formData
        });
        
        const data = await res.json();
        if (res.ok) {
            showToast(`Success! ${data.files.length} references saved. Cache updated!`, 4000);
            
            // Clean up inputs
            selectedRefFiles = [];
            document.getElementById("ref-file-input").value = "";
            document.getElementById("selected-ref-files-count").classList.add("hidden");
            
            await refreshCatalogs();
            await loadReferenceGallery(selectedLabel);
        } else {
            showToast(`Upload failed: ${data.error}`, 4000);
        }
    } catch (e) {
        showToast("Failed to upload reference files", 4000);
    } finally {
        setUploadState(false);
    }
}

function setUploadState(isLoading) {
    const btn = document.getElementById("btn-upload-references");
    const dropzone = document.getElementById("ref-dropzone");
    
    if (isLoading) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner">⏳</span> Processing Vision Embeddings...`;
        dropzone.style.pointerEvents = "none";
        dropzone.style.opacity = "0.6";
    } else {
        btn.innerHTML = `⚡ Upload & Process Embeddings`;
        dropzone.style.pointerEvents = "auto";
        dropzone.style.opacity = "1";
    }
}

// Load photo gallery for selected object
async function loadReferenceGallery(label) {
    const gallerySection = document.getElementById("ref-gallery-section");
    const galleryTitle = document.getElementById("gallery-title-label");
    const galleryGrid = document.getElementById("ref-gallery-grid");
    
    galleryTitle.textContent = label;
    galleryGrid.innerHTML = "Loading gallery...";
    gallerySection.classList.remove("hidden");
    
    // Smooth scroll gallery into view
    gallerySection.scrollIntoView({ behavior: "smooth" });
    
    try {
        const res = await fetch(`/api/reference/${label}`);
        if (!res.ok) throw new Error("Failed to load details");
        
        const data = await res.json();
        galleryGrid.innerHTML = "";
        
        if (data.photos.length === 0) {
            galleryGrid.innerHTML = `<p style="grid-column: 1/-1; text-align: center; color: var(--text-muted); font-size: 13px;">No reference photos saved for this target yet.</p>`;
            return;
        }
        
        data.photos.forEach(url => {
            const img = document.createElement("img");
            img.src = url;
            img.className = "ref-gallery-img";
            img.alt = label;
            galleryGrid.appendChild(img);
        });
    } catch (e) {
        galleryGrid.innerHTML = "Failed to load reference images.";
    }
}

function closeRefGallery() {
    document.getElementById("ref-gallery-section").classList.add("hidden");
}

// Rebuild Embeddings Database cache
async function rebuildEmbeddings() {
    showToast("Rebuilding all embeddings in background... Please stand by.", 12000);
    try {
        const res = await fetch("/api/reference/rebuild", { method: "POST" });
        const data = await res.json();
        if (res.ok) {
            showToast("Successfully rebuilt all embeddings caches!", 4000);
            await refreshCatalogs();
        } else {
            showToast(`Rebuild failed: ${data.error}`, 4000);
        }
    } catch (e) {
        showToast("Error rebuilding embeddings cache.", 4000);
    }
}

// --- TAB 2: SCENE RECOGNITION FLOW ---

// File Selection: Scene Image
function handleSceneFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    selectedSceneFile = file;
    const filenameEl = document.getElementById("selected-scene-filename");
    const btnDetect = document.getElementById("btn-detect");
    
    filenameEl.textContent = `Scene: ${file.name}`;
    filenameEl.classList.remove("hidden");
    btnDetect.disabled = false;
}

// Slider update UI
function updateThresholdValue(val) {
    document.getElementById("threshold-val").textContent = val;
}

// Run Scene Recognition
async function runRecognition() {
    if (!selectedSceneFile) {
        showToast("Please capture or upload a scene image first!", 4000);
        return;
    }
    
    const sliderVal = document.getElementById("threshold-slider").value;
    const filterVal = document.getElementById("filter-label-select").value;
    
    const formData = new FormData();
    formData.append("image", selectedSceneFile);
    formData.append("threshold", sliderVal);
    formData.append("target_label", filterVal);
    
    setRecognizeState(true);
    showToast("Extracting contours & comparing crop embeddings using CLIP...", 15000);
    
    try {
        const res = await fetch("/api/recognize", {
            method: "POST",
            body: formData
        });
        
        const data = await res.json();
        if (res.ok) {
            showToast("Detection scanning completed successfully!", 3000);
            renderRecognitionResults(data);
        } else {
            showToast(`Scan failed: ${data.error}`, 5000);
        }
    } catch (e) {
        showToast("Network error running recognition engine.", 5000);
    } finally {
        setRecognizeState(false);
    }
}

function setRecognizeState(isLoading) {
    const btn = document.getElementById("btn-detect");
    const dropzone = document.getElementById("scene-dropzone");
    
    if (isLoading) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner">⏳</span> Running CLIP Vision engine...`;
        dropzone.style.pointerEvents = "none";
        dropzone.style.opacity = "0.6";
    } else {
        btn.innerHTML = `🔍 Run Recognition Engine`;
        dropzone.style.pointerEvents = "auto";
        dropzone.style.opacity = "1";
    }
}

// Render Results on Page
function renderRecognitionResults(data) {
    // Hide empty state
    document.getElementById("recognition-empty-state").classList.add("hidden");
    
    // Show results section
    const resultsContainer = document.getElementById("recognition-results");
    resultsContainer.classList.remove("hidden");
    
    // Scroll results into view
    resultsContainer.scrollIntoView({ behavior: "smooth" });
    
    // Update annotated image
    const resultImg = document.getElementById("result-image-view");
    // Add dummy timestamp query parameter to prevent browser caching old image results
    resultImg.src = `${data.result_url}?t=${new Date().getTime()}`;
    
    // Update item counting statistics
    const statsGrid = document.getElementById("stats-summary-grid");
    statsGrid.innerHTML = "";
    
    const countKeys = Object.keys(data.counts);
    if (countKeys.length === 0) {
        statsGrid.innerHTML = `
            <div style="text-align: center; color: var(--text-rose); font-size: 13px; font-weight: 500; padding: 10px;">
                ⚠️ No matching target objects detected in scene.
            </div>
        `;
    } else {
        countKeys.forEach(label => {
            const count = data.counts[label];
            const row = document.createElement("div");
            row.className = "stat-row";
            row.innerHTML = `
                <span class="stat-label">${label}</span>
                <span class="stat-value">${count} found</span>
            `;
            statsGrid.appendChild(row);
        });
    }
    
    // Update vision debugger statistics
    document.getElementById("dbg-raw").textContent = data.debug.proposals_count;
    document.getElementById("dbg-merged").textContent = data.debug.merged_contour_count;
    document.getElementById("dbg-fallback").textContent = data.debug.using_fallback_grid ? "ACTIVE" : "NONE";
    document.getElementById("dbg-final").textContent = data.debug.final_analyzed_candidates;
    
    // Update JSON panel
    document.getElementById("json-raw-output").textContent = JSON.stringify(data, null, 2);
}

// Helper: Show status toasts
let toastTimeout;
function showToast(message, duration = 3000) {
    const toast = document.getElementById("status-toast");
    const toastMsg = document.getElementById("toast-message");
    
    toastMsg.textContent = message;
    toast.classList.remove("hidden");
    
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        toast.classList.add("hidden");
    }, duration);
}
