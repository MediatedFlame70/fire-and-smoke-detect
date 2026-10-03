document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const confidenceSlider = document.getElementById('confidenceSlider');
    const confidenceValue = document.getElementById('confidenceValue');
    const detectBtn = document.getElementById('detectBtn');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.querySelector('.btn-loader');
    
    const resultsEmpty = document.getElementById('resultsEmpty');
    const resultsContent = document.getElementById('resultsContent');
    const resultImage = document.getElementById('resultImage');
    const bboxCanvas = document.getElementById('bboxCanvas');
    const imageWrapper = document.querySelector('.image-wrapper');
    
    // Status
    const statusText = document.getElementById('statusText');
    const deviceText = document.getElementById('deviceText');
    const modelStatusText = document.getElementById('modelStatusText');
    const statusDot = document.querySelector('.status-dot');
    
    let currentFile = null;
    let currentResult = null;

    // Check System Status
    async function checkStatus() {
        try {
            const res = await fetch('/status');
            const data = await res.json();
            
            if (data.model_loaded) {
                statusText.textContent = "System Online";
                statusDot.style.backgroundColor = "#10B981"; // Green
                modelStatusText.textContent = "Active (CN2VF)";
                modelStatusText.className = "metric-val highlight-blue";
            } else {
                statusText.textContent = "Model Offline";
                statusDot.style.backgroundColor = "#EF4444"; // Red
                modelStatusText.textContent = "Not Loaded";
                modelStatusText.className = "metric-val";
            }
            
            deviceText.textContent = data.device.toUpperCase();
        } catch (err) {
            console.error("Status check failed", err);
            statusText.textContent = "Connection Error";
            statusDot.style.backgroundColor = "#EF4444";
        }
    }
    
    checkStatus();

    // Slider
    confidenceSlider.addEventListener('input', (e) => {
        confidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
        if (currentResult) {
            drawBoundingBox(currentResult);
        }
    });

    // Upload Handlers
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        
        currentFile = file;
        
        // Show preview in upload box
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPlaceholder.innerHTML = `
                <img src="${e.target.result}" style="max-height: 120px; border-radius: 8px; margin-bottom: 1rem;">
                <h3>${file.name}</h3>
                <p>${(file.size / 1024).toFixed(1)} KB</p>
            `;
            detectBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Run Detection
    detectBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        
        // UI Loading state
        detectBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'block';
        
        resultsEmpty.style.display = 'none';
        resultsContent.style.display = 'flex';
        
        // Setup image wrapper for scanning animation
        imageWrapper.classList.add('scanning');
        
        // Load image into preview
        resultImage.src = URL.createObjectURL(currentFile);
        
        // Clear canvas
        const ctx = bboxCanvas.getContext('2d');
        ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);

        try {
            const formData = new FormData();
            formData.append('file', currentFile);
            formData.append('confidence', confidenceSlider.value);
            
            const res = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || 'Prediction failed');
            }
            
            const data = await res.json();
            currentResult = data;
            
            // Wait for image to load to draw bbox correctly
            if (resultImage.complete) {
                updateDashboard(data);
            } else {
                resultImage.onload = () => updateDashboard(data);
            }
            
        } catch (err) {
            alert("Error: " + err.message);
            resultsEmpty.style.display = 'flex';
            resultsContent.style.display = 'none';
        } finally {
            detectBtn.disabled = false;
            btnText.style.display = 'block';
            btnLoader.style.display = 'none';
            imageWrapper.classList.remove('scanning');
        }
    });

    function updateDashboard(data) {
        // Draw BBox
        drawBoundingBox(data);
        
        // Badge
        const badge = document.getElementById('detectionBadge');
        document.getElementById('predictedClass').textContent = data.class;
        document.getElementById('confidencePercent').textContent = (data.confidence * 100).toFixed(1) + '%';
        
        const icons = ['🔥', '💨', '✓'];
        const colors = ['rgba(239, 68, 68, 0.2)', 'rgba(156, 163, 175, 0.2)', 'rgba(16, 185, 129, 0.2)'];
        const borderColors = ['#EF4444', '#9CA3AF', '#10B981'];
        
        document.getElementById('badgeIcon').textContent = icons[data.class_idx];
        badge.style.backgroundColor = colors[data.class_idx];
        badge.style.borderColor = borderColors[data.class_idx];
        
        // Probabilities
        const probs = data.all_probabilities;
        
        const updateBar = (id, name) => {
            const val = probs[name] * 100;
            document.getElementById(id + 'ProbValue').textContent = val.toFixed(1) + '%';
            document.getElementById(id + 'ProbBar').style.width = val + '%';
        };
        
        updateBar('fire', 'Fire');
        updateBar('smoke', 'Smoke');
        updateBar('neutral', 'Neutral');
        
        // BBox Coords
        document.getElementById('coordX').textContent = data.bbox[0].toFixed(3);
        document.getElementById('coordY').textContent = data.bbox[1].toFixed(3);
        document.getElementById('coordW').textContent = data.bbox[2].toFixed(3);
        document.getElementById('coordH').textContent = data.bbox[3].toFixed(3);
        
        // Alert
        const alertBox = document.getElementById('alertBox');
        if (data.is_detection && (data.class_idx === 0 || data.class_idx === 1)) {
            alertBox.style.display = 'flex';
            document.getElementById('alertText').textContent = `${data.class} detected with ${(data.confidence*100).toFixed(1)}% confidence.`;
            alertBox.style.borderColor = borderColors[data.class_idx];
            alertBox.style.backgroundColor = colors[data.class_idx];
        } else {
            alertBox.style.display = 'none';
        }
    }

    function drawBoundingBox(data) {
        // Match canvas size to image display size
        bboxCanvas.width = resultImage.width;
        bboxCanvas.height = resultImage.height;
        
        const ctx = bboxCanvas.getContext('2d');
        ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
        
        // If neutral or below threshold, don't draw box
        const threshold = parseFloat(confidenceSlider.value);
        if (data.class_idx === 2 || data.confidence < threshold) return;
        
        const [x_c, y_c, w, h] = data.bbox;
        const width = bboxCanvas.width;
        const height = bboxCanvas.height;
        
        const x_px = (x_c - w/2) * width;
        const y_px = (y_c - h/2) * height;
        const w_px = w * width;
        const h_px = h * height;
        
        const color = data.class_idx === 0 ? '#EF4444' : '#9CA3AF';
        
        // Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x_px, y_px, w_px, h_px);
        
        // Corner brackets
        const len = 15;
        ctx.beginPath();
        // Top Left
        ctx.moveTo(x_px, y_px + len); ctx.lineTo(x_px, y_px); ctx.lineTo(x_px + len, y_px);
        // Top Right
        ctx.moveTo(x_px + w_px - len, y_px); ctx.lineTo(x_px + w_px, y_px); ctx.lineTo(x_px + w_px, y_px + len);
        // Bottom Right
        ctx.moveTo(x_px + w_px, y_px + h_px - len); ctx.lineTo(x_px + w_px, y_px + h_px); ctx.lineTo(x_px + w_px - len, y_px + h_px);
        // Bottom Left
        ctx.moveTo(x_px + len, y_px + h_px); ctx.lineTo(x_px, y_px + h_px); ctx.lineTo(x_px, y_px + h_px - len);
        
        ctx.lineWidth = 5;
        ctx.stroke();
        
        // Label
        const label = `${data.class} ${(data.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 14px "Inter", sans-serif';
        const txtWidth = ctx.measureText(label).width;
        
        ctx.fillStyle = color;
        ctx.fillRect(x_px, y_px - 25, txtWidth + 16, 25);
        
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, x_px + 8, y_px - 8);
    }
});
