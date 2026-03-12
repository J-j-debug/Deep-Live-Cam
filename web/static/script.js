const sourceDrop = document.getElementById('source-drop');
const targetDrop = document.getElementById('target-drop');
const sourceInput = document.getElementById('source-input');
const targetInput = document.getElementById('target-input');
const startBtn = document.getElementById('start-btn');
const opacitySlider = document.getElementById('opacity');
const opacityVal = document.getElementById('opacity-val');

let sourceFile = null;
let targetFile = null;
let sessionId = null;
let socket = null;

// UI Helpers
opacitySlider.oninput = () => opacityVal.innerText = opacitySlider.value;

function setupDropZone(zone, input, type) {
    zone.onclick = () => input.click();
    zone.ondragover = (e) => { e.preventDefault(); zone.style.borderColor = '#2ecc71'; };
    zone.ondragleave = () => { zone.style.borderColor = 'rgba(255, 255, 255, 0.1)'; };
    zone.ondrop = (e) => {
        e.preventDefault();
        zone.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        handleFiles(e.dataTransfer.files[0], type);
    };
    input.onchange = () => handleFiles(input.files[0], type);
}

function handleFiles(file, type) {
    if (!file) return;
    if (type === 'source') {
        sourceFile = file;
        showPreview(file, 'source-preview');
    } else {
        targetFile = file;
        showPreview(file, 'target-preview');
    }
}

function showPreview(file, elementId) {
    const el = document.getElementById(elementId);
    el.innerHTML = '';
    if (file.type.startsWith('image/')) {
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        el.appendChild(img);
    } else {
        el.innerText = file.name;
    }
}

setupDropZone(sourceDrop, sourceInput, 'source');
setupDropZone(targetDrop, targetInput, 'target');

// WebSocket Setup
function connectWS(id) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${protocol}//${window.location.host}/ws/${id}`);
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'progress') {
            document.getElementById('progress-fill').style.width = `${data.value}%`;
            document.getElementById('status-text').innerText = data.status;
        } else if (data.type === 'complete') {
            showResult(data.url);
        }
    };
}

// Upload and Start
startBtn.onclick = async () => {
    if (!sourceFile || !targetFile) {
        alert("Please select both source and target files.");
        return;
    }

    startBtn.disabled = true;
    document.getElementById('progress-box').style.display = 'block';
    
    try {
        // Upload Source
        const sourceData = new FormData();
        sourceData.append('file', sourceFile);
        const resSource = await fetch('/upload', { method: 'POST', body: sourceData });
        const { filename: sourceName, session_id: id } = await resSource.json();
        sessionId = id;

        // Upload Target
        const targetData = new FormData();
        targetData.append('file', targetFile);
        const resTarget = await fetch('/upload', { method: 'POST', body: targetData });
        const { filename: targetName } = await resTarget.json();

        connectWS(sessionId);

        // Start Processing
        const options = {
            session_id: sessionId,
            source: sourceName,
            target: targetName,
            enhancer: document.getElementById('face-enhancer').checked,
            poisson: document.getElementById('poisson-blend').checked,
            opacity: parseFloat(opacitySlider.value)
        };

        await fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(options)
        });

    } catch (err) {
        alert("An error occurred: " + err.message);
        startBtn.disabled = false;
    }
};

function showResult(url) {
    document.getElementById('progress-box').style.display = 'none';
    document.getElementById('result-container').style.display = 'block';
    const finalMedia = document.getElementById('final-media');
    finalMedia.innerHTML = '';
    
    if (url.endsWith('.mp4')) {
        const video = document.createElement('video');
        video.src = url;
        video.controls = true;
        video.style.width = '100%';
        finalMedia.appendChild(video);
    } else {
        const img = document.createElement('img');
        img.src = url;
        img.style.width = '100%';
        finalMedia.appendChild(img);
    }
    
    document.getElementById('download-btn').href = url;
    startBtn.disabled = false;
}
