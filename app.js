// 1. SETUP DOM ELEMENTS
const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');

const liveBox = document.getElementById('liveBox');       // Shows "Seeing: Monday" commented out the html 
const sentenceBox = document.getElementById('sentenceBox'); // Shows "Sentence: Monday Tuesday"

// Create a hidden canvas specifically for downscaling and compressing the image
const hiddenCanvas = document.createElement('canvas');
const hiddenCtx = hiddenCanvas.getContext('2d');
hiddenCanvas.width = 640;  // Lock resolution to keep payloads tiny
hiddenCanvas.height = 480;



let idleTimer = null;
let wordCounter = 1;
let lastConfirmedWord = "";
const IDLE_TIME_LIMIT = 4000;

function clearTranslator() {
    // Reset the internal counter
    wordCounter = 1;
    lastConfirmedWord = "";
    previousHistory = "";

    // Clear the Output Box
    if (sentenceBox) sentenceBox.innerText = "";

    // Clear the Words History List
    const wordsList = document.getElementById('confirmed-words-list');
    if (wordsList) {
        wordsList.innerHTML = "";
    }

    console.log("Translator Cleared: Device was idle too long.");
}

function addWordToHistory(word) {
    const list = document.getElementById('confirmed-words-list');
    if (!list) return;
    const row = document.createElement('div');
    row.className = 'word-history-row'; 
    const paddedIndex = wordCounter.toString().padStart(2, '0');

    row.innerHTML = `
        <span class="history-label">${paddedIndex}</span>
        <span class="history-value">${word.toUpperCase()}</span>
    `;
    list.appendChild(row);
    wordCounter++;
}

// TRAFFIC CONTROL: Prevents network lag and maintains local FPS
let isProcessing = false;
let previousHistory = "";

// 2. THE MAIN LOOP (Replaces MediaPipe onResults)
async function processVideoFrame() {
    // Only process if the camera is ready and the server is NOT busy
    if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA && !isProcessing) {
        isProcessing = true; // 🔴 LOCK (Stop other frames from piling up)

        // A. Downscale and Compress
        // Draw the current video frame to our hidden canvas
        hiddenCtx.drawImage(videoElement, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
        // Compress to JPEG at 50% quality for massive speedup
        const base64Image = hiddenCanvas.toDataURL('image/jpeg', 0.5); 

        // B. Send to Flask Backend
        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Image })
            });
            
            const data = await response.json();
            

        // 1. UPDATE THE TOP 3 PREDICTIONS (The code from your screenshot)
        if (data.top_3) {
            data.top_3.forEach((item, index) => {
                const rank = index + 1;
                const wordEl = document.getElementById(`word-${rank}`);
                const confEl = document.getElementById(`conf-${rank}`);

                if (wordEl && confEl) {
                    wordEl.innerText = item.label;
                    confEl.innerText = item.conf.toFixed(1) + "%";
                    wordEl.style.color = (rank === 1) ? "#4770B0" : "#252b3b";
                    confEl.style.color = (rank === 1) ? "#4770B0" : "#252b3b";
                }
            });
        }

        // 2. UPDATE THE STATUS LABEL
const statusLabel = document.getElementById('status-label');
statusLabel.innerText = data.state;

if (data.state === 'SIGNING' || data.state === 'EVALUATE') {
    // A. If the user is actively signing or the model is thinking...
    statusLabel.style.color = (data.state === 'SIGNING') ? "#ff4444" : "#ffbb00";
    
    // STOP the countdown immediately!
    if (idleTimer) {
        clearTimeout(idleTimer);
        idleTimer = null;
        console.log("Timer Reset: User is active.");
    }
} else {
    // B. This is the 'IDLE' or 'READY' state (No hand/sign detected)
    statusLabel.style.color = "#3878ec"; 
    
    // START the countdown only if it's not already running
    if (!idleTimer) {
        console.log("Timer Started: Waiting for inactivity...");
        idleTimer = setTimeout(() => {
            clearTranslator();
            idleTimer = null;
        }, IDLE_TIME_LIMIT); 
    }
}
        // 3. UPDATE FPS (Performance metric)
        const now = Date.now();
        const fps = Math.round(1000 / (now - (window.lastTime || now)));
        window.lastTime = now;
        document.getElementById('fps-display').innerText = `${fps} FPS`

            // --- 1. LIVE PREDICTION (Yellow) ---
            if (data.new_sign && data.new_sign !== "...") {
                liveBox.innerText = "Input: " + data.new_sign;
                liveBox.style.color = "yellow";
            } else {
                liveBox.innerText = "...";
                liveBox.style.color = "gray";
            }

            // --- 2. VALIDATION & FINAL SENTENCE (Green Flash) ---
            if (data.history !== previousHistory) {
    // This block runs ONLY when a new word is confirmed by the backend
    
    // Split the history string into an array to get the latest word added
    const historyArray = data.history.trim().split(" ");
    const latestWord = historyArray[historyArray.length - 1];

    if (latestWord && latestWord !== "...") {
        addWordToHistory(latestWord); 
    }

    if (data.sentence && data.sentence !== "..." && data.sentence !== "") {
        sentenceBox.innerText = data.sentence;
    } else {
        sentenceBox.innerText = data.history; 
    }
    
    previousHistory = data.history;
    
    // Visual Flash Effect
    // sentenceBox.style.backgroundColor = "white"; // Slight grey flash
    setTimeout(() => { sentenceBox.style.backgroundColor = "#dde4ee"; }, 500);
}
        } catch (err) {
            console.error("Server error or timeout:", err);
        } finally {
            isProcessing = false; // 🟢 UNLOCK (Next frame allowed)
        }
    }

    // C. Draw Visual Feedback for the User
    // Mirror the video to the visible canvas so the user can see themselves
    if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        // --- MIRROR EFFECT ---
    // Move to the right side of the canvas
    canvasCtx.translate(canvasElement.width, 0);
    // Flip the image horizontally
    canvasCtx.scale(-1, 1);
    
    // Draw the video
    canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    
    canvasCtx.restore();
    }

    

    // Loop continuously before the next browser repaint
    requestAnimationFrame(processVideoFrame);
}



// 3. START CAMERA (Standard HTML5 WebRTC)
navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
    .then(stream => {
        videoElement.srcObject = stream;
        videoElement.play();
        // Start the loop once the camera is live
        requestAnimationFrame(processVideoFrame);
    })
    .catch(err => {
        console.error("Camera access denied!", err);
        alert("Please allow camera access to use the translator.");
    });