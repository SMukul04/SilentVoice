import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs";
import { LandmarkExtractor } from "./landmark_extractor.js";

/**
 * SilentVoice Frontend Dashboard Prototype Logic
 * Handles real-time video, Speech Recognition (with fallback), simulation routines, and state indicators.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const btnStartRecognition = document.getElementById('btnStartRecognition');
    const btnStopRecognition = document.getElementById('btnStopRecognition');
    const btnSpeak = document.getElementById('btnSpeak');
    const btnSpeakText = document.getElementById('btnSpeakText');
    const btnClear = document.getElementById('btnClear');
    
    // Status Bar & Pill Elements
    const cameraStatusBadge = document.getElementById('cameraStatusBadge');
    const cameraStatusDot = document.getElementById('cameraStatusDot');
    const cameraStatusText = document.getElementById('cameraStatusText');
    const handsDetectedText = document.getElementById('handsDetectedText');
    
    const avatarStatusBadge = document.getElementById('avatarStatusBadge');
    const avatarStatusDot = document.getElementById('avatarStatusDot');
    const avatarStatusText = document.getElementById('avatarStatusText');
    
    const barCameraDot = document.getElementById('barCameraDot');
    const barCameraText = document.getElementById('barCameraText');
    const barMicDot = document.getElementById('barMicDot');
    const barMicText = document.getElementById('barMicText');
    const barAvatarDot = document.getElementById('barAvatarDot');
    const barAvatarText = document.getElementById('barAvatarText');
    const barModelText = document.getElementById('barModelText');
    const barModelDot = document.getElementById('barModelDot');
    
    // Panel Elements
    const webcamVideo = document.getElementById('webcamVideo');
    const cameraPlaceholder = document.getElementById('cameraPlaceholder');
    const avatarImage = document.getElementById('avatarImage');
    const avatarPlaceholder = document.getElementById('avatarPlaceholder');
    
    const signOutput = document.getElementById('signOutput');
    const confidenceOutput = document.getElementById('confidenceOutput');
    const confidenceProgress = document.getElementById('confidenceProgress');
    const fpsOutput = document.getElementById('fpsOutput');
    
    const avatarStateBadge = document.getElementById('avatarStateBadge');
    const avatarDetailedStatus = document.getElementById('avatarDetailedStatus');
    
    const chatWindow = document.getElementById('chatWindow');
    const chatCounterBadge = document.getElementById('chatCounterBadge');
    
    // Settings Elements
    const selectModel = document.getElementById('selectModel');
    const sliderThreshold = document.getElementById('sliderThreshold');
    const sliderThresholdVal = document.getElementById('sliderThresholdVal');
    const btnSaveSettings = document.getElementById('btnSaveSettings');
    
    // State Variables
    let isCameraActive = false;
    let isListening = false;
    let cameraStream = null;
    let recognition = null;
    let avatarTimer = null;
    let lastFrameTimestamp = 0;
    
    // MediaPipe Variables
    let handLandmarker = null;
    let extractor = new LandmarkExtractor();
    let lastVideoTime = -1;
    let animationFrameId = null;
    let isMediaPipeReady = false;
    let isPredictionRequestPending = false;
    
    // Canvas Elements
    const landmarkCanvas = document.getElementById('landmarkCanvas');
    const canvasCtx = landmarkCanvas ? landmarkCanvas.getContext('2d') : null;
    const drawingUtils = canvasCtx ? new DrawingUtils(canvasCtx) : null;

    // Initialize timestamps on load to reflect current local time
    initializeTimestamps();
    updateChatCounter();

    // 1. Settings Threshold Slider
    sliderThreshold.addEventListener('input', (e) => {
        sliderThresholdVal.textContent = e.target.value + '%';
    });

    btnSaveSettings.addEventListener('click', () => {
        const selectedModelText = selectModel.options[selectModel.selectedIndex].text;
        barModelText.textContent = selectedModelText.split(' ')[0] + " (" + selectedModel.value.toUpperCase() + ")";
        showNotification("Settings applied. Model configured to " + selectedModel.value.toUpperCase());
    });

    // 2. Camera Controls Handlers
    btnStartRecognition.addEventListener('click', startCamera);
    btnStopRecognition.addEventListener('click', stopCamera);

    async function startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showNotification("Camera access is not supported by this browser.");
            return;
        }

        try {
            cameraStatusText.textContent = "Starting camera...";
            
            // Request real camera stream
            const constraints = {
                video: true,
                audio: false
            };
            
            cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            webcamVideo.srcObject = cameraStream;
            webcamVideo.classList.remove('d-none');
            cameraPlaceholder.classList.add('d-none');
            
            // Update States
            isCameraActive = true;
            btnStartRecognition.disabled = true;
            btnStopRecognition.disabled = false;
            
            // Camera Status Bar / Pill Updates
            cameraStatusDot.classList.add('active');
            cameraStatusText.textContent = "Camera active";
            barCameraDot.classList.add('active');
            barCameraText.textContent = "Connected";
            barCameraText.className = "text-success";
            
            // Avatar status goes online when camera is on
            avatarStatusDot.classList.add('active');
            avatarStatusText.textContent = "Online";
            barAvatarDot.classList.add('active');
            barAvatarText.textContent = "Online";
            barAvatarText.className = "text-success";
            avatarImage.classList.remove('d-none');
            avatarPlaceholder.classList.add('d-none');
            avatarStateBadge.textContent = "ACTIVE";
            avatarStateBadge.className = "badge bg-success bg-opacity-25 text-success border border-success border-opacity-25";
            
            // Initialize MediaPipe and Start Processing Loop
            await initializeMediaPipe();
            
            // Show canvas
            if (landmarkCanvas) {
                landmarkCanvas.width = webcamVideo.clientWidth || 640;
                landmarkCanvas.height = webcamVideo.clientHeight || 360;
                landmarkCanvas.classList.remove('d-none');
            }
            
            lastVideoTime = -1;
            // Wait for video to be ready before drawing
            if (webcamVideo.readyState >= 2) {
                renderLoop();
                scheduleNextVideoFrame();
            } else {
                webcamVideo.addEventListener("loadeddata", () => {
                    renderLoop();
                    scheduleNextVideoFrame();
                }, { once: true });
            }
            
        } catch (err) {
            let errorMsg = "Unable to access the camera.";
            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                errorMsg = "Camera permission was denied.";
            } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                errorMsg = "No camera was found.";
            } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
                errorMsg = "The camera is currently unavailable.";
            } else if (err.name === "SecurityError") {
                errorMsg = "Camera access is blocked by security settings.";
            }
            showNotification(errorMsg);
            
            cameraStatusText.textContent = "Camera Off";
        }
    }

    function stopCamera() {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
        }
        webcamVideo.srcObject = null;
        webcamVideo.classList.add('d-none');
        cameraPlaceholder.classList.remove('d-none');
        
        if (landmarkCanvas) {
            landmarkCanvas.classList.add('d-none');
            if (canvasCtx) {
                canvasCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
            }
        }
        
        // Reset Backend Prediction State
        fetch('/predict/reset', { method: 'POST' }).catch(err => console.error("Reset failed", err));
        
        // Reset UI Outputs
        signOutput.textContent = "Waiting...";
        confidenceOutput.textContent = "--";
        confidenceProgress.style.width = "0%";
        fpsOutput.textContent = "0.0";
        if (handsDetectedText) handsDetectedText.textContent = "Hands detected: 0";
        isPredictionRequestPending = false;
        
        // Update States
        isCameraActive = false;
        btnStartRecognition.disabled = false;
        btnStopRecognition.disabled = true;
        
        // Camera Status Bar / Pill Updates
        cameraStatusDot.classList.remove('active');
        cameraStatusText.textContent = "Camera stopped";
        barCameraDot.classList.remove('active', 'warning');
        barCameraText.textContent = "Disconnected";
        barCameraText.className = "text-muted";
        
        // Avatar Status Bar / Pill Offline
        avatarStatusDot.classList.remove('active');
        avatarStatusText.textContent = "Offline";
        barAvatarDot.classList.remove('active');
        barAvatarText.textContent = "Offline";
        barAvatarText.className = "text-muted";
        avatarImage.classList.add('d-none');
        avatarPlaceholder.classList.remove('d-none');
        avatarStateBadge.textContent = "IDLE";
        avatarStateBadge.className = "badge bg-purple bg-opacity-25 text-purple border border-purple border-opacity-25";
        avatarDetailedStatus.textContent = "Waiting for input...";
    }

    async function initializeMediaPipe() {
        if (handLandmarker) {
            isMediaPipeReady = true;
            return;
        }
        
        try {
            showNotification("Loading MediaPipe HandLandmarker...");
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
            );
            
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "/models/mediapipe/hand_landmarker.task",
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                numHands: 2,
                minHandDetectionConfidence: 0.5,
                minHandPresenceConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            
            isMediaPipeReady = true;
            showNotification("MediaPipe HandLandmarker loaded successfully.");
        } catch (error) {
            console.error("Failed to load MediaPipe:", error);
            showNotification("Failed to load HandLandmarker model.");
        }
    }
    
    let latestResults = null;
    let isDetecting = false;
    let rVFCId = null;

    function renderLoop() {
        if (!isCameraActive) return;
        
        if (landmarkCanvas && webcamVideo.videoWidth) {
            if (landmarkCanvas.width !== webcamVideo.videoWidth) {
                landmarkCanvas.width = webcamVideo.videoWidth;
            }
            if (landmarkCanvas.height !== webcamVideo.videoHeight) {
                landmarkCanvas.height = webcamVideo.videoHeight;
            }
        }
        
        if (canvasCtx && landmarkCanvas) {
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
            
            if (latestResults && latestResults.landmarks) {
                for (const landmarks of latestResults.landmarks) {
                    drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
                        color: "#bc34fa",
                        lineWidth: 3
                    });
                    drawingUtils.drawLandmarks(landmarks, {
                        color: "#0dcaf0",
                        lineWidth: 2,
                        radius: 3
                    });
                }
            }
            canvasCtx.restore();
        }
        
        animationFrameId = window.requestAnimationFrame(renderLoop);
    }

    async function onVideoFrame(now, metadata) {
        if (!isCameraActive || !isMediaPipeReady) return;
        
        // Drop stale frames: only process if we are not already busy processing a previous frame
        if (!isDetecting) {
            isDetecting = true;
            
            let startTimeMs = metadata ? metadata.presentationTime : performance.now();
            
            if (lastFrameTimestamp > 0) {
                const elapsed = startTimeMs - lastFrameTimestamp;
                if (elapsed > 0) {
                    const fps = 1000 / elapsed;
                    fpsOutput.textContent = fps.toFixed(1);
                }
            }
            lastFrameTimestamp = startTimeMs;
            
            // Yield to the browser's event loop via setTimeout(0)
            // This ensures the browser has a chance to visually paint this video frame to the screen
            // BEFORE we block the main thread with MediaPipe's synchronous detectForVideo.
            await new Promise(resolve => setTimeout(resolve, 0));
            
            if (isCameraActive) {
                // Synchronously process the exact frame presentation time
                latestResults = handLandmarker.detectForVideo(webcamVideo, startTimeMs);
                
                const extracted = extractor.extract(latestResults);
                
                if (handsDetectedText) {
                    handsDetectedText.textContent = `Hands detected: ${extracted.handsDetected}`;
                }
                
                if (!isPredictionRequestPending) {
                    isPredictionRequestPending = true;
                    
                    fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ features: Array.from(extracted.features) })
                    })
                    .then(res => {
                        if (!res.ok) throw new Error("Backend error " + res.status);
                        return res.json();
                    })
                    .then(data => {
                        isPredictionRequestPending = false;
                        if (data.sequence_ready) {
                            signOutput.textContent = data.predicted_class;
                            const confPct = (data.confidence * 100).toFixed(2);
                            confidenceOutput.textContent = confPct + "%";
                            confidenceProgress.style.width = confPct + "%";
                        } else {
                            signOutput.textContent = "Collecting frames...";
                            confidenceOutput.textContent = "--";
                            confidenceProgress.style.width = "0%";
                        }
                    })
                    .catch(err => {
                        isPredictionRequestPending = false;
                        console.error("Prediction error:", err);
                        signOutput.textContent = "Prediction service unavailable.";
                        confidenceOutput.textContent = "--";
                    });
                }
            }
            
            isDetecting = false;
        }
        
        scheduleNextVideoFrame();
    }
    
    function scheduleNextVideoFrame() {
        if (!isCameraActive) return;
        
        if ('requestVideoFrameCallback' in webcamVideo) {
            rVFCId = webcamVideo.requestVideoFrameCallback(onVideoFrame);
        } else {
            // Fallback for older browsers
            rVFCId = requestAnimationFrame((now) => {
                if (lastVideoTime !== webcamVideo.currentTime) {
                    lastVideoTime = webcamVideo.currentTime;
                    onVideoFrame(now, { presentationTime: now });
                } else {
                    scheduleNextVideoFrame();
                }
            });
        }
    }

    // 3. Web Speech API (Microphone Transcription) Handler
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            isListening = true;
            btnSpeakText.textContent = "Listening...";
            btnSpeak.classList.add('active');
            btnSpeak.querySelector('i').classList.add('pulse-mic');
            
            barMicDot.className = "status-dot active";
            barMicText.textContent = "Listening";
            barMicText.className = "text-success";
        };

        recognition.onend = () => {
            isListening = false;
            btnSpeakText.textContent = "Start Speaking";
            btnSpeak.classList.remove('active');
            btnSpeak.querySelector('i').classList.remove('pulse-mic');
            
            barMicDot.className = "status-dot";
            barMicText.textContent = "Disconnected";
            barMicText.className = "text-muted";
        };

        recognition.onerror = (e) => {
            console.error("Speech Recognition Error:", e);
            isListening = false;
            btnSpeakText.textContent = "Start Speaking";
            btnSpeak.classList.remove('active');
            btnSpeak.querySelector('i').classList.remove('pulse-mic');
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            if (transcript.trim() !== '') {
                appendChatMessage("Hearing", transcript);
                // Trigger Avatar translation animation simulation
                simulateAvatarResponse(transcript);
            }
        };
    }

    btnSpeak.addEventListener('click', () => {
        if (!SpeechRecognition) {
            // Speech recognition not supported in browser, fallback to custom prompt dialog
            const customInput = prompt("Web Speech API is not supported in this browser. Enter simulated speech text to post as Hearing user:");
            if (customInput && customInput.trim() !== '') {
                appendChatMessage("Hearing", customInput);
                simulateAvatarResponse(customInput);
            }
            return;
        }

        if (!isListening) {
            // Set dynamic language from settings
            const selectLanguage = document.getElementById('selectLanguage');
            recognition.lang = selectLanguage.value;
            recognition.start();
        } else {
            recognition.stop();
        }
    });


    // 5. Avatar translation simulator
    function simulateAvatarResponse(sentence) {
        if (!isCameraActive) {
            // If camera/avatar isn't active, activate it silently
            toggleCamera().then(() => playAvatarSequence(sentence));
        } else {
            playAvatarSequence(sentence);
        }
    }

    function playAvatarSequence(sentence) {
        clearTimeout(avatarTimer);
        
        avatarStateBadge.textContent = "TRANSLATING";
        avatarStateBadge.className = "badge bg-info bg-opacity-25 text-info border border-info border-opacity-25";
        avatarDetailedStatus.textContent = `Translating: "${sentence}" into sign sequence...`;
        
        // Pulse avatar viewport borders
        const avatarContainer = document.getElementById('avatarContainer');
        avatarContainer.style.boxShadow = "0 0 25px rgba(188, 52, 250, 0.4)";
        
        avatarTimer = setTimeout(() => {
            avatarStateBadge.textContent = "ACTIVE";
            avatarStateBadge.className = "badge bg-success bg-opacity-25 text-success border border-success border-opacity-25";
            avatarDetailedStatus.textContent = "Avatar Idle. Waiting for input...";
            avatarContainer.style.boxShadow = "inset 0 0 30px rgba(0, 0, 0, 0.8)";
        }, 3500);
    }

    // 6. Chat and Messages Operations
    btnClear.addEventListener('click', () => {
        chatWindow.innerHTML = `
            <div class="d-flex flex-column align-items-center justify-content-center h-100 text-muted py-5" id="chatEmptyState">
                <i class="fa-regular fa-comment-dots fa-3x mb-3 text-secondary text-opacity-50"></i>
                <h6>No messages yet</h6>
                <p class="small text-center px-4">Start speaking or turn on the camera to begin translating conversations.</p>
            </div>
        `;
        updateChatCounter();
        showNotification("Conversation cleared.");
    });

    function appendChatMessage(sender, text) {
        // Remove empty state if present
        const emptyState = document.getElementById('chatEmptyState');
        if (emptyState) {
            emptyState.remove();
        }

        const now = new Date();
        const timeString = now.toTimeString().split(' ')[0].substring(0, 5); // HH:MM

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender.toLowerCase()}`;
        
        if (sender === "Deaf") {
            messageDiv.innerHTML = `
                <div class="message-avatar">🤟</div>
                <div>
                    <div class="message-bubble">
                        ${text}
                    </div>
                    <div class="message-meta">
                        <span>🤟 Deaf</span>
                        <span>•</span>
                        <span>${timeString}</span>
                    </div>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-avatar">🎤</div>
                <div>
                    <div class="message-bubble">
                        ${text}
                    </div>
                    <div class="message-meta">
                        <span>${timeString}</span>
                        <span>•</span>
                        <span>🎤 Hearing</span>
                    </div>
                </div>
            `;
        }

        chatWindow.appendChild(messageDiv);
        scrollChatToBottom();
        updateChatCounter();
    }

    function scrollChatToBottom() {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function updateChatCounter() {
        const messageCount = chatWindow.querySelectorAll('.chat-message').length;
        chatCounterBadge.textContent = `${messageCount} Message${messageCount !== 1 ? 's' : ''}`;
    }

    function initializeTimestamps() {
        const timeElements = chatWindow.querySelectorAll('.msg-time');
        const now = new Date();
        
        timeElements.forEach(el => {
            const offsetMinutes = parseInt(el.getAttribute('data-offset') || '0', 10);
            const timeVal = new Date(now.getTime() - offsetMinutes * 60000);
            el.textContent = timeVal.toTimeString().split(' ')[0].substring(0, 5);
        });
        scrollChatToBottom();
    }

    // Custom notification display utility
    function showNotification(message) {
        // Check if an alert toast already exists
        let toast = document.getElementById('systemToast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'systemToast';
            toast.className = 'toast align-items-center text-white bg-dark border border-info border-opacity-25 position-fixed top-0 start-50 translate-middle-x m-3 p-1';
            toast.style.zIndex = '1090';
            toast.style.borderRadius = '12px';
            toast.style.backdropFilter = 'blur(10px)';
            toast.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.5)';
            toast.innerHTML = `
                <div class="d-flex">
                    <div class="toast-body d-flex align-items-center gap-2">
                        <i class="fa-solid fa-circle-info text-info"></i>
                        <span id="toastMessage"></span>
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            `;
            document.body.appendChild(toast);
        }
        
        document.getElementById('toastMessage').textContent = message;
        const bsToast = new bootstrap.Toast(toast, { delay: 3000 });
        bsToast.show();
    }

    // 7. Page Cleanup
    window.addEventListener('beforeunload', () => {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
        }
    });
});
