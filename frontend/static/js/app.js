/**
 * SilentVoice Frontend Dashboard Prototype Logic
 * Handles real-time video, Speech Recognition (with fallback), simulation routines, and state indicators.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const btnCamera = document.getElementById('btnCamera');
    const btnCameraText = document.getElementById('btnCameraText');
    const btnSpeak = document.getElementById('btnSpeak');
    const btnSpeakText = document.getElementById('btnSpeakText');
    const btnClear = document.getElementById('btnClear');
    
    // Status Bar & Pill Elements
    const cameraStatusBadge = document.getElementById('cameraStatusBadge');
    const cameraStatusDot = document.getElementById('cameraStatusDot');
    const cameraStatusText = document.getElementById('cameraStatusText');
    
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
    let fpsInterval = null;
    let simulationInterval = null;
    let avatarTimer = null;
    
    // Simulated gesture glossary for Deaf communication
    const simulatedGestures = [
        "Hello",
        "Thank you",
        "Nice to meet you",
        "How are you",
        "Yes",
        "No",
        "I need help",
        "SilentVoice",
        "Awesome",
        "I love you"
    ];

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

    // 2. Camera Toggle Handler
    btnCamera.addEventListener('click', toggleCamera);

    async function toggleCamera() {
        if (!isCameraActive) {
            try {
                // Request real camera stream
                const constraints = {
                    video: { width: 640, height: 360, facingMode: "user" }
                };
                
                cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
                webcamVideo.srcObject = cameraStream;
                webcamVideo.classList.remove('d-none');
                cameraPlaceholder.classList.add('d-none');
                
                // Update States
                isCameraActive = true;
                btnCameraText.textContent = "Stop Camera";
                btnCamera.classList.add('active');
                
                // Camera Status Bar / Pill Updates
                cameraStatusDot.classList.add('active');
                cameraStatusText.textContent = "Live Feed";
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
                
                // Start Simulated Recognition Stats Loop
                startSimulation();
                showNotification("Camera feed and 3D Avatar connected successfully.");
                
            } catch (err) {
                console.error("Camera access failed:", err);
                // Graceful fallback for browsers/environments without webcam
                alert("Could not access camera (or permission denied). Running simulated video feed placeholder.");
                
                // Simulate Active Camera anyway for demonstration
                isCameraActive = true;
                btnCameraText.textContent = "Stop Camera";
                btnCamera.classList.add('active');
                
                cameraStatusDot.classList.add('active');
                cameraStatusText.textContent = "Simulated Camera";
                barCameraDot.classList.add('warning');
                barCameraText.textContent = "Simulated";
                barCameraText.className = "text-warning";
                
                // Avatar status goes online
                avatarStatusDot.classList.add('active');
                avatarStatusText.textContent = "Online";
                barAvatarDot.classList.add('active');
                barAvatarText.textContent = "Online";
                barAvatarText.className = "text-success";
                avatarImage.classList.remove('d-none');
                avatarPlaceholder.classList.add('d-none');
                avatarStateBadge.textContent = "ACTIVE";
                avatarStateBadge.className = "badge bg-success bg-opacity-25 text-success border border-success border-opacity-25";
                
                startSimulation();
            }
        } else {
            // Stop Camera
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
            }
            webcamVideo.srcObject = null;
            webcamVideo.classList.add('d-none');
            cameraPlaceholder.classList.remove('d-none');
            
            // Update States
            isCameraActive = false;
            btnCameraText.textContent = "Start Camera";
            btnCamera.classList.remove('active');
            
            // Camera Status Bar / Pill Updates
            cameraStatusDot.classList.remove('active');
            cameraStatusText.textContent = "Camera Off";
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
            
            stopSimulation();
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

    // 4. Live Simulation loop for Left Panel metrics and gestures
    function startSimulation() {
        // Fluctuate FPS and Confidence values
        fpsInterval = setInterval(() => {
            const randomFps = (29.2 + Math.random() * 1.3).toFixed(1);
            const randomConfidence = Math.floor(94 + Math.random() * 6);
            
            fpsOutput.textContent = randomFps;
            confidenceOutput.textContent = randomConfidence + "%";
            confidenceProgress.style.width = randomConfidence + "%";
        }, 300);

        // Every 8 seconds, simulate gesture recognition and append to chat
        simulationInterval = setInterval(() => {
            const randomGestureIndex = Math.floor(Math.random() * simulatedGestures.length);
            const recognizedSign = simulatedGestures[randomGestureIndex];
            
            // Set recognized sign
            signOutput.textContent = recognizedSign;
            signOutput.classList.add('glow-text-blue');
            setTimeout(() => {
                signOutput.classList.remove('glow-text-blue');
            }, 1000);

            // Append "Deaf" sign message to conversation
            appendChatMessage("Deaf", recognizedSign);
        }, 7500);
    }

    function stopSimulation() {
        clearInterval(fpsInterval);
        clearInterval(simulationInterval);
        
        // Reset metrics outputs
        signOutput.textContent = "--";
        confidenceOutput.textContent = "0%";
        confidenceProgress.style.width = "0%";
        fpsOutput.textContent = "0.0";
    }

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
});
