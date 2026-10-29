/**
 * SilentVoice Avatar Web Adapter
 * Handles serialization, event routing, and communication with the Unity WebGL build.
 */

class AvatarWebAdapter {
    constructor(unityInstanceName = 'unityInstance', gameObjectName = 'WebGLBridge') {
        this.unityInstanceName = unityInstanceName;
        this.gameObjectName = gameObjectName;
        this.unityInstance = null;
        this.isReady = false;

        // Listeners for avatar events
        this.eventListeners = [];

        // Global callback for Unity -> JS
        window.onAvatarEvent = this.handleUnityEvent.bind(this);
    }

    /**
     * Call this when the Unity WebGL build has finished loading.
     */
    initialize(unityInstance) {
        this.unityInstance = unityInstance;
        this.isReady = true;
        console.log("[AvatarWebAdapter] Initialized with Unity instance.");
    }

    addEventListener(callback) {
        this.eventListeners.push(callback);
    }

    handleUnityEvent(eventJson) {
        try {
            const evt = JSON.parse(eventJson);
            console.debug("[AvatarWebAdapter] Received event:", evt);
            this.eventListeners.forEach(cb => cb(evt));
        } catch (e) {
            console.error("[AvatarWebAdapter] Failed to parse Unity event:", eventJson, e);
        }
    }

    _sendCommand(commandObj) {
        if (!this.isReady || !this.unityInstance) {
            console.error("[AvatarWebAdapter] Cannot send command; Unity is not ready.");
            return;
        }

        try {
            const jsonStr = JSON.stringify(commandObj);
            this.unityInstance.SendMessage(this.gameObjectName, 'ReceiveCommand', jsonStr);
            console.debug("[AvatarWebAdapter] Sent command:", jsonStr);
        } catch (e) {
            console.error("[AvatarWebAdapter] Failed to send command to Unity:", e);
        }
    }

    /**
     * API: Play a sign
     */
    playSign(signId) {
        this._sendCommand({
            command: "play",
            sign_id: signId
        });
    }

    /**
     * API: Stop the avatar
     */
    stop() {
        this._sendCommand({
            command: "stop"
        });
    }

    /**
     * API: Reset the avatar state
     */
    reset() {
        this._sendCommand({
            command: "reset"
        });
    }

    /**
     * API: Request current status
     */
    requestStatus() {
        this._sendCommand({
            command: "status"
        });
    }
}

// Export for module systems, or attach to window for classic scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AvatarWebAdapter;
} else {
    window.AvatarWebAdapter = AvatarWebAdapter;
}
