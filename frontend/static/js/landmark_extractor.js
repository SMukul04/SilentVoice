/**
 * Landmark Extractor for SilentVoice Frontend
 * 
 * Replicates the exact Python feature extraction and normalization logic:
 * 1. Identifies Left vs Right hands
 * 2. Normalizes (translates wrist to origin, scales by reference distance)
 * 3. Constructs a 126-d feature vector: [Left Hand (63), Right Hand (63)]
 * 4. Fills missing hands with zeros.
 */

export class LandmarkExtractor {
    constructor() {
        this.FEATURE_DIMENSION = 126;
        this.HAND_DIMENSION = 63;
    }

    /**
     * Normalizes a single hand's 21 landmarks.
     * @param {Array} handLandmarks - Array of 21 objects {x, y, z}
     * @returns {Float32Array} Normalized 63-value array
     */
    _normalizeHand(handLandmarks) {
        if (!handLandmarks || handLandmarks.length !== 21) {
            return new Float32Array(this.HAND_DIMENSION);
        }

        const normalized = new Float32Array(this.HAND_DIMENSION);
        
        // 1. Translation: Move wrist (landmark 0) to origin
        const wrist = handLandmarks[0];
        const translated = [];
        for (let i = 0; i < 21; i++) {
            translated.push({
                x: handLandmarks[i].x - wrist.x,
                y: handLandmarks[i].y - wrist.y,
                z: handLandmarks[i].z - wrist.z
            });
        }

        // 2. Scaling: Calculate reference distance from wrist to middle MCP (landmark 9)
        // In translated coords, wrist is (0,0,0) so distance is just magnitude of landmark 9
        const middleMCP = translated[9];
        const refDistance = Math.sqrt(
            middleMCP.x * middleMCP.x + 
            middleMCP.y * middleMCP.y + 
            middleMCP.z * middleMCP.z
        );

        let scale = 1.0;
        if (refDistance > 1e-6) {
            scale = 1.0 / refDistance;
        }

        // 3. Apply scaling and flatten
        let idx = 0;
        for (let i = 0; i < 21; i++) {
            // Ensure values are finite
            let nx = translated[i].x * scale;
            let ny = translated[i].y * scale;
            let nz = translated[i].z * scale;
            
            if (!Number.isFinite(nx)) nx = 0.0;
            if (!Number.isFinite(ny)) ny = 0.0;
            if (!Number.isFinite(nz)) nz = 0.0;

            normalized[idx++] = nx;
            normalized[idx++] = ny;
            normalized[idx++] = nz;
        }

        return normalized;
    }

    /**
     * Extracts and flattens landmarks into the 126-d feature vector.
     * @param {Object} detectionResult - Result from MediaPipe HandLandmarker.detectForVideo
     * @returns {Object} Structured output with features and hand detection state
     */
    extract(detectionResult) {
        const result = {
            features: new Float32Array(this.FEATURE_DIMENSION),
            handsDetected: 0,
            hasHand: false,
            handednesses: []
        };

        if (!detectionResult || !detectionResult.landmarks || detectionResult.landmarks.length === 0) {
            result.handsDetected = 0;
            result.hasHand = false;
            return result;
        }
        
        result.hasHand = true;

        let leftHandLandmarks = null;
        let rightHandLandmarks = null;

        const numHands = Math.min(detectionResult.landmarks.length, 2); // Max 2 hands
        result.handsDetected = numHands;

        for (let i = 0; i < numHands; i++) {
            const landmarks = detectionResult.landmarks[i];
            
            // Note: MediaPipe returns "Right" or "Left" based on the camera mirror logic.
            // Our dataset expects Left hand in the first 63 features, Right hand in the next 63.
            let handednessLabel = "Right";
            if (detectionResult.handednesses && detectionResult.handednesses[i]) {
                const category = detectionResult.handednesses[i][0].category;
                if (category === "Left" || category === "Right") {
                    handednessLabel = category;
                }
            }

            result.handednesses.push(handednessLabel);

            if (handednessLabel === "Left") {
                leftHandLandmarks = landmarks;
            } else {
                rightHandLandmarks = landmarks;
            }
        }

        // Normalize Left Hand
        const leftNorm = this._normalizeHand(leftHandLandmarks);
        // Normalize Right Hand
        const rightNorm = this._normalizeHand(rightHandLandmarks);

        // Construct final 126-d vector [left(63), right(63)]
        result.features.set(leftNorm, 0);
        result.features.set(rightNorm, this.HAND_DIMENSION);

        return result;
    }
}
