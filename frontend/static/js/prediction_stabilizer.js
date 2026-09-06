/**
 * Module 8.5: Temporal Stabilization & Sign Confirmation
 * 
 * Filters noisy predictions from the backend and emits confirmed sign events.
 */

export class PredictionStabilizer {
    constructor() {
        this.CONFIDENCE_THRESHOLD = 0.60;
        this.HISTORY_SIZE = 5;
        this.MIN_CONSISTENT_PREDICTIONS = 3;
        this.CONFIRMED_PREDICTIONS = 3;

        this.history = [];
        this.stablePrediction = null;
        this.stablePredictionCount = 0;
        this.confirmedPrediction = null;
        this.lastConfirmedPrediction = null;
    }

    /**
     * Feeds a raw prediction into the stabilizer.
     * @param {string} predictedClass - The predicted sign label.
     * @param {number} confidence - The raw confidence value.
     * @returns {Object} The current stabilization state.
     */
    addPrediction(predictedClass, confidence) {
        let isNewConfirmation = false;
        
        // 1. Threshold filter
        if (confidence >= this.CONFIDENCE_THRESHOLD) {
            // 2. Add to history
            this.history.push({ class: predictedClass, confidence: confidence });
            if (this.history.length > this.HISTORY_SIZE) {
                this.history.shift(); // Remove oldest
            }

            // 3. Determine if stable
            // Count occurrences of each class in history
            const counts = {};
            let candidateStable = null;
            let candidateMax = 0;
            
            for (const item of this.history) {
                counts[item.class] = (counts[item.class] || 0) + 1;
                if (counts[item.class] > candidateMax) {
                    candidateMax = counts[item.class];
                    candidateStable = item.class;
                }
            }

            // Check if max count meets threshold for stability
            let previousStable = this.stablePrediction;
            
            if (candidateMax >= this.MIN_CONSISTENT_PREDICTIONS) {
                this.stablePrediction = candidateStable;
            } else {
                this.stablePrediction = null;
            }

            // 4. Confirmation logic
            if (this.stablePrediction !== previousStable) {
                // Stable prediction changed, reset consecutive confirmation count
                this.stablePredictionCount = this.stablePrediction ? 1 : 0;
            } else if (this.stablePrediction) {
                // Stable prediction stayed the same
                this.stablePredictionCount++;
            }

            // If consecutive updates reach the requirement, trigger confirmation
            if (this.stablePrediction && this.stablePredictionCount >= this.CONFIRMED_PREDICTIONS) {
                this.confirmedPrediction = this.stablePrediction;
                
                // Prevent duplicate confirmations
                if (this.confirmedPrediction !== this.lastConfirmedPrediction) {
                    isNewConfirmation = true;
                    this.lastConfirmedPrediction = this.confirmedPrediction;
                }
            }
        }

        // Return state
        return {
            rawPrediction: predictedClass,
            rawConfidence: confidence,
            stablePrediction: this.stablePrediction,
            stableConfidence: this.calculateStableConfidence(),
            confirmedPrediction: this.confirmedPrediction,
            isNewConfirmation: isNewConfirmation
        };
    }

    /**
     * Calculates the average confidence of the current stable prediction.
     */
    calculateStableConfidence() {
        if (!this.stablePrediction) return null;
        
        let sum = 0;
        let count = 0;
        for (const item of this.history) {
            if (item.class === this.stablePrediction) {
                sum += item.confidence;
                count++;
            }
        }
        return count > 0 ? (sum / count) : null;
    }

    /**
     * Resets the temporal state. 
     * Call this when recognition stops or no hands are detected.
     */
    reset() {
        this.history = [];
        this.stablePrediction = null;
        this.stablePredictionCount = 0;
        this.confirmedPrediction = null;
        this.lastConfirmedPrediction = null;
    }
}
