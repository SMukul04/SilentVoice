import json

class PredictionStabilizer:
    def __init__(self):
        self.CONFIDENCE_THRESHOLD = 0.60
        self.HISTORY_SIZE = 5
        self.MIN_CONSISTENT_PREDICTIONS = 3
        self.CONFIRMED_PREDICTIONS = 3

        self.history = []
        self.stablePrediction = None
        self.stablePredictionCount = 0
        self.confirmedPrediction = None
        self.lastConfirmedPrediction = None

    def addPrediction(self, predictedClass, confidence):
        isNewConfirmation = False
        
        if confidence >= self.CONFIDENCE_THRESHOLD:
            self.history.append({'class': predictedClass, 'confidence': confidence})
            if len(self.history) > self.HISTORY_SIZE:
                self.history.pop(0)

            counts = {}
            candidateStable = None
            candidateMax = 0
            
            for item in self.history:
                cls = item['class']
                counts[cls] = counts.get(cls, 0) + 1
                if counts[cls] > candidateMax:
                    candidateMax = counts[cls]
                    candidateStable = cls

            previousStable = self.stablePrediction
            
            if candidateMax >= self.MIN_CONSISTENT_PREDICTIONS:
                self.stablePrediction = candidateStable
            else:
                self.stablePrediction = None

            if self.stablePrediction != previousStable:
                self.stablePredictionCount = 1 if self.stablePrediction else 0
            elif self.stablePrediction:
                self.stablePredictionCount += 1

            if self.stablePrediction and self.stablePredictionCount >= self.CONFIRMED_PREDICTIONS:
                self.confirmedPrediction = self.stablePrediction
                
                if self.confirmedPrediction != self.lastConfirmedPrediction:
                    isNewConfirmation = True
                    self.lastConfirmedPrediction = self.confirmedPrediction

        return {
            'rawPrediction': predictedClass,
            'stablePrediction': self.stablePrediction,
            'confirmedPrediction': self.confirmedPrediction,
            'isNewConfirmation': isNewConfirmation,
            'stablePredictionCount': self.stablePredictionCount
        }

stab = PredictionStabilizer()

print("--- ALIVE ---")
for i in range(5):
    res = stab.addPrediction("ALIVE", 0.9)
    print(f"frame {i}: newConf={res['isNewConfirmation']}, conf={res['confirmedPrediction']}, stable={res['stablePrediction']}, count={res['stablePredictionCount']}")

print("--- STRONG ---")
for i in range(5):
    res = stab.addPrediction("STRONG", 0.9)
    print(f"frame {i}: newConf={res['isNewConfirmation']}, conf={res['confirmedPrediction']}, stable={res['stablePrediction']}, count={res['stablePredictionCount']}")
