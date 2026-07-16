"""Configuration settings for the SilentVoice project."""

# MediaPipe Hand Detector settings
HAND_STATIC_IMAGE_MODE: bool = False  # False optimizes for tracking across sequential frames
HAND_MAX_NUM_HANDS: int = 2          # Maximum number of hands to detect
HAND_MODEL_COMPLEXITY: int = 1        # 0 or 1. 1 is higher accuracy, 0 is lower latency
HAND_MIN_DETECTION_CONFIDENCE: float = 0.5
HAND_MIN_TRACKING_CONFIDENCE: float = 0.5

# Handedness configuration
SWAP_HANDEDNESS: bool = True  # Swap Left/Right labels to correct for non-mirrored webcam feed

