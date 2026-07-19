"""Standalone verification script for testing LandmarkNormalizer two-hand support."""

import os
import sys
import numpy as np

# Add the workspace root to sys.path to resolve backend imports
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from backend.sign_recognition.hand_features import HandFeatures
from backend.sign_recognition.frame_features import FrameFeatures
from backend.sign_recognition.normalizer import LandmarkNormalizer


def verify_configuration(name: str, frame_features: FrameFeatures, normalizer: LandmarkNormalizer) -> None:
    """Verifies a single FrameFeatures configuration, printing shapes, sizes, and values."""
    print("=" * 60)
    print(f"VERIFYING CONFIGURATION: {name}")
    print("=" * 60)
    
    # Run normalization
    output = normalizer.normalize(frame_features)
    
    # Split the output into left and right halves
    left_half = output[:63]
    right_half = output[63:]
    
    # Count non-zero values
    left_non_zero = np.count_nonzero(left_half)
    right_non_zero = np.count_nonzero(right_half)
    
    # Print the requested statistics
    print(f"Output shape                     : {output.shape}")
    print(f"Left half size                   : {left_half.shape[0]}")
    print(f"Right half size                  : {right_half.shape[0]}")
    print(f"Number of non-zero values (Left) : {left_non_zero}")
    print(f"Number of non-zero values (Right): {right_non_zero}")
    
    # Print first 10 values of each half formatted cleanly
    left_first_10 = [f"{val:.4f}" for val in left_half[:10]]
    right_first_10 = [f"{val:.4f}" for val in right_half[:10]]
    
    print(f"First 10 values (Left)           : [{', '.join(left_first_10)}]")
    print(f"First 10 values (Right)          : [{', '.join(right_first_10)}]")
    print()


def main() -> None:
    normalizer = LandmarkNormalizer()
    
    # Mock data setup
    left_coords = np.arange(63, dtype=np.float32) + 1.0 # Sequential values so math is predictable
    # Set wrist at coords (0,1,2) to [1, 1, 1], and Middle MCP at coords (27, 28, 29) to [1, 5, 1] for reference distance 4
    left_coords[0:3] = [1.0, 1.0, 1.0]
    left_coords[27:30] = [1.0, 5.0, 1.0]
    
    right_coords = np.arange(63, dtype=np.float32) + 100.0
    right_coords[0:3] = [2.0, 2.0, 2.0]
    right_coords[27:30] = [2.0, 6.0, 2.0]
    
    left_hand = HandFeatures(
        landmarks=left_coords,
        handedness="Left",
        confidence=0.99
    )
    
    right_hand = HandFeatures(
        landmarks=right_coords,
        handedness="Right",
        confidence=0.95
    )
    
    # 1. No Hands
    verify_configuration("No Hands", FrameFeatures(left_hand=None, right_hand=None), normalizer)
    
    # 2. Left Hand Only
    verify_configuration("Left Hand Only", FrameFeatures(left_hand=left_hand, right_hand=None), normalizer)
    
    # 3. Right Hand Only
    verify_configuration("Right Hand Only", FrameFeatures(left_hand=None, right_hand=right_hand), normalizer)
    
    # 4. Both Hands
    verify_configuration("Both Hands", FrameFeatures(left_hand=left_hand, right_hand=right_hand), normalizer)


if __name__ == "__main__":
    main()
