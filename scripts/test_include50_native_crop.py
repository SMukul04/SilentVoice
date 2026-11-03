import cv2
from pathlib import Path

from backend.dataset.frame_sampler import FrameSampler
from backend.sign_recognition.mediapipe_detector import MediaPipeDetector

VIDEOS = [
    Path("datasets/include50/videos/Seasons/64. Fall/MVI_4999.MOV"),
    Path("datasets/include50/videos/Seasons/64. Fall/MVI_5000.MOV"),
    Path("datasets/include50/videos/Places/28. Store or Shop/MVI_3387.MOV"),
]

sampler = FrameSampler(sequence_length=32)

def evaluate(detector, video, crop=False):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    indices = [
        int(p.name)
        for p in sampler.sample(
            [Path(str(i)) for i in range(total)]
        )
    ]

    detected = 0
    two_hands = 0

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()

        if not ok:
            continue

        if crop:
            h, w = frame.shape[:2]
            frame = frame[:, w // 4: 3 * w // 4]

        timestamp = int(index * 1000 / fps) if fps > 0 else index
        detector._timestamp_ms = max(
            detector._timestamp_ms,
            timestamp - 1
        )

        result = detector.detect(frame)
        detected += result["num_hands"] > 0
        two_hands += result["num_hands"] == 2

    cap.release()
    return detected, two_hands

for video in VIDEOS:
    print("\nVIDEO:", video.name)

    with MediaPipeDetector() as detector:
        original = evaluate(detector, video, crop=False)

    with MediaPipeDetector() as detector:
        cropped = evaluate(detector, video, crop=True)

    print(f"Original: {original[0]}/32 detected, {original[1]}/32 two hands")
    print(f"Native crop: {cropped[0]}/32 detected, {cropped[1]}/32 two hands")

print("\nEXPERIMENT COMPLETE")
