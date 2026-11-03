import cv2
from pathlib import Path

videos = [
    Path("datasets/include50/videos/Seasons/64. Fall/MVI_4999.MOV"),
    Path("datasets/include50/videos/Seasons/64. Fall/MVI_5000.MOV"),
    Path("datasets/include50/videos/Places/28. Store or Shop/MVI_3387.MOV"),
]

for path in videos:
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        print("Cannot open:", path)
        continue

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\nVideo:", path.name)
    print("Resolution:", width, "x", height)
    print("FPS:", fps)
    print("Total frames:", frames)
    print("Duration:", round(frames / fps, 2) if fps > 0 else "Unknown")

    cap.release()
