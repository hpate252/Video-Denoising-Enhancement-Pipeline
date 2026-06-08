import cv2
import numpy as np
import os


def add_gaussian_noise(frame, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, frame.shape).astype(np.float32)
    noisy_frame = frame.astype(np.float32) + noise
    noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
    return noisy_frame


input_path = "input/clean_video.mp4"
output_path = "input/noisy_video.mp4"

if not os.path.exists(input_path):
    raise FileNotFoundError("input/clean_video.mp4 not found.")

cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    raise RuntimeError("Could not open clean_video.mp4. Use a valid MP4 video.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Width:", width)
print("Height:", height)
print("FPS:", fps)
print("Frames:", total_frames)

if width == 0 or height == 0 or fps == 0 or total_frames == 0:
    raise RuntimeError("Invalid video metadata. Try another MP4 file.")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not writer.isOpened():
    raise RuntimeError("Could not create noisy_video.mp4.")

frame_count = 0

max_frames = 100
frame_count = 0

while frame_count < max_frames:
    ret, frame = cap.read()

    if not ret:
        break

    noisy_frame = add_gaussian_noise(frame)
    writer.write(noisy_frame)
    frame_count += 1

cap.release()
writer.release()

print(f"Noisy video created: {output_path}")
print(f"Frames written: {frame_count}")