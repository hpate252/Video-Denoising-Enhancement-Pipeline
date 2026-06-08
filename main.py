import cv2
import time
import torch
import numpy as np
from tqdm import tqdm

from models.denoising_cnn import DenoisingCNN
from enhance import enhance_frame
from metrics import calculate_metrics


def frame_to_tensor(frame, device):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb.astype(np.float32) / 255.0

    tensor = torch.from_numpy(frame_normalized)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return tensor.to(device)


def tensor_to_frame(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)

    frame_bgr = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)

    return frame_bgr


def process_video(noisy_video_path, clean_video_path, output_video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenoisingCNN().to(device)
    model.eval()

    noisy_cap = cv2.VideoCapture(noisy_video_path)
    clean_cap = cv2.VideoCapture(clean_video_path)

    if not noisy_cap.isOpened():
        raise RuntimeError("Could not open noisy video.")

    if not clean_cap.isOpened():
        raise RuntimeError("Could not open clean video.")

    width = int(noisy_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(noisy_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = noisy_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(noisy_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    psnr_scores = []
    ssim_scores = []
    latency_scores = []

    with torch.no_grad():
        for _ in tqdm(range(total_frames)):
            noisy_ret, noisy_frame = noisy_cap.read()
            clean_ret, clean_frame = clean_cap.read()

            if not noisy_ret:
                break

            if not clean_ret:
                break

            clean_frame = cv2.resize(clean_frame, (width, height))

            input_tensor = frame_to_tensor(noisy_frame, device)

            start_time = time.time()

            output_tensor = model(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()

            latency = (end_time - start_time) * 1000
            latency_scores.append(latency)

            denoised_frame = tensor_to_frame(output_tensor)
            enhanced_frame = enhance_frame(denoised_frame)

            writer.write(enhanced_frame)

            psnr, ssim = calculate_metrics(clean_frame, enhanced_frame)
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

    noisy_cap.release()
    clean_cap.release()
    writer.release()

    avg_latency = np.mean(latency_scores)
    avg_fps = 1000.0 / avg_latency

    print("Processing complete.")
    print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Average Latency: {avg_latency:.2f} ms/frame")
    print(f"Estimated Inference FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    process_video(
        noisy_video_path="input/noisy_video.mp4",
        clean_video_path="input/clean_video.mp4",
        output_video_path="output/denoised_video.mp4"
    )