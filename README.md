# Frame-Level Video Denoising and Enhancement Pipeline

This project implements a video denoising and enhancement pipeline using PyTorch and OpenCV. The system extracts frames from noisy video input, applies a CNN-based denoising model, enhances contrast using CLAHE, reconstructs the output video, and evaluates visual quality using PSNR and SSIM. It also profiles inference latency per frame to assess suitability for edge deployment.

## Features

- Frame-by-frame video processing
- PyTorch CNN denoising model
- OpenCV video I/O
- CLAHE-based contrast enhancement
- PSNR and SSIM quality evaluation
- Inference latency profiling
- Edge deployment analysis

Output:
output:
<img width="1595" height="313" alt="image" src="https://github.com/user-attachments/assets/17256654-0c77-4e7a-b83b-91682a884054" />

## Run

```bash
python main.py
