# Frame-Level Video Denoising and Enhancement Pipeline

This project implements a video denoising and enhancement pipeline using PyTorch and OpenCV. The system extracts frames from noisy video input, applies a CNN-based denoising model, enhances contrast using CLAHE, reconstructs the output video, and evaluates visual quality using PSNR and SSIM. It also profiles inference latency per frame to assess suitability for edge deployment.

This video denoising and enhancement pipeline connects to my AI/ML research work on MOSAIC, a multi-agent AI coaching system focused on sensing-grounded personalization for social health. MOSAIC integrates behavioral sensing, EMA/self-report data, memory retrieval, GROW-based coaching, summarization, and adaptive response generation to support personalized AI coaching conversations.

While MOSAIC primarily works with behavioral and user-profile data, this project explores a related sensing pipeline from the computer vision side: extracting frame-level information from video, enhancing noisy visual input, evaluating output quality, and profiling inference latency for deployment feasibility. The same engineering ideas used here—data preprocessing, model inference, evaluation metrics, latency profiling, and deployment-aware optimization—are directly applicable to MOSAIC’s broader goal of building reliable AI systems that can process real-world multimodal data.

In MOSAIC, behavioral sensing data such as sleep, social interaction frequency, mobility, EMA responses, and user history are used to build a personalized understanding of the user. The MOSAIC paper describes the system as a multi-agent architecture that assembles user context from behavioral sensing, self-report, and conversational history to support personalized GROW-based coaching. This project extends that mindset to video data by treating each frame as a sensor-derived input that must be cleaned, enhanced, evaluated, and processed efficiently before it can be useful for downstream AI tasks.

This project also supports the deployment-oriented side of MOSAIC research. MOSAIC requires efficient AI components for profiling, memory retrieval, summarization, adaptive response generation, and evaluation. Similarly, this pipeline measures PSNR, SSIM, and per-frame inference latency to understand whether a PyTorch-based model can run efficiently on limited hardware. The profiling results can inform future work on quantization, pruning, model compression, and edge deployment, which are also relevant to running smaller transformer-based or multimodal models in real-world health AI systems.

Overall, this project demonstrates practical ML system-building skills that support MOSAIC research: building modular pipelines, processing noisy real-world data, evaluating model quality, measuring runtime performance, and preparing models for deployment-aware AI applications.

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
